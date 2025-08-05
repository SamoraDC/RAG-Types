import os
import tempfile
import uuid
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from datetime import datetime
import time

# Carregando variáveis de ambiente
load_dotenv()

# Configuração da página Streamlit
st.set_page_config(
    page_title="Híbrido RAG - Sistema de Consulta Avançado",
    page_icon="🔍",
    layout="wide"
)

# Função para inicializar a sessão
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_loaded" not in st.session_state:
        st.session_state.document_loaded = False
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = {}

# Inicializar sessão
init_session_state()

# Função para processamento de PDF com RAG híbrido
def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Carregamento do PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        # Dividir texto em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # RETRIEVER 1: Criar vetores e armazenar no FAISS (busca semântica)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        vector_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # RETRIEVER 2: Configurar BM25 para busca léxica (baseada em keywords)
        texts = [doc.page_content for doc in splits]
        bm25_retriever = BM25Retriever.from_texts(texts=texts)
        bm25_retriever.k = 4
        
        # Converter documentos do BM25 para o formato adequado
        def format_docs(docs):
            return [
                Document(
                    page_content=doc.page_content,
                    metadata={"source": f"chunk_{i}", "score": doc.metadata.get("score", 0)}
                )
                for i, doc in enumerate(docs)
            ]
        
        # Wrapper para o BM25 para garantir compatibilidade
        bm25_compatible = RunnableLambda(lambda x: format_docs(bm25_retriever.get_relevant_documents(x)))
        
        return {
            "vector_retriever": vector_retriever,
            "bm25_retriever": bm25_compatible,
            "splits": splits
        }
    
    finally:
        # Limpar arquivo temporário
        os.unlink(tmp_path)

# Função para aplicar Reciprocal Rank Fusion
def reciprocal_rank_fusion(results_lists, k=60):
    """
    Implementa o algoritmo Reciprocal Rank Fusion para combinar resultados de múltiplos retrievers.
    
    Args:
        results_lists: Lista de listas de documentos de diferentes retrievers
        k: Constante de suavização (60 é valor padrão na literatura)
        
    Returns:
        Lista de documentos ordenados por pontuação RRF
    """
    # Dicionário para armazenar documentos e suas pontuações
    doc_scores = {}
    
    # Para cada lista de resultados
    for i, results in enumerate(results_lists):
        # Para cada documento na lista
        for rank, doc in enumerate(results):
            # Calcula a pontuação RRF para este documento nesta lista
            # Rank começa em 0, então adicionamos 1
            rrf_score = 1 / (k + rank + 1)
            
            # Identifica o documento por conteúdo (como chave para o dicionário)
            doc_key = doc.page_content
            
            # Adiciona a pontuação RRF para este documento
            if doc_key not in doc_scores:
                doc_scores[doc_key] = {"doc": doc, "score": 0}
            doc_scores[doc_key]["score"] += rrf_score
    
    # Ordenar documentos por pontuação RRF em ordem decrescente
    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    # Retornar apenas os documentos, não as pontuações
    return [item["doc"] for item in sorted_docs]

# Função para configurar modelo LLM
def setup_llm():
    # Verificar se a API KEY está disponível
    if not os.getenv("GROQ_API_KEY"):
        st.error("API KEY da Groq não encontrada. Configure GROQ_API_KEY no arquivo .env")
        return None
    
    # Usando Groq como LLM
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    return llm

# Função para criar o pipeline de RAG Híbrido
def create_hybrid_rag_pipeline(vector_retriever, bm25_retriever, llm):
    # Função de RAG híbrida com Reciprocal Rank Fusion
    def retrieve_and_fuse(query):
        # Obter resultados dos dois retrievers
        vector_docs = vector_retriever.get_relevant_documents(query)
        bm25_docs = bm25_retriever.invoke(query)
        
        # Aplicar Reciprocal Rank Fusion
        fused_docs = reciprocal_rank_fusion([vector_docs, bm25_docs])
        
        # Limitar a 6 documentos para não sobrecarregar o contexto
        return fused_docs[:6]
    
    # Template de prompt aprimorado para RAG com histórico
    template = """
    Você é um assistente especializado em análise de documentos com memória de conversação.
    
    Use apenas o contexto abaixo e o histórico de conversas para responder à pergunta.
    Se a informação não estiver no contexto, responda que não consegue encontrar a informação solicitada.
    
    Histórico de conversas:
    {chat_history}
    
    Contexto:
    {context}
    
    Pergunta: {question}
    
    Resposta:
    """
    
    # Criar prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    # Função para formatar o histórico de chat
    def format_chat_history(chat_history):
        if not chat_history:
            return "Nenhum histórico de conversas anterior."
        formatted_history = ""
        for i, (q, a) in enumerate(chat_history):
            formatted_history += f"Pergunta {i+1}: {q}\nResposta {i+1}: {a}\n\n"
        return formatted_history
    
    # Função para formatar o contexto
    def format_context(docs):
        return "\n\n".join([f"Documento {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    # Montar pipeline RAG com histórico
    rag_chain = (
        {
            "context": lambda x: format_context(retrieve_and_fuse(x["question"])),
            "question": lambda x: x["question"],
            "chat_history": lambda x: format_chat_history(x["chat_history"])
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Interface principal estilizada como chat
def main():
    st.title("🔍 Sistema Avançado de Híbrido RAG")
    
    # Sidebar para carregar documento
    with st.sidebar:
        st.header("Configurações")
        pdf_file = st.file_uploader("Carregar PDF", type=["pdf"])
        
        if pdf_file:
            # Mostrar nome do arquivo
            st.success(f"Documento carregado: {pdf_file.name}")
            
            # Processar documento
            with st.spinner("Processando documento com análise híbrida..."):
                retrievers = process_pdf(pdf_file)
                llm = setup_llm()
                
                if llm is None:
                    return
                
                rag_chain = create_hybrid_rag_pipeline(
                    retrievers["vector_retriever"],
                    retrievers["bm25_retriever"],
                    llm
                )
                
                # Guardar componentes na sessão
                st.session_state.rag_chain = rag_chain
                st.session_state.document_loaded = True
                st.session_state.document_name = pdf_file.name
        
        # Sistema de histórico de conversas
        st.header("Histórico de Conversas")
        
        if "chat_log" in st.session_state and st.session_state.chat_log:
            # Mostrar conversas anteriores (até 5)
            conversations = list(st.session_state.chat_log.items())
            conversations.sort(key=lambda x: x[1]['timestamp'], reverse=True)
            
            for i, (conv_id, conv_data) in enumerate(conversations[:5]):
                if st.button(f"{conv_data['document'][:20]}... ({conv_data['timestamp'].strftime('%d/%m %H:%M')})", key=f"conv_{conv_id}"):
                    # Carregar histórico selecionado
                    st.session_state.conversation_id = conv_id
                    st.session_state.messages = conv_data['messages']
                    st.session_state.chat_history = conv_data['chat_history']
                    st.experimental_rerun()
        
        # Botão para nova conversa
        if st.button("Nova Conversa", key="new_conversation"):
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.experimental_rerun()
        
        # Exibir explicação sobre o Híbrido RAG
        st.markdown("### Como funciona o Híbrido RAG")
        st.markdown("""
        1. **Recuperação Vetorial**: Busca por similaridade semântica
        2. **Recuperação BM25**: Busca baseada em palavras-chave
        3. **Fusão de Rankings**: Combina resultados usando RRF
        4. **Memória de Conversação**: Mantém contexto entre perguntas
        """)

    # Container para mensagens de chat
    chat_container = st.container()
    
    # Formulário para envio de mensagem
    with st.form(key="message_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "Digite sua pergunta:",
                placeholder="O que você gostaria de saber sobre o documento?",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Enviar")
    
    # Processar mensagem quando enviada
    if submit_button and user_input and st.session_state.document_loaded:
        # Adicionar mensagem do usuário ao histórico de exibição
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Mostrar indicador de "digitando"
        with chat_container:
            messages_placeholder = st.empty()
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
            
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Pensando...")
        
        # Executar pipeline RAG
        with st.spinner(""):
            try:
                response = st.session_state.rag_chain.invoke({
                    "question": user_input,
                    "chat_history": st.session_state.chat_history
                })
                
                # Adicionar à memória de conversa (mantenha apenas as últimas 5 interações)
                st.session_state.chat_history.append((user_input, response))
                if len(st.session_state.chat_history) > 5:
                    st.session_state.chat_history.pop(0)
                
                # Adicionar mensagem do assistente ao histórico de exibição
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Salvar no log de conversas
                st.session_state.chat_log[st.session_state.conversation_id] = {
                    "document": st.session_state.document_name,
                    "timestamp": datetime.now(),
                    "messages": st.session_state.messages,
                    "chat_history": st.session_state.chat_history
                }
                
                # Atualizar a mensagem de "digitando" com a resposta real
                message_placeholder.markdown(response)
                
            except Exception as e:
                st.error(f"Erro ao processar a resposta: {str(e)}")
    
    elif submit_button and not st.session_state.document_loaded:
        st.warning("Por favor, carregue um documento PDF primeiro.")
    
    # Exibir histórico de mensagens no container de chat
    with chat_container:
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            # Mensagem inicial quando não há histórico
            st.markdown("""
            ### 👋 Bem-vindo ao Sistema Híbrido RAG!
            
            Para começar:
            1. Carregue um documento PDF no painel lateral
            2. Faça perguntas sobre o conteúdo do documento
            3. O sistema combinará busca semântica e léxica para encontrar as informações mais relevantes
            
            Este sistema mantém memória das últimas 5 interações para fornecer respostas contextualizadas.
            """)

# Executar aplicação
if __name__ == "__main__":
    main()