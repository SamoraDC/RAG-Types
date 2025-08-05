# app.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Carregando variáveis de ambiente
load_dotenv()

# Configuração da página Streamlit
st.set_page_config(
    page_title="Naive RAG - Consulta de Documentos",
    page_icon="📚",
    layout="wide"
)

# Função para carregar e processar PDF
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
        
        # Criar vetores e armazenar no FAISS
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        
        # Criar retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        return retriever
    
    finally:
        # Limpar arquivo temporário
        os.unlink(tmp_path)
        
# Função para configurar modelo LLM
def setup_llm():
    # Verificar se a API KEY está disponível
    if not os.getenv("GROQ_API_KEY"):
        st.error("API KEY da Groq não encontrada. Configure GROQ_API_KEY no arquivo .env")
        return None
    
    # Usando Groq como LLM
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    return llm

# Função para criar o pipeline de RAG
def create_rag_pipeline(retriever, llm):
    # Template de prompt para RAG
    template = """
    Você é um assistente especializado em analisar documentos e responder perguntas com base no conteúdo fornecido.
    
    Use apenas o contexto abaixo para responder à pergunta. Se a informação não estiver no contexto, responda que não consegue encontrar a informação solicitada.
    
    Contexto:
    {context}
    
    Pergunta: {question}
    
    Resposta:
    """
    
    from langchain_core.prompts import ChatPromptTemplate
    
    # Criar prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    # Montar pipeline RAG
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Interface principal
def main():
    st.title("📚 Naive RAG - Consulta de Documentos PDF")
    st.markdown(
        """
        Este aplicativo permite carregar um documento PDF e fazer perguntas sobre seu conteúdo utilizando RAG (Retrieval Augmented Generation).
        """
    )
    
    # Sidebar para carregar documento
    with st.sidebar:
        st.header("Configurações")
        pdf_file = st.file_uploader("Carregar PDF", type=["pdf"])
        
        # Exibir dicas sobre o funcionamento
        st.markdown("### Como funciona")
        st.markdown("""
        1. Carregue um documento PDF
        2. O sistema divide o documento em pequenos pedaços
        3. Quando você faz uma pergunta, o sistema:
           - Busca as partes mais relevantes do documento
           - Usa um LLM para gerar uma resposta contextualizada
        """)
    
    # Verificar se um documento foi carregado
    if pdf_file is not None:
        # Mostrar nome do arquivo
        st.success(f"Documento carregado: {pdf_file.name}")
        
        # Processar documento
        with st.spinner("Processando documento..."):
            retriever = process_pdf(pdf_file)
            llm = setup_llm()
            
            if llm is None:
                return
            
            rag_chain = create_rag_pipeline(retriever, llm)
            
            # Guardar componentes na sessão
            st.session_state.retriever = retriever
            st.session_state.rag_chain = rag_chain
            st.session_state.document_loaded = True
    
    # Interface para fazer perguntas
    st.header("Pergunte sobre o documento")
    
    # Exemplo de perguntas
    st.markdown("#### Exemplos de perguntas:")
    example_cols = st.columns(2)
    with example_cols[0]:
        st.markdown("- Qual é o assunto principal do documento?")
        st.markdown("- Quais são os pontos principais abordados?")
    with example_cols[1]:
        st.markdown("- Existe alguma conclusão importante?")
        st.markdown("- Quem são as pessoas mencionadas no documento?")
    
    # Input para pergunta
    query = st.text_input("Digite sua pergunta:")
    
    # Processar pergunta
    if query and ('document_loaded' in st.session_state):
        with st.spinner("Buscando resposta..."):
            # Executar pipeline RAG
            response = st.session_state.rag_chain.invoke(query)
            
            # Exibir resposta
            st.header("Resposta:")
            st.markdown(response)
    elif query:
        st.warning("Por favor, carregue um documento PDF primeiro.")

if __name__ == "__main__":
    main()