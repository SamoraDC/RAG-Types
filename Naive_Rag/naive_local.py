# naive_local.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# --- Langchain Imports ---
# Core
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda # RunnableLambda needed for routing
from langchain_core.runnables.history import RunnableWithMessageHistory
# Importação atualizada para ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Document Loaders & Vector Stores
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Ollama Specific Imports
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings

# Carregando variáveis de ambiente (opcional)
load_dotenv()

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Naive-RAG com Ollama (Local)",
    page_icon="💬",
    layout="wide"
)

# --- Funções Principais ---

@st.cache_resource(show_spinner="Processando PDF e criando embeddings...")
def process_pdf(pdf_file_content, pdf_filename):
    """Carrega, divide e vetoriza o conteúdo do PDF usando Ollama Embeddings."""
    tmp_path = None # Inicializa para garantir que existe no bloco finally
    try:
        # Salva o conteúdo em um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file_content)
            tmp_path = tmp_file.name

        # Carregamento do PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        if not documents:
            st.error(f"Não foi possível carregar o conteúdo do PDF: {pdf_filename}")
            return None # Retorna None se o carregamento falhar

        # Dividir texto em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )
        splits = text_splitter.split_documents(documents)

        # Criar vetores com Ollama Embeddings e armazenar no FAISS
        st.info(f"Criando embeddings para '{pdf_filename}' com Ollama (nomic-embed-text)... Pode levar um tempo.")
        try:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        except Exception as e:
            st.error(f"Erro ao criar embeddings ou vector store: {e}")
            st.error("Verifique se o Ollama está rodando e o modelo 'nomic-embed-text:latest' está disponível.")
            return None # Retorna None se a criação de embeddings falhar

        # Criar retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2} # Número de chunks a recuperar
        )
        st.success(f"Documento '{pdf_filename}' processado e vetorizado com sucesso!")
        return retriever

    except Exception as e:
        st.error(f"Erro geral ao processar o PDF '{pdf_filename}': {e}")
        return None # Retorna None em caso de erro geral
    finally:
        # Limpar arquivo temporário APÓS o uso ou em caso de erro
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@st.cache_resource(show_spinner="Configurando o modelo de linguagem Ollama...")
def setup_llm():
    """Configura o modelo de linguagem OllamaLLM."""
    try:
        llm = OllamaLLM(
            model="gemma3:4b",
            max_tokens=256, # Seu modelo Ollama LLM
            temperature=0.1, # Baixa temperatura para respostas mais factuais
            # base_url="http://localhost:11434" # Descomente/ajuste se necessário
        )
        # Teste rápido (opcional, pode ser removido se causar lentidão)
        # llm.invoke("Responda com 'Ok'")
        return llm
    except Exception as e:
        st.error(f"Erro ao instanciar o Ollama LLM: {e}")
        st.error("Verifique se o Ollama está rodando, o modelo 'phi4-mini:latest' está disponível e o pacote 'langchain-ollama' está instalado.")
        return None

def format_docs(docs):
    """Formata os documentos recuperados em uma string única."""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource(show_spinner="Montando a cadeia de RAG...")
def create_rag_chain(_llm, _retriever):
    """Cria a cadeia RAG base (sem gerenciamento de histórico explícito aqui)."""

    # 1. Prompt para contextualizar a pergunta (se houver histórico)
    contextualize_q_system_prompt = """Given a conversation (chat_history) and the user's latest question (question), 
    which may reference context from the conversation, 
    formulate an independent question that can be understood without the conversation. 
    Do not answer the question; only rephrase it if necessary; otherwise, return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | _llm | StrOutputParser()

    # 2. Prompt principal para gerar a resposta com base no contexto
    qa_system_prompt = """​You are a helpful assistant that answers questions based on a provided context. 
    Use ONLY the context below to formulate your response. DO NOT use prior knowledge. 
    If the information is not present in the context, clearly state "Based on the provided context, 
    I did not find the information." Be concise. And ALWAYS ANSWEAR IN PORTUGUESE PT-BR.

Contexto:
{context}
"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    # 3. Define como obter a string da pergunta para o retriever
    def get_question_string(input_dict):
        if input_dict.get("chat_history"):
            # Executa a cadeia para obter a string contextualizada
            return contextualize_q_chain
        else:
            # Retorna um Runnable simples que extrai a pergunta original
            return RunnableLambda(lambda x: x['question'], name="GetOriginalQuestion")

    # 4. Monta a cadeia RAG principal com a lógica corrigida
    rag_chain_base = (
        RunnablePassthrough.assign(
            # O valor de 'context' será o resultado desta sub-cadeia:
            context=RunnableLambda(get_question_string, name="RouteQuestion") # Decide qual caminho seguir
                    | _retriever # Executa o retriever com a string resultante
                    | format_docs # Formata os documentos recuperados
        ).with_config(run_name="RetrieveDocs") # Adiciona nome para clareza no trace (opcional)
        | qa_prompt # Passa o dicionário para o prompt final
        | _llm
        | StrOutputParser() # Garante saída como string
    )
    return rag_chain_base

# --- Gerenciamento de Memória da Sessão ---
def get_session_history(session_id: str) -> ChatMessageHistory:
    """Obtém o histórico de chat para uma dada ID de sessão."""
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]

# --- Interface Principal do Streamlit ---
def main():
    st.title("💬 Chat RAG com Ollama (Local)")
    st.markdown(
        """
        Carregue um documento PDF e converse sobre ele. O chatbot usará o conteúdo do PDF
        e o histórico da conversa para responder suas perguntas. Modelos Ollama phi4-mini é o usada localmente.
        **Certifique-se que o documento é um PDF de texto selecionável**
        """
    )

    # Inicializar estado da sessão se necessário
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {} # Dicionário para guardar históricos por sessão
    if "rag_chain_with_history" not in st.session_state:
        st.session_state.rag_chain_with_history = None
    if "document_loaded" not in st.session_state:
         st.session_state.document_loaded = False
    if "pdf_filename" not in st.session_state:
        st.session_state.pdf_filename = None
    if "processed_file_identifier" not in st.session_state:
        st.session_state.processed_file_identifier = None


    # --- Sidebar para Carregar Documento ---
    with st.sidebar:
        st.header("📄 Configuração do Documento")
        uploaded_file = st.file_uploader("Carregar PDF", type=["pdf"], key="pdf_upload_widget")

        # Lógica para processar um NOVO arquivo carregado
        if uploaded_file is not None:
            new_file_identifier = f"{uploaded_file.name}-{uploaded_file.size}"
            # Processa apenas se for um arquivo diferente do último processado
            if st.session_state.processed_file_identifier != new_file_identifier:
                st.info(f"Novo arquivo detectado: {uploaded_file.name}")
                # Limpar estado antigo relacionado ao chat e documento anterior
                st.session_state.messages = []
                st.session_state.chat_histories = {}
                st.session_state.rag_chain_with_history = None
                st.session_state.document_loaded = False
                st.session_state.pdf_filename = None

                # Ler o conteúdo do arquivo para passar para a função cacheada
                pdf_content = uploaded_file.getvalue()
                retriever = process_pdf(pdf_content, uploaded_file.name) # Passa conteúdo e nome

                if retriever:
                    llm = setup_llm() # Configura LLM (cacheado)
                    if llm:
                        base_rag_chain = create_rag_chain(llm, retriever) # Cria cadeia base (cacheada)
                        # Cria a cadeia com gerenciamento de histórico
                        st.session_state.rag_chain_with_history = RunnableWithMessageHistory(
                            base_rag_chain,
                            get_session_history,
                            input_messages_key="question",
                            history_messages_key="chat_history",
                            # <<< CORREÇÃO: output_messages_key="answer" REMOVIDO daqui >>>
                        )
                        st.session_state.document_loaded = True
                        st.session_state.pdf_filename = uploaded_file.name
                        st.session_state.processed_file_identifier = new_file_identifier
                        # Mensagem inicial apenas após carregar TUDO
                        st.session_state.messages.append({"role": "assistant", "content": f"Olá! O documento '{st.session_state.pdf_filename}' está pronto. Pode perguntar."})
                        st.rerun() # Força o recarregamento para exibir a mensagem e limpar spinners
                    else:
                        st.error("Falha ao configurar o LLM. Chat desabilitado.")
                        st.session_state.document_loaded = False # Garante que o chat não funcione
                else:
                    st.error("Falha ao processar o PDF. Chat desabilitado.")
                    st.session_state.document_loaded = False # Garante que o chat não funcione
            # else: Arquivo é o mesmo já processado, não faz nada


        if st.session_state.document_loaded:
            st.success(f"Documento '{st.session_state.pdf_filename}' carregado.")
            if st.button("Limpar Chat e Recomeçar"):
                 # Limpa apenas o histórico e mensagens, mantém o documento carregado
                 st.session_state.messages = [{"role": "assistant", "content": f"Chat reiniciado para '{st.session_state.pdf_filename}'. Faça sua pergunta."}]
                 st.session_state.chat_histories = {} # Limpa todos os históricos da sessão
                 st.rerun()

        st.divider()
        st.markdown("### Como funciona:")
        st.markdown("""
        1.  **Carregue um PDF:** O conteúdo é extraído e dividido.
        2.  **Embeddings:** O modelo local `nomic-embed-text` do Ollama cria vetores dos trechos.
        3.  **Chat:** Você faz uma pergunta.
        4.  **Contextualização (com Histórico):** Se necessário, o modelo `phi4-mini` reescreve sua pergunta considerando o histórico.
        5.  **Recuperação:** A pergunta (original ou reescrita) é usada para buscar os trechos mais relevantes no PDF.
        6.  **Geração:** O modelo `phi4-mini` usa o histórico, sua pergunta e os trechos recuperados para gerar a resposta final.
        """)
        st.divider()
        st.info("Pré-requisitos: Ollama rodando com modelos 'phi4-mini' e 'nomic-embed-text', `uv add langchain-ollama langchain-community faiss-cpu pypdf`")


    # --- Lógica do Chat ---
    # Exibe o chat apenas se um documento foi carregado com sucesso E a cadeia está pronta
    if st.session_state.document_loaded and st.session_state.rag_chain_with_history:
        # Exibir mensagens existentes
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Campo de entrada do chat
        if prompt := st.chat_input(f"Pergunte sobre {st.session_state.pdf_filename}..."):
            # Adicionar mensagem do usuário à UI
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Preparar para chamar a cadeia RAG com histórico
            with st.spinner("Pensando..."):
                # ID de sessão simples para este exemplo
                session_id = "my_streamlit_session" # Pode ser mais elaborado se necessário
                config = {"configurable": {"session_id": session_id}}

                try:
                    # Invocar a cadeia com a pergunta e a configuração da sessão
                    response = st.session_state.rag_chain_with_history.invoke(
                        {"question": prompt},
                        config=config
                    )

                    # <<< CORREÇÃO: Lógica para extrair a resposta >>>
                    if isinstance(response, str):
                        assistant_response = response
                    elif isinstance(response, dict):
                         st.warning(f"Estrutura de resposta inesperada (dict): {response}")
                         # Tenta obter de chaves comuns como fallback
                         assistant_response = response.get('output', response.get('answer', "Erro: Resposta em formato de dicionário inesperado."))
                    else:
                         st.warning(f"Tipo de resposta inesperado: {type(response)}")
                         assistant_response = str(response) # Tenta converter para string como último recurso
                    # <<< FIM DA CORREÇÃO >>>


                    # Adicionar resposta do assistente à UI
                    # O histórico lógico já foi atualizado pelo RunnableWithMessageHistory
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                    # Re-renderizar a página para mostrar a nova mensagem
                    st.rerun()

                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {e}")
                    error_message = f"Ocorreu um erro ao processar sua pergunta. Verifique os logs ou se o Ollama está respondendo corretamente."
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
                    st.rerun()

    # Mensagens informativas se o chat não estiver pronto
    elif 'document_loaded' in st.session_state and not st.session_state.document_loaded:
        st.warning("Houve um erro ao carregar o documento ou configurar o LLM. Verifique as mensagens de erro.")
    elif not st.session_state.get('processed_file_identifier'): # Se nenhum arquivo foi processado ainda
         st.info("⬅️ Carregue um documento PDF na barra lateral para começar.")


if __name__ == "__main__":
    # Verificações Iniciais de importação
    try:
        from langchain_ollama.llms import OllamaLLM
        from langchain_ollama.embeddings import OllamaEmbeddings
        from langchain_community.chat_message_histories import ChatMessageHistory
        from langchain_community.vectorstores import FAISS
        # Adicione outras importações críticas se necessário
    except ImportError as e:
        st.error(f"Erro Crítico de Importação: {e}")
        st.error("Certifique-se de ter instalado todos os pacotes necessários:")
        st.code("pip install streamlit langchain langchain-core langchain-community langchain-ollama faiss-cpu pypdf python-dotenv")
        st.stop() # Impede a execução do app se faltar pacote essencial

    main()