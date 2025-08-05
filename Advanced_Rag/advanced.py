# --- Core Imports ---
import os
import tempfile
import uuid
import json
import time
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
import re # For robust JSON parsing

# --- Environment Setup (VERY IMPORTANT: Set these EARLY) ---
# Load environment variables first
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))

# Configuração para desativar o aviso de symlinks do HuggingFace
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Configuração para reduzir conflitos entre Streamlit e PyTorch (tentativa)
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"
# Desabilitar JIT e Caching do PyTorch que podem conflitar com Streamlit
os.environ["PYTORCH_JIT"] = "0"
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
# Opcional: Para evitar que Streamlit observe arquivos temporários demais
os.environ["STREAMLIT_WATCH_TMP_FILES"] = "false"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# --- Library Imports ---
# IMPORTANT: Ensure necessary libraries are installed
# pip install streamlit torch transformers langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu pypdf python-dotenv accelerate bitsandbytes bm25-retriever sentencepiece
import streamlit as st
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM, # Import Seq2Seq as well
    pipeline,
    BitsAndBytesConfig
)
from huggingface_hub import login, HfFolder

# LangChain Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_huggingface import HuggingFacePipeline

# Re-ranking Import
from sentence_transformers import CrossEncoder

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# Allow overriding default model via environment variable
# >>> CHANGED TO GEMMA-7B AS REQUESTED <<<
DEFAULT_MODEL_ID = os.getenv("DEFAULT_LLM_MODEL", "google/gemma-7b")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
MAX_CHAT_HISTORY = 5 # Limit context window for history
ITERATIVE_RETRIEVAL_MAX_ITER = 2 # Max iterations for refinement

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="RAG Avançado - Consulta Inteligente",
    page_icon="🧠",
    layout="wide"
)

# --- CSS Styling ---
def load_css():
    # Using a default style here for simplicity, assuming style.css might not exist
    custom_css = """
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start; /* Align items to the top */
    }
    .stChatMessage:nth-child(odd) { /* User messages */
        background-color: #f0f2f6; /* Lighter gray */
    }
    .stChatMessage:nth-child(even) { /* Assistant messages */
        background-color: #e6f7ff; /* Light blue */
    }
    .stChatMessage .stMarkdown { /* Ensure markdown takes full width */
        width: 100%;
    }
    .stChatInput { /* Style the input box */
        margin-top: 1rem;
    }
    .debug-info { /* Style for debug section */
        background-color: #fffbeb; /* Light yellow */
        border: 1px solid #fde68a; /* Yellow border */
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        font-size: 0.9em;
    }
    .debug-info h4 {
        margin-top: 0;
        margin-bottom: 0.5rem;
        color: #ca8a04; /* Darker yellow */
    }
    .debug-info code {
        background-color: #f3f4f6; /* Light gray for code blocks */
        padding: 0.2rem 0.4rem;
        border-radius: 0.3rem;
    }
    """
    st.markdown(f'<style>{custom_css}</style>', unsafe_allow_html=True)

load_css()

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "chat_history": [],
        "document_loaded": False,
        "conversation_id": str(uuid.uuid4()),
        "messages": [],
        "document_name": None,
        "processing_times": {}, # Store metrics per query
        "evaluation_metrics": [], # History of metrics
        "feedback_history": [],
        "debug_mode": False,
        "llm": None,
        "model_loaded": False,
        "vector_retriever": None,
        "bm25_retriever": None,
        "retrieval_system": None,
        "rag_pipeline": None,
        "iterative_rag": None,
        "doc_stats": {},
        "selected_model_id": DEFAULT_MODEL_ID # Track selected model
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Core Classes ---

class DocumentProcessor:
    """Handles PDF loading, splitting, and basic statistics."""
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len, # Use character length
            is_separator_regex=False,
        )
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def process_pdf(self, pdf_file) -> Optional[List[Document]]:
        """Processes an uploaded PDF file and returns document chunks."""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            logger.info(f"PDF saved to temporary file: {tmp_path}")

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            if not documents:
                logger.warning("PyPDFLoader returned no documents.")
                return None

            logger.info(f"Loaded {len(documents)} pages from PDF.")
            splits = self.text_splitter.split_documents(documents)
            logger.info(f"Split document into {len(splits)} chunks.")

            # Add metadata like source chunk number
            for i, split in enumerate(splits):
                split.metadata["source_chunk"] = i

            return splits

        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            st.error(f"Erro ao processar PDF: {e}")
            return None
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                    logger.info(f"Temporary file {tmp_path} deleted.")
                except Exception as e:
                    logger.warning(f"Could not delete temporary file {tmp_path}: {e}")

    def get_document_stats(self, splits: List[Document]) -> Dict[str, Any]:
        """Calculates statistics about the document chunks."""
        if not splits:
            return {"total_chunks": 0, "total_chars": 0, "avg_chunk_chars": 0}

        total_chunks = len(splits)
        total_chars = sum(len(doc.page_content) for doc in splits)
        avg_chunk_chars = total_chars / total_chunks if total_chunks > 0 else 0

        return {
            "total_chunks": total_chunks,
            "total_chars": total_chars,
            "avg_chunk_chars": avg_chunk_chars
        }

class RetrievalSystem:
    """Manages embedding model, retrievers (vector and BM25), and re-ranking."""
    def __init__(self, embedding_model_name=DEFAULT_EMBEDDING_MODEL, reranker_model_name=DEFAULT_RERANKER_MODEL):
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.embeddings = None
        self.reranker = None
        self._load_models()

    def _load_models(self):
        """Loads embedding and re-ranking models."""
        try:
            # Determine device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {device} for embedding and reranker models")

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': device}
            )
            logger.info(f"Loading re-ranker model: {self.reranker_model_name}")
            self.reranker = CrossEncoder(
                self.reranker_model_name,
                device=device
            )
            logger.info("Embedding and re-ranker models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading retrieval models: {e}", exc_info=True)
            st.error(f"Erro ao inicializar sistema de recuperação (embeddings/reranker): {e}")
            raise # Propagate error to stop initialization if critical models fail

    def setup_retrievers(self, splits: List[Document], k_retrieval=15) -> Tuple[Optional[Any], Optional[BM25Retriever]]:
        """Configures FAISS vector retriever and BM25 retriever."""
        if not splits:
            logger.warning("No document splits provided to setup_retrievers.")
            st.error("Nenhum chunk de documento disponível para configurar os retrievers.")
            return None, None
        if not self.embeddings:
             logger.error("Embeddings not loaded, cannot create vector retriever.")
             st.error("Modelo de embedding não carregado. Não é possível criar o retriever vetorial.")
             return None, None

        try:
            logger.info(f"Setting up retrievers with k={k_retrieval}...")
            # Vector Retriever (FAISS)
            logger.info("Creating FAISS index...")
            vectorstore = FAISS.from_documents(documents=splits, embedding=self.embeddings)
            vector_retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k_retrieval}
            )
            logger.info("FAISS vector retriever created.")

            # BM25 Retriever
            logger.info("Creating BM25 retriever...")
            # Pass original documents to preserve metadata
            bm25_retriever = BM25Retriever.from_documents(documents=splits)
            bm25_retriever.k = k_retrieval
            logger.info("BM25 retriever created.")

            return vector_retriever, bm25_retriever

        except Exception as e:
            logger.error(f"Error setting up retrievers: {e}", exc_info=True)
            st.error(f"Erro ao configurar retrievers: {e}")
            return None, None

    def rerank_documents(self, query: str, documents: List[Document], top_k=6) -> List[Document]:
        """Re-ranks documents using the CrossEncoder model."""
        if not documents:
            logger.warning("No documents provided for re-ranking.")
            return []
        if not self.reranker:
            logger.warning("Re-ranker not available, returning original top_k documents.")
            return documents[:min(top_k, len(documents))]

        try:
            logger.info(f"Re-ranking {len(documents)} documents for query: '{query[:50]}...'")
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.reranker.predict(pairs)

            # Combine documents with scores and sort
            scored_docs = list(zip(documents, scores))
            # Sort by score descending
            reranked_docs_with_scores = sorted(scored_docs, key=lambda x: x[1], reverse=True)

            # Add rerank score to metadata
            final_docs = []
            for doc, score in reranked_docs_with_scores[:top_k]:
                doc.metadata['rerank_score'] = float(score) # Ensure score is float
                final_docs.append(doc)

            logger.info(f"Re-ranked documents, returning top {len(final_docs)}.")
            return final_docs

        except Exception as e:
            logger.error(f"Error during document re-ranking: {e}", exc_info=True)
            st.error(f"Erro ao re-rankear documentos: {e}")
            # Fallback: return the original top_k documents without re-ranking scores
            return documents[:min(top_k, len(documents))]

class QueryExpander:
    """Uses an LLM to expand the user query for better retrieval."""
    def __init__(self, llm):
        self.llm = llm
        logger.info("QueryExpander initialized.")

    def expand_query(self, query: str, chat_history: Optional[List[Tuple[str, str]]] = None) -> str:
        """Expands the query using the LLM."""
        if not query:
            return ""
        if not self.llm:
            logger.warning("LLM not available for query expansion. Returning original query.")
            return query

        # More specific template
        template = """You are an expert in query understanding and reformulation for information retrieval.
Your task is to expand the user's query to improve the retrieval of relevant documents.
Generate 3-5 alternative formulations or related sub-queries based on the original query and chat history.
Focus on synonyms, related concepts, and potential ambiguities. Keep the core intent.
Combine these into a single, comprehensive query string suitable for searching.

Original Query: {query}

Chat History (Recent turns):
{history_context}

Respond ONLY with the expanded query string, nothing else.
Expanded Query:"""

        history_context = "No history available."
        if chat_history:
            history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]]) # Last 3 turns
            if history_text:
                 history_context = history_text

        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            logger.info(f"Expanding query: '{query}'")
            start_time = time.time()
            expanded_query = chain.invoke({
                "query": query,
                "history_context": history_context
            }).strip()
            duration = time.time() - start_time
            logger.info(f"Query expanded in {duration:.2f}s. Expanded query: '{expanded_query[:100]}...'")

            # Basic validation: ensure it's not empty and different from original (if possible)
            if expanded_query and expanded_query.lower() != query.lower():
                return expanded_query
            else:
                logger.warning("Query expansion resulted in empty or identical query. Using original.")
                return query

        except Exception as e:
            logger.error(f"Error during query expansion: {e}", exc_info=True)
            st.warning(f"Erro na expansão de query: {e}. Usando a query original.")
            return query # Fallback to original query

class IterativeRAG:
    """Handles iterative retrieval by analyzing results and refining the query."""
    def __init__(self, llm, retrieval_system: RetrievalSystem):
        self.llm = llm
        self.retrieval_system = retrieval_system
        self.max_iterations = ITERATIVE_RETRIEVAL_MAX_ITER
        logger.info(f"IterativeRAG initialized with max_iterations={self.max_iterations}")

    def _robust_json_parse(self, json_string: str) -> Optional[Dict]:
        """Attempts to parse JSON, even if embedded in other text."""
        try:
            # Try direct parsing first
            return json.loads(json_string)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON using regex
            logger.warning(f"Direct JSON parsing failed for: {json_string[:100]}... Trying regex extraction.")
            # Regex to find JSON object starting with { and ending with }
            match = re.search(r'\{.*\}', json_string, re.DOTALL)
            if match:
                extracted_json = match.group(0)
                try:
                    return json.loads(extracted_json)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse extracted JSON: {extracted_json[:100]}... Error: {e}")
                    return None
            else:
                logger.error(f"Could not find JSON block in string: {json_string[:100]}...")
                return None

    def analyze_retrieval_quality(self, query: str, docs: List[Document]) -> Dict[str, Any]:
        """Analyzes retrieved documents using the LLM to decide if refinement is needed."""
        default_analysis = {
            "relevance_score": 5, # Neutral default
            "needs_refinement": False,
            "missing_aspects": [],
            "suggested_query": query
        }
        if not self.llm:
            logger.warning("LLM not available for retrieval analysis.")
            return default_analysis
        if not docs:
            logger.warning("No documents retrieved, analysis indicates refinement needed.")
            return {**default_analysis, "needs_refinement": True, "missing_aspects": ["No documents found"]}

        # Simpler prompt, less prone to LLM errors, focusing on the decision
        template = """Analyze the relevance of the retrieved documents for the user's query.
Query: {query}

Retrieved Documents Snippets:
{docs_snippets}

Based ONLY on the snippets, answer these questions in JSON format:
1.  `relevance_score` (1-10): How relevant are these documents overall to the query?
2.  `needs_refinement` (true/false): Is the query likely missing key aspects or could it be improved for better results?
3.  `suggested_query` (string): If refinement is needed, suggest a better query. Otherwise, repeat the original query.

Respond ONLY with the JSON object. Example:
{{"relevance_score": 8, "needs_refinement": false, "suggested_query": "{query}"}}
OR
{{"relevance_score": 4, "needs_refinement": true, "suggested_query": "alternative query focusing on X"}}

JSON Response:"""

        # Use only snippets to avoid overwhelming the LLM context
        docs_snippets = "\n\n".join([f"Doc {i+1} (Score: {doc.metadata.get('rerank_score', 'N/A'):.2f}):\n{doc.page_content[:200]}..."
                                     for i, doc in enumerate(docs)])

        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            logger.info(f"Analyzing retrieval quality for query: '{query[:50]}...'")
            start_time = time.time()
            analysis_text = chain.invoke({
                "query": query,
                "docs_snippets": docs_snippets
            }).strip()
            duration = time.time() - start_time
            logger.info(f"Retrieval analysis completed in {duration:.2f}s.")

            # Robust JSON parsing
            analysis = self._robust_json_parse(analysis_text)
            if analysis and isinstance(analysis, dict):
                 # Validate expected keys, provide defaults if missing
                 validated_analysis = {
                     "relevance_score": analysis.get("relevance_score", 5),
                     "needs_refinement": analysis.get("needs_refinement", False),
                     "missing_aspects": analysis.get("missing_aspects", []), # Keep if provided, else empty
                     "suggested_query": analysis.get("suggested_query", query)
                 }
                 logger.info(f"Parsed analysis: {validated_analysis}")
                 return validated_analysis
            else:
                logger.error(f"Failed to parse valid JSON from analysis response: {analysis_text}")
                st.warning("Falha ao analisar a qualidade da recuperação (resposta LLM inválida).")
                return default_analysis # Fallback

        except Exception as e:
            logger.error(f"Error during retrieval quality analysis: {e}", exc_info=True)
            st.warning(f"Erro na análise de qualidade da recuperação: {e}")
            return default_analysis # Fallback

    def refine_query(self, current_query: str, analysis: Dict[str, Any]) -> Tuple[str, bool]:
        """Determines the next query based on the analysis."""
        needs_refinement = analysis.get("needs_refinement", False)
        suggested_query = analysis.get("suggested_query", current_query).strip()

        if needs_refinement and suggested_query and suggested_query.lower() != current_query.lower():
            logger.info(f"Refining query based on analysis. New query: '{suggested_query[:100]}...'")
            return suggested_query, True
        else:
            logger.info("No refinement needed or suggested query is same/empty.")
            return current_query, False

    def iterative_retrieval(self, vector_retriever, bm25_retriever, initial_query: str) -> Tuple[List[Document], List[Dict]]:
        """Performs the iterative retrieval loop."""
        if not vector_retriever or not bm25_retriever:
            logger.error("Retrievers not available for iterative retrieval.")
            st.error("Retrievers não configurados corretamente para busca iterativa.")
            return [], []
        if not self.retrieval_system:
             logger.error("RetrievalSystem not available for iterative retrieval.")
             st.error("Sistema de recuperação não disponível para busca iterativa.")
             return [], []

        current_query = initial_query
        best_docs = []
        best_score = -1 # Use -1 to ensure first valid result is stored
        iterations = 0
        refinement_history = []

        while iterations < self.max_iterations:
            iterations += 1
            logger.info(f"--- Iterative Retrieval: Iteration {iterations}, Query: '{current_query[:50]}...' ---")

            try:
                # --- Step 1: Retrieve ---
                start_time_retrieval = time.time()
                vector_docs = vector_retriever.get_relevant_documents(current_query)
                bm25_docs = bm25_retriever.get_relevant_documents(current_query)

                # Combine and deduplicate based on page_content
                all_docs_dict = {doc.page_content: doc for doc in vector_docs + bm25_docs}
                combined_docs = list(all_docs_dict.values())
                retrieval_time = time.time() - start_time_retrieval
                logger.info(f"Iteration {iterations}: Retrieved {len(vector_docs)} vector docs, {len(bm25_docs)} BM25 docs. Combined to {len(combined_docs)} unique docs in {retrieval_time:.2f}s.")

                if not combined_docs:
                    logger.warning(f"Iteration {iterations}: No documents retrieved for query.")
                    refinement_history.append({
                        "iteration": iterations, "query": current_query, "relevance_score": 0,
                        "retrieval_time": retrieval_time, "reranking_time": 0, "num_retrieved": 0, "num_reranked": 0,
                        "needs_refinement": True, "suggested_query": current_query # Assume refinement needed if nothing found
                    })
                    # Analyze quality (will likely suggest refinement)
                    analysis = self.analyze_retrieval_quality(current_query, []) # Pass empty list
                    # Decide next step based on analysis
                    next_query, needs_refinement = self.refine_query(current_query, analysis)
                    if not needs_refinement or iterations >= self.max_iterations:
                        break
                    current_query = next_query
                    continue # Skip reranking/analysis for this iteration

                # --- Step 2: Re-rank ---
                start_time_rerank = time.time()
                reranked_docs = self.retrieval_system.rerank_documents(current_query, combined_docs)
                reranking_time = time.time() - start_time_rerank
                logger.info(f"Iteration {iterations}: Re-ranked to {len(reranked_docs)} docs in {reranking_time:.2f}s.")

                # --- Step 3: Analyze Quality ---
                analysis = self.analyze_retrieval_quality(current_query, reranked_docs)
                current_relevance = analysis.get("relevance_score", 0)

                # Log iteration details
                refinement_history.append({
                    "iteration": iterations,
                    "query": current_query,
                    "relevance_score": current_relevance,
                    "retrieval_time": retrieval_time,
                    "reranking_time": reranking_time,
                    "num_retrieved": len(combined_docs),
                    "num_reranked": len(reranked_docs),
                    "needs_refinement": analysis.get("needs_refinement", False),
                    "suggested_query": analysis.get("suggested_query", current_query)
                })

                # Update best results found so far
                if current_relevance > best_score:
                    logger.info(f"Iteration {iterations}: New best score ({current_relevance}) found. Updating best docs.")
                    best_docs = reranked_docs
                    best_score = current_relevance
                elif not best_docs and reranked_docs: # If no best docs yet, take the first ranked set
                    logger.info(f"Iteration {iterations}: No best docs yet, using current reranked docs.")
                    best_docs = reranked_docs
                    best_score = current_relevance # Set initial best score


                # --- Step 4: Decide whether to refine and continue ---
                next_query, needs_refinement = self.refine_query(current_query, analysis)

                if not needs_refinement or iterations >= self.max_iterations:
                    logger.info(f"Iteration {iterations}: Stopping iteration. Needs refinement: {needs_refinement}. Max iterations reached: {iterations >= self.max_iterations}")
                    break # Exit loop

                # Update query for the next iteration
                current_query = next_query

            except Exception as e:
                logger.error(f"Error during iterative retrieval iteration {iterations}: {e}", exc_info=True)
                st.error(f"Erro na iteração {iterations} da recuperação: {e}")
                break # Stop iterating on error

        if not best_docs and refinement_history:
            logger.warning("Iterative retrieval finished but no documents were deemed best. Returning empty list.")
            st.warning("Não foi possível encontrar documentos relevantes após as tentativas de refinamento.")
        elif best_docs:
             logger.info(f"Iterative retrieval finished. Returning {len(best_docs)} best documents found.")

        return best_docs, refinement_history

# --- LLM Loading ---
@st.cache_resource(show_spinner="Carregando modelo de linguagem base...")
def load_base_model_and_tokenizer(model_id: str):
    """Loads the base LLM model and tokenizer, applying quantization if applicable."""
    logger.info(f"Attempting to load model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = None
    quantization_config = None
    device_map = "auto"
    # Default to float16 for efficiency if GPU available, else float32 for CPU compatibility
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Check for Hugging Face token for gated models (like Gemma)
    hf_token = os.getenv("HUGGINGFACE_API_KEY") or HfFolder.get_token()
    if not hf_token and "gemma" in model_id: # Gemma requires login
        logger.warning("Hugging Face token not found. Gemma model requires login.")
        st.warning("Token do Hugging Face não encontrado. É necessário para carregar o modelo Gemma. Verifique seu arquivo .env ou faça login via CLI.")
        # Optionally raise error or return None early
        # return None, None, False

    # Check CUDA availability
    has_gpu = torch.cuda.is_available()
    if not has_gpu:
        logger.warning(f"--- NO GPU DETECTED --- Loading large model '{model_id}' on CPU.")
        logger.warning("--- This will be VERY SLOW and require significant RAM. ---")
        st.warning(f"⚠️ Sem GPU detectada! Carregar '{model_id}' na CPU será MUITO LENTO e consumirá muita RAM.")

    try:
        # Determine model type (Seq2Seq or CausalLM) - crude check based on name
        is_seq2seq = any(name in model_id for name in ["t5", "bart"])

        # Quantization for larger Causal LMs on GPU
        # Add more model names here if needed
        needs_quantization = any(name in model_id for name in ["phi-3", "llama", "mistral", "gemma"])

        if needs_quantization and has_gpu:
            logger.info("GPU detected. Applying 4-bit quantization.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            # device_map = "auto" # Keep auto for multi-GPU or large models
        elif not has_gpu:
            logger.info("No GPU detected. Loading in default precision (FP32).")
            # device_map = "cpu" # Explicitly set to CPU if no CUDA - 'auto' should handle this with accelerate
            pass # Keep torch_dtype as float32 set earlier

        model_kwargs = {
            "device_map": device_map, # Requires accelerate
            "trust_remote_code": True, # Be cautious with this
            "token": hf_token,
            "torch_dtype": torch_dtype
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        else:
             # Use lower memory usage on CPU if no quantization
             # This requires accelerate to be installed
             model_kwargs["low_cpu_mem_usage"] = True


        if is_seq2seq:
            logger.info(f"Loading Seq2Seq model: {model_id}")
            # Remove quantization for Seq2Seq if it causes issues, T5 usually doesn't need it as much
            if "quantization_config" in model_kwargs:
                 logger.warning("Removing quantization config for Seq2Seq model.")
                 del model_kwargs["quantization_config"]
            # low_cpu_mem_usage might not be applicable or as effective for Seq2Seq
            if "low_cpu_mem_usage" in model_kwargs and not has_gpu:
                 logger.info("Using low_cpu_mem_usage for Seq2Seq on CPU.")

            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **model_kwargs)
        else:
            logger.info(f"Loading CausalLM model: {model_id}")
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        logger.info(f"Model {model_id} loaded successfully.")
        # Log device placement if possible
        try:
            # device_map is more informative if used
            effective_device_map = getattr(model, 'hf_device_map', None)
            if effective_device_map:
                logger.info(f"Model device map: {effective_device_map}")
            else:
                 logger.info(f"Model device: {model.device}") # Fallback
        except Exception:
             logger.warning("Could not determine model device placement details.")

        return model, tokenizer, is_seq2seq

    except ImportError as e:
         # Specific check for accelerate missing
         if "Accelerate" in str(e):
              logger.error(f"ImportError: {e}. The 'accelerate' library is required for efficient model loading, especially with device_map or low_cpu_mem_usage.")
              st.error("Erro: A biblioteca 'accelerate' é necessária. Por favor, instale com: pip install accelerate")
         else:
              logger.error(f"Error loading model {model_id}: {e}", exc_info=True)
              st.error(f"Erro de importação ao carregar modelo {model_id}: {e}")
         return None, None, False
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}", exc_info=True)
        st.error(f"Erro crítico ao carregar modelo {model_id}: {e}")
        return None, None, False

# --- LLM Pipeline Setup ---
# Separate function to create the pipeline AFTER the model is loaded
def create_llm_pipeline(model, tokenizer, is_seq2seq):
    """Creates the HuggingFacePipeline based on the loaded model."""
    if model is None or tokenizer is None:
        return None

    try:
        task = "text2text-generation" if is_seq2seq else "text-generation"
        logger.info(f"Creating pipeline for task: {task}")

        # Adjust pipeline parameters
        pipe_kwargs = {
            "model": model,
            "tokenizer": tokenizer,
            "temperature": 0.7, # Add some creativity control
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        if task == "text-generation":
            pipe_kwargs["max_new_tokens"] = 512
            # Ensure pad_token_id is set if EOS is used
            if tokenizer.pad_token_id is None:
                logger.warning("Tokenizer does not have a pad token ID. Setting to EOS token ID.")
                # Common practice, but check model docs if issues arise
                tokenizer.pad_token_id = tokenizer.eos_token_id
                # Update pipeline's tokenizer instance as well
                # pipe_kwargs["tokenizer"].pad_token_id = tokenizer.eos_token_id # This might modify the cached tokenizer, be careful
                # It's generally better to ensure the base tokenizer loaded has it,
                # or handle padding explicitly in the prompt/generation call if needed.
                # For now, setting it on the tokenizer object passed to pipeline is usually sufficient.
                model.config.pad_token_id = tokenizer.eos_token_id # Also update model config

        elif task == "text2text-generation":
             pipe_kwargs["max_length"] = 512 # T5 style models use max_length

        pipe = pipeline(task, **pipe_kwargs)

        llm_pipeline = HuggingFacePipeline(pipeline=pipe)
        logger.info("HuggingFacePipeline created successfully.")
        return llm_pipeline

    except Exception as e:
        logger.error(f"Error creating LLM pipeline: {e}", exc_info=True)
        st.error(f"Erro ao criar o pipeline LLM: {e}")
        return None

# --- RAG Pipeline Class ---
class RAGPipeline:
    """Orchestrates the entire RAG process: expansion, retrieval, generation."""
    def __init__(self, llm, query_expander: QueryExpander):
        self.llm = llm
        self.query_expander = query_expander
        logger.info("RAGPipeline initialized.")

    def generate_response(self, query: str, docs: List[Document], chat_history: Optional[List[Tuple[str, str]]] = None, refinement_history: Optional[List[Dict]] = None) -> str:
        """Generates the final response using the LLM and retrieved context."""
        if not self.llm:
            logger.error("LLM not available for response generation.")
            return "Desculpe, o modelo de linguagem não está disponível no momento."
        if not docs:
            logger.warning("No documents provided to generate_response. Returning canned response.")
            return "Não encontrei informações relevantes nos documentos recuperados para responder à sua pergunta. Poderia tentar reformular?"

        # Enhanced prompt emphasizing context grounding and citation
        template = """You are a helpful AI assistant answering questions based ONLY on the provided context documents and chat history.
Do not use any prior knowledge. If the answer is not found in the context, state that clearly.
Cite the source document number [Document N] where the information was found.

Chat History:
{chat_history}

Context Documents:
{context}

User Query: {question}

{refinement_info}

Answer (based ONLY on context):"""

        # Format chat history
        chat_history_text = "No previous conversation."
        if chat_history:
            history_items = []
            for q, a in chat_history[-(MAX_CHAT_HISTORY):]: # Use constant
                history_items.append(f"User: {q}\nAssistant: {a}")
            if history_items:
                chat_history_text = "\n\n".join(history_items)

        # Format context with clear numbering and scores
        context_items = []
        for i, doc in enumerate(docs):
            score_info = ""
            if 'rerank_score' in doc.metadata:
                score_info = f" (Relevance Score: {doc.metadata['rerank_score']:.2f})"
            context_items.append(f"[Document {i+1}]{score_info}\n{doc.page_content}")
        context_text = "\n\n---\n\n".join(context_items) # Use separator

        # Add refinement info if available
        refinement_info = ""
        if refinement_history:
            num_iterations = len(refinement_history)
            if num_iterations > 1: # Only mention if refinement actually happened
                 refinement_info = f"Note: The search was refined over {num_iterations} iterations to improve results."

        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        try:
            logger.info(f"Generating response for query: '{query[:50]}...' using {len(docs)} documents.")
            start_time = time.time()
            response = chain.invoke({
                "question": query,
                "context": context_text,
                "chat_history": chat_history_text,
                "refinement_info": refinement_info
            }).strip()
            duration = time.time() - start_time
            logger.info(f"Response generated in {duration:.2f}s.")
            return response

        except Exception as e:
            logger.error(f"Error during response generation: {e}", exc_info=True)
            st.error(f"Erro na geração da resposta: {e}")
            return "Desculpe, ocorreu um erro interno ao gerar a resposta."

    def process_query(self, query: str, vector_retriever, bm25_retriever, iterative_rag: IterativeRAG, chat_history: Optional[List[Tuple[str, str]]] = None) -> Tuple[str, Dict, List[Document]]:
        """Executes the full RAG pipeline for a given query."""
        if not self.llm:
            return "Modelo de linguagem não disponível.", {}, []
        if not vector_retriever or not bm25_retriever:
            return "Sistema de recuperação não configurado.", {}, []
        if not iterative_rag:
             return "Sistema de RAG iterativo não configurado.", {}, []

        full_start_time = time.time()
        metrics = {}
        retrieved_docs = []
        response = "Ocorreu um erro inesperado no processamento." # Default error response

        try:
            # --- Step 1: Query Expansion ---
            logger.info("--- RAG Pipeline: Step 1 - Query Expansion ---")
            start_time = time.time()
            expanded_query = self.query_expander.expand_query(query, chat_history)
            metrics["query_expansion_time"] = time.time() - start_time
            metrics["original_query"] = query
            metrics["expanded_query"] = expanded_query
            logger.info(f"Query expansion took {metrics['query_expansion_time']:.2f}s.")

            # --- Step 2: Iterative Retrieval & Re-ranking ---
            logger.info("--- RAG Pipeline: Step 2 - Iterative Retrieval ---")
            start_time = time.time()
            # Use the expanded query for the initial retrieval
            best_docs, refinement_history = iterative_rag.iterative_retrieval(
                vector_retriever,
                bm25_retriever,
                expanded_query # Start iteration with expanded query
            )
            metrics["retrieval_total_time"] = time.time() - start_time # Includes all iterations
            metrics["refinement_history"] = refinement_history
            metrics["num_docs_retrieved_final"] = len(best_docs)
            retrieved_docs = best_docs # Use the best docs found
            logger.info(f"Iterative retrieval took {metrics['retrieval_total_time']:.2f}s.")

            # --- Step 3: Response Generation ---
            logger.info("--- RAG Pipeline: Step 3 - Response Generation ---")
            start_time = time.time()
            # Use the ORIGINAL query for generation, but with context found via expanded/refined query
            response = self.generate_response(query, retrieved_docs, chat_history, refinement_history)
            metrics["generation_time"] = time.time() - start_time
            logger.info(f"Response generation took {metrics['generation_time']:.2f}s.")

            metrics["total_time"] = time.time() - full_start_time
            logger.info(f"--- RAG Pipeline Completed in {metrics['total_time']:.2f}s ---")
            return response, metrics, retrieved_docs

        except Exception as e:
            logger.error(f"Error in RAG process_query: {e}", exc_info=True)
            st.error(f"Erro ao processar a consulta: {e}")
            metrics["error"] = str(e)
            metrics["total_time"] = time.time() - full_start_time
            return response, metrics, retrieved_docs # Return default error and any metrics collected

# --- Streamlit UI ---

def display_debug_info(metrics: Dict, retrieved_docs: List[Document]):
    """Displays debugging information in an expander."""
    with st.expander("🔍 Detalhes do Processamento (Debug)", expanded=False):
        st.markdown('<div class="debug-info">', unsafe_allow_html=True) # Apply custom class

        st.markdown("#### Métricas Gerais")
        col1, col2, col3 = st.columns(3)
        col1.metric("Expansão Query", f"{metrics.get('query_expansion_time', 0):.2f}s")
        col2.metric("Retrieval Total", f"{metrics.get('retrieval_total_time', 0):.2f}s")
        col3.metric("Geração", f"{metrics.get('generation_time', 0):.2f}s")
        st.metric("Tempo Total", f"{metrics.get('total_time', 0):.2f}s", label_visibility="visible")
        st.text(f"Documentos usados na resposta: {metrics.get('num_docs_retrieved_final', 0)}")

        st.markdown("---")
        st.markdown("#### Query Expansion")
        st.text("Original:")
        st.code(metrics.get('original_query', 'N/A'), language=None)
        st.text("Expandida/Inicial:")
        st.code(metrics.get('expanded_query', 'N/A'), language=None)

        st.markdown("---")
        st.markdown("#### Recuperação Iterativa")
        history = metrics.get('refinement_history', [])
        if not history:
            st.text("Nenhuma iteração de refinamento registrada.")
        else:
            for i, iteration in enumerate(history):
                st.markdown(f"**Iteração {i+1}**")
                st.text(f"Query: {iteration.get('query', 'N/A')[:150]}...") # Show snippet
                st.text(f"Score Relevância (LLM): {iteration.get('relevance_score', 'N/A')}/10")
                st.text(f"Tempo: {iteration.get('retrieval_time', 0):.2f}s (Ret) + {iteration.get('reranking_time', 0):.2f}s (Rank)")
                st.text(f"Docs: {iteration.get('num_retrieved', 0)} (Ret) -> {iteration.get('num_reranked', 0)} (Rank)")
                if iteration.get('needs_refinement', False):
                    st.text(f"Refinamento Sugerido: {iteration.get('suggested_query', 'N/A')[:100]}...")
                else:
                     st.text("Refinamento: Não necessário nesta iteração.")
                if i < len(history) - 1:
                    st.divider()

        st.markdown("---")
        st.markdown(f"#### Documentos Finais Recuperados ({len(retrieved_docs)})")
        if not retrieved_docs:
            st.text("Nenhum documento final recuperado.")
        else:
            for i, doc in enumerate(retrieved_docs):
                 score = doc.metadata.get('rerank_score', None)
                 score_str = f"(Score: {score:.2f})" if score is not None else "(Score: N/A)"
                 with st.popover(f"Doc {i+1} {score_str}", use_container_width=True):
                     st.markdown(f"**Documento {i+1} {score_str}**")
                     # Use markdown code block for better formatting and scrollability
                     st.markdown(f"```\n{doc.page_content}\n```")
                     st.caption(f"Metadados: {doc.metadata}")

        st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main Streamlit application function."""
    st.title("🧠 Sistema RAG Avançado")
    st.caption("Faça perguntas sobre o documento PDF carregado.")

    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Controles")

        st.info(f"Modelo LLM configurado: `{st.session_state.selected_model_id}`")
        if "gemma-7b" in st.session_state.selected_model_id and not torch.cuda.is_available():
             st.warning("⚠️ Gemma-7B na CPU será muito lento!")

        # PDF Upload
        pdf_file = st.file_uploader("Carregar Documento PDF", type=["pdf"], key="pdf_uploader")

        # Document Processing Logic
        if pdf_file and (not st.session_state.document_loaded or pdf_file.name != st.session_state.document_name):
            st.session_state.document_loaded = False # Mark as not loaded until success
            st.session_state.messages = [] # Clear chat on new doc
            st.session_state.chat_history = []
            logger.info(f"New PDF uploaded: {pdf_file.name}. Starting processing.")
            with st.spinner(f"Processando '{pdf_file.name}'... (Modelo grande pode demorar na 1ª vez)"):
                try:
                    # --- 1. Load LLM (if not already loaded) ---
                    if not st.session_state.model_loaded or not st.session_state.llm:
                        logger.info("LLM not loaded. Attempting to load...")
                        model, tokenizer, is_seq2seq = load_base_model_and_tokenizer(st.session_state.selected_model_id)
                        if model and tokenizer:
                            llm_pipeline = create_llm_pipeline(model, tokenizer, is_seq2seq)
                            if llm_pipeline:
                                st.session_state.llm = llm_pipeline
                                st.session_state.model_loaded = True
                                logger.info("LLM Pipeline successfully created and stored in session state.")
                            else:
                                raise ValueError("Falha ao criar o pipeline LLM após carregar o modelo.")
                        else:
                             # Error message already shown in load_base_model_and_tokenizer
                             raise ValueError(f"Falha crítica ao carregar o modelo base {st.session_state.selected_model_id}.")

                    # --- 2. Process Document ---
                    doc_processor = DocumentProcessor()
                    splits = doc_processor.process_pdf(pdf_file)

                    if not splits:
                        st.error("Não foi possível extrair conteúdo do PDF.")
                        raise ValueError("Document processing failed, no splits generated.")

                    st.session_state.doc_stats = doc_processor.get_document_stats(splits)

                    # --- 3. Setup Retrieval System ---
                    retrieval_system = RetrievalSystem() # Loads embedding/reranker models
                    vector_retriever, bm25_retriever = retrieval_system.setup_retrievers(splits)

                    if not vector_retriever or not bm25_retriever:
                         raise ValueError("Falha ao configurar os sistemas de recuperação (vector/BM25).")

                    st.session_state.vector_retriever = vector_retriever
                    st.session_state.bm25_retriever = bm25_retriever
                    st.session_state.retrieval_system = retrieval_system

                    # --- 4. Setup RAG Components (depend on LLM and Retrievers) ---
                    if st.session_state.llm:
                        query_expander = QueryExpander(st.session_state.llm)
                        iterative_rag = IterativeRAG(st.session_state.llm, st.session_state.retrieval_system)
                        rag_pipeline = RAGPipeline(st.session_state.llm, query_expander)

                        st.session_state.iterative_rag = iterative_rag
                        st.session_state.rag_pipeline = rag_pipeline
                        logger.info("Full RAG pipeline components initialized.")
                    else:
                        logger.error("LLM is None after attempting load. RAG pipeline cannot be fully initialized.")
                        st.error("Erro: Modelo LLM não está disponível. Funcionalidade RAG completa desativada.")
                        st.session_state.iterative_rag = None
                        st.session_state.rag_pipeline = None


                    # --- 5. Finalize ---
                    st.session_state.document_loaded = True
                    st.session_state.document_name = pdf_file.name
                    st.success(f"Documento '{pdf_file.name}' processado!")
                    logger.info(f"Document '{pdf_file.name}' processed successfully.")

                    # Display Doc Stats
                    stats = st.session_state.doc_stats
                    st.markdown("##### Estatísticas do Documento")
                    st.write(f"- Chunks: {stats.get('total_chunks', 'N/A')}")
                    st.write(f"- Caracteres: {stats.get('total_chars', 'N/A')}")
                    st.write(f"- Média Carac./Chunk: {stats.get('avg_chunk_chars', 0):.1f}")

                    # >>> CHANGED TO st.rerun() <<<
                    st.rerun()

                except Exception as e:
                    st.session_state.document_loaded = False # Ensure it's marked as failed
                    st.session_state.document_name = None
                    logger.error(f"Failed to process document '{pdf_file.name}': {e}", exc_info=True)
                    st.error(f"Erro ao processar documento: {e}")
                    # Clear potentially partially initialized state
                    keys_to_clear = ["vector_retriever", "bm25_retriever", "retrieval_system", "iterative_rag", "rag_pipeline", "doc_stats", "llm", "model_loaded"]
                    for key in keys_to_clear:
                         if key in st.session_state:
                              del st.session_state[key]


        # Display document status
        if st.session_state.document_loaded and st.session_state.document_name:
            st.success(f"Documento carregado: **{st.session_state.document_name}**")
        elif not pdf_file:
             st.info("⬆️ Carregue um documento PDF para começar.")


        # Debug Mode Toggle
        st.toggle("Modo de Depuração", key="debug_mode", value=st.session_state.debug_mode)

        # New Conversation Button
        if st.button("🗑️ Nova Conversa"):
            logger.info("Starting new conversation.")
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.conversation_id = str(uuid.uuid4())
            # Clear last query metrics if any
            st.session_state.processing_times = {}
            # >>> CHANGED TO st.rerun() <<<
            st.rerun()

        # About Section
        with st.expander("ℹ️ Sobre", expanded=False):
            st.markdown("""
            Este sistema usa **Retrieval-Augmented Generation (RAG)** para responder perguntas sobre um documento PDF.

            **Técnicas Avançadas:**
            *   **Hybrid Search:** Combina busca vetorial (semântica) e BM25 (palavras-chave).
            *   **Query Expansion:** Reformula sua pergunta usando um LLM para buscar melhor.
            *   **Re-ranking:** Usa um modelo Cross-Encoder para priorizar os resultados mais relevantes.
            *   **Iterative Retrieval:** Analisa os resultados e refina a busca automaticamente se necessário.
            """)

    # --- Main Chat Area ---
    chat_container = st.container()

    with chat_container:
        # Display existing messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"]) # Use markdown for better rendering
                # Display debug info below assistant messages if enabled
                if msg["role"] == "assistant" and st.session_state.debug_mode and "metrics" in msg:
                    display_debug_info(msg["metrics"], msg.get("retrieved_docs", []))


    # Chat Input - Disable if document not loaded or RAG pipeline not ready
    input_disabled = not st.session_state.document_loaded or not st.session_state.get("rag_pipeline")
    user_input = st.chat_input(
        "Faça sua pergunta sobre o documento...",
        key="chat_input",
        disabled=input_disabled
    )

    if input_disabled and st.session_state.document_loaded and not st.session_state.get("rag_pipeline"):
         # Check if LLM failed loading specifically
         if not st.session_state.get("llm") and not st.session_state.get("model_loaded"):
              st.error("Falha ao carregar o modelo LLM. Verifique os logs e as dependências.")
         else:
              st.warning("O sistema RAG ainda não está pronto. Verifique o carregamento do documento e do modelo.")


    # Process user input
    if user_input:
        logger.info(f"User query received: {user_input}")
        # Add user message to state and display it immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
             with st.chat_message("user"):
                  st.markdown(user_input)

        # Show thinking indicator
        with chat_container:
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                thinking_placeholder.markdown("🧠 Pensando...")

        # Process query using RAG pipeline
        try:
            rag_pipeline = st.session_state.rag_pipeline
            vector_retriever = st.session_state.vector_retriever
            bm25_retriever = st.session_state.bm25_retriever
            iterative_rag = st.session_state.iterative_rag

            if not all([rag_pipeline, vector_retriever, bm25_retriever, iterative_rag]):
                 # Check specifically if LLM is missing
                 if not st.session_state.llm:
                      raise ValueError("Modelo LLM não está carregado ou inicializado corretamente.")
                 else:
                      raise ValueError("Componentes RAG essenciais não estão inicializados na sessão.")

            response, metrics, retrieved_docs = rag_pipeline.process_query(
                user_input,
                vector_retriever,
                bm25_retriever,
                iterative_rag,
                st.session_state.chat_history # Pass current history
            )

            # Store results and update UI
            st.session_state.processing_times = metrics # Store metrics for the last query
            st.session_state.evaluation_metrics.append(metrics) # Keep history if needed

            # Add assistant response to state
            assistant_message = {
                "role": "assistant",
                "content": response,
                "metrics": metrics, # Attach metrics for debug display
                "retrieved_docs": retrieved_docs
            }
            st.session_state.messages.append(assistant_message)

            # Update chat history (limited length)
            st.session_state.chat_history.append((user_input, response))
            if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
                st.session_state.chat_history.pop(0) # Remove oldest

            # Update the placeholder with the actual response
            thinking_placeholder.markdown(response)

            # Display debug info if enabled (will appear below the new message)
            # We need to rerun for the debug info expander to be placed correctly after the message
            if st.session_state.debug_mode:
                 # >>> CHANGED TO st.rerun() <<<
                 st.rerun() # Rerun to render the debug info correctly below the message


        except Exception as e:
            logger.error(f"Error processing user query '{user_input}': {e}", exc_info=True)
            error_message = f"Desculpe, ocorreu um erro ao processar sua pergunta: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            thinking_placeholder.error(error_message)
            # Optionally rerun to show the error message properly if layout issues occur
            # st.rerun()


# --- Application Entry Point ---
if __name__ == "__main__":
    # Basic check for dependencies maybe?
    try:
        import langchain
        import transformers
        import sentence_transformers
        import accelerate # Check if accelerate is installed
        logger.info(f"Langchain: {langchain.__version__}, Transformers: {transformers.__version__}, SentenceTransformers: {sentence_transformers.__version__}, Accelerate: {accelerate.__version__}")
        logger.info(f"Torch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
             logger.warning("CUDA not available. Running on CPU.")
    except ImportError as e:
        st.error(f"Erro de importação. Verifique as dependências: {e}. Certifique-se de instalar 'accelerate' e 'bitsandbytes': pip install accelerate bitsandbytes")
        logger.error(f"Import error: {e}", exc_info=True)
        sys.exit(1) # Stop if core libs are missing

    try:
        main()
    except Exception as e:
        # Catch unexpected errors during app execution
        logger.critical(f"Critical error in Streamlit app execution: {e}", exc_info=True)
        st.error(f"Erro crítico na aplicação: {e}")
        st.warning("Tente recarregar a página ou reiniciar o aplicativo.")