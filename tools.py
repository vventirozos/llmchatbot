# tools.py
import os
import json
import asyncio
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, List

from langchain.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain_community.tools import WikipediaQueryRun, QuerySQLDatabaseTool
from langchain_tavily import TavilySearch
from langchain_community.utilities import WikipediaAPIWrapper, SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.retriever import BaseRetriever
from langchain.schema.document import Document


import logging

logger = logging.getLogger(__name__)

# --- RAG Setup ---
# Note: You may need to install faiss-cpu: pip install faiss-cpu
RAG_DATA_DIR = "rag_documents"
FAISS_INDEX_PATH = "faiss_index"

class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        return []
    async def _aget_relevant_documents(self, query: str, *, run_manager) -> List[Document]:
        return []

def setup_rag_retriever() -> BaseRetriever:
    """
    Sets up the RAG retriever. Creates a FAISS index if it doesn't exist.
    """
    if not os.path.exists(RAG_DATA_DIR) or not os.listdir(RAG_DATA_DIR):
        logger.warning(f"RAG data directory '{RAG_DATA_DIR}' is empty or does not exist. RAG tool will not be effective.")
        # Return a dummy retriever that does nothing
        return DummyRetriever()

    embeddings = OllamaEmbeddings(model=os.getenv("DEFAULT_MODEL", 'granite4:micro-h'))
    
    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("Loading existing FAISS index.")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.info("Creating new FAISS index.")
        loader = DirectoryLoader(RAG_DATA_DIR, glob="**/*.txt")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(FAISS_INDEX_PATH)
        logger.info(f"FAISS index created and saved to {FAISS_INDEX_PATH}")

    return vector_store.as_retriever()

# Initialize the RAG retriever on startup
rag_retriever = setup_rag_retriever()

@tool
def rag_search(query: str) -> Dict[str, Any]:
    """
    Searches the local RAG (Retrieval-Augmented Generation) document collection.
    Use this to answer questions about information stored locally.
    """
    try:
        if not isinstance(rag_retriever, DummyRetriever):
            results = rag_retriever.invoke(query)
            content = "\n".join([doc.page_content for doc in results])
            logger.info(f"RAG search completed for query: '{query}'")
            return {
                "success": True,
                "results": content,
                "document_count": len(results)
            }
        else:
            return {
                "success": False,
                "error": "RAG system is not configured. The document directory is empty."
            }
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return {"success": False, "error": str(e)}


# --- Existing Tools ---

# Database setup
db = SQLDatabase.from_uri(os.getenv("DATABASE_URL"))

# Security configuration
ALLOWED_DIRS = [os.getenv("ALLOWED_WORK_DIR", "./workspace")]
os.makedirs(ALLOWED_DIRS[0], exist_ok=True)

# Dangerous Python patterns to block
DANGEROUS_PATTERNS = []


@tool
def sql_query(query: str) -> Dict[str, Any]:
    """
    Executes a READ-ONLY SQL query against the PostgreSQL database.
    Returns structured result with success status.
    """
    if query.strip().upper().startswith("DROP"):
        logger.warning(f"Blocked DROP query: {query[:50]}...")
        return {
            "success": False,
            "error": "DROP queries are not allowed for security reasons."
        }
    
    try:
        tool = QuerySQLDatabaseTool(db=db)
        result = tool.run(query)
        logger.info(f"SQL query executed successfully: {query[:50]}...")
        return {
            "success": True,
            "result": result,
            "row_count": len(result.split('\n')) if result else 0
        }
    except Exception as e:
        logger.error(f"SQL query error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@tool
def search(query: str) -> Dict[str, Any]:
    """
    Searches the web for the given query using Tavily.
    Returns structured results.
    """
    try:
        search_wrapper = TavilySearch()
        results = search_wrapper.invoke(query)
        logger.info(f"Web search completed: '{query}'")
        return {
            "success": True,
            "results": results,
            "result_count": len(results) if isinstance(results, list) else 0
        }
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@tool
@lru_cache(maxsize=100)
def wikipedia(query: str) -> Dict[str, Any]:
    """
    Searches Wikipedia for the given query with result caching.
    Returns structured content.
    """
    try:
        api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=4000)
        wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
        result = wiki.run(query)
        logger.info(f"Wikipedia search completed: '{query}'")
        return {
            "success": True,
            "content": result,
            "query": query
        }
    except Exception as e:
        logger.error(f"Wikipedia error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@tool
def web_fetch(url: str) -> Dict[str, Any]:
    """
    Fetches content from a URL with timeout and size limits.
    Returns structured content or error.
    """
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        content = "".join(doc.page_content for doc in docs)[:8000]
        logger.info(f"Web fetch completed: {url}")
        return {
            "success": True,
            "content": content,
            "url": url,
            "length": len(content)
        }
    except Exception as e:
        logger.error(f"Web fetch error for {url}: {e}")
        return {
            "success": False,
            "error": str(e),
            "url": url
        }

@tool
def python_interpreter(code: str) -> Dict[str, Any]:
    """
    Executes Python code safely in a restricted environment with timeout.
    Blocks dangerous operations and limits execution time.
    """
    for pattern in DANGEROUS_PATTERNS:
        if pattern in code:
            logger.warning(f"Blocked dangerous code pattern: {pattern}")
            return {
                "success": False,
                "error": f"Security error: Prohibited operation detected ({pattern})"
            }
    
    try:
        import subprocess
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=ALLOWED_DIRS[0],
            env={**os.environ, "PYTHONPATH": ""}
        )
        
        output = result.stdout if result.returncode == 0 else result.stderr
        logger.info(f"Python code executed (exit code: {result.returncode})")
        return {
            "success": result.returncode == 0,
            "output": output[:8000],
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        logger.warning("Python execution timed out")
        return {
            "success": False,
            "error": "Execution timed out after 10 seconds"
        }
    except Exception as e:
        logger.error(f"Python execution error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@tool
def read_file(file_path: str) -> Dict[str, Any]:
    """
    Reads a file from the allowed workspace directory only.
    Prevents directory traversal attacks.
    """
    try:
        safe_path = Path(ALLOWED_DIRS[0]) / Path(file_path).name
        safe_path = safe_path.resolve()
        
        if not str(safe_path).startswith(str(Path(ALLOWED_DIRS[0]).resolve())):
            logger.warning(f"Blocked path traversal attempt: {file_path}")
            return {
                "success": False,
                "error": "Access denied: Cannot read files outside workspace"
            }
        
        content = safe_path.read_text()
        logger.info(f"File read successfully: {safe_path}")
        return {
            "success": True,
            "content": content[:8000],
            "path": str(safe_path),
            "size": len(content)
        }
    except Exception as e:
        logger.error(f"File read error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@tool
def write_file(file_path: str, content: str) -> Dict[str, Any]:
    """
    Writes content to a file in the allowed workspace directory only.
    Prevents directory traversal attacks.
    """
    try:
        safe_path = Path(ALLOWED_DIRS[0]) / Path(file_path).name
        safe_path = safe_path.resolve()
        
        if not str(safe_path).startswith(str(Path(ALLOWED_DIRS[0]).resolve())):
            logger.warning(f"Blocked path traversal attempt: {file_path}")
            return {
                "success": False,
                "error": "Access denied: Cannot write files outside workspace"
            }
        
        safe_path.write_text(content)
        logger.info(f"File written successfully: {safe_path}")
        return {
            "success": True,
            "message": f"File {safe_path.name} written successfully",
            "path": str(safe_path),
            "size": len(content)
        }
    except Exception as e:
        logger.error(f"File write error: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_tools():
    """Returns a list of all available tools."""
    return [search, wikipedia, web_fetch, python_interpreter, read_file, write_file, sql_query, rag_search]