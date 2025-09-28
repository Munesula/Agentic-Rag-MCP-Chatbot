# mcp_agents.py

import os
from typing import Dict, Any, List
# Core RAG Utilities
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION (OPTIMIZED for speed on local CPU) ---
# Reliable public embedding model for local RAG
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
# SWAPPED to the lighter 2.0 GB model for faster inference
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2") 
DB_PATH = "./faiss_index"

# --- Model Context Protocol (MCP) Helper ---
def create_mcp_message(sender: str, receiver: str, mcp_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Creates a structured message object for agent-to-agent communication."""
    return {
        "sender": sender,
        "receiver": receiver,
        "type": mcp_type,
        "trace_id": os.urandom(4).hex(),
        "payload": payload
    }

# --- 1. IngestionAgent: Parses & Preprocesses Documents ---
class IngestionAgent:
    """Handles multi-format document loading and text splitting."""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )

    def process_file(self, file_path: str) -> List[Any]:
        """Loads and chunks a document based on its extension."""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        # Handles TXT, Markdown, and CSV as plain text
        elif file_path.endswith(('.txt', '.md', '.csv')): 
            loader = TextLoader(file_path)
        else:
            print(f"Skipping unsupported file format: {file_path}")
            return []
        
        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        return chunks

    def execute(self, mcp_in: Dict[str, Any]) -> Dict[str, Any]:
        file_paths = mcp_in['payload']['file_paths']
        all_chunks = []
        for path in file_paths:
            all_chunks.extend(self.process_file(path))
        
        return create_mcp_message(
            sender="IngestionAgent",
            receiver="RetrievalAgent",
            mcp_type="CONTEXT_READY",
            payload={"chunks": all_chunks, "query": None} 
        )

# --- 2. RetrievalAgent: Embeddings, Vector Store, and Semantic Search ---
class RetrievalAgent:
    """Manages the Vector Store (FAISS) and retrieves relevant context chunks."""

    def __init__(self):
        # NOTE: SSL check is bypassed by environment variables set in the terminal
        self.embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    def initialize_db(self, chunks: List[Any]):
        """Creates and saves the FAISS vector store."""
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(folder_path=DB_PATH)
    
    def retrieve_context(self, query: str) -> Dict[str, List[str]]:
        """Performs semantic search to retrieve context."""
        try:
            # Loads DB from disk after creation
            vector_store = FAISS.load_local(DB_PATH, self.embeddings, allow_dangerous_deserialization=True)
            retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return {"context": [], "sources": []}

        retrieved_docs = retriever.invoke(query)
        
        context_list = []
        sources_list = []
        for doc in retrieved_docs:
            source = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', 'N/A')
            context_list.append(f"Source: {source} (Page {page}) | Content: {doc.page_content}")
            sources_list.append(f"{source} (Page {page})")
            
        return {"context": context_list, "sources": sources_list}

    def execute(self, mcp_in: Dict[str, Any]) -> Dict[str, Any]:
        """Handles both DB creation (CONTEXT_READY) and search (RETRIEVAL_REQUEST) modes."""
        
        if mcp_in['type'] == "CONTEXT_READY":
            # PHASE 1: Build DB
            self.initialize_db(mcp_in['payload']['chunks'])
            return create_mcp_message(
                sender="RetrievalAgent", receiver="Coordinator", mcp_type="DB_INITIALIZED",
                payload={"status": "Vector database created successfully."}
            )
        
        elif mcp_in['type'] == "RETRIEVAL_REQUEST":
            # PHASE 2: Retrieve Context
            query = mcp_in['payload']['query']
            context_data = self.retrieve_context(query)
            
            # Send RETRIEVAL_RESULT message to LLMResponseAgent
            return create_mcp_message(
                sender="RetrievalAgent", receiver="LLMResponseAgent", mcp_type="RETRIEVAL_RESULT",
                payload={"retrieved_context": context_data['context'], "sources": context_data['sources'], "query": query}
            )
        
        raise ValueError(f"RetrievalAgent received unknown MCP type: {mcp_in['type']}")


# mcp_agents.py (ONLY the LLMResponseAgent Class)

# ... (rest of imports and RetrievalAgent class above) ...

# --- 3. LLMResponseAgent: Final Query Construction and Answer Generation ---
class LLMResponseAgent:
    """Forms the final prompt with context and calls the LLM (Ollama)."""

    def __init__(self):
        # --- OPTIMIZED FOR MAXIMUM SPEED ON LOCAL CPU ---
        self.llm = Ollama(
            model=LLM_MODEL, 
            temperature=0.4,          # Maintained temperature for balance
            num_predict=150,          # CRUCIAL: Limits response length for FAST answers (Max 150 tokens)
            stop=["Final Answer:", "\n\n", "---"], # Stops generation early to save time
        )

    def create_final_prompt(self, context: List[str], query: str) -> str:
        """Custom prompt to instruct the LLM on RAG behavior."""
        prompt_template = """
        You are an expert Q&A system. Your sole function is to provide an accurate, 
        concise answer to the user's question based *ONLY* on the provided context.
        Ensure your answer is brief and to the point.
        If the context does not contain the answer, you MUST state, "The required 
        information is not available in the provided documents."
        
        --- CONTEXT FROM DOCUMENTS ---
        {context}
        ---
        
        User Question: {query}
        
        Final Answer:
        """
        return prompt_template.format(context="\n---\n".join(context), query=query)

    def execute(self, mcp_in: Dict[str, Any]) -> Dict[str, Any]:
        """Generates the final answer and sends FINAL_ANSWER message to Coordinator."""
        context = mcp_in['payload']['retrieved_context']
        sources = mcp_in['payload']['sources']
        query = mcp_in['payload']['query']
        
        final_prompt = self.create_final_prompt(context, query)
        
        # Direct LLM invocation using the local Ollama server (llama3.2)
        response_text = self.llm.invoke(final_prompt)
        
        return create_mcp_message(
            sender="LLMResponseAgent", receiver="Coordinator", mcp_type="FINAL_ANSWER",
            payload={"answer": response_text.strip(), "sources": list(set(sources))}
        )