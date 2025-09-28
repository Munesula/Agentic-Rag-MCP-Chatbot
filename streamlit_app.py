# streamlit_app.py

import streamlit as st
import tempfile
import shutil
import os
# Ensure mcp_agents.py is in the same directory
from mcp_agents import IngestionAgent, RetrievalAgent, LLMResponseAgent, create_mcp_message

# --- Agent Initialization ---
# Use st.cache_resource to initialize agents only once
@st.cache_resource
def initialize_agents():
    """Initializes agents once for efficiency."""
    return IngestionAgent(), RetrievalAgent(), LLMResponseAgent()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Agentic RAG Chatbot (MCP)", layout="wide")
st.title("Agentic RAG Chatbot for Multi-Format QA")
st.caption("Architecture: 3 Agents using Model Context Protocol (MCP)")

# Initialize state and agents
if "agents" not in st.session_state:
    st.session_state.agents = initialize_agents()
    st.session_state.ingestion_complete = False
    st.session_state.messages = []
    st.session_state.file_paths = []

ingestion_agent, retrieval_agent, llm_agent = st.session_state.agents

# --- UI Sidebar for Ingestion (COORDINATOR Flow) ---
with st.sidebar:
    st.header("1. Document Ingestion")
    
    # File Uploader component
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, DOCX, TXT, CSV, MD)", 
        type=['pdf', 'docx', 'txt', 'csv', 'md'], 
        accept_multiple_files=True
    )
    
    if st.button("Process & Build Vector DB", use_container_width=True, type="primary", disabled=st.session_state.ingestion_complete):
        if not uploaded_files:
            st.error("Please upload at least one document.")
            st.stop()

        with st.spinner("Executing IngestionAgent & RetrievalAgent..."):
            # Save uploaded files to a temp directory
            temp_dir = tempfile.mkdtemp()
            st.session_state.file_paths = []
            for uploaded_file in uploaded_files:
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.session_state.file_paths.append(temp_path)

            try:
                # --- COORDINATOR Step 1: Trigger IngestionAgent ---
                with st.status("IngestionAgent: Parsing and chunking documents...", expanded=True) as status_ingestion:
                    ingestion_mcp_in = create_mcp_message(
                        sender="Coordinator", receiver="IngestionAgent", mcp_type="INGEST_DOCS",
                        payload={"file_paths": st.session_state.file_paths}
                    )
                    retrieval_mcp_in = ingestion_agent.execute(ingestion_mcp_in)
                    status_ingestion.update(label="IngestionAgent: Documents chunked.", state="complete")
                    
                # --- COORDINATOR Step 2: Trigger RetrievalAgent (DB Init) ---
                with st.status("RetrievalAgent: Creating Vector DB (FAISS)...", expanded=True) as status_retrieval:
                    retrieval_agent.execute(retrieval_mcp_in)
                    st.session_state.ingestion_complete = True
                    status_retrieval.update(label="RetrievalAgent: Vector DB created and indexed.", state="complete")
                    
                st.success(f"Ingestion Complete. Ready to chat! ({len(st.session_state.file_paths)} files processed)")

            except Exception as e:
                st.error(f"An error occurred during ingestion. Is Ollama running? Error: {e}")
                st.session_state.ingestion_complete = False
            finally:
                # Clean up the temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                
    st.markdown("---")
    st.button("Clear Chat History", on_click=lambda: st.session_state.messages.clear(), use_container_width=True)

# --- Main Chat Interface (COORDINATOR Flow for Q&A) ---
if not st.session_state.ingestion_complete:
    st.info("👆 Please upload and process documents in the sidebar to begin chatting.")
else:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                # Renders the sources for all *past* messages
                with st.expander(" Sources Used (Required Detail)"):
                    # FINAL FIX: Uses robust Markdown for display
                    if message["sources"]:
                        st.markdown("\n".join([f"- **{s}**" for s in message["sources"]]))
                    else:
                        st.markdown("*No specific source documents were retrieved for this query.*")


    # Handle user input (Multi-turn question support)
    if prompt := st.chat_input("Ask a question about the documents..."):
        # Append user message to chat history for re-run
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Start the assistant process block
        with st.chat_message("assistant"):
            with st.spinner("Coordinating Agents..."):
                try:
                    # --- COORDINATOR Step 3: Trigger RetrievalAgent (Query Mode) ---
                    with st.status("RetrievalAgent: Searching Vector DB for context...", expanded=True) as status_retrieval:
                        retrieval_request_mcp = create_mcp_message(
                            sender="Coordinator", receiver="RetrievalAgent", mcp_type="RETRIEVAL_REQUEST",
                            payload={"query": prompt}
                        )
                        llm_mcp_in = retrieval_agent.execute(retrieval_request_mcp)
                        status_retrieval.update(label="RetrievalAgent: Context retrieved.", state="complete")
                        
                    # --- COORDINATOR Step 4: Trigger LLMResponseAgent ---
                    with st.status("LLMResponseAgent: Generating final answer with context...", expanded=True) as status_llm:
                        final_mcp_out = llm_agent.execute(llm_mcp_in)
                        status_llm.update(label="LLMResponseAgent: Answer generated.", state="complete")

                    # --- COORDINATOR Step 5: Deliver FINAL_ANSWER to UI ---
                    response_payload = final_mcp_out['payload']
                    answer = response_payload['answer']
                    sources = response_payload['sources']
                    
                    # Display the answer in the current chat container
                    st.markdown(answer)
                    
                    # Display the sources directly below the answer
                    with st.expander(" Sources Used (Required Detail)"):
                        if sources:
                            st.markdown("\n".join([f"- **{s}**" for s in sources]))
                        else:
                            st.markdown("*No specific source documents were retrieved for this query.*")
                        
                    # CRUCIAL FIX: Append the complete message object for persistence
                    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
                    
                except Exception as e:
                    error_message = f"An error occurred during query: {e}. Check if the Ollama server is running."
                    st.error(error_message)

                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I encountered an internal error. Details: {error_message}"})
