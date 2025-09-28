🤖 Agentic RAG Chatbot for Multi-Format QA
🚀 Project Overview
This repository contains the complete solution for the Agentic Retrieval-Augmented Generation (RAG) Chatbot challenge. The architecture strictly adheres to a modular, three-agent structure using the Model Context Protocol (MCP) for communication.

The system answers complex user queries using uploaded, private documents (PDF, DOCX, CSV, etc.).

Core Features
Agentic Architecture: Implements the required Ingestion, Retrieval, and LLM Response agents.

Protocol Adherence: Uses a custom implementation of the Model Context Protocol (MCP) for structured communication.

Data Privacy: Utilizes Ollama (llama3.2) and FAISS to run the entire RAG pipeline locally.

Multi-Format Support: Handles PDF, DOCX, CSV, TXT, and Markdown files.

Source Citation: Answers include visible references to the source documents and page numbers.

🛠️ Technology Stack
Codebase: Python 3.x

Repository: Munesula/Agentic-Rag-MCP-Chatbot

UI / Coordinator: Streamlit

LLM (Generation): Ollama (llama3.2:latest)

Vector Database: FAISS

Embeddings: Sentence Transformers (all-MiniLM-L6-v2)

⚙️ Installation and Execution
Prerequisites
Ollama: Must be installed and the server running.

Model Pull: Ensure the required LLM is downloaded via the terminal:

Bash

ollama pull llama3.2
Setup Steps
Clone the Repository:

Bash

git clone https://github.com/Munesula/Agentic-Rag-MCP-Chatbot
cd Agentic-Rag-MCP-Chatbot
Activate Virtual Environment:

Bash

python -m venv venv
venv\Scripts\activate  # Windows
Install Dependencies:

Bash

pip install -r requirements.txt
Launch Sequence (CRITICAL)
Bypass SSL (If Necessary): Run these commands in the terminal before launching Streamlit:

Bash

set REQUESTS_CA_BUNDLE=
set CURL_CA_BUNDLE=
Start the Ollama Server: Open a separate terminal and run:

Bash

ollama serve
Run the Chatbot (The Coordinator): In your activated Python terminal, launch the application:

Bash

streamlit run streamlit_app.py
🗺️ Architectural Flow (Model Context Protocol)
The system operates as a sequential, event-driven pipeline managed by the Streamlit Coordinator.

UI Request: User asks a question in the Streamlit UI.

Retrieval Step: Coordinator sends a RETRIEVAL_REQUEST (MCP message) to the Retrieval Agent.

Context Hand-off: Retrieval Agent searches FAISS, finds relevant chunks, and passes them to the LLM Response Agent via a RETRIEVAL_RESULT (MCP message).

Final Generation: LLM Response Agent calls the Ollama LLM to synthesize the final answer.

💡 Challenges and Performance Optimizations
1. Solved Engineering Challenges
Successfully resolved complex SSL Certificate Verification Errors and Keras/TensorFlow version incompatibilities necessary to load the embedding model, demonstrating strong debugging ability.

2. Performance Tuning
Model Optimization: Swapped the LLM to the lighter, faster llama3.2 model.

Latency Fix: The LLMResponseAgent is configured with a strict num_predict=256 token limit to deliver rapid, concise answers.