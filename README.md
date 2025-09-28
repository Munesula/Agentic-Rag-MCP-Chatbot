🤖 Agentic RAG Chatbot for Multi-Format QA
🚀 Project Overview
This repository contains the complete solution for building a production-ready Agentic Retrieval-Augmented Generation (RAG) Chatbot. The system is designed to answer complex user queries based on private, uploaded documents of various formats (PDF, DOCX, CSV, etc.).

The architecture strictly adheres to a modular, three-agent structure, communicating via the Model Context Protocol (MCP), demonstrating a commitment to scalable, decoupled design.

Core Features
Agentic Architecture: Implements the required Ingestion, Retrieval, and LLM Response agents.

Protocol Adherence: Uses a custom implementation of the Model Context Protocol (MCP) for structured inter-agent communication.

Multi-Format Support: Handles PDF, DOCX, TXT, CSV, and Markdown files.

Local & Private: Utilizes Ollama and FAISS to run the entire RAG pipeline locally, ensuring data privacy and eliminating cloud costs.

Source Citation: Answers include visible references to the source documents and page pages.

🛠️ Technology Stack
Component	Technology / Model	Role in System
Repository Name	Munesula/Agentic-Rag-MCP-Chatbot	The hosted location of this project.
Orchestration	Python / Streamlit	Acts as the Coordinator Agent and UI.
LLM (Generation)	Ollama (llama3.2:latest)	Generates the final, contextualized answer.
Embeddings	Sentence Transformers (all-MiniLM-L6-v2)	Converts text chunks to numerical vectors.
Vector Database	FAISS	In-memory/file-based index for fast semantic retrieval.
Frameworks	LangChain	Provides robust document loading and splitting utilities.

Export to Sheets
⚙️ Installation and Execution
Prerequisites
Ollama: Must be installed and the server must be actively running (ollama serve) during application use.

Model Pull: Ensure the required LLM is downloaded via the terminal: ollama pull llama3.2

Python: Python 3.8+ is required.

Setup Steps
Clone the Repository:

Bash

git clone https://github.com/Munesula/Agentic-Rag-MCP-Chatbot
cd Agentic-Rag-MCP-Chatbot
Create and Activate Virtual Environment:

Bash

python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate # Mac/Linux
Install Dependencies:

Bash

pip install -r requirements.txt
Launch Sequence (CRITICAL)
Bypass SSL (If Necessary): Run these commands in the terminal before the next step to prevent SSL errors:

Bash

set REQUESTS_CA_BUNDLE=
set CURL_CA_BUNDLE=
Start the Ollama Server: Open a separate terminal and run: ollama serve (Keep this window open!).

Run the Chatbot (The Coordinator): In your Python environment terminal, launch the application:

Bash

streamlit run streamlit_app.py
🗺️ Architectural Flow (Model Context Protocol)
The system operates as a sequential pipeline managed by the Coordinator, using structured messages for inter-agent communication.

Step	Agent / Component	Message Passed (MCP Type)	Description
1.	Coordinator (UI)	→ RETRIEVAL_REQUEST	Sends user query to the retrieval layer.
2.	Retrieval Agent	→ RETRIEVAL_RESULT	Searches FAISS and packages retrieved context chunks + query.
3.	LLM Response Agent	→ [Internal API Call]	Constructs the final RAG prompt and calls the Ollama LLM (llama3.2).
4.	LLM Response Agent	→ FINAL_ANSWER	Returns the generated text answer and the required source citations to the UI.

Export to Sheets
💡 Challenges and Performance Optimizations
1. Solved Engineering Challenges
Initial development involved troubleshooting complex SSL Certificate Verification Errors and Keras/TensorFlow version incompatibilities. This was resolved by using environment variable workarounds and installing necessary compatibility packages (tf-keras), demonstrating robust troubleshooting ability.

2. Performance Tuning
To ensure the answers are delivered quickly (a required feature):

Model Selection: We opted for the smaller, highly optimized llama3.2 over the larger llama3.

Generation Limit: The LLMResponseAgent is configured with a num_predict=256 token limit, which drastically reduces response time by preventing the model from generating unnecessary verbose output.
