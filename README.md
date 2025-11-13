# CLI Chat with Ollama and LangChain Tools

This project is a Python-based command-line interface (CLI) for interacting with large language models (LLMs) through Ollama. It leverages the `langchain` library to provide a powerful and extensible toolset, enabling the model to perform actions like web searches, database queries, and more.

## Features

*   **Interactive Chat:** A familiar CLI-based chat experience.
*   **Ollama Integration:** Connects to any model served by Ollama.
*   **Extensible Toolset:** Easily add new tools using `langchain`.
*   **Streaming Responses:** See the model's response as it's being generated.
*   **Conversation Management:** Save, clear, or exit conversations with simple commands.
*   **Retrieval-Augmented Generation (RAG):** Search local documents to augment model responses.
*   **Secure by Default:** Tools like the Python interpreter and file I/O are restricted to a designated workspace directory.

## Requirements

The project requires Python 3.x and the dependencies listed in `requirements.txt`.

Key dependencies include:
*   `langchain` and its ecosystem (`langchain-ollama`, `langchain-community`, etc.)
*   `ollama`
*   `faiss-cpu` for RAG
*   `SQLAlchemy` and `psycopg2-binary` for database interaction
*   `tavily-python` for web search

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Now, edit the `.env` file and add your specific credentials and settings.

    **Required variables:**
    *   `TAVILY_API_KEY`: Your API key for Tavily search.
    *   `DATABASE_URL`: The connection string for your PostgreSQL database (e.g., `postgresql://user:password@host:port/dbname`).

5.  **Set up RAG (Optional):**
    To use the Retrieval-Augmented Generation (RAG) feature, place your text documents (`.txt` files) into the `rag_documents` directory. The first time you run the application, it will create a FAISS vector index from these documents.

## Usage

To start the chat application, run:
```bash
python chat.py
```

You can also specify a different Ollama model to use:
```bash
python chat.py --model <model-name>
```
(e.g., `python chat.py --model granite4:latest`)

### In-Chat Commands

*   `/help`: Show the list of available commands.
*   `/clear`: Clear the current conversation history.
*   `/save`: Save the current conversation to a Markdown file in the `conversations` directory.
*   `/exit`: Exit the application.

## Configuration

The application is configured via environment variables in the `.env` file:

*   `DEFAULT_MODEL`: The default Ollama model to use (e.g., `granite4:tiny-h`).
*   `TAVILY_API_KEY`: API key for Tavily web search.
*   `DATABASE_URL`: PostgreSQL database connection URL.
*   `SYSTEM_PROMPT`: An optional custom system prompt for the AI.
*   `ALLOWED_WORK_DIR`: The directory where tools like `read_file`, `write_file`, and `python_interpreter` are allowed to operate. Defaults to `./workspace`.
*   `MAX_CONVO_MESSAGES`: The maximum number of messages to keep in the conversation history. Defaults to `20`.

## Available Tools

The AI assistant can use the following tools:

*   `search`: Searches the web using Tavily.
*   `wikipedia`: Searches Wikipedia.
*   `web_fetch`: Fetches the content of a URL.
*   `sql_query`: Executes a read-only SQL query against the configured database.
*   `python_interpreter`: Executes Python code in a restricted environment.
*   `read_file`: Reads a file from the workspace directory.
*   `write_file`: Writes a file to the workspace directory.
*   `rag_search`: Searches the local document collection.

## Project Structure

```
.
├── chat.py               # Main application logic
├── config.py             # Configuration and constants
├── tools.py              # Tool definitions for langchain
├── requirements.txt      # Python dependencies
├── workspace/            # Safe working directory for tools
├── rag_documents/        # Source documents for RAG
├── faiss_index/          # Stored FAISS index for RAG
└── conversations/        # Saved conversation histories
```
