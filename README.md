# AI Chatbot

This project is a command-line-based AI chatbot that leverages local large language models (LLMs) through Ollama. It is designed to be an extensible assistant, capable of performing a variety of tasks by using a suite of integrated tools. The chatbot features streaming responses, conversation management, and a strong emphasis on security.

## Features

- **Interactive CLI:** A user-friendly command-line interface for seamless interaction with the AI.
- **Ollama Integration:** Natively supports any model served by Ollama, allowing for flexibility and local execution.
- **Streaming Responses:** AI responses are streamed token-by-token for a real-time experience.
- **Rich Toolset:** The AI can autonomously use the following tools to perform complex tasks:
  - **`search`**: Performs web searches using the Tavily API.
  - **`wikipedia`**: Queries Wikipedia for detailed articles.
  - **`web_fetch`**: Fetches and reads the content of any given URL.
  - **`sql_query`**: Executes **read-only** SQL queries against a connected PostgreSQL database.
  - **`python_interpreter`**: Executes Python code in a sandboxed environment to perform calculations, data manipulation, and more.
  - **`read_file` / `write_file`**: Reads from and writes to a designated secure workspace directory.
- **Conversation Management:**
  - **Save History:** Save the current conversation to a timestamped Markdown file.
  - **Clear History:** Reset the conversation with a simple command.
  - **Context Trimming:** Automatically trims the conversation history to stay within token and message limits, ensuring long conversations remain efficient.
- **Security First:**
  - **Sandboxed Workspace:** File operations are restricted to a specific `workspace` directory to prevent unauthorized file access.
  - **Restricted Python Execution:** The Python interpreter blocks potentially dangerous modules and functions (`os`, `subprocess`, `open`, etc.).
  - **Read-Only Database Access:** The SQL tool is hardcoded to only allow `SELECT` queries, preventing any data modification or deletion.

## Getting Started

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running.
- A running PostgreSQL database instance.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd llmchatbot
   ```

2. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a file named `.env` in the root of the project directory and add the following variables.

   ```env
   # Required
   TAVILY_API_KEY="your_tavily_api_key"
   DATABASE_URL="postgresql://user:password@host:port/database"

   # Optional
   DEFAULT_MODEL="granite4:latest" # The default Ollama model to use
   ALLOWED_WORK_DIR="./workspace"    # The secure directory for file I/O tools
   MAX_CONVO_MESSAGES="20"           # Max messages to keep in history
   ## suggested is 65k
   MAX_CONVO_TOKENS="8000"           # Max tokens to keep in history
   USER_AGENT="YourCustomUserAgent/1.0" # User agent for web requests
   # SYSTEM_PROMPT="Your custom system prompt. The current date is {datetime}."
   ```

   - **`TAVILY_API_KEY`**: Required for the `search` tool. Get a free key from [Tavily AI](https://tavily.com/).
   - **`DATABASE_URL`**: The connection string for your PostgreSQL database.

## Usage

1. **Start the Chatbot:**
   Run the `chat.py` script from your terminal.
   ```bash
   python chat.py
   ```

2. **Specify a Different Model:**
   You can override the default model (granite4) by using the `--model` command-line argument.
   ```bash
   python chat.py --model "llama3:latest"
   ```

3. **Interact with the Chatbot:**
   Once started, you can type your messages directly. To use a command, type one of the following in the prompt:

   - `/help`: Shows the list of available commands.
   - `/save`: Saves the current conversation to the `conversations` directory.
   - `/clear`: Clears the current session's conversation history.
   - `/exit`: Exits the application gracefully.
   - `Ctrl+C`: Interrupts the current AI response generation.

## How It Works

The application is built around the `langchain` ecosystem.

- **`chat.py`**: The main entry point that manages the chat loop, handles user input, and orchestrates the flow between the user, the model, and the tools.
- **`tools.py`**: Defines the suite of tools that the LLM can use. Each tool is decorated with `@tool` and includes robust error handling and security constraints.
- **`config.py`**: Manages configuration, including environment variables, color codes for the CLI, and the system prompt that guides the AI's behavior.
- **`langchain-ollama`**: Provides the connection to the locally running Ollama instance. The model is bound to the tools, allowing it to decide when and how to use them to fulfill a user's request. When the model decides to use a tool, the application executes it, sends the result back to the model, and then streams the final, synthesized answer.
