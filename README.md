# CLI Chat with Ollama and LangChain

This is a command-line chat application that leverages local language models via Ollama and LangChain. It provides a persistent, interactive chat session with tool-use capabilities, conversation management, and performance monitoring.

## Features

*   **Interactive Chat:** A terminal-based interface for conversing with an LLM.
*   **Ollama Integration:** Connects to any model served by a local Ollama instance.
*   **Tool Usage:** The model can invoke local tools to perform actions. The response from the tools is then fed back to the model.
*   **Streaming Responses:** AI responses are streamed token-by-token for a real-time experience.
*   **Conversation Management:**
    *   Maintains conversation history within a session.
    *   Automatically trims long conversations to manage token limits.
    *   Saves conversations to Markdown files.
    *   Clears the current session's history.
*   **Performance Metrics:** Displays token generation speed (`tokens_per_second`), evaluation duration, and other performance data after each response.
*   **Secure and Local:** Runs entirely on your local machine, ensuring privacy.

## Requirements

*   Python 3.8+
*   An active Ollama instance with a downloaded model (e.g., Llama 3, Mistral).
*   A `requirements.txt` file with necessary packages.

## Installation

1.  Ensure you have a running Ollama instance.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application is configured via a `.env` file. Create a `.env` file in the root of the project. The following variables are used:

```env
# The base URL for the Ollama API
OLLAMA_HOST="http://localhost:11434"

# The maximum number of messages to keep in the conversation history
MAX_CONVO_MESSAGES=20

# The maximum number of tokens to allow in the conversation history
MAX_CONVO_TOKENS=8000
```

The script also expects a `config.py` file to provide a system prompt and color codes, and a `tools.py` file to define the available tools for the model.

## Usage

Run the chat application from your terminal. You can specify which Ollama model to use with the `--model` flag.

```bash
python chat.py --model <your-model-name>
```

For example, to use the `llama3:8b` model:
```bash
python chat.py --model llama3:8b
```

If no model is specified, it will use the default model defined in the script (`gemma:2b`).

## In-Chat Commands

Once the chat is running, you can use these commands:

*   `/help`: Show the list of available commands.
*   `/clear`: Clear the current conversation history.
*   `/save`: Save the current conversation to a timestamped Markdown file in the `conversations/` directory.
*   `/exit`: Exit the chat application.
*   `Ctrl+C`: Interrupt the current AI response generation or wait for user input.
