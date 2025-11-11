# chat.py (streaming-enabled fix)
import asyncio
import logging
import readline
import argparse
import sys
import os
import json
import re  # Added for JSON extraction fix
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
)

from config import (
    RED, YELLOW, DEEP_BLUE, GRAY, WHITE, RESET,
    DEFAULT_MODEL, check_env_vars, get_system_prompt
)
from tools import get_tools

# Load environment variables and configure logging
load_dotenv()

# Define custom formatter for colored logs
class ColorFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages based on level."""
    
    LEVEL_COLORS = {
        logging.INFO: GRAY,
        logging.WARNING: GRAY,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }
    
    def format(self, record):
        # Get color for this log level
        color = self.LEVEL_COLORS.get(record.levelno, RESET)
        
        # Format the message
        formatted = super().format(record)
        
        # Apply color and reset
        return f"{color}{formatted}{RESET}"

# Configure logging with custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(
    level=logging.INFO,
    handlers=[console_handler]
)
logger = logging.getLogger(__name__)


class ChatManager:
    """Manages the chat interaction, including model, tools, and messages."""

    def __init__(self, model_name: str):
        """
        Initializes the ChatManager.

        Args:
            model_name: The name of the Ollama model to use.
        """
        check_env_vars()
        self.tools = get_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.model = ChatOllama(model=model_name)
        self.llm_with_tools = self.model.bind_tools(self.tools)
        self.messages: List[BaseMessage] = [SystemMessage(content=get_system_prompt())]
        self.max_messages = int(os.getenv("MAX_CONVO_MESSAGES", "20"))
        self.max_tokens = int(os.getenv("MAX_CONVO_TOKENS", "8000"))
        self.running = False
        logger.info(f"ChatManager initialized with model: {model_name}")

    def _trim_history(self):
        """Keep only recent messages under configured limits."""
        if len(self.messages) > self.max_messages:
            logger.info(f"Trimming conversation history from {len(self.messages)} to {self.max_messages} messages")
            self.messages = [self.messages[0]] + self.messages[-self.max_messages:]

    def _print_metrics(self, response_metadata: Optional[Dict[str, Any]]):
        """Prints performance metrics from the model's response."""
        if not response_metadata:
            return

        metrics = {
            "total_duration_ms": response_metadata.get('total_duration', 0) / 1_000_000,
            "load_duration_ms": response_metadata.get('load_duration', 0) / 1_000_000,
            "prompt_eval_count": response_metadata.get('prompt_eval_count', 0),
            "prompt_eval_duration_ms": response_metadata.get('prompt_eval_duration', 0) / 1_000_000,
            "eval_count": response_metadata.get('eval_count', 0),
            "eval_duration_ms": response_metadata.get('eval_duration', 0) / 1_000_000,
        }
        if metrics["eval_count"] > 0 and metrics["eval_duration_ms"] > 0:
            metrics["tokens_per_second"] = metrics["eval_count"] / (metrics["eval_duration_ms"] / 1000)
        else:
            metrics["tokens_per_second"] = 0

        print(f"\n{WHITE}Performance Metrics:{RESET}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{WHITE}- {key}: {value:.2f}{RESET}")
            else:
                print(f"{WHITE}- {key}: {value}{RESET}")

    def _print_help(self):
        """Display available commands."""
        help_text = f"""
{WHITE}
Available Commands:
/help     - Show this help message
/clear    - Clear conversation history
/save     - Save conversation to file
/exit     - Exit the application
Ctrl+C    - Interrupt current operation{RESET}
        """
        print(help_text)

    def _save_conversation(self):
        """Save conversation to timestamped markdown file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = Path(f"conversations/convo_{timestamp}.md")
        filename.parent.mkdir(exist_ok=True)

        content = f"# Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        for msg in self.messages:
            role = msg.type.upper()
            content += f"## {role}\n\n{msg.content}\n\n"

        filename.write_text(content)
        print(f"{WHITE}Conversation saved to {filename}{RESET}")
        logger.info(f"Conversation saved to {filename}")

    async def _handle_tool_calls(self, ai_msg: AIMessage):
        """Handles tool calls from the AI message asynchronously."""
        logger.info(f"Tool calls: {ai_msg.tool_calls}")

        for tool_call in ai_msg.tool_calls:
            # Handle both dict and object formats
            tool_name = tool_call.get('name') if isinstance(tool_call, dict) else tool_call.name
            tool_id = tool_call.get('id') if isinstance(tool_call, dict) else tool_call.id
            tool_args = tool_call.get('args') if isinstance(tool_call, dict) else tool_call.args

            tool_to_call = self.tool_map.get(tool_name)
            if tool_to_call:
                try:
                    # Execute tool asynchronously
                    result = await asyncio.to_thread(
                        tool_to_call.invoke,
                        tool_args
                    )
                    logger.info(f"Tool result: {str(result)[:100]}...")
                    self.messages.append(ToolMessage(
                        content=str(result)[:8000],
                        tool_call_id=tool_id
                    ))
                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    self.messages.append(ToolMessage(
                        content=f"Tool execution error: {e}",
                        tool_call_id=tool_id
                    ))
            else:
                logger.warning(f"Tool {tool_name} not found")
                self.messages.append(ToolMessage(
                    content=f"Tool {tool_name} not found",
                    tool_call_id=tool_id
                ))

    async def _stream_response(self):
        """Streams response and handles tool calls correctly."""
        buffer = ""
        tool_calls_by_index = {}

        logger.info("Starting response stream")
        async for chunk in self.llm_with_tools.astream(self.messages):
            if chunk.content:
                print(f"{YELLOW}{chunk.content}{RESET}", end="", flush=True)
                buffer += chunk.content

            # Properly handle tool call chunks
            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                for tool_chunk in chunk.tool_call_chunks:
                    # Get index, default to 0 if not present
                    index = tool_chunk.get('index', 0)

                    # Initialize if this is the first chunk for this index
                    if index not in tool_calls_by_index:
                        tool_calls_by_index[index] = {
                            'id': tool_chunk.get('id'),
                            'name': tool_chunk.get('name'),
                            'args': ''
                        }

                    # Accumulate arguments
                    args_delta = tool_chunk.get('args')
                    if args_delta:
                        tool_calls_by_index[index]['args'] += args_delta

        print()  # New line after streaming

        # Convert accumulated tool calls to proper format
        tool_calls = []
        for idx in sorted(tool_calls_by_index.keys()):
            call_data = tool_calls_by_index[idx]
            try:
                # Parse JSON args with error handling for malformed data
                if call_data['args']:
                    raw_args = call_data['args']
                    logger.debug(f"Raw tool args: {raw_args}")

                    # Try to parse the full string first
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError as e:
                        # Handle "Extra data" error by extracting first JSON object
                        if "Extra data" in str(e):
                            match = re.search(r'\{.*?\}', raw_args, re.DOTALL)
                            if match:
                                json_str = match.group(0)
                                logger.warning(f"Extracted valid JSON from malformed args: {json_str}")
                                parsed_args = json.loads(json_str)
                            else:
                                raise
                        else:
                            raise
                else:
                    parsed_args = {}

                call_data['args'] = parsed_args
                tool_calls.append(call_data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse tool args for {call_data.get('name')}: {e}")
                logger.error(f"Raw args string: {repr(call_data['args'])}")
                # Skip malformed tool calls to prevent crashing
                continue

        # Create the final message
        ai_msg = AIMessage(content=buffer, tool_calls=tool_calls)

        self.messages.append(ai_msg)
        logger.info("Response stream completed")
        return ai_msg

    async def run(self):
        """Starts the main chat loop with clean shutdown handling."""
        print(f"{GRAY}Chat started. Type /help for commands.{RESET}")
        self.running = True

        try:
            while self.running:
                try:
                    user_input = await asyncio.to_thread(input, f"{RED}User >{RESET} ")
                    user_input = user_input.strip()
                except (KeyboardInterrupt, EOFError):
                    print(f"\n{WHITE}Use /exit to quit gracefully.{RESET}")
                    continue

                if not user_input:
                    continue

                # Command handling
                if user_input.startswith("/"):
                    if user_input == "/exit":
                        print("Bye.\n")
                        self.running = False
                        break
                    elif user_input == "/clear":
                        self.messages = [self.messages[0]]
                        print(f"{WHITE}History cleared.{RESET}")
                        logger.info("Conversation history cleared")
                        continue
                    elif user_input == "/help":
                        self._print_help()
                        continue
                    elif user_input == "/save":
                        self._save_conversation()
                        continue
                    else:
                        print(f"{GRAY}Unknown command: {user_input}. Type /help for options.{RESET}")
                        continue

                self.messages.append(HumanMessage(content=user_input))

                while True:
                    try:
                        ai_msg = await self._stream_response()

                        if not ai_msg.tool_calls:
                            self._print_metrics(ai_msg.response_metadata)
                            break

                        print(f"{DEEP_BLUE}Executing tools...{RESET}")
                        await self._handle_tool_calls(ai_msg)
                    except KeyboardInterrupt:
                        print(f"\n{GRAY}Operation interrupted. Continuing...{RESET}")
                        break
                    except Exception as e:
                        logger.error(f"Error during response generation: {e}")
                        print(f"{RED}Error: {e}{RESET}")
                        break

                # Trim history after each turn
                self._trim_history()

        except Exception as e:
            logger.exception(f"Fatal error in main loop: {e}")
            print(f"{RED}Fatal error: {e}{RESET}")
        finally:
            # Ensure clean shutdown
            logger.info("Chat session ending")
            self.running = False


def main():
    """Main function to run the chat application."""
    parser = argparse.ArgumentParser(description="A secure chat application with integrated tools.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"The Ollama model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    chat_manager = None
    try:
        chat_manager = ChatManager(model_name=args.model)
        asyncio.run(chat_manager.run())
    except KeyboardInterrupt:
        # Handle Ctrl+C during startup or shutdown
        print(f"\n{WHITE}Shutting down gracefully...{RESET}")
        if chat_manager:
            chat_manager.running = False
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"{RED}Configuration Error: {e}{RESET}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"{RED}An unexpected error occurred: {e}{RESET}")
        sys.exit(1)
    finally:
        # Ensure we exit cleanly
        print(f"{WHITE}Goodbye!{RESET}")


if __name__ == "__main__":
    main()
    