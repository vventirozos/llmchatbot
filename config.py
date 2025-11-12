# config.py
import os
import datetime

# Color codes
RED = '\033[31m'
YELLOW = '\033[93m'
DEEP_BLUE = '\033[34m'
GRAY = '\033[90m'
WHITE = '\033[37m'
RESET = '\033[0m'

# Model configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", 'granite4:latest')

# Environment variables
def check_env_vars():
    """Check for required environment variables and set defaults."""
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("TAVILY_API_KEY environment variable not set")
    if not os.getenv("DATABASE_URL"):
        raise ValueError("DATABASE_URL environment variable not set")
    if not os.getenv("USER_AGENT"):
        os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    
    # Create allowed workspace directory
    workspace_dir = os.getenv("ALLOWED_WORK_DIR", "./workspace")
    os.makedirs(workspace_dir, exist_ok=True)

# System Prompt
def get_system_prompt():
    """Returns the system prompt from env or uses default."""
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Use environment variable for custom prompt, fallback to default
    if custom_prompt := os.getenv("SYSTEM_PROMPT"):
        return custom_prompt.replace("{datetime}", current_datetime)
    
    # Default prompt - less aggressive than original
    return f"""System: Your name is Ghost in the machine you are an AI assistant and the current date and time is {current_datetime}. 
    Instructions: Be direct, concise, and helpful. Avoid unnecessary fluff. Focus on providing accurate, actionable information."""
