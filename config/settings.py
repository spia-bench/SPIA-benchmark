"""
Configuration settings loaded from environment variables.

Create a .env file in the project root with your API keys:
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    OLLAMA_BASE_URL=http://localhost:11434
    DEFAULT_API_PROVIDER=openai
    DEFAULT_MODEL_NAME=gpt-4.1
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

# Default settings
DEFAULT_API_PROVIDER = os.environ.get('DEFAULT_API_PROVIDER', 'openai')
DEFAULT_MODEL_NAME = os.environ.get('DEFAULT_MODEL_NAME', 'gpt-4.1')
