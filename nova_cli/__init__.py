"""
NOVA CLI - Advanced AI-Powered CLI Assistant
🚀 Professional AI agents for coding, business, medical advice, and more
"""

__version__ = "1.0.0"
__author__ = "Aryan Kakade"

# Make main function available
try:
    from .main import main
    __all__ = ["main"]
except ImportError:
    __all__ = []