"""
🚀 NOVA - Next-Generation AI CLI with Enterprise-Grade Features
Copyright (c) 2024 Aryan Kakade
"""

__version__ = "1.0.0"
__author__ = "Aryan Kakade"
__email__ = "aryankakade143@gmail.com"
__description__ = "Next-Gen AI CLI with intelligent multi-agent orchestration"

import sys
import os

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import main function from parent directory's feature.py
try:
    from feature import main
    print("✅ NOVA CLI loaded successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    def main():
        print("NOVA CLI main function not available")

# Create placeholder classes (no more yellow lines!)
class NovaSystem:
    def __init__(self):
        self.version = __version__
        self.name = "NOVA CLI"
    
    def start(self):
        return main()

class EnhancedProductionAPIManager:
    def __init__(self):
        self.version = __version__
        print("⚠️ API Manager placeholder loaded")

__all__ = [
    "main",
    "NovaSystem", 
    "EnhancedProductionAPIManager",
    "__version__",
    "__author__",
]

def get_welcome_message():
    return f"""
🚀 NOVA CLI v{__version__}
Next-Gen AI Assistant by {__author__}
    
Quick Start:
- Run 'nova' to start
- Press Ctrl+P for command palette
- Visit: https://github.com/Aryankakade/NOVA-CLI-CHATBOT
"""

if __name__ == "__main__":
    main()
