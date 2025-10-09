#!/usr/bin/env python3
"""
NOVA CLI - Advanced AI-Powered CLI Assistant
Entry point for the application
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Safe config import with environment guard
try:
    from . import config
except ImportError:
    try:
        import config
    except ImportError:
        print("⚠️ Warning: Config module not found, proceeding without auto-setup")
        config = None

def main():
    """Main entry point for nova-cli command"""
    try:
        # Setup environment variables only once
        if config is not None and not os.environ.get("NOVA_ENV_LOADED"):
            config.setup_environment()
            os.environ["NOVA_ENV_LOADED"] = "1"

        # Import your main CLI file
        from NOVA_CLI import main as cli_main

        # Run the CLI
        cli_main()

    except KeyboardInterrupt:
        print("\n👋 Thanks for using NOVA CLI!")
        sys.exit(0)

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all required dependencies are installed.")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error starting NOVA CLI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Protect against double execution when using `-m`
    if not os.environ.get("NOVA_MAIN_RUNNING"):
        os.environ["NOVA_MAIN_RUNNING"] = "1"
        main()
