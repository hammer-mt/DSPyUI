#!/usr/bin/env python3
"""
Gradio Spaces entry point for DSPyUI.

This file is specifically for deployment to Hugging Face Spaces.
For local deployment, use webui.sh or run interface.py directly.
"""

import os
import sys

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and launch the Gradio interface
from interface import demo

if __name__ == "__main__":
    # Launch with Spaces-appropriate settings
    demo.launch(
        server_name="0.0.0.0",  # Required for Spaces
        server_port=7860,       # Standard Gradio port
        show_error=True,        # Show errors in UI for debugging
        share=False             # Don't create share link (Spaces handles this)
    )
