"""The `dspyui` command: load API keys from .env and launch the web app."""

import argparse
import os


def _load_dotenv() -> None:
    """Load API keys from a .env file in the folder you launch from, if there is one."""
    if not os.path.exists(".env"):
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Found a .env file but python-dotenv is not installed; skipping it.")
        return
    load_dotenv(".env")
    print("Loaded environment variables from .env")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dspyui",
        description="DSPyUI: a Gradio user interface for DSPy. "
        "Compiled programs are saved in the folder you launch from.",
    )
    parser.add_argument("--host", default=None, help="Address to listen on (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    # Keys must be in the environment before the interface is built
    _load_dotenv()

    from dspyui.core import PROVIDER_KEY_ENV, key_is_set
    from dspyui.interface import demo

    if not any(key_is_set(os.environ.get(name)) for names in PROVIDER_KEY_ENV.values() for name in names):
        print(
            "No API keys found. The recommended setup is one OpenRouter key, which works for every model in the lists:\n"
            "  put OPENROUTER_API_KEY=... in a .env file in this folder, or enter it in the Settings tab.",
            flush=True,
        )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
