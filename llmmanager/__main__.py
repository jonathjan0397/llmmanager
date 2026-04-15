"""Entry point: python -m llmmanager or `llmmanager` CLI command."""

import sys


def main() -> None:
    # Lazy import so startup errors surface cleanly
    try:
        from llmmanager.app import LLMManagerApp
    except ImportError as e:
        print(f"[llmmanager] Failed to import app: {e}", file=sys.stderr)
        print("Run: pip install llmmanager", file=sys.stderr)
        sys.exit(1)

    app = LLMManagerApp()
    app.run()


if __name__ == "__main__":
    main()
