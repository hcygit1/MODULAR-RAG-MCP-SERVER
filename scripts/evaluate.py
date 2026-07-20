"""Backward-compatible wrapper for the packaged evaluate command."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modular_rag_mcp_server.cli_commands.evaluate import *  # noqa: F401,F403
from modular_rag_mcp_server.cli_commands.evaluate import main as _main

if __name__ == "__main__":
    _main()
