"""Backward-compatible wrapper for the packaged MinerU command."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from modular_rag_mcp_server.cli_commands.ingest_mineru import main

if __name__ == "__main__":
    main()
