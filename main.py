#!/usr/bin/env python3
"""Compatibility wrapper for the installed console entry point."""
from modular_rag_mcp_server.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
