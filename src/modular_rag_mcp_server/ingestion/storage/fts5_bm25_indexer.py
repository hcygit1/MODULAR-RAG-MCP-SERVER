"""Compatibility alias for the FTS5 vector-store adapter.

FTS5 is a SQLite storage capability, so its implementation now lives next to
the other vector-store adapters.  Keep this import path for older callers.
"""

import sys

from modular_rag_mcp_server.libs.vector_store import fts5_bm25_indexer as _implementation

sys.modules[__name__] = _implementation
