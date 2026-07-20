"""Compatibility alias for the ingestion integrity module.

The implementation lives under ``ingestion.integrity`` because it tracks
ingestion history rather than loading documents.  This module remains so
existing integrations can keep importing the historical path.
"""

import sys

from modular_rag_mcp_server.ingestion.integrity import file_integrity as _implementation

sys.modules[__name__] = _implementation
