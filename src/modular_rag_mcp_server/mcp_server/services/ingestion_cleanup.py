"""Lifecycle cleanup shared by ingestion service requests."""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


def cleanup_pipeline(
    pipeline: Any,
    trace: Any | None,
    trace_collector_factory: Callable[[], Any],
    *,
    tool_name: str,
    file_path: str,
    collection_name: str,
) -> BaseException | None:
    """Collect trace and close a pipeline, preserving the first cleanup error."""
    first_error: BaseException | None = None
    if trace is not None:
        try:
            trace_collector_factory().collect(trace)
        except BaseException as exc:
            logger.error(
                "%s trace collection failed: path=%s collection=%s error=%s",
                tool_name,
                file_path,
                collection_name,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            first_error = exc
    try:
        pipeline.close()
    except BaseException as exc:
        logger.error(
            "%s pipeline close failed: path=%s collection=%s error=%s",
            tool_name,
            file_path,
            collection_name,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        if first_error is None:
            first_error = exc
    return first_error
