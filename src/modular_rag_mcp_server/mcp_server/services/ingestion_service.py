"""Shared orchestration for document ingestion MCP tools."""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Protocol, cast

from modular_rag_mcp_server.mcp_server.tools.error_utils import ErrorType, build_error_response
from modular_rag_mcp_server.mcp_server.tools.ingest_utils import parse_force

logger = logging.getLogger(__name__)

Parser = Callable[[str, Any], "ParsedIngestion"]


@dataclass(frozen=True)
class ParsedIngestion:
    """A parser result containing exactly one Pipeline payload mode."""

    document: Any | None = None
    chunks: list[Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.document is None) == (self.chunks is None):
            raise ValueError("ParsedIngestion 必须且只能包含 document 或 chunks")


@dataclass(frozen=True)
class SkipResponseContext:
    file_path: str
    collection_name: str
    force: bool


@dataclass(frozen=True)
class SuccessResponseContext:
    file_path: str
    collection_name: str
    force: bool
    chunk_count: int
    metadata: Mapping[str, Any]


class SkipResponseBuilder(Protocol):
    def __call__(self, context: SkipResponseContext, /) -> Dict[str, Any]: ...


class SuccessResponseBuilder(Protocol):
    def __call__(self, context: SuccessResponseContext, /) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class IngestionToolSpec:
    """Immutable strategy supplied by an ingestion tool."""

    tool_name: str
    supported_suffixes: frozenset[str]
    loader_name_resolver: Callable[[Path], str]
    parser: Parser
    skip_response_builder: SkipResponseBuilder
    success_response_builder: SuccessResponseBuilder
    runtime_error_prefix: str
    generic_error_prefix: str


class IngestionResponseError(Exception):
    """Request an exact MCP error response from a parser strategy."""

    def __init__(self, error_type: ErrorType, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


@dataclass
class _LockEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    references: int = 0


class _KeyedLockRegistry:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._entries: dict[tuple[str, str, str], _LockEntry] = {}

    @contextmanager
    def hold(self, key: tuple[str, str, str]) -> Iterator[None]:
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _LockEntry()
                self._entries[key] = entry
            entry.references += 1

        try:
            with entry.lock:
                yield
        finally:
            with self._guard:
                entry.references -= 1
                if entry.references == 0 and self._entries.get(key) is entry:
                    del self._entries[key]

    def size(self) -> int:
        with self._guard:
            return len(self._entries)


_LOCK_REGISTRY = _KeyedLockRegistry()


def _default_pipeline_factory(settings: Any, checker: Any) -> Any:
    from modular_rag_mcp_server.ingestion.pipeline import IngestionPipeline

    return IngestionPipeline(settings, integrity_checker=checker)


def _default_trace_factory(**kwargs: Any) -> Any:
    from modular_rag_mcp_server.core.trace.trace_context import TraceContext

    return TraceContext(**kwargs)


def _default_trace_collector_factory() -> Any:
    from modular_rag_mcp_server.observability.logger import get_trace_collector

    return get_trace_collector()


class IngestionService:
    """Execute one ingestion request using a tool-provided strategy."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], Any],
        integrity_checker_factory: Callable[[], Any],
        processing_profile_builder: Callable[..., str],
        pipeline_factory: Callable[[Any, Any], Any] | None = None,
        trace_factory: Callable[..., Any] | None = None,
        trace_collector_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.settings_loader = settings_loader
        self._integrity_checker_factory = integrity_checker_factory
        self._processing_profile_builder = processing_profile_builder
        self._pipeline_factory = pipeline_factory or _default_pipeline_factory
        self._trace_factory = trace_factory or _default_trace_factory
        self._trace_collector_factory = (
            trace_collector_factory or _default_trace_collector_factory
        )

    @classmethod
    def active_lock_count(cls) -> int:
        """Return the current registry size for lifecycle verification."""
        return _LOCK_REGISTRY.size()

    def execute(
        self,
        arguments: Dict[str, Any],
        spec: IngestionToolSpec,
    ) -> Dict[str, Any]:
        file_path_value = arguments.get("file_path")
        if not file_path_value or not str(file_path_value).strip():
            return self._error("INVALID_PARAMS", "参数 file_path 不能为空")

        file_path = str(file_path_value).strip()
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return self._error("RESOURCE_NOT_FOUND", f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        if spec.supported_suffixes and suffix not in spec.supported_suffixes:
            supported = ", ".join(sorted(spec.supported_suffixes))
            return self._error(
                "INVALID_PARAMS",
                f"不支持的文件类型: {path.suffix}，支持：{supported}",
            )

        collection_name = "<unknown>"
        try:
            settings = self.settings_loader()
            collection_value = arguments.get("collection_name")
            if not collection_value or not str(collection_value).strip():
                collection_name = str(settings.vector_store.collection_name)
            else:
                collection_name = str(collection_value).strip()
            force = parse_force(arguments.get("force", False))
            try:
                checker = self._integrity_checker_factory()
                processing_profile = self._processing_profile_builder(
                    settings,
                    loader_name=spec.loader_name_resolver(path),
                )
                file_hash = checker.compute_sha256(file_path)
            except Exception as exc:
                logger.exception(
                    "%s integrity check failed: path=%s collection=%s",
                    spec.tool_name,
                    file_path,
                    collection_name,
                )
                return self._error(
                    "INTERNAL_ERROR",
                    f"完整性检查失败: {exc}",
                    integrity=True,
                )

            lock_key = (file_hash, collection_name, processing_profile)
            with _LOCK_REGISTRY.hold(lock_key):
                try:
                    should_skip = (
                        False
                        if force
                        else checker.should_skip(
                            file_hash,
                            collection_name,
                            processing_profile,
                        )
                    )
                except Exception as exc:
                    logger.exception(
                        "%s integrity check failed: path=%s collection=%s",
                        spec.tool_name,
                        file_path,
                        collection_name,
                    )
                    return self._error(
                        "INTERNAL_ERROR",
                        f"完整性检查失败: {exc}",
                        integrity=True,
                    )

                if should_skip:
                    return spec.skip_response_builder(
                        SkipResponseContext(
                            file_path=file_path,
                            collection_name=collection_name,
                            force=force,
                        )
                    )

                parsed = spec.parser(file_path, settings)
                pipeline = self._pipeline_factory(settings, checker)
                trace = None
                primary_error: BaseException | None = None
                primary_traceback = None
                try:
                    trace = self._trace_factory(operation="ingestion")
                    if parsed.document is not None:
                        chunk_count = pipeline.process_document(
                            parsed.document,
                            collection_name,
                            trace=trace,
                        )
                    else:
                        chunk_count = pipeline.process_chunks(
                            parsed.chunks,
                            collection_name,
                            trace=trace,
                        )
                    response = spec.success_response_builder(
                        SuccessResponseContext(
                            file_path=file_path,
                            collection_name=collection_name,
                            force=force,
                            chunk_count=chunk_count,
                            metadata=dict(parsed.metadata),
                        )
                    )
                except BaseException as exc:
                    primary_error = exc
                    primary_traceback = exc.__traceback__

                cleanup_error = self._cleanup_pipeline(
                    pipeline,
                    trace,
                    tool_name=spec.tool_name,
                    file_path=file_path,
                    collection_name=collection_name,
                )
                if primary_error is not None:
                    raise primary_error.with_traceback(primary_traceback)
                if cleanup_error is not None:
                    raise cleanup_error.with_traceback(cleanup_error.__traceback__)

                checker.mark_success(
                    file_hash,
                    collection_name,
                    processing_profile,
                )
                return response
        except IngestionResponseError as exc:
            return self._error(exc.error_type, exc.message)
        except FileNotFoundError as exc:
            return self._error("RESOURCE_NOT_FOUND", str(exc))
        except ValueError as exc:
            return self._error("INVALID_PARAMS", str(exc))
        except RuntimeError as exc:
            logger.exception(
                "%s runtime failure: path=%s collection=%s error=%s",
                spec.tool_name,
                file_path,
                collection_name,
                exc,
            )
            return self._error(
                "INTERNAL_ERROR",
                f"{spec.runtime_error_prefix}: {exc}",
            )
        except Exception as exc:
            logger.exception(
                "%s failure: path=%s collection=%s error=%s",
                spec.tool_name,
                file_path,
                collection_name,
                exc,
            )
            return self._error(
                "INTERNAL_ERROR",
                f"{spec.generic_error_prefix}: {exc}",
            )

    def _cleanup_pipeline(
        self,
        pipeline: Any,
        trace: Any | None,
        *,
        tool_name: str,
        file_path: str,
        collection_name: str,
    ) -> BaseException | None:
        first_error: BaseException | None = None
        if trace is not None:
            try:
                self._trace_collector_factory().collect(trace)
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

    @staticmethod
    def _error(
        error_type: ErrorType,
        message: str,
        *,
        integrity: bool = False,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {"chunk_count": 0}
        if integrity:
            base["skipped"] = False
        return cast(
            Dict[str, Any],
            build_error_response(
                error_type,
                message,
                structured_content_base=base,
            ),
        )
