"""Shared ingestion service contract tests."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from unittest.mock import MagicMock

import pytest

from modular_rag_mcp_server.mcp_server.services import ingestion_service as ingestion_service_module
from modular_rag_mcp_server.mcp_server.services.ingestion_service import (
    IngestionService,
    IngestionToolSpec,
    ParsedIngestion,
    SkipResponseContext,
    SuccessResponseContext,
)


def _success_response(context: SuccessResponseContext) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": "success"}],
        "structuredContent": {
            "file_path": context.file_path,
            "collection_name": context.collection_name,
            "force": context.force,
            "chunk_count": context.chunk_count,
            "metadata": dict(context.metadata),
        },
        "isError": False,
    }


def _skip_response(context: SkipResponseContext) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": "skipped"}],
        "structuredContent": {
            "file_path": context.file_path,
            "collection_name": context.collection_name,
            "force": context.force,
            "skipped": True,
        },
        "isError": False,
    }


def _spec(
    parser: Callable[[str, Any], ParsedIngestion],
    *,
    supported_suffixes: frozenset[str] = frozenset({".txt"}),
    success_response_builder: Callable[
        [SuccessResponseContext],
        dict[str, Any],
    ] = _success_response,
) -> IngestionToolSpec:
    return IngestionToolSpec(
        tool_name="test_ingestion",
        supported_suffixes=supported_suffixes,
        loader_name_resolver=lambda path: "text",
        parser=parser,
        skip_response_builder=_skip_response,
        success_response_builder=success_response_builder,
        runtime_error_prefix="工具运行失败",
        generic_error_prefix="工具处理失败",
    )


def _settings(collection_name: str = "default-docs") -> SimpleNamespace:
    return SimpleNamespace(
        vector_store=SimpleNamespace(collection_name=collection_name),
    )


def _service(
    *,
    checker: Any | None = None,
    settings: Any | None = None,
    pipeline_factory: Any | None = None,
    trace: Any | None = None,
    collector: Any | None = None,
    profile_builder: Any | None = None,
) -> tuple[IngestionService, Any, Any, Any, Any]:
    if checker is None:
        checker = MagicMock()
        checker.compute_sha256.return_value = "file-hash"
        checker.should_skip.return_value = False
    pipeline_factory = pipeline_factory or MagicMock()
    pipeline_factory.return_value.process_document.return_value = 3
    pipeline_factory.return_value.process_chunks.return_value = 4
    trace = trace or object()
    collector = collector or MagicMock()
    profile_builder = profile_builder or MagicMock(return_value="profile")
    settings = settings or _settings()
    service = IngestionService(
        settings_loader=lambda: settings,
        integrity_checker_factory=lambda: checker,
        processing_profile_builder=profile_builder,
        pipeline_factory=pipeline_factory,
        trace_factory=lambda **_: trace,
        trace_collector_factory=lambda: collector,
    )
    return service, checker, pipeline_factory, trace, collector


def test_response_contexts_are_frozen_and_spec_has_tool_name() -> None:
    skip_context_type = getattr(ingestion_service_module, "SkipResponseContext")
    success_context_type = getattr(
        ingestion_service_module,
        "SuccessResponseContext",
    )
    skip_context = skip_context_type("document.txt", "docs", False)
    success_context = success_context_type(
        "document.txt",
        "docs",
        False,
        2,
        {"parser": "test"},
    )

    with pytest.raises(FrozenInstanceError):
        skip_context.force = True
    with pytest.raises(FrozenInstanceError):
        success_context.chunk_count = 3
    assert "tool_name" in signature(IngestionToolSpec).parameters


@pytest.mark.parametrize(
    ("arguments", "error_type", "message"),
    [
        ({"file_path": ""}, "INVALID_PARAMS", "参数 file_path 不能为空"),
        (
            {"file_path": "/definitely/missing/document.txt"},
            "RESOURCE_NOT_FOUND",
            "文件不存在",
        ),
    ],
)
def test_rejects_empty_and_missing_paths(
    arguments: dict[str, Any],
    error_type: str,
    message: str,
) -> None:
    service, _, pipeline_factory, _, _ = _service()

    result = service.execute(arguments, _spec(MagicMock()))

    assert result["isError"] is True
    assert result["structuredContent"]["errorType"] == error_type
    assert message in result["structuredContent"]["message"]
    pipeline_factory.assert_not_called()


def test_rejects_unsupported_suffix(tmp_path: Path) -> None:
    file_path = tmp_path / "document.csv"
    file_path.write_text("content", encoding="utf-8")
    service, _, pipeline_factory, _, _ = _service()

    result = service.execute({"file_path": str(file_path)}, _spec(MagicMock()))

    assert result["isError"] is True
    assert result["structuredContent"]["errorType"] == "INVALID_PARAMS"
    assert result["structuredContent"]["message"] == (
        "不支持的文件类型: .csv，支持：.txt"
    )
    pipeline_factory.assert_not_called()


def test_uses_default_collection_force_and_loader_profile(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    parser = MagicMock(return_value=ParsedIngestion(document=object()))
    profile_builder = MagicMock(return_value="text-profile")
    service, checker, pipeline_factory, _, _ = _service(
        settings=_settings("configured-docs"),
        profile_builder=profile_builder,
    )

    result = service.execute(
        {"file_path": str(file_path), "force": "true"},
        _spec(parser),
    )

    assert result["isError"] is False
    assert result["structuredContent"]["collection_name"] == "configured-docs"
    assert result["structuredContent"]["force"] is True
    profile_builder.assert_called_once_with(
        service.settings_loader(),
        loader_name="text",
    )
    checker.should_skip.assert_not_called()
    pipeline_factory.assert_called_once()


def test_unchanged_file_skips_parser_pipeline_and_mark_success(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    parser = MagicMock()
    service, checker, pipeline_factory, _, _ = _service()
    checker.should_skip.return_value = True

    result = service.execute({"file_path": str(file_path)}, _spec(parser))

    assert result["isError"] is False
    assert result["structuredContent"]["skipped"] is True
    parser.assert_not_called()
    pipeline_factory.assert_not_called()
    checker.mark_success.assert_not_called()


def test_document_payload_processes_document_and_marks_success(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    document = object()
    parser = MagicMock(
        return_value=ParsedIngestion(
            document=document,
            metadata={"parser": "document"},
        )
    )
    service, checker, pipeline_factory, trace, collector = _service()
    pipeline = pipeline_factory.return_value

    result = service.execute({"file_path": str(file_path)}, _spec(parser))

    assert result["structuredContent"]["chunk_count"] == 3
    assert result["structuredContent"]["metadata"] == {"parser": "document"}
    pipeline.process_document.assert_called_once_with(
        document,
        "default-docs",
        trace=trace,
    )
    pipeline.process_chunks.assert_not_called()
    checker.mark_success.assert_called_once_with(
        "file-hash",
        "default-docs",
        "profile",
    )
    pipeline_factory.assert_called_once_with(service.settings_loader(), checker)
    collector.collect.assert_called_once_with(trace)
    pipeline.close.assert_called_once_with()


def test_chunks_payload_processes_chunks(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    chunks = [object(), object()]
    service, _, pipeline_factory, trace, _ = _service()
    pipeline = pipeline_factory.return_value

    result = service.execute(
        {"file_path": str(file_path)},
        _spec(lambda *_: ParsedIngestion(chunks=chunks)),
    )

    assert result["structuredContent"]["chunk_count"] == 4
    pipeline.process_chunks.assert_called_once_with(
        chunks,
        "default-docs",
        trace=trace,
    )
    pipeline.process_document.assert_not_called()


@pytest.mark.parametrize("failure_site", ["process", "response"])
def test_pipeline_is_traced_and_closed_when_execution_fails(
    failure_site: str,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    service, checker, pipeline_factory, trace, collector = _service()
    pipeline = pipeline_factory.return_value
    response_builder = _success_response
    if failure_site == "process":
        pipeline.process_document.side_effect = RuntimeError("pipeline failed")
    else:
        response_builder = MagicMock(side_effect=KeyError("response failed"))

    result = service.execute(
        {"file_path": str(file_path)},
        _spec(
            lambda *_: ParsedIngestion(document=object()),
            success_response_builder=response_builder,
        ),
    )

    assert result["isError"] is True
    expected_prefix = (
        "工具运行失败" if failure_site == "process" else "工具处理失败"
    )
    assert result["structuredContent"]["message"].startswith(expected_prefix)
    collector.collect.assert_called_once_with(trace)
    pipeline.close.assert_called_once_with()
    assert "test_ingestion" in caplog.text
    assert f"path={file_path}" in caplog.text
    assert "collection=default-docs" in caplog.text
    if failure_site == "process":
        checker.mark_success.assert_not_called()


def test_response_builder_failure_does_not_mark_and_can_retry(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    parser = MagicMock(return_value=ParsedIngestion(document=object()))
    response_attempts = 0

    def flaky_response_builder(
        context: SuccessResponseContext,
    ) -> dict[str, Any]:
        nonlocal response_attempts
        response_attempts += 1
        if response_attempts == 1:
            raise KeyError("response failed")
        return _success_response(context)

    service, checker, pipeline_factory, _, _ = _service()
    spec = _spec(
        parser,
        success_response_builder=flaky_response_builder,
    )

    first_result = service.execute({"file_path": str(file_path)}, spec)

    assert first_result["isError"] is True
    checker.mark_success.assert_not_called()

    second_result = service.execute({"file_path": str(file_path)}, spec)

    assert second_result["isError"] is False
    assert parser.call_count == 2
    assert pipeline_factory.return_value.process_document.call_count == 2
    checker.mark_success.assert_called_once_with(
        "file-hash",
        "default-docs",
        "profile",
    )


def test_success_response_and_cleanup_complete_before_mark_success(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    order: list[str] = []
    collector = MagicMock()

    def success_response_builder(
        context: SuccessResponseContext,
    ) -> dict[str, Any]:
        order.append("response")
        return _success_response(context)

    service, checker, pipeline_factory, _, _ = _service(collector=collector)
    pipeline = pipeline_factory.return_value
    collector.collect.side_effect = lambda _trace: order.append("collect")
    pipeline.close.side_effect = lambda: order.append("close")
    checker.mark_success.side_effect = lambda *_: order.append("mark")

    result = service.execute(
        {"file_path": str(file_path)},
        _spec(
            lambda *_: ParsedIngestion(document=object()),
            success_response_builder=success_response_builder,
        ),
    )

    assert result["isError"] is False
    assert order == ["response", "collect", "close", "mark"]


def test_process_error_survives_collector_and_close_errors(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    collector = MagicMock()
    collector.collect.side_effect = ValueError("collector failed")
    service, checker, pipeline_factory, trace, _ = _service(collector=collector)
    pipeline = pipeline_factory.return_value
    pipeline.process_document.side_effect = RuntimeError("pipeline failed")
    pipeline.close.side_effect = OSError("close failed")

    result = service.execute(
        {"file_path": str(file_path)},
        _spec(lambda *_: ParsedIngestion(document=object())),
    )

    assert result["structuredContent"]["errorType"] == "INTERNAL_ERROR"
    assert result["structuredContent"]["message"] == (
        "工具运行失败: pipeline failed"
    )
    collector.collect.assert_called_once_with(trace)
    pipeline.close.assert_called_once_with()
    checker.mark_success.assert_not_called()
    assert "test_ingestion trace collection failed" in caplog.text
    assert "test_ingestion pipeline close failed" in caplog.text
    assert f"path={file_path}" in caplog.text
    assert "collection=default-docs" in caplog.text


def test_trace_error_survives_close_error(tmp_path: Path) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    checker = MagicMock()
    checker.compute_sha256.return_value = "file-hash"
    checker.should_skip.return_value = False
    pipeline_factory = MagicMock()
    pipeline_factory.return_value.close.side_effect = OSError("close failed")
    collector = MagicMock()
    service = IngestionService(
        settings_loader=_settings,
        integrity_checker_factory=lambda: checker,
        processing_profile_builder=MagicMock(return_value="profile"),
        pipeline_factory=pipeline_factory,
        trace_factory=MagicMock(side_effect=ValueError("trace failed")),
        trace_collector_factory=lambda: collector,
    )

    result = service.execute(
        {"file_path": str(file_path)},
        _spec(lambda *_: ParsedIngestion(document=object())),
    )

    assert result["structuredContent"]["errorType"] == "INVALID_PARAMS"
    assert result["structuredContent"]["message"] == "trace failed"
    collector.collect.assert_not_called()
    pipeline_factory.return_value.close.assert_called_once_with()
    checker.mark_success.assert_not_called()


def test_first_cleanup_error_fails_request_and_prevents_mark(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    collector = MagicMock()
    collector.collect.side_effect = ValueError("collector failed")
    service, checker, pipeline_factory, trace, _ = _service(collector=collector)
    pipeline = pipeline_factory.return_value
    pipeline.close.side_effect = OSError("close failed")

    result = service.execute(
        {"file_path": str(file_path)},
        _spec(lambda *_: ParsedIngestion(document=object())),
    )

    assert result["structuredContent"]["errorType"] == "INVALID_PARAMS"
    assert result["structuredContent"]["message"] == "collector failed"
    collector.collect.assert_called_once_with(trace)
    pipeline.close.assert_called_once_with()
    checker.mark_success.assert_not_called()


def test_integrity_failure_uses_stable_error_message(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    service, checker, pipeline_factory, _, _ = _service()
    checker.compute_sha256.side_effect = OSError("cannot read")

    result = service.execute({"file_path": str(file_path)}, _spec(MagicMock()))

    assert result["structuredContent"]["errorType"] == "INTERNAL_ERROR"
    assert result["structuredContent"]["message"] == (
        "完整性检查失败: cannot read"
    )
    assert result["structuredContent"]["skipped"] is False
    pipeline_factory.assert_not_called()
    assert "test_ingestion integrity check failed" in caplog.text
    assert f"path={file_path}" in caplog.text
    assert "collection=default-docs" in caplog.text


def test_integrity_checker_initialization_failure_uses_integrity_error(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    pipeline_factory = MagicMock()
    service = IngestionService(
        settings_loader=_settings,
        integrity_checker_factory=MagicMock(
            side_effect=OSError("cannot initialize"),
        ),
        processing_profile_builder=MagicMock(),
        pipeline_factory=pipeline_factory,
    )

    result = service.execute({"file_path": str(file_path)}, _spec(MagicMock()))

    assert result["structuredContent"]["errorType"] == "INTERNAL_ERROR"
    assert result["structuredContent"]["message"] == (
        "完整性检查失败: cannot initialize"
    )
    assert result["structuredContent"]["skipped"] is False
    pipeline_factory.assert_not_called()


class _StatefulChecker:
    def __init__(self) -> None:
        self.success = False
        self.compute_calls = 0
        self.should_skip_calls = 0
        self.mark_success_calls = 0
        self._guard = threading.Lock()

    def compute_sha256(self, _file_path: str) -> str:
        with self._guard:
            self.compute_calls += 1
        return "shared-hash"

    def should_skip(self, *_: Any) -> bool:
        with self._guard:
            self.should_skip_calls += 1
            return self.success

    def mark_success(self, *_: Any) -> None:
        with self._guard:
            self.mark_success_calls += 1
            self.success = True


class _ObservingLockRegistry:
    def __init__(self) -> None:
        self._inner = ingestion_service_module._KeyedLockRegistry()
        self.keys: list[tuple[str, str, str]] = []
        self.second_attempted = threading.Event()
        self._guard = threading.Lock()

    @contextmanager
    def hold(self, key: tuple[str, str, str]) -> Iterator[None]:
        with self._guard:
            self.keys.append(key)
            if len(self.keys) == 2:
                self.second_attempted.set()
        with self._inner.hold(key):
            yield

    def size(self) -> int:
        return self._inner.size()


def test_same_hash_collection_and_profile_serializes_across_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for path in paths:
        path.write_text("same content", encoding="utf-8")
    checker = _StatefulChecker()
    registry = _ObservingLockRegistry()
    monkeypatch.setattr(ingestion_service_module, "_LOCK_REGISTRY", registry)
    first_parser_started = threading.Event()
    release_first_parser = threading.Event()
    parser_calls = 0
    parser_calls_lock = threading.Lock()

    def parser(*_: Any) -> ParsedIngestion:
        nonlocal parser_calls
        with parser_calls_lock:
            parser_calls += 1
        first_parser_started.set()
        assert release_first_parser.wait(timeout=5)
        return ParsedIngestion(document=object())

    service, _, _, _, _ = _service(checker=checker)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.execute,
            {"file_path": str(paths[0])},
            _spec(parser),
        )
        assert first_parser_started.wait(timeout=5)
        second = executor.submit(
            service.execute,
            {"file_path": str(paths[1])},
            _spec(parser),
        )
        assert registry.second_attempted.wait(timeout=5)
        try:
            assert registry.keys == [
                ("shared-hash", "default-docs", "profile"),
                ("shared-hash", "default-docs", "profile"),
            ]
            assert checker.should_skip_calls == 1
            assert parser_calls == 1
        finally:
            release_first_parser.set()
        results = [first.result(timeout=5), second.result(timeout=5)]

    assert checker.compute_calls == 2
    assert parser_calls == 1
    assert checker.should_skip_calls == 2
    assert checker.mark_success_calls == 1
    assert sorted(result["structuredContent"].get("skipped", False) for result in results) == [
        False,
        True,
    ]
    assert IngestionService.active_lock_count() == 0


def test_different_keys_can_process_in_parallel(tmp_path: Path) -> None:
    paths = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for path in paths:
        path.write_text("content", encoding="utf-8")
    both_parsers_started = threading.Barrier(2)

    def parser(*_: Any) -> ParsedIngestion:
        both_parsers_started.wait(timeout=5)
        return ParsedIngestion(document=object())

    service, checker, _, _, _ = _service()
    checker.compute_sha256.side_effect = lambda file_path: file_path
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                service.execute,
                {"file_path": str(path)},
                _spec(parser),
            )
            for path in paths
        ]
        results = [future.result(timeout=5) for future in futures]

    assert all(result["isError"] is False for result in results)
    assert IngestionService.active_lock_count() == 0


def test_lock_registry_releases_reference_after_thread_failure(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")
    service, _, pipeline_factory, _, _ = _service()

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            service.execute,
            {"file_path": str(file_path)},
            _spec(MagicMock(side_effect=RuntimeError("parser failed"))),
        ).result(timeout=5)

    assert result["isError"] is True
    pipeline_factory.assert_not_called()
    assert IngestionService.active_lock_count() == 0
