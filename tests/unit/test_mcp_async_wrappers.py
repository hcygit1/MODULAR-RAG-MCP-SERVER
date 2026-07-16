"""FastMCP 公共工具异步包装测试。"""

import asyncio
import importlib
import inspect
import subprocess
import sys
import threading
import textwrap
from types import SimpleNamespace

import pytest

_TOOL_CASES = [
    pytest.param(
        "src.mcp_server.tools.query_knowledge_hub",
        "query_knowledge_hub",
        "execute_query_knowledge_hub",
        {
            "query": "question",
            "collection_name": "documents",
            "top_k": 3,
        },
        {
            "query": "question",
            "collection_name": "documents",
            "top_k": 3,
        },
        id="query-knowledge-hub",
    ),
    pytest.param(
        "src.mcp_server.tools.list_collections",
        "list_collections",
        "execute_list_collections",
        {"base_path": "/deprecated"},
        {},
        id="list-collections",
    ),
    pytest.param(
        "src.mcp_server.tools.ingest_document_normal",
        "ingest_document_normal",
        "execute_ingest_document_normal",
        {
            "file_path": "document.md",
            "collection_name": "documents",
            "force": True,
        },
        {
            "file_path": "document.md",
            "collection_name": "documents",
            "force": True,
        },
        id="ingest-document-normal",
    ),
    pytest.param(
        "src.mcp_server.tools.ingest_document_docling",
        "ingest_document_docling",
        "execute_ingest_document_docling",
        {
            "file_path": "document.pdf",
            "collection_name": "documents",
            "force": True,
        },
        {
            "file_path": "document.pdf",
            "collection_name": "documents",
            "force": True,
        },
        id="ingest-document-docling",
    ),
    pytest.param(
        "src.mcp_server.tools.ingest_document_mineru",
        "ingest_document_mineru",
        "execute_ingest_document_mineru",
        {
            "file_path": "document.pdf",
            "collection_name": "documents",
            "force": True,
        },
        {
            "file_path": "document.pdf",
            "collection_name": "documents",
            "force": True,
        },
        id="ingest-document-mineru",
    ),
]


async def _wait_for_thread_event(
    event: threading.Event,
    timeout: float = 5,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not event.is_set():
        if loop.time() >= deadline:
            raise TimeoutError("worker thread did not reach the expected state")
        await asyncio.sleep(0.001)


def _run_cancellation_script(script: str) -> None:
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=True,
        timeout=2,
    )


def test_run_sync_callable_propagates_inner_cancelled_error() -> None:
    _run_cancellation_script(
        """
        import asyncio

        from src.mcp_server.tools.mcp_utils import run_sync_callable

        inner = asyncio.CancelledError("inner")

        def raise_inner():
            raise inner

        async def main():
            try:
                await run_sync_callable(raise_inner)
            except asyncio.CancelledError as exc:
                assert exc is inner
            else:
                raise AssertionError("inner cancellation was not propagated")

        asyncio.run(main())
        """
    )


def test_run_sync_callable_propagates_cancelled_executor_operation() -> None:
    _run_cancellation_script(
        """
        import asyncio
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from src.mcp_server.tools.mcp_utils import run_sync_callable

        release = threading.Event()
        executor = ThreadPoolExecutor(max_workers=1)
        blocker = executor.submit(release.wait, 5)

        async def main():
            task = asyncio.create_task(
                run_sync_callable(lambda: "unreachable", executor=executor)
            )
            await asyncio.sleep(0)
            executor.shutdown(wait=False, cancel_futures=True)
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("cancelled operation was not propagated")
            finally:
                release.set()
                blocker.result(timeout=5)

        asyncio.run(main())
        """
    )


@pytest.mark.parametrize(
    "module_name,tool_name,execute_name,tool_kwargs,expected_arguments",
    _TOOL_CASES,
)
def test_fastmcp_public_tools_are_async(
    module_name,
    tool_name,
    execute_name,
    tool_kwargs,
    expected_arguments,
) -> None:
    module = importlib.import_module(module_name)
    tool = getattr(module, tool_name)

    assert inspect.iscoroutinefunction(tool)


@pytest.mark.parametrize(
    "module_name,tool_name,execute_name,tool_kwargs,expected_arguments",
    _TOOL_CASES,
)
def test_fastmcp_public_tool_offloads_blocking_executor(
    monkeypatch,
    module_name,
    tool_name,
    execute_name,
    tool_kwargs,
    expected_arguments,
) -> None:
    module = importlib.import_module(module_name)
    tool = getattr(module, tool_name)
    calls = []
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocking_execute(arguments):
        calls.append(arguments)
        started.set()
        try:
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release worker thread")
            return {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {"citations": []},
                "isError": False,
            }
        finally:
            finished.set()

    monkeypatch.setattr(module, execute_name, blocking_execute)

    async def scenario() -> None:
        task = asyncio.create_task(tool(**tool_kwargs))
        await _wait_for_thread_event(started)
        try:
            assert finished.is_set() is False
            assert task.done() is False
        finally:
            release.set()
        result = await task
        assert result.isError is False

    try:
        asyncio.run(scenario())
    finally:
        release.set()
    assert calls == [expected_arguments]


def test_query_tool_serializes_calls_on_one_worker_thread(monkeypatch) -> None:
    module = importlib.import_module(
        "src.mcp_server.tools.query_knowledge_hub"
    )
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    thread_ids: list[int] = []
    calls_lock = threading.Lock()

    def blocking_execute(arguments):
        with calls_lock:
            call_index = len(thread_ids)
            thread_ids.append(threading.get_ident())
        if call_index == 0:
            first_started.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release first query")
        else:
            second_started.set()
        return {
            "content": [{"type": "text", "text": arguments["query"]}],
            "structuredContent": {"citations": []},
            "isError": False,
        }

    monkeypatch.setattr(module, "execute_query_knowledge_hub", blocking_execute)

    async def scenario() -> None:
        first = asyncio.create_task(module.query_knowledge_hub("first"))
        await _wait_for_thread_event(first_started)
        second = asyncio.create_task(module.query_knowledge_hub("second"))
        try:
            await asyncio.sleep(0.05)
            assert second_started.is_set() is False
        finally:
            release_first.set()
            await asyncio.gather(first, second, return_exceptions=True)

    asyncio.run(scenario())
    assert second_started.is_set()
    assert len(thread_ids) == 2
    assert len(set(thread_ids)) == 1


def test_sync_and_async_query_share_the_same_worker_thread(monkeypatch) -> None:
    module = importlib.import_module(
        "src.mcp_server.tools.query_knowledge_hub"
    )
    thread_ids: list[int] = []

    class ThreadAffinePipeline:
        def retrieve(self, **kwargs):
            thread_ids.append(threading.get_ident())
            return []

    settings = SimpleNamespace(
        vector_store=SimpleNamespace(
            backend="memory",
            collection_name="documents",
        ),
        retrieval=SimpleNamespace(sparse_backend="bm25"),
    )
    monkeypatch.setattr(module, "_pipeline", ThreadAffinePipeline())
    monkeypatch.setattr(module, "_cached_settings", settings)

    sync_result = module.execute_query_knowledge_hub({"query": "sync"})
    async_result = asyncio.run(module.query_knowledge_hub("async"))

    assert sync_result["isError"] is False
    assert async_result.isError is False
    assert len(thread_ids) == 2
    assert len(set(thread_ids)) == 1
    assert thread_ids[0] != threading.get_ident()


@pytest.mark.parametrize(
    "module_name,tool_name,execute_name,tool_kwargs,expected_arguments",
    _TOOL_CASES,
)
def test_cancelled_wrapper_waits_for_sync_operation_to_finish(
    monkeypatch,
    module_name,
    tool_name,
    execute_name,
    tool_kwargs,
    expected_arguments,
) -> None:
    module = importlib.import_module(module_name)
    tool = getattr(module, tool_name)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []

    def blocking_execute(arguments):
        calls.append(arguments)
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release worker thread")
        finished.set()
        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {},
            "isError": False,
        }

    monkeypatch.setattr(module, execute_name, blocking_execute)

    async def scenario() -> None:
        task = asyncio.create_task(tool(**tool_kwargs))
        await _wait_for_thread_event(started)
        task.cancel()
        await asyncio.sleep(0.02)
        assert task.done() is False
        task.cancel()
        await asyncio.sleep(0.05)
        assert task.done() is False
        assert finished.is_set() is False

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    try:
        asyncio.run(scenario())
    finally:
        release.set()

    assert calls == [expected_arguments]


def test_cancelled_wrapper_preserves_sync_failure_as_cause(monkeypatch) -> None:
    module = importlib.import_module(
        "src.mcp_server.tools.ingest_document_normal"
    )
    started = threading.Event()
    release = threading.Event()

    def failing_execute(arguments):
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release worker thread")
        raise RuntimeError("write failed")

    monkeypatch.setattr(
        module,
        "execute_ingest_document_normal",
        failing_execute,
    )

    async def scenario() -> None:
        task = asyncio.create_task(
            module.ingest_document_normal("document.md")
        )
        await _wait_for_thread_event(started)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await task
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "write failed"

    try:
        asyncio.run(scenario())
    finally:
        release.set()
