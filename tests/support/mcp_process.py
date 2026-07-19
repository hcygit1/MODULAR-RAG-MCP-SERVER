"""Helpers for reliable JSON-RPC exchanges with MCP subprocesses."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from typing import Any


def send_until_response(
    proc: subprocess.Popen[str],
    payload: str,
    response_id: int,
    timeout: float,
) -> tuple[str, str]:
    """Keep stdin open until the requested response arrives, then reap the process."""
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    reader_errors: list[Exception] = []
    writer_errors: list[Exception] = []
    response_received = threading.Event()
    request_written = threading.Event()

    def read_stdout() -> None:
        try:
            for line in proc.stdout:
                stdout_lines.append(line)
                try:
                    response = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if response.get("id") == response_id:
                    response_received.set()
        except Exception as exc:
            reader_errors.append(exc)

    def read_stderr() -> None:
        try:
            stderr_lines.extend(proc.stderr)
        except Exception as exc:
            reader_errors.append(exc)

    def write_stdin() -> None:
        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            writer_errors.append(exc)
        finally:
            request_written.set()

    stdout_reader = threading.Thread(target=read_stdout, daemon=True)
    stderr_reader = threading.Thread(target=read_stderr, daemon=True)
    stdin_writer = threading.Thread(target=write_stdin, daemon=True)
    stdout_reader.start()
    stderr_reader.start()
    stdin_writer.start()

    deadline = time.monotonic() + timeout
    failure_reason: str | None = None
    try:
        while not request_written.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure_reason = (
                    f"Timed out writing MCP request after {timeout} seconds"
                )
                break
            request_written.wait(min(0.05, remaining))

        if failure_reason is None and writer_errors:
            failure_reason = f"Failed to write MCP request: {writer_errors[0]}"

        while failure_reason is None and not response_received.is_set():
            if reader_errors:
                failure_reason = f"Failed to read MCP subprocess output: {reader_errors[0]}"
                break
            if proc.poll() is not None:
                stdout_reader.join(timeout=max(0, deadline - time.monotonic()))
                if response_received.is_set():
                    break
                failure_reason = (
                    f"MCP server exited before response id={response_id}; "
                    f"returncode={proc.returncode}"
                )
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure_reason = (
                    f"Timed out waiting for MCP response id={response_id} "
                    f"after {timeout} seconds"
                )
                break
            response_received.wait(min(0.05, remaining))

        if failure_reason is None:
            proc.stdin.close()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure_reason = (
                    f"Timed out waiting for MCP server exit after response id={response_id}"
                )
            else:
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    failure_reason = (
                        f"MCP server did not exit after response id={response_id} "
                        f"within {timeout} seconds"
                    )
    finally:
        if failure_reason is not None and proc.poll() is None:
            proc.kill()
        stdin_writer.join(timeout=1)
        if stdin_writer.is_alive():
            if proc.poll() is None:
                proc.kill()
            stdin_writer.join(timeout=1)
        if stdin_writer.is_alive() and failure_reason is None:
            failure_reason = "MCP stdin writer did not exit"
        if not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError as exc:
                if failure_reason is None:
                    failure_reason = f"Failed to close MCP stdin: {exc}"
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

        stdout_reader.join(timeout=1)
        stderr_reader.join(timeout=1)
        readers = (
            (stdout_reader, proc.stdout),
            (stderr_reader, proc.stderr),
        )
        for reader, stream in readers:
            if reader.is_alive() and not stream.closed:
                stream.close()
                reader.join(timeout=1)
        for reader, stream in readers:
            if not stream.closed:
                stream.close()
            if reader.is_alive() and failure_reason is None:
                failure_reason = f"MCP output reader did not exit: {reader.name}"
        if reader_errors and failure_reason is None:
            failure_reason = f"Failed to read MCP subprocess output: {reader_errors[0]}"

    stdout_data = "".join(stdout_lines)
    stderr_data = "".join(stderr_lines)
    if failure_reason is not None:
        raise AssertionError(
            f"{failure_reason}\nstdout:\n{stdout_data}\nstderr:\n{stderr_data}"
        )
    return stdout_data, stderr_data


def jsonrpc_response(stdout_data: str, response_id: int) -> dict[str, Any]:
    """Return the JSON-RPC response with the requested id."""
    for line in stdout_data.splitlines():
        if not line.strip():
            continue
        response = json.loads(line)
        if response.get("id") == response_id:
            return response
    raise AssertionError(f"Missing JSON-RPC response id={response_id}: {stdout_data}")
