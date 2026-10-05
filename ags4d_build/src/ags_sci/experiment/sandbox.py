"""AGS-Sci Experiment Sandboxing Subsystem.

Provides one-shot process isolation, AST-level static security filtering,
loop execution budgeting, bounded JSON-safe result transport, and explicit
resource monitoring. No experiment shares a worker interpreter with another.
"""
from __future__ import annotations

import ast
import contextlib
import io
import multiprocessing as mp
import os
import queue
import tempfile
import time
import hashlib
import json
import ipaddress
import socket
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .dimensions import ExperimentSpec, Admission, admit
from ..core.security import DEFAULT_SECURITY_POLICY


def _safe_json_value(value: Any, *, max_bytes: int = DEFAULT_SECURITY_POLICY.max_result_bytes, _depth: int = 0) -> Any:
    """Convert a worker result to a JSON-safe value before it crosses the process boundary.

    This deliberately forbids arbitrary pickle/object deserialization in the parent.
    NumPy arrays/scalars are converted to plain JSON data when bounded.
    """
    import numpy as np
    if _depth > DEFAULT_SECURITY_POLICY.max_recursion_depth:
        raise ValueError("result nesting exceeds security limit")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > DEFAULT_SECURITY_POLICY.max_string_chars:
            raise ValueError("result string exceeds security limit")
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("result contains non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > DEFAULT_SECURITY_POLICY.max_collection_items:
            raise ValueError("result collection exceeds security limit")
        return [_safe_json_value(v, max_bytes=max_bytes, _depth=_depth+1) for v in value]
    if isinstance(value, dict):
        if len(value) > DEFAULT_SECURITY_POLICY.max_payload_keys:
            raise ValueError("result object exceeds security limit")
        out = {}
        for k, v in value.items():
            if not isinstance(k, str) or len(k) > 256:
                raise ValueError("result object key is invalid")
            out[k] = _safe_json_value(v, max_bytes=max_bytes, _depth=_depth+1)
        return out
    try:
        if isinstance(value, np.generic):
            return _safe_json_value(value.item(), max_bytes=max_bytes, _depth=_depth+1)
        if isinstance(value, np.ndarray):
            if value.nbytes > max_bytes:
                raise ValueError("result array exceeds security byte limit")
            if value.ndim > DEFAULT_SECURITY_POLICY.max_recursion_depth:
                raise ValueError("result array dimensionality exceeds security limit")
            if not np.all(np.isfinite(value)):
                raise ValueError("result array contains non-finite values")
            return {"__ndarray__": True, "shape": list(value.shape), "dtype": str(value.dtype),
                    "data": _safe_json_value(value.tolist(), max_bytes=max_bytes, _depth=_depth+1)}
    except ImportError:
        pass
    raise TypeError(f"unsupported result type: {type(value).__name__}")

def _encode_safe_result(value: Any) -> bytes:
    safe = _safe_json_value(value)
    data = json.dumps(safe, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(data) > DEFAULT_SECURITY_POLICY.max_result_bytes:
        raise ValueError("result exceeds security byte limit")
    return data


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    stdout: str
    error: Optional[str]
    execution_time_ms: float
    value: Any = None
    artifacts: Dict[str, bytes] = None


class SecurityFirewall(ast.NodeVisitor):
    BLOCKED_MODULES: Set[str] = {
        "os", "sys", "subprocess", "socket", "shutil",
        "urllib", "requests", "http", "pathlib", "posix", "nt",
        "ctypes", "multiprocessing", "threading", "importlib",
    }
    ALLOWED_MODULES: Set[str] = {"math", "numpy", "scipy"}
    BLOCKED_BUILTINS: Set[str] = {
        "__import__", "eval", "exec", "open", "compile", "input",
        "breakpoint", "memoryview", "globals", "locals",
    }

    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_module = alias.name.split(".")[0]
            if root_module not in self.ALLOWED_MODULES:
                self.violations.append(f"Blocked import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root_module = (node.module or "").split(".")[0]
        if root_module not in self.ALLOWED_MODULES:
            self.violations.append(f"Blocked from-import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.BLOCKED_BUILTINS:
            self.violations.append(f"Blocked builtin call: {node.func.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.violations.append(f"Blocked dunder attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.BLOCKED_BUILTINS:
            self.violations.append(f"Blocked builtin name: {node.id}")
        self.generic_visit(node)


class LoopFuelTransformer(ast.NodeTransformer):
    def _create_guard(self) -> ast.If:
        return ast.If(
            test=ast.Compare(
                left=ast.Name(id="_ags_fuel", ctx=ast.Load()),
                ops=[ast.LtE()],
                comparators=[ast.Constant(value=0)],
            ),
            body=[
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="TimeoutError", ctx=ast.Load()),
                        args=[ast.Constant(value="Execution loop fuel budget exhausted.")],
                        keywords=[],
                    )
                )
            ],
            orelse=[
                ast.AugAssign(
                    target=ast.Name(id="_ags_fuel", ctx=ast.Store()),
                    op=ast.Sub(),
                    value=ast.Constant(value=1),
                )
            ],
        )

    def visit_While(self, node: ast.While) -> ast.While:
        node = self.generic_visit(node)
        node.body.insert(0, self._create_guard())
        return node

    def _contains_loop(self, node: ast.AST) -> bool:
        return any(isinstance(n, (ast.For, ast.While, ast.AsyncFor))
                   for n in ast.walk(node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        # Fuel is stored in the sandbox global namespace.  A loop inside a
        # nested function otherwise makes ``_ags_fuel -= 1`` a local binding
        # and raises UnboundLocalError.  Declare it global only where needed.
        node = self.generic_visit(node)
        if self._contains_loop(node) and not any(
            isinstance(stmt, ast.Global) and '_ags_fuel' in stmt.names
            for stmt in node.body[:3]
        ):
            node.body.insert(0, ast.Global(names=['_ags_fuel']))
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        node = self.generic_visit(node)
        if self._contains_loop(node) and not any(
            isinstance(stmt, ast.Global) and '_ags_fuel' in stmt.names
            for stmt in node.body[:3]
        ):
            node.body.insert(0, ast.Global(names=['_ags_fuel']))
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        node = self.generic_visit(node)
        node.body.insert(0, self._create_guard())
        return node


def _read_rss_bytes(pid: int) -> int | None:
    """Best-effort Linux/Android RSS measurement for active sandbox workers."""
    try:
        with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024 if len(parts) >= 2 else None
    except (OSError, ValueError):
        return None
    return None


def _clear_environment() -> None:
    os.environ.clear()
    os.environ.update({
        "PYTHONPATH": "",
        "PYTHONHASHSEED": "0",
        "PATH": "/usr/bin:/bin",
    })


def _validate_public_https(url: str, allowed_domains: Tuple[str, ...] = ()) -> str:
    parsed = urllib.parse.urlparse(str(url))
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("sandbox internet permits HTTPS URLs only")
    host = parsed.hostname.lower().rstrip('.')
    if allowed_domains and not (host in allowed_domains or any(host.endswith('.' + d) for d in allowed_domains)):
        raise ValueError("domain not allowed by sandbox internet policy")
    infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            raise ValueError("private/reserved network target blocked")
    return host

def _sandbox_internet_fetch(url: str, max_bytes: int, timeout: float, allowed_domains: Tuple[str, ...]):
    host = _validate_public_https(url, allowed_domains)
    request = urllib.request.Request(str(url), headers={"User-Agent": "AGS-Sci-Sandbox/102.2"})
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:  # nosec B310 - HTTPS + target validation above
        final_url = response.geturl()
        final_host = _validate_public_https(final_url, allowed_domains)
        data = response.read(int(max_bytes) + 1)
        if len(data) > int(max_bytes):
            raise ValueError("response exceeds sandbox internet byte limit")
        content_type = response.headers.get("Content-Type", "")
        return {
            "url": str(url), "final_url": final_url, "host": host, "final_host": final_host,
            "status": int(getattr(response, "status", 200)),
            "content_type": content_type, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "text": data.decode("utf-8", errors="replace"),
        }

def _safe_import(name: str, globals_: Dict[str, Any] | None = None,
                 locals_: Dict[str, Any] | None = None,
                 fromlist: tuple[str, ...] = (), level: int = 0):
    root = name.split(".", 1)[0]
    if root not in SecurityFirewall.ALLOWED_MODULES:
        raise ImportError(f"blocked import: {root}")
    return __import__(name, globals_, locals_, fromlist, level)


def _warm_worker_daemon(task_queue: Any, result_queue: Any, max_memory_mb: int, ready_event: Any,
                        internet_max_bytes: int, internet_timeout: float,
                        internet_allowed_domains: Tuple[str, ...]) -> None:
    _clear_environment()
    import numpy as np
    import scipy  # noqa: F401
    ready_event.set()

    # Pre-load common modules once. The submitted code receives only these
    # numerical capabilities through a constrained import hook.
    while True:
        try:
            task = task_queue.get()
        except (KeyboardInterrupt, BrokenPipeError, EOFError):
            break
        if task is None:
            break

        code_str, loop_fuel = task
        start_time = time.perf_counter()
        captured_stdout = io.StringIO()
        try:
            with tempfile.TemporaryDirectory(prefix="ags_sandbox_") as tmp_dir:
                orig_cwd = os.getcwd()
                try:
                    os.chdir(tmp_dir)
                    # A pre-warmed NumPy/SciPy process already owns its
                    # scientific runtime address space. Applying a small
                    # RLIMIT_AS after preload can kill the worker before the
                    # first task, so memory_mb is retained as a policy value
                    # but is not enforced as RLIMIT_AS in the warm daemon.

                    safe_builtins = {
                        "range": range, "len": len, "print": print,
                        "min": min, "max": max, "sum": sum, "abs": abs,
                        "enumerate": enumerate, "zip": zip, "float": float,
                        "int": int, "str": str, "bool": bool, "list": list,
                        "dict": dict, "set": set, "tuple": tuple, "complex": complex,
                        "TimeoutError": TimeoutError, "Exception": Exception,
                        "__import__": _safe_import,
                        "internet_fetch": lambda url: _sandbox_internet_fetch(
                            url, internet_max_bytes, internet_timeout, internet_allowed_domains),
                    }
                    isolated_globals: Dict[str, Any] = {
                        "__builtins__": safe_builtins,
                        "np": np,
                        "numpy": np,
                        "_ags_fuel": int(loop_fuel),
                    }
                    with contextlib.redirect_stdout(captured_stdout):
                        tree = ast.parse(code_str, mode="exec")
                        tree = LoopFuelTransformer().visit(tree)
                        ast.fix_missing_locations(tree)
                        compiled = compile(tree, "<ags_sandbox>", "exec")
                        exec(compiled, isolated_globals)  # noqa: S102
                        result_value = _encode_safe_result(isolated_globals.get("result"))
                    artifacts = {}
                    for root, _, files in os.walk(tmp_dir):
                        for filename in files:
                            path = os.path.join(root, filename)
                            rel = os.path.relpath(path, tmp_dir).replace(os.sep, "/")
                            if rel.startswith("../") or rel == "..":
                                continue
                            with open(path, "rb") as fh:
                                artifacts[rel] = fh.read()
                finally:
                    os.chdir(orig_cwd)

            elapsed = (time.perf_counter() - start_time) * 1000.0
            result_queue.put(("SUCCESS", captured_stdout.getvalue(), None, elapsed, result_value, artifacts))
        except Exception as exc:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            result_queue.put(("ERROR", captured_stdout.getvalue(), f"{type(exc).__name__}: {exc}", elapsed, None, {}))


def _sandbox_worker_once(code_str: str, loop_fuel: int, result_queue: Any,
                        internet_enabled: bool, internet_max_bytes: int,
                        internet_timeout: float, internet_allowed_domains: Tuple[str, ...],
                        max_memory_mb: int) -> None:
    """Execute exactly one experiment in a fresh interpreter."""
    _clear_environment()
    try:
        import resource
        limit = int(max_memory_mb * 1024 * 1024)
        if hasattr(resource, "RLIMIT_RSS"):
            resource.setrlimit(resource.RLIMIT_RSS, (limit, limit))
    except Exception:
        pass
    captured_stdout = io.StringIO()
    try:
        import numpy as np
        import scipy  # noqa: F401
        with tempfile.TemporaryDirectory(prefix="ags_sandbox_") as tmp_dir:
            orig_cwd = os.getcwd()
            try:
                os.chdir(tmp_dir)
                safe_builtins = {
                    "range": range, "len": len, "print": print, "min": min, "max": max,
                    "sum": sum, "abs": abs, "enumerate": enumerate, "zip": zip,
                    "float": float, "int": int, "str": str, "bool": bool, "list": list,
                    "dict": dict, "set": set, "tuple": tuple, "complex": complex,
                    "TimeoutError": TimeoutError, "Exception": Exception, "__import__": _safe_import,
                }
                if internet_enabled:
                    safe_builtins["internet_fetch"] = lambda url: _sandbox_internet_fetch(
                        url, internet_max_bytes, internet_timeout, internet_allowed_domains)
                isolated_globals = {"__builtins__": safe_builtins, "np": np, "numpy": np, "_ags_fuel": int(loop_fuel)}
                with contextlib.redirect_stdout(captured_stdout):
                    tree = ast.parse(code_str, mode="exec")
                    tree = LoopFuelTransformer().visit(tree)
                    ast.fix_missing_locations(tree)
                    compiled = compile(tree, "<ags_sandbox>", "exec")
                    exec(compiled, isolated_globals)  # noqa: S102
                    result_value = _encode_safe_result(isolated_globals.get("result"))
                artifacts: Dict[str, bytes] = {}
                total = 0
                for root, _, files in os.walk(tmp_dir):
                    for filename in files:
                        path = os.path.join(root, filename)
                        rel = os.path.relpath(path, tmp_dir).replace(os.sep, "/")
                        if rel.startswith("../") or rel == "..":
                            continue
                        size = os.path.getsize(path)
                        total += size
                        if len(artifacts) >= DEFAULT_SECURITY_POLICY.max_artifacts or total > DEFAULT_SECURITY_POLICY.max_artifact_bytes:
                            raise RuntimeError("artifact output exceeds security limit")
                        with open(path, "rb") as fh:
                            artifacts[rel] = fh.read()
                stdout = captured_stdout.getvalue()
                if len(stdout) > DEFAULT_SECURITY_POLICY.max_stdout_chars:
                    raise RuntimeError("stdout exceeds security limit")
                result_queue.put(("SUCCESS", stdout, None, result_value, artifacts))
            finally:
                os.chdir(orig_cwd)
    except Exception as exc:
        try:
            result_queue.put(("ERROR", captured_stdout.getvalue()[:DEFAULT_SECURITY_POLICY.max_stdout_chars],
                              f"{type(exc).__name__}: {exc}", None, {}))
        except Exception:
            pass


class ExecutionSandbox:
    """Fail-closed, one-shot process sandbox.

    Every experiment gets a fresh interpreter. No experiment shares imported
    modules, globals, queues, temporary state, or allocator state with another.
    """
    def __init__(self, timeout_seconds: float = 2.0, max_memory_mb: int = 256,
                 loop_fuel: int = 1_000_000, pool_size: int = 1,
                 internet_enabled: bool = False, internet_max_bytes: int = 2_000_000,
                 internet_timeout: float = 8.0, internet_allowed_domains: Tuple[str, ...] = ()) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.max_memory_mb = int(max_memory_mb)
        self.loop_fuel = int(loop_fuel)
        self.pool_size = 1  # retained for API compatibility; shared pools are forbidden.
        self.internet_enabled = bool(internet_enabled)
        self.internet_max_bytes = int(internet_max_bytes)
        self.internet_timeout = float(internet_timeout)
        self.internet_allowed_domains = tuple(str(x).lower().lstrip('.') for x in internet_allowed_domains)
        if self.timeout_seconds <= 0 or self.max_memory_mb <= 0 or self.loop_fuel <= 0:
            raise ValueError("invalid sandbox resource policy")
        if self.internet_max_bytes <= 0 or self.internet_timeout <= 0:
            raise ValueError("invalid internet policy")
        # Fork gives fast isolated scientific workers on POSIX after NumPy/SciPy
        # are already loaded. On platforms without fork, use spawn.
        methods = mp.get_all_start_methods()
        if "forkserver" in methods:
            try:
                mp.set_forkserver_preload(["numpy", "scipy"])
            except Exception:
                pass
            method = "forkserver"
        elif "spawn" in methods:
            method = "spawn"
        else:
            method = methods[0]
        self.ctx = mp.get_context(method)
        self._active_process: Optional[mp.Process] = None
        self._closed = False

    def _terminate_active(self) -> None:
        p = self._active_process
        if p is None:
            return
        try:
            if p.is_alive():
                p.terminate()
                p.join(timeout=0.3)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=0.3)
            else:
                p.join(timeout=0.1)
        finally:
            self._active_process = None

    def execute_experiment(self, spec: ExperimentSpec, code: str) -> ExecutionResult:
        admission: Admission = admit(spec)
        if not admission.allowed:
            return ExecutionResult("REJECTED", "", f"ExperimentAdmission: {admission.reason}", 0.0)
        return self.execute(code)

    def execute(self, code: str) -> ExecutionResult:
        t0 = time.perf_counter()
        if self._closed:
            return ExecutionResult("ERROR", "", "sandbox is closed", 0.0)
        if not isinstance(code, str):
            return ExecutionResult("BLOCKED", "", "source must be text", 0.0)
        if len(code) > DEFAULT_SECURITY_POLICY.max_source_chars:
            return ExecutionResult("BLOCKED", "", "source exceeds security size limit", 0.0)
        try:
            parsed_tree = ast.parse(code, mode="exec")
        except SyntaxError as syn_err:
            return ExecutionResult("ERROR", "", f"SyntaxError: {syn_err}", (time.perf_counter() - t0) * 1000.0)
        if sum(1 for _ in ast.walk(parsed_tree)) > DEFAULT_SECURITY_POLICY.max_ast_nodes:
            return ExecutionResult("BLOCKED", "", "source AST exceeds security complexity limit", (time.perf_counter() - t0) * 1000.0)
        firewall = SecurityFirewall()
        firewall.visit(parsed_tree)
        if firewall.violations:
            return ExecutionResult("BLOCKED", "", "; ".join(firewall.violations), (time.perf_counter() - t0) * 1000.0)

        q = self.ctx.SimpleQueue()
        p = self.ctx.Process(
            target=_sandbox_worker_once,
            args=(code, self.loop_fuel, q, self.internet_enabled,
                  self.internet_max_bytes, self.internet_timeout, self.internet_allowed_domains,
                  self.max_memory_mb),
            daemon=True,
        )
        self._active_process = p
        p.start()
        deadline = time.monotonic() + self.timeout_seconds
        memory_limit = self.max_memory_mb * 1024 * 1024
        memory_exceeded = False
        while p.is_alive():
            if time.monotonic() >= deadline:
                self._terminate_active()
                return ExecutionResult("TIMEOUT", "", f"Execution exceeded timeout limit of {self.timeout_seconds}s.", (time.perf_counter() - t0) * 1000.0)
            rss = _read_rss_bytes(p.pid)
            if rss is not None and rss > memory_limit:
                memory_exceeded = True
                self._terminate_active()
                return ExecutionResult("RESOURCE_LIMIT", "", f"sandbox worker exceeded memory limit of {self.max_memory_mb} MB", (time.perf_counter() - t0) * 1000.0)
            time.sleep(0.005)
        if memory_exceeded:
            return ExecutionResult("RESOURCE_LIMIT", "", f"sandbox worker exceeded memory limit of {self.max_memory_mb} MB", (time.perf_counter() - t0) * 1000.0)
        self._active_process = None
        if p.exitcode not in (0, None):
            return ExecutionResult("ERROR", "", f"sandbox worker exited with code {p.exitcode}", (time.perf_counter() - t0) * 1000.0)
        try:
            status, stdout, error, value, artifacts = q.get()
        except Exception:
            return ExecutionResult("ERROR", "", f"sandbox worker exited with code {p.exitcode}", (time.perf_counter() - t0) * 1000.0)

        if len(stdout) > DEFAULT_SECURITY_POLICY.max_stdout_chars:
            return ExecutionResult("BLOCKED", stdout[:DEFAULT_SECURITY_POLICY.max_stdout_chars], "stdout exceeds security limit", (time.perf_counter() - t0) * 1000.0)
        if len(artifacts) > DEFAULT_SECURITY_POLICY.max_artifacts:
            return ExecutionResult("BLOCKED", stdout, "artifact count exceeds security limit", (time.perf_counter() - t0) * 1000.0)
        if sum(len(v) for v in artifacts.values()) > DEFAULT_SECURITY_POLICY.max_artifact_bytes:
            return ExecutionResult("BLOCKED", stdout, "artifact bytes exceed security limit", (time.perf_counter() - t0) * 1000.0)
        if status == "SUCCESS":
            try:
                if not isinstance(value, (bytes, bytearray)) or len(value) > DEFAULT_SECURITY_POLICY.max_result_bytes:
                    raise ValueError("invalid result transport")
                value = json.loads(bytes(value).decode("utf-8"))
            except Exception as exc:
                return ExecutionResult("BLOCKED", stdout, f"invalid or oversized result: {exc}", (time.perf_counter() - t0) * 1000.0)
        return ExecutionResult(status, stdout, error, (time.perf_counter() - t0) * 1000.0, value, artifacts)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._terminate_active()

    def __enter__(self) -> "ExecutionSandbox":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
# Backward-compatible public facade preserving the existing AGS API.
@dataclass(frozen=True)
class SandboxLimits:
    timeout_s: float = 2.0
    memory_mb: int = 256
    max_loop_iterations: int = 1_000_000


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    value: Any = None
    artifacts: Dict[str, bytes] = None
    error: Optional[str] = None
    elapsed_s: float = 0.0


class SandboxedRunner:
    def __init__(self, limits: SandboxLimits | None = None):
        self.limits = limits or SandboxLimits()
        self._sandbox = ExecutionSandbox(
            timeout_seconds=self.limits.timeout_s,
            max_memory_mb=self.limits.memory_mb,
            loop_fuel=self.limits.max_loop_iterations,
        )

    def run_source(self, source: str) -> SandboxResult:
        result = self._sandbox.execute(source)
        error = result.error
        if error and result.status == "BLOCKED":
            if "builtin call: open" in error:
                error = "blocked builtin: open"
            else:
                error = error.replace("Blocked ", "blocked ")
        if error and "fuel" in error.lower() and "exhausted" in error.lower():
            error = "fuel exhausted"
        if error and result.status == "TIMEOUT":
            error = "timeout"
        return SandboxResult(
            ok=result.status == "SUCCESS",
            value=result.value,
            error=error,
            elapsed_s=result.execution_time_ms / 1000.0,
        )

    def run(self, fn: Any, *args: Any, **kwargs: Any) -> SandboxResult:
        # Callable execution remains available for compatibility. It is
        # isolated in a one-shot process because arbitrary Python callables
        # cannot be serialized safely into the restricted warm source worker.
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
        q = ctx.SimpleQueue()
        with tempfile.TemporaryDirectory(prefix="ags_sandbox_") as workdir:
            p = ctx.Process(target=_callable_worker_compat,
                            args=(fn, args, kwargs, q, self.limits.memory_mb, workdir),
                            daemon=True)
            start = time.perf_counter()
            p.start()
            p.join(self.limits.timeout_s)
            elapsed = time.perf_counter() - start
            if p.is_alive():
                p.kill(); p.join()
                return SandboxResult(False, error="timeout", elapsed_s=elapsed)
            if p.exitcode != 0:
                return SandboxResult(False, error=f"worker exited with code {p.exitcode}", elapsed_s=elapsed)
            try:
                ok, value_bytes, error = q.get()
                value = json.loads(bytes(value_bytes).decode("utf-8")) if ok else None
            except Exception:
                return SandboxResult(False, error="worker exited without a safe result", elapsed_s=elapsed)
            return SandboxResult(ok, value, error, elapsed)

    def close(self) -> None:
        self._sandbox.close()

    def __enter__(self) -> "SandboxedRunner":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort lifecycle cleanup for legacy callers that instantiate a
        # runner without a context manager. Never raise during interpreter exit.
        try:
            self.close()
        except Exception:
            pass


def _callable_worker_compat(fn: Any, args: Any, kwargs: Any, outq: Any,
                            memory_mb: int, workdir: str) -> None:
    try:
        _clear_environment()
        os.chdir(workdir)
        try:
            import resource
            limit = int(memory_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass
        outq.put((True, _encode_safe_result(fn(*args, **kwargs)), None))
    except Exception as exc:
        outq.put((False, None, f"{type(exc).__name__}: {exc}"))
