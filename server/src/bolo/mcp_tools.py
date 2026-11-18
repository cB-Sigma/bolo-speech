"""MCP tools for Realtime sessions: connect to Model Context Protocol servers and call their tools.

A Realtime session names a server in ``session.tools`` as OpenAI's Realtime API does::

    {"type": "mcp", "server_label": "docs", "server_url": "https://example.com/mcp",
     "allowed_tools": ["search"], "require_approval": "never", "headers": {"X-Token": "..."}}

or, for a server the operator configured (setting ``mcp_servers``), by its label alone
(``{"type": "mcp", "server_label": "files"}``). Configured servers can also be started as local
processes (``command`` and ``args``: the stdio transport); a session can never start a process.

Each server is opened once per session with the MCP Python SDK (``pip install 'bolo-speech[mcp]'``;
1.x and 2.x both work), over Streamable HTTP, or SSE when the URL ends in ``/sse``. Its tools are
offered to the language model as functions; when the model calls one, bolo calls the server and
hands the result back to the model, so MCP tool calls flow like function calls without the client
doing anything (unless the call needs the client's approval).

The SDK's transports run in task groups that must be entered and left by the same task, so every
connection lives in a task of its own and is driven through a queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("bolo.mcp")

INSTALL = "pip install 'bolo-speech[mcp]'"


class McpError(Exception):
    pass


@dataclass
class McpServerSpec:
    label: str
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    allowed_tools: list[str] | None = None
    #: "never", "always", or {"never": {"tool_names": [...]}, "always": {"tool_names": [...]}}
    require_approval: Any = "always"
    description: str | None = None
    #: from the operator's configuration (may use a command), not from a session
    configured: bool = False

    def key(self) -> str:
        """What identifies the connection: a change reopens it."""
        return json.dumps(
            [self.url, sorted(self.headers.items()), self.command, self.args, sorted((self.env or {}).items())]
        )

    def public(self) -> dict[str, Any]:
        """The tool as the session reports it (headers redacted)."""
        out: dict[str, Any] = {"type": "mcp", "server_label": self.label}
        if self.url and not self.configured:
            out["server_url"] = self.url
        if self.headers:
            out["headers"] = dict.fromkeys(self.headers, "***")
        if self.allowed_tools is not None:
            out["allowed_tools"] = list(self.allowed_tools)
        out["require_approval"] = self.require_approval
        if self.description:
            out["server_description"] = self.description
        return out

    def needs_approval(self, tool: str) -> bool:
        policy = self.require_approval
        if policy in (None, "never"):
            return False
        if policy == "always":
            return True
        if isinstance(policy, dict):
            never = ((policy.get("never") or {}).get("tool_names")) or []
            always = ((policy.get("always") or {}).get("tool_names")) or []
            if tool in always:
                return True
            if tool in never:
                return False
        return True

    def allows(self, tool: str) -> bool:
        return self.allowed_tools is None or tool in self.allowed_tools


def _names(value: Any, param: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, dict):  # OpenAI's {"tool_names": [...], "read_only": bool}
        value = value.get("tool_names")
        if value is None:
            return None
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{param} must be a list of tool names")
    return list(value)


def _approval(value: Any) -> Any:
    if value is None:
        return "always"
    if value in ("never", "always"):
        return value
    if isinstance(value, dict) and set(value) <= {"never", "always"}:
        for part in value.values():
            if not isinstance(part, dict):
                raise ValueError('require_approval must be "never", "always" or {"never"|"always": {tool_names}}')
            _names(part.get("tool_names"), "require_approval.tool_names")
        return value
    raise ValueError('require_approval must be "never", "always" or {"never"|"always": {"tool_names": [...]}}')


def configured_servers(settings: Any) -> dict[str, McpServerSpec]:
    """The ``mcp_servers`` setting: label -> {url | command, args, env, headers, allowed_tools,
    require_approval, description}."""
    out: dict[str, McpServerSpec] = {}
    for label, entry in dict(getattr(settings, "mcp_servers", None) or {}).items():
        if not isinstance(entry, dict):
            log.warning("ignoring mcp_servers.%s: not a table", label)
            continue
        if not entry.get("url") and not entry.get("command"):
            log.warning("ignoring mcp_servers.%s: it needs a url or a command", label)
            continue
        try:
            out[str(label)] = McpServerSpec(
                label=str(label),
                url=entry.get("url"),
                headers={str(k): str(v) for k, v in (entry.get("headers") or {}).items()},
                command=entry.get("command"),
                args=[str(a) for a in entry.get("args") or []],
                env={str(k): str(v) for k, v in entry["env"].items()} if entry.get("env") else None,
                allowed_tools=_names(entry.get("allowed_tools"), "allowed_tools"),
                require_approval=_approval(entry.get("require_approval", "never")),
                description=entry.get("description"),
                configured=True,
            )
        except ValueError as exc:
            log.warning("ignoring mcp_servers.%s: %s", label, exc)
    return out


LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")


def session_urls_allowed(settings: Any, authenticated: bool) -> bool:
    """Whether a session may name an MCP server by URL (setting ``realtime_mcp_urls``; unset: on a loopback bind,
    or when requests need an API key, so that nobody unknown can make the server connect where they like)."""
    value = getattr(settings, "realtime_mcp_urls", None)
    if value is not None:
        return bool(value)
    return authenticated or getattr(settings, "host", "127.0.0.1") in LOOPBACK


def parse_tool(tool: dict[str, Any], settings: Any, *, authenticated: bool = False) -> McpServerSpec:
    """A ``{"type": "mcp", ...}`` entry of ``session.tools``. Raises ValueError with a message for the client.
    ``authenticated``: requests to this server need an API key."""
    label = tool.get("server_label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("an MCP tool needs a server_label")
    configured = configured_servers(settings)
    url = tool.get("server_url")
    if tool.get("connector_id"):
        raise ValueError("connectors (connector_id) are not supported; give a server_url or a configured server_label")
    if not url:
        base = configured.get(label)
        if base is None:
            names = ", ".join(sorted(configured)) or "none"
            raise ValueError(
                f"no MCP server '{label}' is configured on this server (configured: {names}); give a server_url"
            )
        spec = McpServerSpec(**{**base.__dict__})
        if "allowed_tools" in tool:
            narrowed = _names(tool.get("allowed_tools"), "allowed_tools")
            if narrowed is not None and spec.allowed_tools is not None:
                narrowed = [t for t in narrowed if t in spec.allowed_tools]
            spec.allowed_tools = narrowed if narrowed is not None else spec.allowed_tools
        if "require_approval" in tool:
            spec.require_approval = _approval(tool.get("require_approval"))
        return spec
    if not session_urls_allowed(settings, authenticated):
        raise ValueError(
            "this server only connects to the MCP servers it has configured (realtime_mcp_urls; by default a "
            "server_url is accepted on a loopback address or when requests need an API key)"
        )
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("server_url must be an http:// or https:// address")
    headers = tool.get("headers") or {}
    if not isinstance(headers, dict):
        raise ValueError("headers must be an object")
    headers = {str(k): str(v) for k, v in headers.items()}
    if tool.get("authorization"):
        headers["Authorization"] = f"Bearer {tool['authorization']}"
    return McpServerSpec(
        label=label,
        url=url,
        headers=headers,
        allowed_tools=_names(tool.get("allowed_tools"), "allowed_tools"),
        require_approval=_approval(tool.get("require_approval")),
        description=tool.get("server_description"),
    )


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def tool_info(tool: Any) -> dict[str, Any]:
    """An MCP tool (SDK object) as the Realtime API lists it."""
    annotations = _field(tool, "annotations")
    return {
        "name": tool.name,
        "description": _field(tool, "description", default="") or "",
        "input_schema": _field(tool, "input_schema", "inputSchema", default=None) or {"type": "object"},
        "annotations": annotations.model_dump(exclude_none=True) if hasattr(annotations, "model_dump") else None,
    }


def result_text(result: Any) -> tuple[str, bool]:
    """A ``CallToolResult`` as text for the language model, and whether the tool reported an error."""
    parts: list[str] = []
    for block in _field(result, "content", default=[]) or []:
        kind = getattr(block, "type", None)
        if kind == "text":
            parts.append(str(block.text))
        elif kind == "resource":
            res = getattr(block, "resource", None)
            parts.append(str(getattr(res, "text", None) or getattr(res, "uri", "")))
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} content: {getattr(block, 'mimeType', None) or getattr(block, 'mime_type', '')}]")
        elif kind == "resource_link":
            parts.append(str(getattr(block, "uri", "")))
    structured = _field(result, "structured_content", "structuredContent")
    if not parts and structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False))
    return "\n".join(parts), bool(_field(result, "is_error", "isError", default=False))


@contextlib.asynccontextmanager
async def _transport(spec: McpServerSpec, timeout: float) -> Any:
    """The (read, write) streams for a server, with the installed SDK (1.x or 2.x)."""
    if spec.command:
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=spec.command, args=list(spec.args), env=spec.env)
        async with stdio_client(params) as streams:
            yield streams[0], streams[1]
        return
    assert spec.url is not None
    if spec.url.rstrip("/").endswith("/sse"):
        from mcp.client.sse import sse_client

        async with sse_client(spec.url, headers=spec.headers or None, timeout=timeout) as streams:
            yield streams[0], streams[1]
        return
    from mcp.client import streamable_http

    legacy = getattr(streamable_http, "streamablehttp_client", None)
    modern = getattr(streamable_http, "streamable_http_client", None)
    if modern is not None and hasattr(streamable_http, "create_mcp_http_client"):  # mcp 2.x
        http = streamable_http.create_mcp_http_client(headers=spec.headers or None)
        async with http, modern(spec.url, http_client=http) as streams:
            yield streams[0], streams[1]
        return
    if legacy is None:
        raise McpError("the installed mcp package has no Streamable HTTP client")
    async with legacy(spec.url, headers=spec.headers or None, timeout=timeout) as streams:  # mcp 1.x
        yield streams[0], streams[1]


class McpConnection:
    """One open MCP server: its tools, and calls to them."""

    def __init__(self, spec: McpServerSpec, *, timeout: float = 30.0) -> None:
        self.spec = spec
        self.timeout = timeout
        self.tools: list[dict[str, Any]] = []
        self.error: str | None = None
        self.server_name: str | None = None
        self._jobs: asyncio.Queue[tuple[str, dict[str, Any], asyncio.Future[Any]] | None] = asyncio.Queue()
        self._ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._task: asyncio.Task[None] | None = None

    async def open(self) -> None:
        """Connect and list the tools. Raises :class:`McpError` when the server cannot be used."""
        self._task = asyncio.create_task(self._run(), name=f"bolo-mcp-{self.spec.label}")
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), self.timeout)
        except asyncio.TimeoutError:
            await self.close()
            self.error = f"the MCP server '{self.spec.label}' did not answer within {self.timeout:g} s"
            raise McpError(self.error) from None

    async def _run(self) -> None:
        try:
            from mcp import ClientSession
        except ImportError:
            self._fail(f"MCP tools need the MCP Python SDK: {INSTALL}")
            return
        try:
            async with _transport(self.spec, self.timeout) as (read, write), ClientSession(read, write) as session:
                init = await session.initialize()
                info = _field(init, "server_info", "serverInfo")
                self.server_name = getattr(info, "name", None)
                listed = await session.list_tools()
                self.tools = [tool_info(t) for t in listed.tools if self.spec.allows(t.name)]
                if not self._ready.done():
                    self._ready.set_result(None)
                while True:
                    job = await self._jobs.get()
                    if job is None:
                        return
                    name, arguments, future = job
                    try:
                        result = await session.call_tool(name, arguments)
                        if not future.done():
                            future.set_result(result)
                    except Exception as exc:
                        if not future.done():
                            future.set_exception(McpError(f"{type(exc).__name__}: {exc}"))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # an ExceptionGroup from the transport's task group
            inner: BaseException = exc
            while getattr(inner, "exceptions", None):  # ExceptionGroup (Python 3.11+ or the exceptiongroup backport)
                inner = inner.exceptions[0]  # type: ignore[attr-defined]
            self._fail(f"could not use the MCP server '{self.spec.label}': {type(inner).__name__}: {inner}")
        finally:
            while not self._jobs.empty():
                job = self._jobs.get_nowait()
                if job is not None and not job[2].done():
                    job[2].set_exception(McpError(self.error or "the MCP connection closed"))

    def _fail(self, message: str) -> None:
        self.error = message
        log.warning("%s", message)
        if not self._ready.done():
            self._ready.set_exception(McpError(message))
            self._ready.exception()  # retrieved: no "never retrieved" warning when nobody waits

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[str, bool]:
        if self._task is None or self._task.done():
            raise McpError(self.error or f"the MCP server '{self.spec.label}' is not connected")
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._jobs.put((name, arguments, future))
        try:
            result = await asyncio.wait_for(future, self.timeout)
        except asyncio.TimeoutError:
            raise McpError(f"the MCP tool '{name}' did not answer within {self.timeout:g} s") from None
        return result_text(result)

    async def close(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        await self._jobs.put(None)
        try:
            await asyncio.wait_for(asyncio.shield(task), 5.0)
        except (asyncio.TimeoutError, Exception):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


Emit = Callable[..., Awaitable[Any]]


@dataclass
class McpTool:
    server: str
    name: str


class McpTools:
    """The MCP servers of one Realtime session: opened when ``session.tools`` names them, closed with the session."""

    def __init__(self, *, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.specs: dict[str, McpServerSpec] = {}
        self.connections: dict[str, McpConnection] = {}
        self._opening: dict[str, asyncio.Task[None]] = {}
        self._keys: dict[str, str] = {}
        #: approval requests waiting for the client: approval item id -> (server, tool, arguments)
        self.pending: dict[str, tuple[str, str, str]] = {}

    def configure(self, specs: list[McpServerSpec]) -> None:
        labels = [s.label for s in specs]
        if len(set(labels)) != len(labels):
            raise ValueError("each MCP tool needs a server_label of its own")
        self.specs = {s.label: s for s in specs}

    async def sync(self, emit: Emit) -> None:
        """Open the servers named now, close the others. ``emit(type, **fields)`` sends a Realtime event;
        ``emit("item", item=...)`` adds an item to the conversation."""
        for label in [lbl for lbl in self.connections if lbl not in self.specs]:
            await self._drop(label)
        for label, spec in self.specs.items():
            if label in self.connections and self._keys.get(label) == spec.key():
                conn = self.connections[label]
                conn.spec = spec  # allowed_tools / require_approval may have changed
                continue
            await self._drop(label)
            self._keys[label] = spec.key()
            self._opening[label] = asyncio.create_task(self._open(spec, emit), name=f"bolo-mcp-open-{label}")

    async def _open(self, spec: McpServerSpec, emit: Emit) -> None:
        conn = McpConnection(spec, timeout=self.timeout)
        self.connections[spec.label] = conn
        item_id = await emit("mcp_list_tools.open", server_label=spec.label)
        try:
            await conn.open()
        except McpError as exc:
            await emit("mcp_list_tools.failed", item_id=item_id, server_label=spec.label, message=str(exc))
            return
        await emit("mcp_list_tools.completed", item_id=item_id, server_label=spec.label, tools=conn.tools)

    async def _drop(self, label: str) -> None:
        task = self._opening.pop(label, None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        conn = self.connections.pop(label, None)
        self._keys.pop(label, None)
        if conn is not None:
            await conn.close()

    async def ready(self) -> None:
        """Wait for the servers still being opened (a response needs their tool lists)."""
        tasks = [t for t in self._opening.values() if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def functions(self, taken: set[str]) -> tuple[list[dict[str, Any]], dict[str, McpTool]]:
        """The MCP tools as function tools for the language model, and which server each name belongs to.
        A name that is already a function tool, or a tool of an earlier server, becomes ``<label>__<name>``."""
        functions: list[dict[str, Any]] = []
        mapping: dict[str, McpTool] = {}
        for label, conn in self.connections.items():
            spec = self.specs.get(label)
            if spec is None or conn.error:
                continue
            for tool in conn.tools:
                if not spec.allows(tool["name"]):
                    continue
                name = tool["name"]
                if name in taken or name in mapping:
                    name = f"{label}__{tool['name']}"
                mapping[name] = McpTool(label, tool["name"])
                functions.append(
                    {
                        "type": "function",
                        "name": name,
                        "description": tool["description"]
                        + (f" (from {spec.description})" if spec.description else ""),
                        "parameters": tool["input_schema"],
                    }
                )
        return functions, mapping

    def needs_approval(self, tool: McpTool) -> bool:
        spec = self.specs.get(tool.server)
        return spec is None or spec.needs_approval(tool.name)

    async def call(self, tool: McpTool, arguments: str) -> tuple[str, bool]:
        conn = self.connections.get(tool.server)
        if conn is None:
            raise McpError(f"the MCP server '{tool.server}' is not connected")
        try:
            args = json.loads(arguments or "{}")
        except ValueError:
            raise McpError(f"the arguments for '{tool.name}' are not valid JSON: {arguments[:200]}") from None
        if not isinstance(args, dict):
            raise McpError(f"the arguments for '{tool.name}' must be a JSON object")
        return await conn.call(tool.name, args)

    async def close(self) -> None:
        for label in list({*self.connections, *self._opening}):
            await self._drop(label)
