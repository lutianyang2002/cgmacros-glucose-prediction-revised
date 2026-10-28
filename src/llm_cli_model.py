"""Claude Code CLI adapter for subscription-authenticated expression proposals.

Checks login and model identity, removes API-billing variables from child processes,
and pauses on overage or rate-limit events. Post-invocation failures still count
against the search budget. Calls disable tools and run in an empty temporary folder.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from src.llm_factor_search import PauseSearch

__all__ = ["FORBIDDEN_ENVIRONMENT", "SESSION_ENVIRONMENT", "resolve_claude", "check_environment", "check_login", "scrubbed_environment", "ClaudeCliModel", "CallRecord"]

FORBIDDEN_ENVIRONMENT = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_RESOURCE",
                         "ANTHROPIC_VERTEX_PROJECT_ID", "ANTHROPIC_BEDROCK_BASE_URL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                         "CLAUDE_CODE_SKIP_BEDROCK_AUTH", "CLAUDE_CODE_SKIP_VERTEX_AUTH", "AWS_BEARER_TOKEN_BEDROCK", "CLAUDE_CODE_API_KEY_HELPER_TTL_MS")
SESSION_ENVIRONMENT = ("CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
                       "CLAUDE_AGENT_SDK_VERSION", "CLAUDE_CODE_SESSION_ATTENDED")
NESTED_MARKERS = ("CLAUDECODE", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ID")          # set inside a running Claude Code session; the other session variables are only removed
LIMIT_PATTERN = re.compile(r"(usage limit|limit reached|rate limit|rate_limit|quota|resets? (at|in)|too many requests|overloaded_error)", re.I)
SYSTEM_PROMPT = "You propose feature expressions for a glucose-forecast research task. Answer only in the JSON format that the user message requests. Do not use tools."


class CallRecord(dict):
    """Metadata of one call: model, login source, tokens, estimated cost, rate-limit events, duration (never the prompt or credentials)."""


def resolve_claude(explicit: str | None = None) -> list:
    """The command prefix of the Claude Code executable. On Windows the native `claude.exe` next to the npm shim is preferred to the `.cmd` shim."""
    if explicit:
        return [explicit]
    found = shutil.which("claude.exe") or shutil.which("claude")
    if not found:
        raise FileNotFoundError("the `claude` command was not found on PATH")
    path = Path(found)
    if path.suffix.lower() in (".cmd", ".bat", ""):
        native = path.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
        if native.exists():
            return [str(native)]
    return [str(path)]


def check_environment(environ: dict | None = None) -> dict:
    """Names (never values) of the variables that could route a call to separate billing or to another provider, and of those that mark a nested Claude Code session."""
    environ = os.environ if environ is None else environ
    return {"billing_variables_set": sorted(n for n in FORBIDDEN_ENVIRONMENT if environ.get(n)), "nested_session_variables_set": sorted(n for n in NESTED_MARKERS if environ.get(n))}


def scrubbed_environment(environ: dict | None = None) -> dict:
    environ = dict(os.environ if environ is None else environ)
    for name in (*FORBIDDEN_ENVIRONMENT, *SESSION_ENVIRONMENT):
        environ.pop(name, None)
    return environ


def check_login(command: list, expected_subscription: str = "pro") -> dict:
    """`claude auth status` (no model call). Raises unless the login is a claude.ai subscription of the expected type with the first-party provider."""
    done = subprocess.run([*command, "auth", "status", "--json"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=scrubbed_environment(), timeout=60)
    try:
        status = json.loads(done.stdout)
    except json.JSONDecodeError:
        raise PauseSearch("login_not_verified", "`claude auth status` did not return JSON") from None
    safe = {k: status.get(k) for k in ("loggedIn", "authMethod", "apiProvider", "subscriptionType")}
    if not (safe["loggedIn"] and safe["authMethod"] == "claude.ai" and safe["apiProvider"] == "firstParty" and safe["subscriptionType"] == expected_subscription):
        raise PauseSearch("login_not_a_subscription", json.dumps(safe))
    return safe


def _walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _truthy_overage(events: list) -> bool:
    for event in events:
        for key, value in _walk(event):
            if re.fullmatch(r"is_?using_?overage", key, re.I) and value in (True, "true", 1):
                return True
    return False


def _reset_time(record) -> float | None:
    """End of the usage window (epoch seconds) from a rejected rate-limit event, if the stream gave one."""
    times = [e.get("resetsAt") for e in record.get("rate_limit_events", []) if str(e.get("status", "")).lower() in ("rejected", "blocked", "limit_reached") and isinstance(e.get("resetsAt"), (int, float))]
    return float(max(times)) if times else None


class ClaudeCliModel:
    """`model(request) -> text` through `claude -p`. One call per request; failures other than the pause conditions are ordinary exceptions (counted by the budget policy)."""

    def __init__(self, model_id: str = "claude-sonnet-5", command: list | None = None, timeout_seconds: int = 300, expected_subscription: str = "pro",
                 allowed_key_sources: tuple | None = ("none",), pin_model: bool = True):
        """`allowed_key_sources`: values of `apiKeySource` in the init event that count as the subscription login (None: accept anything and only report it, for the first ping).
        A missing field is accepted: login checks and the scrubbed environment supply the remaining safeguards."""
        self.command = command if command is not None else resolve_claude()
        self.model_id, self.timeout, self.expected_subscription, self.pin_model = model_id, timeout_seconds, expected_subscription, pin_model
        self.allowed_key_sources = allowed_key_sources
        self.last = None

    def preflight(self) -> dict:
        env = check_environment()
        if env["nested_session_variables_set"]:
            raise PauseSearch("nested_session", "run this from an independent terminal, not from inside a Claude Code session: " + ", ".join(env["nested_session_variables_set"]))
        if env["billing_variables_set"]:
            raise PauseSearch("billing_variable_present", "unset these variables first: " + ", ".join(env["billing_variables_set"]))
        return {"environment": env, "login": check_login(self.command, self.expected_subscription)}

    def build_command(self) -> list:
        return [*self.command, "-p", "--model", self.model_id, "--output-format", "stream-json", "--verbose", "--tools", "", "--no-session-persistence", "--safe-mode",
                "--system-prompt", SYSTEM_PROMPT]

    def check(self) -> None:
        """Refusals before an invocation (not calls): nested session, billing variable, login type. The login is checked once per object, the environment before every call."""
        env = check_environment()
        if env["nested_session_variables_set"] or env["billing_variables_set"] or not getattr(self, "_login_ok", False):
            self.preflight()
            self._login_ok = True

    def __call__(self, request: dict) -> str:
        self.last = None
        self.check()
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="llm_search_") as folder:
            try:
                done = subprocess.run(self.build_command(), input=request["prompt"], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=folder,
                                      env=scrubbed_environment(), timeout=self.timeout)
            except subprocess.TimeoutExpired:
                raise TimeoutError(f"no reply within {self.timeout} s") from None
        events = []
        for line in done.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        record = self._record(events, done, time.time() - started)
        self.last = record
        result = next((e for e in reversed(events) if e.get("type") == "result"), None)
        text = (result or {}).get("result", "") or ""
        if _truthy_overage(events):
            raise PauseSearch("extra_usage_detected", "the stream reports usage beyond the plan; no further call is made", invoked=True, text=text)
        source = record.get("api_key_source")
        if source is not None and self.allowed_key_sources is not None and source not in self.allowed_key_sources:
            raise PauseSearch("api_key_source", f"login source {source!r} is not the subscription login", invoked=True, text=text)
        blocked = any(isinstance(e, dict) and e.get("type") == "rate_limit_event" and str((e.get("rate_limit_info") or e).get("status", "")).lower() in ("rejected", "blocked", "limit_reached")
                      for e in events)
        if blocked or (result is not None and result.get("is_error") and LIMIT_PATTERN.search(text + " " + done.stderr)) or (result is None and LIMIT_PATTERN.search(done.stdout + done.stderr)):
            raise PauseSearch("usage_limit_reached", re.sub(r"\s+", " ", (text or done.stderr))[:200], invoked=True, resets_at=_reset_time(record), text=text)
        if result is None or result.get("is_error") or done.returncode != 0:
            raise RuntimeError(f"claude -p failed (exit {done.returncode}): {re.sub(chr(10), ' ', (text or done.stderr))[:200]}")
        if self.pin_model:                                       # a reply is only accepted when every part of the call reports the pinned model
            seen = {record.get("model"), *record.get("assistant_models", [])} - {None}
            if seen != {self.model_id}:
                raise PauseSearch("model_changed", f"expected {self.model_id}, the call reported {sorted(map(str, seen))}", invoked=True, text=text)
        return text

    @staticmethod
    def _record(events: list, done, seconds: float) -> CallRecord:
        init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
        result = next((e for e in reversed(events) if e.get("type") == "result"), {})
        limits = [{k: v for k, v in (e.get("rate_limit_info") or e).items() if k in ("status", "resetsAt", "rateLimitType", "isUsingOverage", "overageStatus", "utilization")}
                  for e in events if e.get("type") == "rate_limit_event"]
        usage = result.get("usage") or {}
        assistant_models = sorted({(e.get("message") or {}).get("model") for e in events if e.get("type") == "assistant" and (e.get("message") or {}).get("model")})
        return CallRecord(model=init.get("model"), assistant_models=assistant_models, models_used=sorted((result.get("modelUsage") or {}).keys()), api_key_source=init.get("apiKeySource"), claude_code_version=init.get("claude_code_version"), tools=init.get("tools"),
                          input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"), cache_read_tokens=usage.get("cache_read_input_tokens"),
                          cache_creation_tokens=usage.get("cache_creation_input_tokens"), estimated_cost_usd=result.get("total_cost_usd"), is_error=result.get("is_error"),
                          result_subtype=result.get("subtype"), rate_limit_events=limits, event_types=sorted({str(e.get("type")) + "/" + str(e.get("subtype")) for e in events}),
                          exit_code=done.returncode, seconds=round(seconds, 2))
