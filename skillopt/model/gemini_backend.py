"""Google Gemini chat backend for optimizer and target paths.

Talks to Gemini through the official ``google-genai`` SDK, on either of its two
transports:

* **Vertex AI** (Gemini Enterprise Agent Platform) — authenticates with
  Application Default Credentials (``gcloud auth application-default login``,
  a service account, or workload identity). Selected when
  ``GOOGLE_GENAI_USE_VERTEXAI`` is true, or when no API key is configured.
  Needs ``GOOGLE_CLOUD_PROJECT``; ``GOOGLE_CLOUD_LOCATION`` defaults to
  ``global``.
* **Gemini Developer API** — authenticates with ``GEMINI_API_KEY`` (or the
  ``GOOGLE_API_KEY`` alias).

The module mirrors the callable surface of the other chat backends so it can be
selected as the optimizer and/or target backend and routed through
:mod:`skillopt.model`. OpenAI-style message lists (system / user / assistant /
tool roles, ``image_url`` parts, and function tools) are translated to Gemini
``Content`` objects, and responses are returned as the shared compatibility
message objects from :mod:`skillopt.model.common`.
"""
from __future__ import annotations

import base64
import collections
import json
import mimetypes
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlparse

from skillopt.model.common import (
    CompatAssistantMessage,
    CompatToolCall,
    CompatToolFunction,
    TokenTracker,
    default_model_for_backend,
)

BACKEND_NAME = "gemini_chat"

_DEFAULT_LOCATION = "global"
# Gemini 2.5 / 3.x Flash and Pro accept up to 65,536 output tokens; thinking
# tokens count against the same budget.
_DEFAULT_MAX_TOKENS = 65536
_DEFAULT_TIMEOUT_SECONDS = 600.0

# Gemini 3 rejects a replayed function call that lacks the thought signature it
# was returned with. Signatures are kept here, keyed by the compatibility tool
# call id, so a caller that round-trips plain OpenAI-style dicts still sends
# them back. The documented bypass value covers calls whose signature is gone
# (evicted, or a history built by another backend).
_SIGNATURE_CACHE_SIZE = 4096
_SKIP_SIGNATURE = b"skip_thought_signature_validator"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
# 4xx responses that retrying cannot fix. 429 (quota) and 5xx are retried.
_NON_RETRYABLE_STATUS = {400, 401, 403, 404}


@dataclass
class GeminiConfig:
    api_key: str
    use_vertexai: bool | None
    project: str
    location: str
    deployment: str
    timeout_seconds: float
    max_tokens: int
    temperature: float | None
    thinking_level: str


def _parse_optional_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value for Gemini configuration: {value!r}")


def _parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    return float(raw) if raw else None


def _env_api_key() -> str:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""


def _initial_config(role: str) -> GeminiConfig:
    role_upper = role.upper()
    deployment_env = "OPTIMIZER_DEPLOYMENT" if role == "optimizer" else "TARGET_DEPLOYMENT"
    return GeminiConfig(
        api_key=_env_api_key(),
        use_vertexai=_parse_optional_bool(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")),
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION") or _DEFAULT_LOCATION,
        deployment=(
            os.environ.get(f"{role_upper}_GEMINI_MODEL")
            or os.environ.get("GEMINI_MODEL")
            or os.environ.get(deployment_env)
            or default_model_for_backend(BACKEND_NAME)
        ),
        timeout_seconds=float(os.environ.get("GEMINI_TIMEOUT_SECONDS") or _DEFAULT_TIMEOUT_SECONDS),
        max_tokens=int(os.environ.get("GEMINI_MAX_TOKENS") or _DEFAULT_MAX_TOKENS),
        temperature=_parse_optional_float(os.environ.get("GEMINI_TEMPERATURE")),
        thinking_level=(os.environ.get("GEMINI_THINKING_LEVEL") or "").strip().lower(),
    )


OPTIMIZER_CONFIG = _initial_config("optimizer")
TARGET_CONFIG = _initial_config("target")
REASONING_EFFORT: str | None = None

_config_lock = threading.Lock()
_client_lock = threading.Lock()
_signature_lock = threading.Lock()
tracker = TokenTracker()

_clients: dict[str, Any] = {}
_signatures: collections.OrderedDict[str, bytes] = collections.OrderedDict()


def _require_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The gemini_chat backend requires the google-genai package. "
            "Install it with `pip install -e \".[gemini]\"` or `pip install google-genai`."
        ) from exc
    return genai, types


def _config_for(role: str) -> GeminiConfig:
    return OPTIMIZER_CONFIG if role == "optimizer" else TARGET_CONFIG


def uses_vertexai(config: GeminiConfig) -> bool:
    """Vertex AI unless explicitly disabled, or an API key is the only credential."""
    if config.use_vertexai is not None:
        return config.use_vertexai
    return not config.api_key


def _build_client(config: GeminiConfig):
    genai, types = _require_genai()
    http_options = types.HttpOptions(timeout=int(config.timeout_seconds * 1000))
    if uses_vertexai(config):
        kwargs: dict[str, Any] = {"vertexai": True, "location": config.location or _DEFAULT_LOCATION}
        if config.project:
            kwargs["project"] = config.project
        try:
            return genai.Client(http_options=http_options, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised with setup guidance
            raise RuntimeError(
                "Could not create a Vertex AI Gemini client. Set GOOGLE_CLOUD_PROJECT "
                "(and optionally GOOGLE_CLOUD_LOCATION, default 'global') and make "
                "Application Default Credentials available, e.g. with "
                "`gcloud auth application-default login`. To use the Gemini Developer "
                f"API instead, set GEMINI_API_KEY. Underlying error: {exc}"
            ) from exc
    if not config.api_key:
        raise RuntimeError(
            "gemini_chat is configured for the Gemini Developer API but no API key is set. "
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY), or set GOOGLE_GENAI_USE_VERTEXAI=true "
            "to authenticate with Application Default Credentials on Vertex AI."
        )
    return genai.Client(api_key=config.api_key, http_options=http_options)


def _get_client(role: str):
    with _client_lock:
        client = _clients.get(role)
        if client is None:
            client = _build_client(_config_for(role))
            _clients[role] = client
        return client


def _reset_clients() -> None:
    with _client_lock:
        _clients.clear()


# ── Thinking configuration ───────────────────────────────────────────────────

_GEMINI_VERSION_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?")
_THINKING_LEVELS = ("minimal", "low", "medium", "high")
# Gemini 2.5 exposes a token budget instead of a level. Values follow the
# documented ranges; Pro cannot disable thinking, so its floor is 128.
_THINKING_BUDGETS = {"minimal": 0, "low": 1024, "medium": 8192, "high": 24576}


def _gemini_major_version(model: str) -> int | None:
    match = _GEMINI_VERSION_RE.search(str(model or "").lower())
    return int(match.group(1)) if match else None


def _normalize_effort(effort: str | None) -> str:
    value = str(effort or "").strip().lower()
    if value in {"", "off"}:
        return ""
    if value == "none":
        return "minimal"
    if value in {"xhigh", "max"}:
        return "high"
    return value if value in _THINKING_LEVELS else ""


def _thinking_config(model: str, effort: str | None, override: str = ""):
    """Translate SkillOpt's shared reasoning effort to a Gemini ThinkingConfig.

    Returns ``None`` (use the model's default) when no effort is set or the
    model has no thinking controls.
    """
    _, types = _require_genai()
    level = _normalize_effort(override or effort)
    if not level:
        return None
    lowered = str(model or "").lower()
    major = _gemini_major_version(lowered)
    if major is None:
        return None
    if major >= 3:
        # Gemini 3 Pro supports only low/high; minimal and medium are Flash-only.
        if "flash" not in lowered:
            level = {"minimal": "low", "medium": "high"}.get(level, level)
        return types.ThinkingConfig(thinking_level=level.upper())
    if major == 2 and "2.5" in lowered:
        budget = _THINKING_BUDGETS[level]
        if "pro" in lowered:
            budget = max(budget, 128)
        return types.ThinkingConfig(thinking_budget=budget)
    return None


# ── Message translation ──────────────────────────────────────────────────────


def _remember_signature(tool_call_id: str, signature: bytes | None) -> None:
    if not tool_call_id or not signature:
        return
    with _signature_lock:
        _signatures[tool_call_id] = signature
        _signatures.move_to_end(tool_call_id)
        while len(_signatures) > _SIGNATURE_CACHE_SIZE:
            _signatures.popitem(last=False)


def _lookup_signature(tool_call_id: str) -> bytes | None:
    with _signature_lock:
        return _signatures.get(tool_call_id)


def _image_part(url: str):
    _, types = _require_genai()
    if url.startswith("data:") and ";base64," in url:
        header, encoded = url.split(",", 1)
        mime = header[5:].split(";", 1)[0] or "image/png"
        return types.Part.from_bytes(data=base64.b64decode(encoded), mime_type=mime)
    if url.startswith(("http://", "https://", "gs://")):
        mime = mimetypes.guess_type(urlparse(url).path)[0] or "image/png"
        return types.Part.from_uri(file_uri=url, mime_type=mime)
    path = unquote(urlparse(url).path) if url.startswith("file://") else url
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image attachment not found: {path}")
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return types.Part.from_bytes(data=f.read(), mime_type=mime)


def _content_parts(content: Any) -> list[Any]:
    _, types = _require_genai()
    if content is None:
        return []
    if isinstance(content, str):
        return [types.Part.from_text(text=content)] if content else []
    if not isinstance(content, list):
        return [types.Part.from_text(text=str(content))]
    parts: list[Any] = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append(types.Part.from_text(text=item))
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"text", "input_text", "output_text"}:
            text = str(item.get("text", "") or "")
            if text:
                parts.append(types.Part.from_text(text=text))
        elif item_type == "image_url":
            image_url = item.get("image_url") or {}
            url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
            if url:
                parts.append(_image_part(str(url)))
        elif item_type == "input_image":
            url = str(item.get("image_url", "") or "")
            if url:
                parts.append(_image_part(url))
    return parts


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return "" if content is None else str(content)


def _parse_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    raw = str(arguments or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"value": raw}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _function_response_payload(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"output": text}
    return parsed if isinstance(parsed, dict) else {"output": parsed}


def _messages_to_contents(messages: list[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Split OpenAI-style messages into a system instruction and Gemini contents."""
    _, types = _require_genai()
    system_parts: list[str] = []
    contents: list[Any] = []
    tool_names: dict[str, str] = {}

    def _append(role: str, parts: list[Any]) -> None:
        if not parts:
            return
        # Gemini expects parallel function responses in one turn; merging
        # adjacent same-role turns also keeps plain histories well-formed.
        if contents and contents[-1].role == role:
            contents[-1].parts.extend(parts)
        else:
            contents.append(types.Content(role=role, parts=list(parts)))

    for message in messages:
        role = str(message.get("role", "user") or "user")
        content = message.get("content", "")
        if role in {"system", "developer"}:
            text = _text_of(content).strip()
            if text:
                system_parts.append(text)
            continue
        if role == "assistant":
            parts = _content_parts(content)
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", {}) or {}
                name = str(function.get("name", "") or "")
                call_id = str(tool_call.get("id", "") or "")
                if call_id:
                    tool_names[call_id] = name
                part = types.Part(
                    function_call=types.FunctionCall(
                        name=name,
                        args=_parse_arguments(function.get("arguments")),
                    ),
                    thought_signature=_lookup_signature(call_id) or _SKIP_SIGNATURE,
                )
                parts.append(part)
            _append("model", parts)
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id", "") or "")
            name = str(message.get("name") or tool_names.get(call_id) or "tool")
            part = types.Part.from_function_response(
                name=name,
                response=_function_response_payload(_text_of(content)),
            )
            _append("user", [part])
            continue
        _append("user", _content_parts(content))

    return "\n\n".join(system_parts), contents


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[Any] | None:
    if not tools:
        return None
    _, types = _require_genai()
    declarations: list[Any] = []
    extra_tools: list[Any] = []
    for tool in tools:
        tool_type = str(tool.get("type", "function") or "function")
        if tool_type == "function":
            function = tool.get("function", tool)
            kwargs: dict[str, Any] = {
                "name": str(function.get("name", "") or ""),
                "description": str(function.get("description", "") or ""),
            }
            parameters = function.get("parameters")
            if parameters:
                kwargs["parameters_json_schema"] = parameters
            declarations.append(types.FunctionDeclaration(**kwargs))
        elif tool_type in {"web_search", "web_search_preview"}:
            extra_tools.append(types.Tool(google_search=types.GoogleSearch()))
        else:
            raise ValueError(f"gemini_chat does not support tool type {tool_type!r}")
    converted = list(extra_tools)
    if declarations:
        converted.insert(0, types.Tool(function_declarations=declarations))
    return converted


def _convert_tool_choice(tool_choice: str | dict[str, Any] | None):
    if tool_choice is None:
        return None
    _, types = _require_genai()
    mode: str
    allowed: list[str] | None = None
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function", {}) or {}
        name = str(function.get("name", "") or tool_choice.get("name", "") or "")
        mode = "ANY"
        allowed = [name] if name else None
    else:
        mode = {"auto": "AUTO", "none": "NONE", "required": "ANY"}.get(str(tool_choice).lower(), "AUTO")
    return types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=mode,
            allowed_function_names=allowed,
        )
    )


# ── Response parsing ─────────────────────────────────────────────────────────


def usage_from_gemini_usage(usage: Any) -> dict[str, int]:
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) + int(
        getattr(usage, "thoughts_token_count", 0) or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _message_from_response(resp: Any) -> CompatAssistantMessage:
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        feedback = getattr(resp, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None) if feedback else None
        raise RuntimeError(f"Gemini returned no candidates (block_reason={reason})")
    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    text_parts: list[str] = []
    tool_calls: list[CompatToolCall] = []
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            call_id = str(getattr(function_call, "id", "") or "") or f"gemini_{uuid.uuid4().hex[:16]}"
            _remember_signature(call_id, getattr(part, "thought_signature", None))
            tool_calls.append(
                CompatToolCall(
                    id=call_id,
                    function=CompatToolFunction(
                        name=str(getattr(function_call, "name", "") or ""),
                        arguments=json.dumps(dict(getattr(function_call, "args", None) or {}), ensure_ascii=False),
                    ),
                )
            )
            continue
        if getattr(part, "thought", False):
            continue
        text = getattr(part, "text", None)
        if text:
            text_parts.append(text)
    finish_reason = getattr(candidate, "finish_reason", None)
    finish_name = str(getattr(finish_reason, "name", finish_reason) or "")
    if not text_parts and not tool_calls and finish_name not in {"", "STOP", "MAX_TOKENS"}:
        raise RuntimeError(f"Gemini returned an empty response (finish_reason={finish_name})")
    metadata = {"finish_reason": finish_name} if finish_name else {}
    return CompatAssistantMessage(content="".join(text_parts), tool_calls=tool_calls, metadata=metadata)


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


# ── Core call ────────────────────────────────────────────────────────────────


def _chat_messages_impl(
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    retries: int,
    stage: str,
    *,
    role: str,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    deployment: str | None = None,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> tuple[Any, dict[str, int]]:
    _, types = _require_genai()
    config = _config_for(role)
    model = deployment or config.deployment
    system, contents = _messages_to_contents(messages)
    if not contents:
        raise ValueError("gemini_chat requires at least one non-system message")

    config_kwargs: dict[str, Any] = {
        "max_output_tokens": min(int(max_completion_tokens), config.max_tokens),
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if system:
        config_kwargs["system_instruction"] = system
    if config.temperature is not None:
        config_kwargs["temperature"] = config.temperature
    effort = reasoning_effort if reasoning_effort is not None else REASONING_EFFORT
    thinking = _thinking_config(model, effort, config.thinking_level)
    if thinking is not None:
        config_kwargs["thinking_config"] = thinking
    converted_tools = _convert_tools(tools)
    if converted_tools:
        config_kwargs["tools"] = converted_tools
        tool_config = _convert_tool_choice(tool_choice)
        if tool_config is not None:
            config_kwargs["tool_config"] = tool_config
    if timeout is not None:
        config_kwargs["http_options"] = types.HttpOptions(timeout=int(float(timeout) * 1000))
    generate_config = types.GenerateContentConfig(**config_kwargs)

    last_err: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            client = _get_client(role)
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_config,
            )
            message = _message_from_response(resp)
            usage_info = usage_from_gemini_usage(getattr(resp, "usage_metadata", None))
            tracker.record(stage, usage_info["prompt_tokens"], usage_info["completion_tokens"])
            if return_message:
                return message, usage_info
            return message.content, usage_info
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if _status_code(exc) in _NON_RETRYABLE_STATUS:
                break
            if isinstance(exc, (ImportError, ValueError)):
                break
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Gemini chat call to {model!r} failed after {retries} attempt(s): {last_err}")


# ── Public API (mirrors the other chat backends) ─────────────────────────────


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, int]]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        role="optimizer",
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_target(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> tuple[str, dict[str, int]]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        role="target",
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: float | None = None,
) -> tuple[Any, dict[str, int]]:
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        role="optimizer",
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


def chat_target_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "target",
    reasoning_effort: str | None = None,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: float | None = None,
) -> tuple[Any, dict[str, int]]:
    return _chat_messages_impl(
        messages,
        max_completion_tokens,
        retries,
        stage,
        role="target",
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
    )


# ── Configuration / lifecycle ────────────────────────────────────────────────


def configure_gemini_chat(
    *,
    api_key: str | None = None,
    use_vertexai: bool | str | None = None,
    project: str | None = None,
    location: str | None = None,
    temperature: float | str | None = None,
    timeout_seconds: float | str | None = None,
    max_tokens: int | str | None = None,
    thinking_level: str | None = None,
    optimizer_model: str | None = None,
    target_model: str | None = None,
) -> None:
    """Configure the Gemini backend at runtime.

    Connection settings are shared by both roles; ``optimizer_model`` and
    ``target_model`` pick each role's model. ``None`` leaves a value unchanged.
    """
    parsed_vertex = _parse_optional_bool(use_vertexai) if use_vertexai is not None else None
    parsed_temperature = _parse_optional_float(temperature) if temperature is not None else None
    level = None
    if thinking_level is not None:
        level = str(thinking_level).strip().lower()
        if level and level not in _THINKING_LEVELS:
            raise ValueError(
                f"Invalid gemini_chat thinking level {thinking_level!r}; "
                f"expected one of {', '.join(_THINKING_LEVELS)} or blank"
            )
    with _config_lock:
        for config in (OPTIMIZER_CONFIG, TARGET_CONFIG):
            if api_key is not None:
                config.api_key = str(api_key).strip()
            if use_vertexai is not None:
                config.use_vertexai = parsed_vertex
            if project is not None:
                config.project = str(project).strip()
            if location is not None:
                config.location = str(location).strip() or _DEFAULT_LOCATION
            if temperature is not None:
                config.temperature = parsed_temperature
            if timeout_seconds is not None:
                config.timeout_seconds = float(timeout_seconds)
            if max_tokens is not None:
                config.max_tokens = int(max_tokens)
            if level is not None:
                config.thinking_level = level
        if optimizer_model:
            OPTIMIZER_CONFIG.deployment = str(optimizer_model).strip()
        if target_model:
            TARGET_CONFIG.deployment = str(target_model).strip()
    _reset_clients()


def _blank_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def configure_gemini_chat_from_config(config: dict[str, Any]) -> None:
    """Apply flattened ``gemini_*`` train/eval keys; blank values keep env settings."""
    configure_gemini_chat(
        api_key=_blank_to_none(config.get("gemini_api_key")),
        use_vertexai=_blank_to_none(config.get("gemini_use_vertexai")),
        project=_blank_to_none(config.get("gemini_project")),
        location=_blank_to_none(config.get("gemini_location")),
        temperature=_blank_to_none(config.get("gemini_temperature")),
        timeout_seconds=_blank_to_none(config.get("gemini_timeout_seconds")),
        max_tokens=_blank_to_none(config.get("gemini_max_tokens")),
        thinking_level=_blank_to_none(config.get("gemini_thinking_level")),
    )


def get_gemini_chat_config() -> dict[str, Any]:
    """Resolved, credential-free view of the configuration (for run metadata)."""
    return {
        "transport": "vertexai" if uses_vertexai(TARGET_CONFIG) else "gemini_api",
        "project": TARGET_CONFIG.project,
        "location": TARGET_CONFIG.location,
        "optimizer_model": OPTIMIZER_CONFIG.deployment,
        "target_model": TARGET_CONFIG.deployment,
        "api_key_set": bool(TARGET_CONFIG.api_key),
        "thinking_level": TARGET_CONFIG.thinking_level,
    }


def get_max_tokens() -> int:
    return TARGET_CONFIG.max_tokens


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    global REASONING_EFFORT
    REASONING_EFFORT = effort or None


def set_target_deployment(deployment: str) -> None:
    TARGET_CONFIG.deployment = deployment or default_model_for_backend(BACKEND_NAME)


def set_optimizer_deployment(deployment: str) -> None:
    OPTIMIZER_CONFIG.deployment = deployment or default_model_for_backend(BACKEND_NAME)
