"""Tests for the google-genai backed gemini_chat backend (no live calls)."""
from __future__ import annotations

import base64
import copy
import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("google.genai")

import skillopt.model as model  # noqa: E402
from skillopt.model import backend_config  # noqa: E402
from skillopt.model import gemini_backend as gemini  # noqa: E402
from skillopt.model.common import (  # noqa: E402
    CompatAssistantMessage,
    default_model_for_backend,
    normalize_backend_name,
)


@pytest.fixture(autouse=True)
def restore_gemini_state(monkeypatch) -> Iterator[None]:
    optimizer_config = copy.deepcopy(gemini.OPTIMIZER_CONFIG)
    target_config = copy.deepcopy(gemini.TARGET_CONFIG)
    effort = gemini.REASONING_EFFORT
    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    monkeypatch.setattr(gemini.time, "sleep", lambda _seconds: None)
    gemini.tracker.reset()
    gemini._reset_clients()
    yield
    gemini.OPTIMIZER_CONFIG.__dict__.update(optimizer_config.__dict__)
    gemini.TARGET_CONFIG.__dict__.update(target_config.__dict__)
    gemini.REASONING_EFFORT = effort
    gemini.tracker.reset()
    gemini._reset_clients()
    with gemini._signature_lock:
        gemini._signatures.clear()
    backend_config.set_optimizer_backend(optimizer_backend)
    backend_config.set_target_backend(target_backend)


def _part(**kwargs: Any) -> SimpleNamespace:
    defaults = {"text": None, "thought": False, "function_call": None, "thought_signature": None}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _response(parts: list[SimpleNamespace], *, prompt: int = 11, output: int = 5, thoughts: int = 7, finish: str = "STOP"):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=parts),
                finish_reason=SimpleNamespace(name=finish),
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=output,
            thoughts_token_count=thoughts,
        ),
        prompt_feedback=None,
    )


class _FakeModels:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _install_client(monkeypatch, responses: list[Any]) -> _FakeModels:
    models = _FakeModels(responses)
    monkeypatch.setattr(gemini, "_get_client", lambda role: SimpleNamespace(models=models))
    return models


class _StatusError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"status {code}")
        self.code = code


# ── Registration ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("alias", ["gemini", "gemini_chat", "google", "vertex", "vertex_ai"])
def test_aliases_normalize_to_gemini_chat(alias: str) -> None:
    assert normalize_backend_name(alias) == "gemini_chat"


def test_gemini_chat_is_whitelisted_for_both_roles() -> None:
    backend_config.set_optimizer_backend("gemini_chat")
    backend_config.set_target_backend("gemini_chat")
    assert backend_config.is_optimizer_chat_backend()
    assert backend_config.is_target_chat_backend()
    assert not backend_config.is_target_exec_backend()
    assert default_model_for_backend("gemini_chat") == "gemini-3.5-flash"


def test_set_backend_maps_both_roles() -> None:
    assert model.set_backend("gemini") == "gemini_chat"
    assert model.get_optimizer_backend() == "gemini_chat"
    assert model.get_target_backend() == "gemini_chat"
    assert model.get_backend_name() == "gemini_chat"


# ── Message translation ──────────────────────────────────────────────────────


def test_messages_become_system_instruction_and_contents() -> None:
    png = base64.b64encode(b"\x89PNG fake").decode()
    system, contents = gemini._messages_to_contents([
        {"role": "system", "content": "Be terse."},
        {"role": "developer", "content": "Answer in English."},
        {"role": "user", "content": [
            {"type": "text", "text": "What is in the image?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
        ]},
        {"role": "user", "content": "Second user turn merges."},
    ])
    assert system == "Be terse.\n\nAnswer in English."
    assert len(contents) == 1
    assert contents[0].role == "user"
    parts = contents[0].parts
    assert parts[0].text == "What is in the image?"
    assert parts[1].inline_data.mime_type == "image/png"
    assert parts[1].inline_data.data == b"\x89PNG fake"
    assert parts[2].text == "Second user turn merges."


def test_local_image_path_is_inlined(tmp_path) -> None:
    image = tmp_path / "page.jpg"
    image.write_bytes(b"jpeg-bytes")
    _, contents = gemini._messages_to_contents([
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": str(image)}}]},
    ])
    part = contents[0].parts[0]
    assert part.inline_data.mime_type == "image/jpeg"
    assert part.inline_data.data == b"jpeg-bytes"


def test_tool_round_trip_uses_function_parts_and_signatures() -> None:
    gemini._remember_signature("call_known", b"real-signature")
    _, contents = gemini._messages_to_contents([
        {"role": "user", "content": "Look it up."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_known", "type": "function", "function": {"name": "search", "arguments": '{"q": "a"}'}},
            {"id": "call_foreign", "type": "function", "function": {"name": "search", "arguments": '{"q": "b"}'}},
        ]},
        {"role": "tool", "tool_call_id": "call_known", "content": '{"hits": 3}'},
        {"role": "tool", "tool_call_id": "call_foreign", "content": "plain text result"},
    ])
    assert [content.role for content in contents] == ["user", "model", "user"]
    first_call, second_call = contents[1].parts
    assert first_call.function_call.name == "search"
    assert first_call.function_call.args == {"q": "a"}
    assert first_call.thought_signature == b"real-signature"
    assert second_call.thought_signature == gemini._SKIP_SIGNATURE
    first_result, second_result = contents[2].parts
    assert first_result.function_response.name == "search"
    assert first_result.function_response.response == {"hits": 3}
    assert second_result.function_response.response == {"output": "plain text result"}


def test_tools_and_tool_choice_are_converted() -> None:
    tools = gemini._convert_tools([
        {"type": "function", "function": {
            "name": "search",
            "description": "Search the corpus",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        }},
        {"type": "web_search"},
    ])
    assert tools[0].function_declarations[0].name == "search"
    assert tools[0].function_declarations[0].parameters_json_schema["required"] == ["q"]
    assert tools[1].google_search is not None

    forced = gemini._convert_tool_choice({"type": "function", "function": {"name": "search"}})
    assert forced.function_calling_config.mode.name == "ANY"
    assert forced.function_calling_config.allowed_function_names == ["search"]
    assert gemini._convert_tool_choice("required").function_calling_config.mode.name == "ANY"
    assert gemini._convert_tool_choice("none").function_calling_config.mode.name == "NONE"

    with pytest.raises(ValueError, match="does not support tool type"):
        gemini._convert_tools([{"type": "code_interpreter"}])


# ── Thinking configuration ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model_name", "effort", "level", "budget"),
    [
        ("gemini-3.5-flash", "medium", "MEDIUM", None),
        ("gemini-3.8-flash", "xhigh", "HIGH", None),
        ("gemini-3.5-flash", "none", "MINIMAL", None),
        ("gemini-3.1-pro-preview", "medium", "HIGH", None),
        ("gemini-3.1-pro-preview", "minimal", "LOW", None),
        ("gemini-2.5-flash", "low", None, 1024),
        ("gemini-2.5-pro", "minimal", None, 128),
    ],
)
def test_reasoning_effort_maps_to_thinking_config(model_name, effort, level, budget) -> None:
    config = gemini._thinking_config(model_name, effort)
    assert config is not None
    if level:
        assert config.thinking_level.name == level
    else:
        assert config.thinking_budget == budget


def test_thinking_config_is_omitted_when_unset_or_unknown() -> None:
    assert gemini._thinking_config("gemini-3.5-flash", None) is None
    assert gemini._thinking_config("gemini-3.5-flash", "") is None
    assert gemini._thinking_config("gemma-3-27b-it", "high") is None


def test_explicit_thinking_level_overrides_effort() -> None:
    config = gemini._thinking_config("gemini-3.5-flash", "high", override="low")
    assert config.thinking_level.name == "LOW"


# ── Calls ────────────────────────────────────────────────────────────────────


def test_chat_optimizer_returns_text_and_tracks_usage(monkeypatch) -> None:
    models = _install_client(monkeypatch, [
        _response([_part(text="internal reasoning", thought=True), _part(text="final answer")]),
    ])
    gemini.set_optimizer_deployment("gemini-3.5-flash")
    gemini.set_reasoning_effort("high")

    text, usage = gemini.chat_optimizer("sys prompt", "user prompt", max_completion_tokens=2048, stage="reflect")

    assert text == "final answer"
    assert usage == {"prompt_tokens": 11, "completion_tokens": 12, "total_tokens": 23}
    call = models.calls[0]
    assert call["model"] == "gemini-3.5-flash"
    assert call["config"].system_instruction == "sys prompt"
    assert call["config"].max_output_tokens == 2048
    assert call["config"].thinking_config.thinking_level.name == "HIGH"
    assert call["config"].automatic_function_calling.disable is True
    summary = gemini.get_token_summary()
    assert summary["reflect"]["calls"] == 1
    assert summary["_total"]["total_tokens"] == 23


def test_per_call_reasoning_effort_and_timeout_win(monkeypatch) -> None:
    models = _install_client(monkeypatch, [_response([_part(text="ok")])])
    gemini.set_target_deployment("gemini-3.5-flash")
    gemini.set_reasoning_effort("high")

    gemini.chat_target("s", "u", reasoning_effort="low", timeout=42)

    config = models.calls[0]["config"]
    assert config.thinking_config.thinking_level.name == "LOW"
    assert config.http_options.timeout == 42000


def test_message_call_returns_tool_calls_and_remembers_signature(monkeypatch) -> None:
    function_call = SimpleNamespace(id="", name="search", args={"q": "gemini"})
    models = _install_client(monkeypatch, [
        _response([_part(function_call=function_call, thought_signature=b"sig-1")]),
        _response([_part(text="done")]),
    ])
    tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]

    message, _ = gemini.chat_target_messages(
        [{"role": "user", "content": "find it"}],
        tools=tools,
        tool_choice="required",
        return_message=True,
    )

    assert isinstance(message, CompatAssistantMessage)
    assert message.content == ""
    [tool_call] = message.tool_calls
    assert tool_call.function.name == "search"
    assert json.loads(tool_call.function.arguments) == {"q": "gemini"}
    assert models.calls[0]["config"].tool_config.function_calling_config.mode.name == "ANY"

    # Callers replay model_dump() dicts; the cached signature must go back out.
    history = [
        {"role": "user", "content": "find it"},
        message.model_dump(),
        {"role": "tool", "tool_call_id": tool_call.id, "content": "result"},
    ]
    text, _ = gemini.chat_target_messages(history, tools=tools)
    assert text == "done"
    replayed = models.calls[1]["contents"][1].parts[0]
    assert replayed.thought_signature == b"sig-1"


def test_server_errors_are_retried(monkeypatch) -> None:
    models = _install_client(monkeypatch, [_StatusError(503), _StatusError(429), _response([_part(text="ok")])])
    text, _ = gemini.chat_target("s", "u", retries=3)
    assert text == "ok"
    assert len(models.calls) == 3


def test_client_errors_fail_fast(monkeypatch) -> None:
    models = _install_client(monkeypatch, [_StatusError(400), _response([_part(text="unused")])])
    with pytest.raises(RuntimeError, match="failed after 5 attempt"):
        gemini.chat_target("s", "u", retries=5)
    assert len(models.calls) == 1


def test_blocked_prompt_raises(monkeypatch) -> None:
    blocked = SimpleNamespace(
        candidates=[],
        usage_metadata=None,
        prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
    )
    _install_client(monkeypatch, [blocked])
    with pytest.raises(RuntimeError, match="block_reason=SAFETY"):
        gemini.chat_target("s", "u", retries=1)


def test_routing_through_public_api(monkeypatch) -> None:
    _install_client(monkeypatch, [_response([_part(text="opt")]), _response([_part(text="tgt")])])
    model.set_optimizer_backend("gemini_chat")
    model.set_target_backend("gemini_chat")
    model.reset_token_tracker()

    assert model.chat_optimizer("s", "u")[0] == "opt"
    assert model.chat_target_messages([{"role": "user", "content": "u"}])[0] == "tgt"
    summary = model.get_token_summary()
    assert summary["optimizer"]["calls"] == 1
    assert summary["target"]["calls"] == 1
    assert summary["_total"]["calls"] == 2


# ── Client / configuration ───────────────────────────────────────────────────


def test_transport_selection() -> None:
    config = copy.deepcopy(gemini.TARGET_CONFIG)
    config.api_key, config.use_vertexai = "", None
    assert gemini.uses_vertexai(config)
    config.api_key = "key"
    assert not gemini.uses_vertexai(config)
    config.use_vertexai = True
    assert gemini.uses_vertexai(config)


def test_build_client_uses_vertex_or_api_key(monkeypatch) -> None:
    created: list[dict[str, Any]] = []

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

    monkeypatch.setattr("google.genai.Client", _Client)
    config = copy.deepcopy(gemini.TARGET_CONFIG)
    config.api_key, config.use_vertexai, config.project, config.location = "", None, "my-proj", "us-central1"
    gemini._build_client(config)
    assert created[-1]["vertexai"] is True
    assert created[-1]["project"] == "my-proj"
    assert created[-1]["location"] == "us-central1"

    config.api_key = "secret"
    gemini._build_client(config)
    assert created[-1]["api_key"] == "secret"
    assert "vertexai" not in created[-1]


def test_missing_api_key_error_is_actionable() -> None:
    config = copy.deepcopy(gemini.TARGET_CONFIG)
    config.api_key, config.use_vertexai = "", False
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        gemini._build_client(config)


def test_configure_from_config_keeps_env_for_blank_values() -> None:
    gemini.configure_gemini_chat(project="env-project", location="europe-west4")
    gemini.configure_gemini_chat_from_config({
        "gemini_project": "",
        "gemini_location": None,
        "gemini_use_vertexai": "true",
        "gemini_thinking_level": "low",
        "gemini_max_tokens": 4096,
    })
    assert gemini.TARGET_CONFIG.project == "env-project"
    assert gemini.OPTIMIZER_CONFIG.location == "europe-west4"
    assert gemini.TARGET_CONFIG.use_vertexai is True
    assert gemini.TARGET_CONFIG.thinking_level == "low"
    assert gemini.OPTIMIZER_CONFIG.max_tokens == 4096
    public = gemini.get_gemini_chat_config()
    assert public["transport"] == "vertexai"
    assert "api_key" not in public


def test_invalid_thinking_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="thinking level"):
        gemini.configure_gemini_chat(thinking_level="extreme")


def test_yaml_keys_flatten() -> None:
    from skillopt.config import flatten_config

    flat = flatten_config({"model": {"gemini_project": "p", "gemini_thinking_level": "high"}})
    assert flat["gemini_project"] == "p"
    assert flat["gemini_thinking_level"] == "high"
