"""Tests for the Antigravity CLI (agy) exec target and optimizer backends."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

import skillopt.model as model
from skillopt.engine.trainer import _configure_trace_to_optimizer_gates, _resolve_role_backends
from skillopt.model import agy_backend, backend_config
from skillopt.model import codex_harness as harness
from skillopt.model.common import CompatAssistantMessage, default_model_for_backend, normalize_backend_name

_AGY_KEYS = (
    "AGY_EXEC_PATH",
    "AGY_EXEC_SKIP_PERMISSIONS",
    "AGY_EXEC_SANDBOX",
    "AGY_EXEC_MODE",
    "AGY_EXEC_EFFORT",
    "AGY_CHAT_TIMEOUT",
    "OPTIMIZER_BACKEND",
    "TARGET_BACKEND",
    "REFLACT_AGY_TRACE_TO_OPTIMIZER",
)


@pytest.fixture(autouse=True)
def restore_agy_state(monkeypatch) -> Iterator[None]:
    saved = {
        name: getattr(backend_config, name)
        for name in (
            "OPTIMIZER_BACKEND",
            "TARGET_BACKEND",
            "AGY_EXEC_PATH",
            "AGY_EXEC_SKIP_PERMISSIONS",
            "AGY_EXEC_SANDBOX",
            "AGY_EXEC_MODE",
            "AGY_EXEC_EFFORT",
            "AGY_CHAT_TIMEOUT",
            "EXEC_EMPTY_RESPONSE_RETRIES",
        )
    }
    env = {key: os.environ.get(key) for key in _AGY_KEYS}
    agy_state = (agy_backend.OPTIMIZER_DEPLOYMENT, agy_backend.TARGET_DEPLOYMENT, agy_backend.REASONING_EFFORT)
    monkeypatch.setattr(agy_backend.time, "sleep", lambda _seconds: None)
    agy_backend.tracker.reset()
    yield
    for name, value in saved.items():
        setattr(backend_config, name, value)
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    agy_backend.OPTIMIZER_DEPLOYMENT, agy_backend.TARGET_DEPLOYMENT, agy_backend.REASONING_EFFORT = agy_state
    agy_backend.tracker.reset()


def _stream(response: str = "<answer>42</answer>", *, status: str = "SUCCESS", extra: dict | None = None) -> str:
    events = [
        {"event": "init", "conversation_id": "conv-1", "init": {"model": "gemini-3.5-flash", "cwd": "/w"}},
        {"event": "step_update", "step_update": {
            "conversation_id": "conv-1", "step_index": 1, "state": "ACTIVE", "step_type": "tool",
            "tool_name": "view_file", "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/w/task.md"}},
        }},
        {"event": "step_update", "step_update": {
            "conversation_id": "conv-1", "step_index": 1, "state": "DONE", "step_type": "tool",
            "tool_name": "view_file", "tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/w/task.md"}},
        }},
        {"event": "step_update", "step_update": {
            "conversation_id": "conv-1", "step_index": 2, "state": "ERROR", "step_type": "tool",
            "tool_name": "run_command",
            "tool_info": {"name": "run_command", "parameters": {"CommandLine": "ls"},
                          "error": {"type": "TOOL_ERROR", "message": "permission check failed"}},
        }},
    ]
    result = {
        "conversation_id": "conv-1",
        "status": status,
        "response": response,
        "num_turns": 1,
        "duration_seconds": 1.5,
        "usage": {"input_tokens": 100, "output_tokens": 20, "thinking_tokens": 5, "cache_read_tokens": 3},
    }
    result.update(extra or {})
    events.append({"event": "result", "result": result})
    return "\n".join(json.dumps(event) for event in events) + "\n"


class _Runner:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": list(cmd), **kwargs})
        item = self.outputs.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _proc(stdout: str, returncode: int = 0, stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def _workspace(tmp_path) -> str:
    work_dir = tmp_path / "pred" / "agy_exec"
    harness.prepare_workspace(work_dir=str(work_dir), skill_md="# skill", task_text="What is 6*7?")
    return str(work_dir)


def _stdin_prompt(call: dict[str, Any]) -> str:
    payload = json.loads(call["input"])
    assert payload["event"] == "user"
    return payload["message"]["content"]


# ── Registration and configuration ───────────────────────────────────────────


@pytest.mark.parametrize("alias", ["agy", "agy_exec", "antigravity", "antigravity_cli", "antigravity-cli"])
def test_aliases_normalize_to_agy_exec(alias: str) -> None:
    assert normalize_backend_name(alias) == "agy_exec"


def test_agy_exec_roles() -> None:
    backend_config.set_target_backend("agy_exec")
    backend_config.set_optimizer_backend("agy_exec")
    assert backend_config.is_target_exec_backend()
    assert not backend_config.is_target_chat_backend()
    assert backend_config.is_optimizer_chat_backend()
    assert default_model_for_backend("agy_exec") == "gemini-3.5-flash"


def test_set_backend_pairs_agy_target_with_gemini_optimizer() -> None:
    assert model.set_backend("antigravity") == "agy_exec"
    assert model.get_optimizer_backend() == "gemini_chat"
    assert model.get_target_backend() == "agy_exec"
    assert model.get_backend_name() == "agy_exec"


@pytest.mark.parametrize(
    ("backend", "optimizer", "target", "expected"),
    [
        ("agy", "openai_chat", "openai_chat", ("gemini_chat", "agy_exec")),
        ("agy_exec", "agy_exec", "openai_chat", ("agy_exec", "agy_exec")),
        ("agy_exec", "claude_chat", None, ("claude_chat", "agy_exec")),
        ("gemini", "openai_chat", "openai_chat", ("gemini_chat", "gemini_chat")),
        ("vertex", None, "agy_exec", ("gemini_chat", "agy_exec")),
    ],
)
def test_trainer_role_resolution(backend, optimizer, target, expected) -> None:
    assert _resolve_role_backends(backend, optimizer, target) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", ""), ("none", ""), ("off", ""), ("minimal", "low"), ("medium", "medium"), ("xhigh", "max"), ("MAX", "max")],
)
def test_normalize_agy_effort(value, expected) -> None:
    assert backend_config.normalize_agy_effort(value) == expected


def test_invalid_agy_settings_are_rejected() -> None:
    with pytest.raises(ValueError, match="agy_exec effort"):
        backend_config.normalize_agy_effort("ultra")
    with pytest.raises(ValueError, match="mode"):
        backend_config.configure_agy_exec(mode="yolo")
    with pytest.raises(ValueError, match="agy_exec_sandbox"):
        backend_config.configure_agy_exec(sandbox="maybe")
    with pytest.raises(ValueError, match="positive integer"):
        backend_config.configure_agy_exec(chat_timeout="0")


def test_configure_from_config_falls_back_to_reasoning_effort() -> None:
    backend_config.configure_agy_exec(mode="plan")
    backend_config.configure_agy_exec_from_config({
        "reasoning_effort": "high",
        "agy_exec_effort": "",
        "agy_exec_mode": "",
        "agy_exec_skip_permissions": "false",
        "agy_exec_sandbox": True,
        "agy_chat_timeout": 120,
    })
    config = backend_config.get_agy_exec_config()
    assert config["effort"] == "high"
    assert config["mode"] == "plan"  # blank keeps the previous value
    assert config["skip_permissions"] is False
    assert config["sandbox"] is True
    assert config["chat_timeout"] == 120

    backend_config.configure_agy_exec_from_config({"reasoning_effort": "high", "agy_exec_effort": "none"})
    assert backend_config.get_agy_exec_config()["effort"] == ""


def test_default_path_falls_back_to_installer_location(tmp_path, monkeypatch) -> None:
    binary = tmp_path / ".local" / "bin" / "agy"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert backend_config._resolve_agy_path("agy") == str(binary)
    assert backend_config._resolve_agy_path("/opt/custom/agy") == "/opt/custom/agy"


@pytest.mark.parametrize(
    ("model_name", "effort", "expected"),
    [
        ("gemini-3.8-flash", "low", ["--model", "gemini-3.8-flash", "--effort", "low"]),
        ("gemini-3.8-flash-low", "high", ["--model", "gemini-3.8-flash-low"]),
        ("gemini-3-flash-preview", "", ["--model", "gemini-3-flash-preview"]),
        ("", "medium", ["--effort", "medium"]),
    ],
)
def test_agy_model_args(model_name, effort, expected) -> None:
    assert harness.agy_model_args(model_name, effort) == expected


def test_yaml_keys_flatten() -> None:
    from skillopt.config import flatten_config

    flat = flatten_config({"model": {"agy_exec_effort": "low", "agy_trace_to_optimizer": False}})
    assert flat["agy_exec_effort"] == "low"
    assert flat["agy_trace_to_optimizer"] is False


# ── Stream parsing ───────────────────────────────────────────────────────────


def test_parse_stream_keeps_latest_step_snapshot() -> None:
    parsed = harness.parse_agy_stream(_stream())
    assert parsed["init"]["model"] == "gemini-3.5-flash"
    assert [step["state"] for step in parsed["steps"]] == ["DONE", "ERROR"]
    assert parsed["result"]["response"] == "<answer>42</answer>"


def test_trace_steps_and_summary() -> None:
    steps = harness.format_agy_trace_steps(_stream())
    assert '[1] view_file({"AbsolutePath": "/w/task.md"}) -> DONE' in steps
    assert "[2] run_command" in steps and "permission check failed" in steps

    summary = harness._build_agy_trace_summary(
        _stream(extra={"denied_actions": [{"action": "command", "display_name": "RunCommand"}]}),
        "<answer>42</answer>",
    )
    assert "- model: gemini-3.5-flash" in summary
    assert "- tool calls: 2" in summary
    assert "- denied actions: RunCommand" in summary
    assert "input=100 output=20 thinking=5" in summary
    assert "- final answer format: tagged" in summary


# ── Target exec ──────────────────────────────────────────────────────────────


def test_run_agy_exec_builds_command_and_persists_artifacts(tmp_path, monkeypatch) -> None:
    work_dir = _workspace(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    backend_config.configure_agy_exec(path="/bin/agy-test", effort="medium", skip_permissions=True, sandbox=True, mode="")
    runner = _Runner([_proc(_stream())])
    monkeypatch.setattr(harness.subprocess, "run", runner)
    monkeypatch.setenv("GOOGLE_API_KEY", "alias-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    backend_config.set_target_backend("agy_exec")
    response, raw = harness.run_target_exec(
        work_dir=work_dir,
        prompt="Read task.md and answer.",
        model="gemini-3.8-flash",
        timeout=90,
        data_dirs=[str(data_dir)],
    )

    assert response == "<answer>42</answer>"
    call = runner.calls[0]
    cmd = call["cmd"]
    assert cmd[:8] == [
        "/bin/agy-test", "--print", "", "--input-format", "stream-json",
        "--output-format", "stream-json", "--disable-slash-commands",
    ]
    assert cmd[cmd.index("--print-timeout") + 1] == "90s"
    assert cmd[cmd.index("--model") + 1] == "gemini-3.8-flash"
    assert cmd[cmd.index("--effort") + 1] == "medium"
    assert "--dangerously-skip-permissions" in cmd
    assert "--sandbox" in cmd
    assert "--mode" not in cmd
    add_dirs = [cmd[i + 1] for i, token in enumerate(cmd) if token == "--add-dir"]
    assert add_dirs == [os.path.realpath(work_dir), os.path.realpath(str(data_dir))]
    assert call["cwd"] == work_dir
    assert call["timeout"] == 120
    assert call["env"]["GEMINI_API_KEY"] == "alias-key"
    prompt = _stdin_prompt(call)
    assert ".agents/skills/skillopt-target/SKILL.md" in prompt
    assert "Read task.md and answer." in prompt
    assert "AGY CLI ATTEMPT 1" in raw

    pred_dir = os.path.dirname(work_dir)
    with open(os.path.join(pred_dir, "agy_trace_summary.txt"), encoding="utf-8") as f:
        assert "Antigravity CLI Trace Summary" in f.read()
    with open(os.path.join(pred_dir, "agy_trace_steps.txt"), encoding="utf-8") as f:
        assert "view_file" in f.read()
    assert os.path.exists(os.path.join(pred_dir, "agy_raw.txt"))


def test_run_agy_exec_omits_permission_bypass_when_disabled(tmp_path, monkeypatch) -> None:
    work_dir = _workspace(tmp_path)
    backend_config.configure_agy_exec(skip_permissions=False, sandbox=False, mode="plan", effort="none")
    runner = _Runner([_proc(_stream())])
    monkeypatch.setattr(harness.subprocess, "run", runner)

    harness.run_agy_exec(work_dir=work_dir, prompt="p", model="gemini-3-flash-preview", timeout=30)

    cmd = runner.calls[0]["cmd"]
    assert "--dangerously-skip-permissions" not in cmd
    assert "--sandbox" not in cmd
    assert "--effort" not in cmd
    assert cmd[cmd.index("--mode") + 1] == "plan"


def test_empty_response_is_retried_with_hint(tmp_path, monkeypatch) -> None:
    work_dir = _workspace(tmp_path)
    backend_config.EXEC_EMPTY_RESPONSE_RETRIES = 1
    runner = _Runner([_proc(_stream("")), _proc(_stream("<answer>7</answer>"))])
    monkeypatch.setattr(harness.subprocess, "run", runner)

    response, raw = harness.run_agy_exec(work_dir=work_dir, prompt="p", model="m", timeout=30)

    assert response == "<answer>7</answer>"
    assert len(runner.calls) == 2
    assert "Previous execution returned an empty final response" in _stdin_prompt(runner.calls[1])
    assert "AGY CLI ATTEMPT 2" in raw


def test_all_empty_attempts_return_empty(tmp_path, monkeypatch) -> None:
    work_dir = _workspace(tmp_path)
    backend_config.EXEC_EMPTY_RESPONSE_RETRIES = 1
    runner = _Runner([_proc(_stream("")), _proc(_stream(""))])
    monkeypatch.setattr(harness.subprocess, "run", runner)

    response, _ = harness.run_agy_exec(work_dir=work_dir, prompt="p", model="m", timeout=30)
    assert response == ""
    assert os.path.exists(os.path.join(os.path.dirname(work_dir), "agy_trace_summary.txt"))


def test_setup_errors_fail_fast(tmp_path, monkeypatch) -> None:
    work_dir = _workspace(tmp_path)
    backend_config.EXEC_EMPTY_RESPONSE_RETRIES = 3
    error = json.dumps({"event": "result", "result": {
        "conversation_id": "", "status": "ERROR", "response": "", "num_turns": 0,
        "error": "invalid model selection: model nope is not recognized",
    }})
    runner = _Runner([_proc(error + "\n", returncode=1)])
    monkeypatch.setattr(harness.subprocess, "run", runner)

    with pytest.raises(RuntimeError, match="not recognized"):
        harness.run_agy_exec(work_dir=work_dir, prompt="p", model="nope", timeout=30)
    assert len(runner.calls) == 1


def test_timeout_returns_empty_and_keeps_partial_trace(tmp_path, monkeypatch) -> None:
    work_dir = _workspace(tmp_path)
    partial = _stream().splitlines()[0] + "\n"
    runner = _Runner([subprocess.TimeoutExpired(cmd="agy", timeout=60, output=partial.encode(), stderr=b"slow")])
    monkeypatch.setattr(harness.subprocess, "run", runner)

    response, raw = harness.run_agy_exec(work_dir=work_dir, prompt="p", model="m", timeout=30)

    assert response == ""
    assert "(timeout)" in raw and "slow" in raw


def test_missing_binary_error_is_actionable(tmp_path, monkeypatch) -> None:
    work_dir = _workspace(tmp_path)
    runner = _Runner([FileNotFoundError("agy")])
    monkeypatch.setattr(harness.subprocess, "run", runner)
    with pytest.raises(RuntimeError, match="install.sh"):
        harness.run_agy_exec(work_dir=work_dir, prompt="p", model="m", timeout=30)


def test_trace_gate_and_reflector_injection(tmp_path) -> None:
    from skillopt.gradient.reflect import fmt_minibatch_trajectories

    _configure_trace_to_optimizer_gates("agy_exec", {"agy_trace_to_optimizer": True})
    assert os.environ["REFLACT_AGY_TRACE_TO_OPTIMIZER"] == "1"
    _configure_trace_to_optimizer_gates("codex_exec", {"agy_trace_to_optimizer": True})
    assert os.environ["REFLACT_AGY_TRACE_TO_OPTIMIZER"] == "0"

    pred = tmp_path / "t1"
    pred.mkdir()
    (pred / "conversation.json").write_text(
        json.dumps([{"type": "message", "turn": 1, "content": "<answer>41</answer>"}]),
        encoding="utf-8",
    )
    (pred / "agy_trace_summary.txt").write_text("Antigravity CLI Trace Summary\n- tool calls: 2", encoding="utf-8")
    (pred / "agy_trace_steps.txt").write_text("[1] view_file -> DONE", encoding="utf-8")
    items = [{"id": "t1", "hard": 0, "fail_reason": "wrong"}]

    os.environ["REFLACT_AGY_TRACE_TO_OPTIMIZER"] = "1"
    text = fmt_minibatch_trajectories(items, str(tmp_path))
    assert "#### Antigravity Trace Summary" in text
    assert "[1] view_file -> DONE" in text

    os.environ["REFLACT_AGY_TRACE_TO_OPTIMIZER"] = "0"
    assert "Antigravity" not in fmt_minibatch_trajectories(items, str(tmp_path))


# ── Optimizer chat ───────────────────────────────────────────────────────────


def test_run_agy_chat_plain_text(monkeypatch) -> None:
    backend_config.configure_agy_exec(path="agy-bin", effort="")
    runner = _Runner([_proc(_stream("Reflection text"))])
    monkeypatch.setattr(harness.subprocess, "run", runner)

    text, usage = harness.run_agy_chat(system="You are an analyst.", prompt="Analyze.", model="gemini-3.8-flash-high")

    assert text == "Reflection text"
    assert usage == {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}
    call = runner.calls[0]
    cmd = call["cmd"]
    assert "--dangerously-skip-permissions" not in cmd
    assert "--add-dir" not in cmd
    assert "--json-schema" not in cmd
    assert cmd[cmd.index("--print-timeout") + 1] == "900s"
    assert not os.path.exists(call["cwd"])  # scratch dir is cleaned up
    prompt = _stdin_prompt(call)
    assert "<system_instructions>\nYou are an analyst.\n</system_instructions>" in prompt
    assert prompt.endswith("Analyze.")


def test_run_agy_chat_structured_output_and_errors(monkeypatch) -> None:
    schema = {"type": "object", "properties": {"content": {"type": "string"}}}
    structured = _stream('{"content":"x","toolAction":"extra"}', extra={"structured_output": {"content": "x"}})
    runner = _Runner([
        _proc(structured),
        _proc(_stream("", status="ERROR", extra={"error": "quota exhausted"}), returncode=1),
        _proc("not json at all", returncode=2, stderr="boom"),
    ])
    monkeypatch.setattr(harness.subprocess, "run", runner)

    text, _ = harness.run_agy_chat(system="", prompt="p", model="m", schema=schema, effort="low", timeout=60)
    assert json.loads(text) == {"content": "x"}
    cmd = runner.calls[0]["cmd"]
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == schema
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert cmd[cmd.index("--print-timeout") + 1] == "60s"

    with pytest.raises(RuntimeError, match="quota exhausted"):
        harness.run_agy_chat(system="", prompt="p", model="m")
    with pytest.raises(RuntimeError, match="boom"):
        harness.run_agy_chat(system="", prompt="p", model="m")


def test_optimizer_routing_with_tools(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        payload = {"content": "", "tool_calls": [{"name": "apply_edit", "arguments": '{"op": "add"}'}]}
        return json.dumps(payload), {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12}

    monkeypatch.setattr(agy_backend, "run_agy_chat", fake_chat)
    model.set_optimizer_backend("agy_exec")
    model.set_optimizer_deployment("gemini-3.8-flash-high")
    model.reset_token_tracker()

    message, usage = model.chat_optimizer_messages(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "edit the skill"}],
        tools=[{"type": "function", "function": {"name": "apply_edit", "parameters": {"type": "object"}}}],
        tool_choice="required",
        return_message=True,
        stage="reflect",
    )

    assert isinstance(message, CompatAssistantMessage)
    assert message.tool_calls[0].function.name == "apply_edit"
    assert usage["total_tokens"] == 12
    assert calls[0]["model"] == "gemini-3.8-flash-high"
    assert calls[0]["system"] == "sys"
    assert calls[0]["schema"]["required"] == ["content", "tool_calls"]
    assert model.get_token_summary()["reflect"]["calls"] == 1


def test_optimizer_plain_call_retries_then_fails(monkeypatch) -> None:
    attempts: list[int] = []

    def flaky(**kwargs):
        attempts.append(1)
        raise RuntimeError("Antigravity CLI returned an empty response")

    monkeypatch.setattr(agy_backend, "run_agy_chat", flaky)
    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        agy_backend.chat_optimizer("s", "u", retries=3)
    assert len(attempts) == 3


def test_optimizer_rejects_images(monkeypatch) -> None:
    monkeypatch.setattr(agy_backend, "run_agy_chat", lambda **kwargs: pytest.fail("should not run"))
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]}]
    with pytest.raises(RuntimeError, match="image attachments"):
        agy_backend.chat_optimizer_messages(messages, retries=3)


# ── Script-level config resolution ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--backend", "agy"], ("gemini_chat", "agy_exec", "gemini-3.5-flash", "gemini-3.5-flash")),
        (["--backend", "gemini"], ("gemini_chat", "gemini_chat", "gemini-3.5-flash", "gemini-3.5-flash")),
        (
            ["--backend", "agy", "--target_model", "gemini-3.8-flash-high"],
            ("gemini_chat", "agy_exec", "gemini-3.5-flash", "gemini-3.8-flash-high"),
        ),
        (
            ["--cfg-options", "model.optimizer_backend=agy_exec", "model.target_backend=agy_exec"],
            ("agy_exec", "agy_exec", "gemini-3.5-flash", "gemini-3.5-flash"),
        ),
    ],
)
def test_train_script_resolves_backends_and_default_models(monkeypatch, tmp_path, argv, expected) -> None:
    from scripts import train

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/searchqa/default.yaml", "--out_root", str(tmp_path), *argv],
    )
    flat = train.load_config(train.parse_args())
    assert (
        flat["optimizer_backend"],
        flat["target_backend"],
        flat["optimizer_model"],
        flat["target_model"],
    ) == expected
    assert flat["agy_trace_to_optimizer"] is True
