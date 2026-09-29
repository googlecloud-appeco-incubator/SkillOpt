"""Antigravity CLI (``agy``) chat backend for SkillOpt (optimizer role).

Runs the same ``agy`` binary that powers the ``agy_exec`` target backend as a
plain chat model for reflection, so a run can use agy's authentication (Google
sign-in, ADC via ``AGY_ADC_AUTH=true``, or ``GEMINI_API_KEY``) for both roles.
Tool-calling message calls are emulated with agy's ``--json-schema`` structured
output, the same way the ``claude_code_exec`` optimizer does it.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from skillopt.model import codex_backend as _codex
from skillopt.model.claude_backend import _build_prompt_from_messages
from skillopt.model.codex_harness import run_agy_chat
from skillopt.model.common import TokenTracker, default_model_for_backend

BACKEND_NAME = "agy_exec"

OPTIMIZER_DEPLOYMENT = os.environ.get("OPTIMIZER_DEPLOYMENT", default_model_for_backend(BACKEND_NAME))
TARGET_DEPLOYMENT = os.environ.get("TARGET_DEPLOYMENT", default_model_for_backend(BACKEND_NAME))
REASONING_EFFORT: str | None = None
tracker = TokenTracker()


def _chat_messages_impl(
    model: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    retries: int,
    stage: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[Any, dict[str, int]]:
    del max_completion_tokens  # agy does not expose a completion-token cap
    structured_output = bool(tools) or return_message
    schema = _codex._assistant_message_schema() if structured_output else None
    # An explicit per-call effort wins; otherwise fall back to the value set via
    # set_reasoning_effort. None leaves agy_exec_effort in charge.
    effort = reasoning_effort if reasoning_effort is not None else REASONING_EFFORT
    last_err: Exception | None = None

    for attempt in range(max(1, retries)):
        try:
            system, prompt, attachments = _build_prompt_from_messages(
                messages,
                tools=tools,
                tool_choice=tool_choice,
                structured_output=structured_output,
            )
            if attachments:
                raise ValueError("agy_exec optimizer backend does not support image attachments")
            raw_text, usage_info = run_agy_chat(
                system=system,
                prompt=prompt,
                model=model,
                timeout=timeout,
                schema=schema,
                effort=effort,
            )
            tracker.record(stage, usage_info["prompt_tokens"], usage_info["completion_tokens"])
            if not structured_output:
                return raw_text, usage_info
            payload = json.loads(raw_text)
            compat = _codex._compat_message_from_payload(payload, tool_choice=tool_choice)
            return (compat if return_message else compat.content), usage_info
        except ValueError as exc:
            last_err = exc
            if not isinstance(exc, json.JSONDecodeError):
                break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        if attempt + 1 < retries:
            time.sleep(min(2 ** attempt, 30))

    raise RuntimeError(f"Antigravity CLI call failed after {retries} attempt(s): {last_err}")


def chat_with_model(
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "custom",
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, int]]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return _chat_messages_impl(
        model,
        messages,
        max_completion_tokens,
        retries,
        stage,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def chat_optimizer(
    system: str,
    user: str,
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, dict[str, int]]:
    return chat_with_model(
        model=OPTIMIZER_DEPLOYMENT,
        system=system,
        user=user,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage=stage,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def chat_optimizer_messages(
    messages: list[dict[str, Any]],
    max_completion_tokens: int = 16384,
    retries: int = 5,
    stage: str = "optimizer",
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    return_message: bool = False,
    timeout: int | None = None,
    reasoning_effort: str | None = None,
) -> tuple[Any, dict[str, int]]:
    return _chat_messages_impl(
        OPTIMIZER_DEPLOYMENT,
        messages,
        max_completion_tokens,
        retries,
        stage,
        tools=tools,
        tool_choice=tool_choice,
        return_message=return_message,
        timeout=timeout,
        reasoning_effort=reasoning_effort,
    )


def get_token_summary() -> dict[str, dict[str, int]]:
    return tracker.summary()


def reset_token_tracker() -> None:
    tracker.reset()


def set_reasoning_effort(effort: str | None) -> None:
    global REASONING_EFFORT
    REASONING_EFFORT = effort or None


def set_optimizer_deployment(deployment: str) -> None:
    global OPTIMIZER_DEPLOYMENT
    OPTIMIZER_DEPLOYMENT = deployment or default_model_for_backend(BACKEND_NAME)


def set_target_deployment(deployment: str) -> None:
    global TARGET_DEPLOYMENT
    TARGET_DEPLOYMENT = deployment or default_model_for_backend(BACKEND_NAME)
