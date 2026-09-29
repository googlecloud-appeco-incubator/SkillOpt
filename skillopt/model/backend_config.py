"""Runtime backend configuration for optimizer/target model calls."""
from __future__ import annotations

import json
import os
import shutil
import warnings
from collections.abc import Mapping
from typing import Any

from skillopt.model.common import normalize_backend_name


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_bool_setting(value: Any, *, name: str) -> bool:
    """Parse a config boolean without treating non-empty strings as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    elif isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(
        f"Invalid {name}: {value!r}. Expected a boolean or one of "
        "true/false, yes/no, on/off, or 1/0."
    )


def _resolve_cli_path(value: str) -> str:
    """Resolve a CLI name/executable via PATH + PATHEXT.

    On Windows these npm CLIs install as ``.cmd`` shims, and CreateProcess does
    not search PATHEXT for a bare name (so a bare ``codex`` spawn raises
    WinError 2). ``shutil.which`` finds the real executable; fall back to the
    given value so a configured path still passes through unchanged when it
    cannot be resolved (e.g. a name that is not on this PATH).
    """
    resolved = shutil.which(value)
    return resolved or value


OPTIMIZER_BACKEND = normalize_backend_name(os.environ.get("OPTIMIZER_BACKEND", "openai_chat"))
TARGET_BACKEND = normalize_backend_name(os.environ.get("TARGET_BACKEND", "openai_chat"))

CODEX_EXEC_PATH = _resolve_cli_path(os.environ.get("CODEX_EXEC_PATH") or os.environ.get("CODEX_CLI_BIN") or os.environ.get("CODEX_PATH") or "codex")
CODEX_EXEC_SANDBOX = os.environ.get("CODEX_EXEC_SANDBOX") or os.environ.get("CODEX_SANDBOX_MODE") or os.environ.get("CODEX_SANDBOX") or "workspace-write"
CODEX_EXEC_PROFILE = os.environ.get("CODEX_EXEC_PROFILE", "")
_CODEX_EXEC_FULL_AUTO_ENV = os.environ.get("CODEX_EXEC_FULL_AUTO")
# Kept as a compatibility symbol only.  The retired --full-auto flag is never
# emitted, regardless of the deprecated environment/config value.
CODEX_EXEC_FULL_AUTO = False
CODEX_EXEC_REASONING_EFFORT = os.environ.get("CODEX_EXEC_REASONING_EFFORT", "none")
CODEX_EXEC_USE_SDK = os.environ.get("CODEX_EXEC_USE_SDK", "auto")
CODEX_EXEC_NETWORK_ACCESS = _parse_bool(os.environ.get("CODEX_EXEC_NETWORK_ACCESS"), False)
CODEX_EXEC_WEB_SEARCH = _parse_bool(os.environ.get("CODEX_EXEC_WEB_SEARCH"), False)
CODEX_EXEC_APPROVAL_POLICY = os.environ.get("CODEX_EXEC_APPROVAL_POLICY", "never")
CLAUDE_CODE_EXEC_PATH = _resolve_cli_path(os.environ.get("CLAUDE_CODE_EXEC_PATH", "claude"))
CLAUDE_CODE_EXEC_PROFILE = os.environ.get("CLAUDE_CODE_EXEC_PROFILE", "")
CLAUDE_CODE_EXEC_USE_SDK = os.environ.get("CLAUDE_CODE_EXEC_USE_SDK", "auto")
CLAUDE_CODE_EXEC_EFFORT = os.environ.get("CLAUDE_CODE_EXEC_EFFORT", "medium")
CURSOR_EXEC_PATH = _resolve_cli_path(os.environ.get("CURSOR_EXEC_PATH", "cursor-agent"))
CURSOR_EXEC_SANDBOX = os.environ.get("CURSOR_EXEC_SANDBOX", "enabled")
COPILOT_EXEC_PATH = _resolve_cli_path(os.environ.get("COPILOT_EXEC_PATH", "copilot"))
COPILOT_EXEC_HOME = os.environ.get("COPILOT_EXEC_HOME", "")
COPILOT_EXEC_ALLOW_ALL_TOOLS = (
    "1" if _parse_bool(os.environ.get("COPILOT_EXEC_ALLOW_ALL_TOOLS"), False) else "0"
)
COPILOT_CHAT_OPTIMIZER_MODEL = os.environ.get("COPILOT_CHAT_OPTIMIZER_MODEL", "")
COPILOT_CHAT_TARGET_MODEL = os.environ.get("COPILOT_CHAT_TARGET_MODEL", "")
COPILOT_CHAT_TIMEOUT = os.environ.get("COPILOT_CHAT_TIMEOUT", "600")


def _resolve_agy_path(value: str) -> str:
    """Resolve the Antigravity CLI, falling back to its installer's default dir.

    ``curl -fsSL https://antigravity.google/cli/install.sh | bash`` drops the
    binary in ``~/.local/bin``, which is often missing from non-login PATHs.
    """
    resolved = shutil.which(value)
    if resolved:
        return resolved
    if value == "agy":
        candidate = os.path.join(os.path.expanduser("~"), ".local", "bin", "agy")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return value


AGY_EXEC_PATH = _resolve_agy_path(os.environ.get("AGY_EXEC_PATH") or os.environ.get("AGY_CLI_BIN") or "agy")
# Headless agy auto-denies every tool that needs approval (shell commands in
# particular), which usually ends the turn with an empty response. Skipping
# the prompt matches the claude_code_exec default (bypassPermissions).
AGY_EXEC_SKIP_PERMISSIONS = _parse_bool(os.environ.get("AGY_EXEC_SKIP_PERMISSIONS"), True)
AGY_EXEC_SANDBOX = _parse_bool(os.environ.get("AGY_EXEC_SANDBOX"), False)
AGY_EXEC_MODE = os.environ.get("AGY_EXEC_MODE", "")
AGY_EXEC_EFFORT = os.environ.get("AGY_EXEC_EFFORT", "")

if _CODEX_EXEC_FULL_AUTO_ENV is not None:
    warnings.warn(
        "CODEX_EXEC_FULL_AUTO is deprecated and ignored; use "
        "CODEX_EXEC_SANDBOX and CODEX_EXEC_APPROVAL_POLICY instead",
        FutureWarning,
        stacklevel=2,
    )


# A train/eval configuration is a complete runtime snapshot.  Preserve the
# environment-derived values from process startup so loading a later config
# cannot inherit path or permission settings applied by an earlier run in the
# same process.
_CODEX_EXEC_BASELINE = {
    "path": CODEX_EXEC_PATH,
    "sandbox": CODEX_EXEC_SANDBOX,
    "profile": CODEX_EXEC_PROFILE,
    "reasoning_effort": CODEX_EXEC_REASONING_EFFORT,
    "use_sdk": CODEX_EXEC_USE_SDK,
    "network_access": CODEX_EXEC_NETWORK_ACCESS,
    "web_search": CODEX_EXEC_WEB_SEARCH,
    "approval_policy": CODEX_EXEC_APPROVAL_POLICY,
}
_CODEX_EXEC_MUTATED_ENV_KEYS = (
    "CODEX_EXEC_PATH",
    "CODEX_CLI_BIN",
    "CODEX_EXEC_SANDBOX",
    "CODEX_SANDBOX_MODE",
    "CODEX_EXEC_PROFILE",
    "CODEX_EXEC_REASONING_EFFORT",
    "CODEX_EXEC_USE_SDK",
    "CODEX_EXEC_NETWORK_ACCESS",
    "CODEX_EXEC_WEB_SEARCH",
    "CODEX_EXEC_APPROVAL_POLICY",
)
_CODEX_EXEC_ENV_BASELINE = {
    key: os.environ[key]
    for key in _CODEX_EXEC_MUTATED_ENV_KEYS
    if key in os.environ
}


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        return default


EXEC_EMPTY_RESPONSE_RETRIES = max(0, _parse_int(os.environ.get("EXEC_EMPTY_RESPONSE_RETRIES"), 1))
AGY_CHAT_TIMEOUT = max(1, _parse_int(os.environ.get("AGY_CHAT_TIMEOUT"), 900))
CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS = max(
    0,
    _parse_int(os.environ.get("CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS"), 16384),
)


def set_optimizer_backend(backend: str) -> None:
    global OPTIMIZER_BACKEND
    OPTIMIZER_BACKEND = normalize_backend_name(backend or "openai_chat")
    if OPTIMIZER_BACKEND not in {
        "openai_chat",
        "claude_chat",
        "qwen_chat",
        "minimax_chat",
        "openai_compatible",
        "copilot_chat",
        "codex_exec",
        "claude_code_exec",
        "gemini_chat",
        "agy_exec",
    }:
        raise ValueError(
            f"Unsupported optimizer backend: {OPTIMIZER_BACKEND!r}. "
            "Supported values are 'openai_chat', 'claude_chat', 'qwen_chat', 'minimax_chat', "
            "'openai_compatible', 'copilot_chat', 'codex_exec', 'claude_code_exec', "
            "'gemini_chat', and 'agy_exec'."
        )
    os.environ["OPTIMIZER_BACKEND"] = OPTIMIZER_BACKEND


def get_optimizer_backend() -> str:
    return OPTIMIZER_BACKEND


def set_target_backend(backend: str) -> None:
    global TARGET_BACKEND
    TARGET_BACKEND = normalize_backend_name(backend or "openai_chat")
    if TARGET_BACKEND not in {"openai_chat", "claude_chat", "qwen_chat", "minimax_chat", "openai_compatible", "copilot_chat", "gemini_chat", "codex_exec", "claude_code_exec", "cursor_exec", "copilot_exec", "agy_exec"}:
        raise ValueError(
            f"Unsupported target backend: {TARGET_BACKEND!r}. "
            "Supported values are 'openai_chat', 'claude_chat', 'qwen_chat', 'minimax_chat', "
            "'openai_compatible', 'copilot_chat', 'gemini_chat', 'codex_exec', 'claude_code_exec', "
            "'cursor_exec', 'copilot_exec', and 'agy_exec'."
        )
    os.environ["TARGET_BACKEND"] = TARGET_BACKEND


def get_target_backend() -> str:
    return TARGET_BACKEND


def is_target_exec_backend() -> bool:
    return TARGET_BACKEND in {"codex_exec", "claude_code_exec", "cursor_exec", "copilot_exec", "agy_exec"}


def is_optimizer_chat_backend() -> bool:
    return OPTIMIZER_BACKEND in {
        "openai_chat",
        "claude_chat",
        "qwen_chat",
        "minimax_chat",
        "openai_compatible",
        "copilot_chat",
        "codex_exec",
        "claude_code_exec",
        "gemini_chat",
        "agy_exec",
    }


def is_target_chat_backend() -> bool:
    return TARGET_BACKEND in {"openai_chat", "claude_chat", "qwen_chat", "minimax_chat", "openai_compatible", "copilot_chat", "gemini_chat"}


_ALLOWED_CODEX_SANDBOXES = frozenset({"read-only", "workspace-write", "danger-full-access"})


def validate_exec_sandbox(sandbox: str) -> str:
    s = str(sandbox).strip()
    if s not in _ALLOWED_CODEX_SANDBOXES:
        raise ValueError(
            f"Invalid codex_exec sandbox: {sandbox!r}. "
            f"Allowed values are: {sorted(_ALLOWED_CODEX_SANDBOXES)}"
        )
    return s


def configure_codex_exec(
    *,
    path: str | None = None,
    sandbox: str | None = None,
    profile: str | None = None,
    full_auto: bool | None = None,
    reasoning_effort: str | None = None,
    use_sdk: str | None = None,
    network_access: bool | str | int | None = None,
    web_search: bool | str | int | None = None,
    approval_policy: str | None = None,
) -> None:
    global CODEX_EXEC_PATH, CODEX_EXEC_SANDBOX, CODEX_EXEC_PROFILE, CODEX_EXEC_REASONING_EFFORT, CODEX_EXEC_USE_SDK, CODEX_EXEC_NETWORK_ACCESS, CODEX_EXEC_WEB_SEARCH, CODEX_EXEC_APPROVAL_POLICY
    parsed_network_access = (
        None
        if network_access is None
        else _coerce_bool_setting(network_access, name="codex_exec_network_access")
    )
    parsed_web_search = (
        None
        if web_search is None
        else _coerce_bool_setting(web_search, name="codex_exec_web_search")
    )
    if path is not None:
        CODEX_EXEC_PATH = _resolve_cli_path(str(path).strip() or "codex")
        os.environ["CODEX_EXEC_PATH"] = CODEX_EXEC_PATH
        os.environ["CODEX_CLI_BIN"] = CODEX_EXEC_PATH
    if sandbox is not None:
        val = str(sandbox).strip() or "workspace-write"
        validate_exec_sandbox(val)
        CODEX_EXEC_SANDBOX = val
        os.environ["CODEX_EXEC_SANDBOX"] = CODEX_EXEC_SANDBOX
        os.environ["CODEX_SANDBOX_MODE"] = CODEX_EXEC_SANDBOX
    if profile is not None:
        CODEX_EXEC_PROFILE = str(profile).strip()
        os.environ["CODEX_EXEC_PROFILE"] = CODEX_EXEC_PROFILE
    if full_auto is not None:
        warnings.warn(
            "codex_exec_full_auto is deprecated and ignored; use "
            "codex_exec_sandbox and codex_exec_approval_policy instead",
            FutureWarning,
            stacklevel=2,
        )
    if reasoning_effort is not None:
        CODEX_EXEC_REASONING_EFFORT = str(reasoning_effort).strip() or "none"
        os.environ["CODEX_EXEC_REASONING_EFFORT"] = CODEX_EXEC_REASONING_EFFORT
    if use_sdk is not None:
        CODEX_EXEC_USE_SDK = str(use_sdk).strip().lower() or "auto"
        os.environ["CODEX_EXEC_USE_SDK"] = CODEX_EXEC_USE_SDK
    if parsed_network_access is not None:
        CODEX_EXEC_NETWORK_ACCESS = parsed_network_access
        os.environ["CODEX_EXEC_NETWORK_ACCESS"] = "true" if CODEX_EXEC_NETWORK_ACCESS else "false"
    if parsed_web_search is not None:
        CODEX_EXEC_WEB_SEARCH = parsed_web_search
        os.environ["CODEX_EXEC_WEB_SEARCH"] = "true" if CODEX_EXEC_WEB_SEARCH else "false"
    if approval_policy is not None:
        CODEX_EXEC_APPROVAL_POLICY = str(approval_policy).strip() or "never"
        os.environ["CODEX_EXEC_APPROVAL_POLICY"] = CODEX_EXEC_APPROVAL_POLICY


def _first_nonempty(config: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = config.get(key)
        if value not in (None, ""):
            return value
    return None


def _restore_codex_exec_baseline() -> None:
    """Restore process-start globals and the exact environment-key state."""
    global CODEX_EXEC_PATH, CODEX_EXEC_SANDBOX, CODEX_EXEC_PROFILE, CODEX_EXEC_REASONING_EFFORT, CODEX_EXEC_USE_SDK, CODEX_EXEC_NETWORK_ACCESS, CODEX_EXEC_WEB_SEARCH, CODEX_EXEC_APPROVAL_POLICY
    CODEX_EXEC_PATH = _CODEX_EXEC_BASELINE["path"]
    CODEX_EXEC_SANDBOX = _CODEX_EXEC_BASELINE["sandbox"]
    CODEX_EXEC_PROFILE = _CODEX_EXEC_BASELINE["profile"]
    CODEX_EXEC_REASONING_EFFORT = _CODEX_EXEC_BASELINE["reasoning_effort"]
    CODEX_EXEC_USE_SDK = _CODEX_EXEC_BASELINE["use_sdk"]
    CODEX_EXEC_NETWORK_ACCESS = _CODEX_EXEC_BASELINE["network_access"]
    CODEX_EXEC_WEB_SEARCH = _CODEX_EXEC_BASELINE["web_search"]
    CODEX_EXEC_APPROVAL_POLICY = _CODEX_EXEC_BASELINE["approval_policy"]
    for key in _CODEX_EXEC_MUTATED_ENV_KEYS:
        if key in _CODEX_EXEC_ENV_BASELINE:
            os.environ[key] = _CODEX_EXEC_ENV_BASELINE[key]
        else:
            os.environ.pop(key, None)


def configure_codex_exec_from_config(config: Mapping[str, Any]) -> None:
    """Configure Codex exec from a flattened train/eval configuration.

    Dedicated ``codex_exec_*`` keys take precedence over legacy aliases.  The
    ordering is centralized here so training and eval-only cannot drift.
    """
    _restore_codex_exec_baseline()
    configure_codex_exec(
        path=_first_nonempty(
            config,
            "codex_exec_path",
            "codex_path",
            "codex_cli_bin",
            "codex_bin",
        ),
        sandbox=_first_nonempty(
            config,
            "codex_exec_sandbox",
            "sandbox",
            "codex_sandbox",
        ),
        profile=_first_nonempty(config, "codex_exec_profile"),
        full_auto=config.get("codex_exec_full_auto"),
        reasoning_effort=_first_nonempty(config, "codex_exec_reasoning_effort"),
        use_sdk=_first_nonempty(config, "codex_exec_use_sdk"),
        network_access=config.get("codex_exec_network_access"),
        web_search=config.get("codex_exec_web_search"),
        approval_policy=_first_nonempty(config, "codex_exec_approval_policy"),
    )


def get_codex_exec_config() -> dict[str, str | bool | int]:
    return {
        "path": CODEX_EXEC_PATH,
        "sandbox": CODEX_EXEC_SANDBOX,
        "profile": CODEX_EXEC_PROFILE,
        "full_auto": CODEX_EXEC_FULL_AUTO,
        "reasoning_effort": CODEX_EXEC_REASONING_EFFORT,
        "use_sdk": CODEX_EXEC_USE_SDK,
        "network_access": CODEX_EXEC_NETWORK_ACCESS,
        "web_search": CODEX_EXEC_WEB_SEARCH,
        "approval_policy": CODEX_EXEC_APPROVAL_POLICY,
        "empty_response_retries": EXEC_EMPTY_RESPONSE_RETRIES,
    }


def build_codex_exec_cli_config_overrides(
    config: Mapping[str, Any] | None = None,
) -> list[str]:
    """Translate shared network/search settings to Codex CLI overrides."""
    effective = get_codex_exec_config() if config is None else config
    network_enabled = _coerce_bool_setting(
        effective.get("network_access", False),
        name="codex_exec_network_access",
    )
    search_enabled = _coerce_bool_setting(
        effective.get("web_search", False),
        name="codex_exec_web_search",
    )
    network_access = "true" if network_enabled else "false"
    web_search = "live" if search_enabled else "disabled"
    return [
        f"sandbox_workspace_write.network_access={network_access}",
        f"web_search={json.dumps(web_search)}",
    ]


def configure_claude_code_exec(
    *,
    path: str | None = None,
    profile: str | None = None,
    use_sdk: str | None = None,
    effort: str | None = None,
    max_thinking_tokens: int | str | None = None,
) -> None:
    global CLAUDE_CODE_EXEC_PATH, CLAUDE_CODE_EXEC_PROFILE, CLAUDE_CODE_EXEC_USE_SDK, CLAUDE_CODE_EXEC_EFFORT, CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS
    if path is not None:
        CLAUDE_CODE_EXEC_PATH = _resolve_cli_path(str(path).strip() or "claude")
        os.environ["CLAUDE_CODE_EXEC_PATH"] = CLAUDE_CODE_EXEC_PATH
    if profile is not None:
        CLAUDE_CODE_EXEC_PROFILE = str(profile).strip()
        os.environ["CLAUDE_CODE_EXEC_PROFILE"] = CLAUDE_CODE_EXEC_PROFILE
    if use_sdk is not None:
        CLAUDE_CODE_EXEC_USE_SDK = str(use_sdk).strip().lower() or "auto"
        os.environ["CLAUDE_CODE_EXEC_USE_SDK"] = CLAUDE_CODE_EXEC_USE_SDK
    if effort is not None:
        CLAUDE_CODE_EXEC_EFFORT = str(effort).strip().lower() or "medium"
        os.environ["CLAUDE_CODE_EXEC_EFFORT"] = CLAUDE_CODE_EXEC_EFFORT
    if max_thinking_tokens is not None:
        CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS = max(
            0,
            _parse_int(str(max_thinking_tokens), 16384),
        )
        os.environ["CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS"] = str(CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS)


def get_claude_code_exec_config() -> dict[str, str | int]:
    return {
        "path": CLAUDE_CODE_EXEC_PATH,
        "profile": CLAUDE_CODE_EXEC_PROFILE,
        "use_sdk": CLAUDE_CODE_EXEC_USE_SDK,
        "effort": CLAUDE_CODE_EXEC_EFFORT,
        "max_thinking_tokens": CLAUDE_CODE_EXEC_MAX_THINKING_TOKENS,
        "empty_response_retries": EXEC_EMPTY_RESPONSE_RETRIES,
    }


def configure_cursor_exec(
    *,
    path: str | None = None,
    sandbox: str | None = None,
) -> None:
    global CURSOR_EXEC_PATH, CURSOR_EXEC_SANDBOX
    if path is not None:
        CURSOR_EXEC_PATH = _resolve_cli_path(str(path).strip() or "cursor-agent")
        os.environ["CURSOR_EXEC_PATH"] = CURSOR_EXEC_PATH
    if sandbox is not None:
        normalized_sandbox = str(sandbox).strip().lower() or "enabled"
        if normalized_sandbox not in {"enabled", "disabled"}:
            raise ValueError("cursor_exec sandbox must be 'enabled' or 'disabled'")
        CURSOR_EXEC_SANDBOX = normalized_sandbox
        os.environ["CURSOR_EXEC_SANDBOX"] = CURSOR_EXEC_SANDBOX


def get_cursor_exec_config() -> dict[str, str | int]:
    if CURSOR_EXEC_SANDBOX not in {"enabled", "disabled"}:
        raise ValueError("cursor_exec sandbox must be 'enabled' or 'disabled'")
    return {
        "path": CURSOR_EXEC_PATH,
        "sandbox": CURSOR_EXEC_SANDBOX,
        "empty_response_retries": EXEC_EMPTY_RESPONSE_RETRIES,
    }


def configure_copilot_exec(
    *,
    path: str | None = None,
    home: str | None = None,
    allow_all_tools: bool | str | None = None,
) -> None:
    """Configure the GitHub Copilot CLI exec backend.

    ``home`` points ``COPILOT_HOME`` at an isolated config dir so the user's MCP
    servers and custom instructions are not loaded during a rollout. Auth lives
    outside ``COPILOT_HOME`` (OS credential store / token env vars), so isolating
    the home does not break authentication.
    """
    global COPILOT_EXEC_PATH, COPILOT_EXEC_HOME, COPILOT_EXEC_ALLOW_ALL_TOOLS
    if path is not None:
        COPILOT_EXEC_PATH = _resolve_cli_path(str(path).strip() or "copilot")
        os.environ["COPILOT_EXEC_PATH"] = COPILOT_EXEC_PATH
    if home is not None:
        COPILOT_EXEC_HOME = str(home).strip()
        os.environ["COPILOT_EXEC_HOME"] = COPILOT_EXEC_HOME
    if allow_all_tools is not None:
        normalized = str(allow_all_tools).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            COPILOT_EXEC_ALLOW_ALL_TOOLS = "1"
        elif normalized in {"0", "false", "no", "off", ""}:
            COPILOT_EXEC_ALLOW_ALL_TOOLS = "0"
        else:
            raise ValueError(
                "copilot_exec allow_all_tools must be a boolean-like value"
            )
        os.environ["COPILOT_EXEC_ALLOW_ALL_TOOLS"] = COPILOT_EXEC_ALLOW_ALL_TOOLS


def get_copilot_exec_config() -> dict[str, str | int]:
    if COPILOT_EXEC_ALLOW_ALL_TOOLS not in {"0", "1"}:
        raise ValueError("copilot_exec allow_all_tools must be '0' or '1'")
    return {
        "path": COPILOT_EXEC_PATH,
        "home": COPILOT_EXEC_HOME,
        "allow_all_tools": COPILOT_EXEC_ALLOW_ALL_TOOLS,
        "empty_response_retries": EXEC_EMPTY_RESPONSE_RETRIES,
    }


def configure_copilot_chat(
    *,
    path: str | None = None,
    home: str | None = None,
    optimizer_model: str | None = None,
    target_model: str | None = None,
    timeout: int | str | None = None,
) -> None:
    """Configure the Copilot CLI *chat* backend (optimizer and/or target).

    Unlike `copilot_exec`, this runs the CLI as a plain chat model so a whole
    run can use CLI authentication without a separate provider API key.
    """
    global COPILOT_EXEC_PATH, COPILOT_EXEC_HOME
    global COPILOT_CHAT_OPTIMIZER_MODEL, COPILOT_CHAT_TARGET_MODEL, COPILOT_CHAT_TIMEOUT
    if path is not None:
        COPILOT_EXEC_PATH = _resolve_cli_path(str(path).strip() or "copilot")
        os.environ["COPILOT_EXEC_PATH"] = COPILOT_EXEC_PATH
    if home is not None:
        COPILOT_EXEC_HOME = str(home).strip()
        os.environ["COPILOT_EXEC_HOME"] = COPILOT_EXEC_HOME
    if optimizer_model is not None:
        COPILOT_CHAT_OPTIMIZER_MODEL = str(optimizer_model).strip()
        os.environ["COPILOT_CHAT_OPTIMIZER_MODEL"] = COPILOT_CHAT_OPTIMIZER_MODEL
    if target_model is not None:
        COPILOT_CHAT_TARGET_MODEL = str(target_model).strip()
        os.environ["COPILOT_CHAT_TARGET_MODEL"] = COPILOT_CHAT_TARGET_MODEL
    if timeout is not None:
        value = str(timeout).strip()
        if not value.isdigit() or int(value) <= 0:
            raise ValueError("copilot_chat timeout must be a positive integer")
        COPILOT_CHAT_TIMEOUT = value
        os.environ["COPILOT_CHAT_TIMEOUT"] = COPILOT_CHAT_TIMEOUT


def get_copilot_chat_config() -> dict[str, str | int]:
    timeout = str(COPILOT_CHAT_TIMEOUT).strip()
    if not timeout.isdigit() or int(timeout) <= 0:
        raise ValueError("copilot_chat timeout must be a positive integer")
    return {
        "path": COPILOT_EXEC_PATH,
        "home": COPILOT_EXEC_HOME,
        "optimizer_model": COPILOT_CHAT_OPTIMIZER_MODEL,
        "target_model": COPILOT_CHAT_TARGET_MODEL,
        "timeout": int(timeout),
    }


_AGY_EFFORTS = frozenset({"low", "medium", "high", "max"})
_AGY_MODES = frozenset({"", "accept-edits", "plan"})


def normalize_agy_effort(value: Any) -> str:
    """Map SkillOpt's shared reasoning vocabulary onto ``agy --effort``.

    Blank/none/off means "do not pass --effort" (use the model slug's default).
    """
    effort = str(value or "").strip().lower()
    if effort in {"", "none", "off"}:
        return ""
    if effort == "minimal":
        return "low"
    if effort == "xhigh":
        return "max"
    if effort not in _AGY_EFFORTS:
        raise ValueError(
            f"Invalid agy_exec effort: {value!r}. Expected one of "
            "low, medium, high, max (or blank)."
        )
    return effort


def configure_agy_exec(
    *,
    path: str | None = None,
    skip_permissions: bool | str | int | None = None,
    sandbox: bool | str | int | None = None,
    mode: str | None = None,
    effort: str | None = None,
    chat_timeout: int | str | None = None,
) -> None:
    """Configure the Antigravity CLI (``agy``) backend (target exec and optimizer chat)."""
    global AGY_EXEC_PATH, AGY_EXEC_SKIP_PERMISSIONS, AGY_EXEC_SANDBOX, AGY_EXEC_MODE, AGY_EXEC_EFFORT, AGY_CHAT_TIMEOUT
    parsed_skip = (
        None
        if skip_permissions is None
        else _coerce_bool_setting(skip_permissions, name="agy_exec_skip_permissions")
    )
    parsed_sandbox = None if sandbox is None else _coerce_bool_setting(sandbox, name="agy_exec_sandbox")
    normalized_mode = None
    if mode is not None:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in _AGY_MODES:
            raise ValueError("agy_exec mode must be blank, 'accept-edits', or 'plan'")
    normalized_effort = None if effort is None else normalize_agy_effort(effort)
    parsed_timeout = None
    if chat_timeout is not None:
        raw = str(chat_timeout).strip()
        if not raw.isdigit() or int(raw) <= 0:
            raise ValueError("agy_chat timeout must be a positive integer")
        parsed_timeout = int(raw)

    if path is not None:
        AGY_EXEC_PATH = _resolve_agy_path(str(path).strip() or "agy")
        os.environ["AGY_EXEC_PATH"] = AGY_EXEC_PATH
    if parsed_skip is not None:
        AGY_EXEC_SKIP_PERMISSIONS = parsed_skip
        os.environ["AGY_EXEC_SKIP_PERMISSIONS"] = "true" if parsed_skip else "false"
    if parsed_sandbox is not None:
        AGY_EXEC_SANDBOX = parsed_sandbox
        os.environ["AGY_EXEC_SANDBOX"] = "true" if parsed_sandbox else "false"
    if normalized_mode is not None:
        AGY_EXEC_MODE = normalized_mode
        os.environ["AGY_EXEC_MODE"] = AGY_EXEC_MODE
    if normalized_effort is not None:
        AGY_EXEC_EFFORT = normalized_effort
        os.environ["AGY_EXEC_EFFORT"] = AGY_EXEC_EFFORT
    if parsed_timeout is not None:
        AGY_CHAT_TIMEOUT = parsed_timeout
        os.environ["AGY_CHAT_TIMEOUT"] = str(parsed_timeout)


def get_agy_exec_config() -> dict[str, str | bool | int]:
    return {
        "path": AGY_EXEC_PATH,
        "skip_permissions": AGY_EXEC_SKIP_PERMISSIONS,
        "sandbox": AGY_EXEC_SANDBOX,
        "mode": AGY_EXEC_MODE,
        "effort": normalize_agy_effort(AGY_EXEC_EFFORT),
        "chat_timeout": AGY_CHAT_TIMEOUT,
        "empty_response_retries": EXEC_EMPTY_RESPONSE_RETRIES,
    }


def configure_agy_exec_from_config(config: Mapping[str, Any]) -> None:
    """Configure agy from a flattened train/eval configuration.

    ``agy_exec_effort`` falls back to the shared ``reasoning_effort``; set it to
    ``none`` to omit ``--effort`` (required for models without effort variants,
    such as ``gemini-3-flash-preview``). Blank values keep the environment.
    """
    effort = _first_nonempty(config, "agy_exec_effort", "reasoning_effort")
    configure_agy_exec(
        path=_first_nonempty(config, "agy_exec_path"),
        skip_permissions=config.get("agy_exec_skip_permissions"),
        sandbox=config.get("agy_exec_sandbox"),
        mode=_first_nonempty(config, "agy_exec_mode"),
        effort=None if effort is None else str(effort),
        chat_timeout=_first_nonempty(config, "agy_chat_timeout"),
    )
