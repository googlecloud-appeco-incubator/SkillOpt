# Configuration Guide

SkillOpt uses YAML configuration files with a hierarchical override system.

## Config Structure

```
configs/
├── _base_/
│   └── default.yaml          # Global defaults
├── searchqa/
│   └── default.yaml          # SearchQA overrides
├── docvqa/
│   └── default.yaml          # DocVQA overrides
└── alfworld/
    └── default.yaml          # ALFWorld overrides
```

Benchmark configs inherit from `_base_/default.yaml` and override specific values.

## Key Parameters

### Model Backends

`optimizer_backend` controls reflection and skill editing;
`target_backend` controls task rollout. The legacy `backend` field remains for
backward compatibility, but explicit role fields are the clearest configuration.

```yaml
model:
  backend: azure_openai          # High-level compatibility label
  optimizer_backend: openai_chat
  target_backend: openai_chat
  optimizer: gpt-5.5             # Optimizer deployment/model
  target: gpt-5.5                # Target deployment/model
  azure_openai_auth_mode: api_key
```

| Backend | Optimizer | Target | Configuration |
|---|:---:|:---:|---|
| `openai_chat` | ✓ | ✓ | Azure OpenAI, or its explicit compatibility auth mode |
| `openai_compatible` | ✓ | ✓ | Generic OpenAI Chat Completions endpoint |
| `claude_chat` | ✓ | ✓ | Claude Code CLI (`claude -p`) |
| `qwen_chat` | ✓ | ✓ | Qwen served through an OpenAI-compatible endpoint (self-hosted vLLM/SGLang or a hosted gateway) |
| `minimax_chat` | ✓ | ✓ | MiniMax API |
| `copilot_chat` | ✓ | ✓ | GitHub Copilot CLI (`copilot -p`); alias `copilot` |
| `codex_exec` | ✓ | ✓ | Codex CLI execution harness |
| `claude_code_exec` | ✓ | ✓ | Claude Code CLI execution harness |
| `cursor_exec` | — | ✓ | Cursor Agent CLI execution harness |
| `copilot_exec` | — | ✓ | GitHub Copilot CLI execution harness |
| `gemini_chat` | ✓ | ✓ | Google Gemini via `google-genai` (Vertex AI with ADC, or a Gemini API key); aliases `gemini`, `vertex` |
| `agy_exec` | ✓ | ✓ | Antigravity CLI (`agy`) execution harness; aliases `agy`, `antigravity` |

`copilot_chat` fills **both** roles through a locally installed,
CLI-authenticated client, so `--backend copilot` runs a complete training loop
without a separate provider API key. Inference still uses the GitHub Copilot
cloud service. Sign in once with `copilot` (GitHub Copilot CLI) beforehand.
Expect roughly 20-40 s per call: the CLI is an agent, not a completions
endpoint, so a full-size run is far slower than a hosted backend. It also
reports no token counts, so usage totals are zero.
It does not support caller-supplied tools or structured tool calls; environments
that require those features must use another chat backend.

The current MiniMax adapter has one shared deployment. Set
`model.minimax_model` when MiniMax is the target; a mixed-backend run cannot
independently select a MiniMax optimizer model and a different target model.

MiniMax is served from region-specific hosts. Select one with
`model.minimax_region` (or `MINIMAX_REGION`) instead of hardcoding a host:
`global_en` (default) resolves to `https://api.minimax.io/v1` and `cn_zh`
resolves to `https://api.minimaxi.com/v1`. An explicit
`model.minimax_base_url` or `MINIMAX_BASE_URL` overrides the region default.

### Gemini and Antigravity (agy)

`gemini_chat` calls Gemini through the official `google-genai` SDK
(`pip install -e ".[gemini]"`). It fills both roles with `--backend gemini`,
supports tool calling (function tools; `web_search` maps to Google Search
grounding), `image_url` inputs, and token accounting. It authenticates the way
`google-genai` does:

- **Vertex AI** (default when no API key is set): Application Default
  Credentials plus `GOOGLE_CLOUD_PROJECT`; `GOOGLE_CLOUD_LOCATION` defaults to
  `global`. Force it with `GOOGLE_GENAI_USE_VERTEXAI=true`.
- **Gemini Developer API**: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).

The shared `reasoning_effort` becomes a Gemini thinking level (Gemini 3:
`minimal`/`low`/`medium`/`high`, with Pro limited to `low`/`high`) or a
thinking budget (Gemini 2.5). Pin a level with `model.gemini_thinking_level`.

`agy_exec` runs the [Antigravity CLI](https://antigravity.google/docs/cli/install)
headlessly inside the benchmark workspace, like `claude_code_exec`. The prompt
is sent on stdin as stream-json, and every tool step is parsed from agy's
stream-json output into `agy_trace_summary.txt` / `agy_trace_steps.txt`, which
the reflector reads when `model.agy_trace_to_optimizer` is on (the default).
`--backend agy` pairs the agy target with a `gemini_chat` optimizer so the whole
run uses Google credentials; set `model.optimizer_backend=agy_exec` to reflect
through agy as well.

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash   # installs ~/.local/bin/agy
export AGY_ADC_AUTH=true                  # headless auth from ADC (or sign in once with `agy`)
export GOOGLE_CLOUD_PROJECT=my-project    # used by gemini_chat on Vertex AI
skillopt-train --config configs/searchqa/default.yaml --backend agy \
  --cfg-options model.target=gemini-3.8-flash-low model.optimizer=gemini-3.5-flash
```

agy model slugs come from `agy models`. A slug with an effort suffix
(`gemini-3.8-flash-low`) is passed as is; for a bare slug, SkillOpt adds
`--effort` from `model.agy_exec_effort`, falling back to `reasoning_effort`.
Set `model.agy_exec_effort=none` for models without effort variants, such as
`gemini-3-flash-preview`. Headless agy auto-denies tools that need approval,
which usually ends the turn with an empty answer, so rollouts pass
`--dangerously-skip-permissions` by default (the same policy as
`claude_code_exec`); set `model.agy_exec_skip_permissions=false` to disable it,
and `model.agy_exec_sandbox=true` to add agy's terminal sandbox. agy reads its
settings, skills, and MCP servers from `~/.gemini/antigravity-cli`, so user
customizations there also apply during rollouts.

For a generic compatible provider, select the role backends explicitly rather
than relying on a high-level shorthand:

```yaml
model:
  optimizer_backend: openai_compatible
  target_backend: openai_compatible
  optimizer: deepseek-chat
  target: deepseek-chat
```

The train/eval entry points apply `model.optimizer` and `model.target` after
backend initialization. For the selected roles, these YAML values override
`OPENAI_COMPATIBLE_MODEL`, `QWEN_CHAT_MODEL`, and their per-role environment
forms. The environment model variables mainly seed direct library use; always
set the role models in a training or evaluation config.

### Training

```yaml
train:
  num_epochs: 4                  # Number of training epochs
  batch_size: 40                 # Tasks per step (batch size)
  accumulation: 1                # Gradient accumulation
  seed: 42
```

### Gradient (Reflection)

```yaml
gradient:
  minibatch_size: 8              # Reflect minibatch size
  analyst_workers: 16            # Parallel reflection workers
  failure_only: false            # Only reflect on failures
```

### Optimizer

```yaml
optimizer:
  learning_rate: 4               # Max edits per step (edit budget)
  min_learning_rate: 2           # Min edits for decay schedulers
  lr_scheduler: cosine           # constant | linear | cosine | autonomous
  use_slow_update: true          # Momentum-like blending at epoch boundary
  slow_update_samples: 20        # Samples for slow update evaluation
  use_meta_skill: true           # Cross-epoch strategy memory
```

### Skill-Aware Reflection (optional, off by default)

EmbodiSkill-style failure routing: the failure analyst classifies each
failure pattern as **SKILL_DEFECT** (the rule is wrong or missing → normal
gated body edit) or **EXECUTION_LAPSE** (a valid rule exists but was not
followed → a short reminder appended to a protected appendix region inside
the skill that step-level edits can never modify).

```yaml
optimizer:
  use_skill_aware_reflection: false    # Master switch (default off = baseline-identical)
  skill_aware_appendix_source: both    # both | failure_only (paper-faithful S_app)
  skill_aware_consolidate_threshold: 0 # >0: LLM-compact the appendix past N notes (experimental)
```

Notes:

- The switch is resolved process-wide from the config
  (`configure_skill_aware_reflection`), so it applies to every benchmark
  with no per-adapter wiring.
- `failure_only` restricts appendix notes to the failure analyst, matching
  the original S_app formulation; `both` additionally lets the success
  analyst re-emphasize existing rules.
- Appendix notes bypass the validation gate by design and accumulate with
  order-preserving dedup; lapse-only steps (no body edits) still flush
  their notes.
- Not supported together with `skill_update_mode=rewrite_from_suggestions`
  or the full-rewrite modes: whole-document rewrites can drop the appendix
  region.

### Evaluation

```yaml
evaluation:
  use_gate: true                 # Validation gating (accept/reject updates)
  gate_metric: hard              # hard | soft | mixed
  gate_mixed_weight: 0.5         # Soft-score weight when metric=mixed
  use_semantic_density: false    # Optional instruction-density bonus
  eval_test: true                # Run test evaluation after training
```

The default and paper-style setting is `use_gate: true`. Setting it to `false`
still records selection scores but force-accepts every candidate, so it changes
the optimization semantics and should be reported explicitly.

### Environment (Data)

```yaml
env:
  name: searchqa                 # Benchmark name
  split_mode: ratio              # ratio | split_dir
  split_ratio: "2:1:7"           # train:val:test ratio
  data_path: ""                  # Path to dataset
  exec_timeout: 120              # Per-task timeout (seconds)
```

## CLI Overrides

Override any config value from the command line:

```bash
python scripts/train.py \
  --config configs/searchqa/default.yaml \
  --cfg-options \
    optimizer.learning_rate=16 \
    optimizer.lr_scheduler=linear \
    gradient.analyst_workers=8
```

## Environment Variables

Model credentials are loaded from environment variables:

| Variable | Backend | Description |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | `openai_chat` | Azure resource URL, or compatibility-mode base URL |
| `AZURE_OPENAI_API_VERSION` | `openai_chat` | Azure API version |
| `AZURE_OPENAI_AUTH_MODE` | `openai_chat` | `api_key`, `azure_cli`, `managed_identity`, or `openai_compatible` |
| `AZURE_OPENAI_API_KEY` | `openai_chat` | Required when auth mode is `api_key` or `openai_compatible` |
| `OPENAI_COMPATIBLE_BASE_URL` | `openai_compatible` | Generic Chat Completions base URL |
| `OPENAI_COMPATIBLE_API_KEY` | `openai_compatible` | Provider API key; optional for local servers |
| `OPENAI_COMPATIBLE_MODEL` | `openai_compatible` | Shared provider model ID for direct library use; train/eval YAML role models take precedence |
| `CLAUDE_CLI_BIN` | `claude_chat` | Optional path to the `claude` executable; defaults to `claude` |
| `ANTHROPIC_API_KEY` | `claude_chat` | Optional authentication method understood by the Claude CLI, not a direct SkillOpt API client |
| `CURSOR_EXEC_PATH` | `cursor_exec` | Optional path to `cursor-agent`; defaults to `cursor-agent` |
| `CURSOR_EXEC_SANDBOX` | `cursor_exec` | Cursor sandbox mode: `enabled` (default) or `disabled` |
| `CURSOR_API_KEY` | `cursor_exec` | Optional authentication method understood directly by Cursor Agent |
| `QWEN_CHAT_BASE_URL` | `qwen_chat` | OpenAI-compatible Qwen endpoint: self-hosted vLLM/SGLang or a hosted gateway |
| `QWEN_CHAT_THINKING_MODE` | `qwen_chat` | `server_default` (default; omit `chat_template_kwargs`), `enabled`, or `disabled`; per-role `OPTIMIZER_`/`TARGET_` variants take precedence |
| `QWEN_CHAT_MODEL` | `qwen_chat` | Served model name for direct library use; train/eval YAML role models take precedence |
| `MINIMAX_REGION` | `minimax_chat` | Service region: `global_en` (default) or `cn_zh`; selects the base URL |
| `MINIMAX_BASE_URL` | `minimax_chat` | MiniMax-compatible base URL; overrides the region default |
| `MINIMAX_API_KEY` | `minimax_chat` | MiniMax API key |
| `COPILOT_EXEC_PATH` | `copilot_chat`, `copilot_exec` | Optional path to `copilot`; defaults to `copilot` |
| `COPILOT_EXEC_HOME` | `copilot_chat`, `copilot_exec` | Optional `COPILOT_HOME` override isolating CLI config; sign-in state lives outside it |
| `COPILOT_EXEC_ALLOW_ALL_TOOLS` | `copilot_exec` | Opt in to `--allow-all-tools` for file-edit rollouts; `false` by default |
| `COPILOT_CHAT_OPTIMIZER_MODEL` | `copilot_chat` | Optional model ID passed as `--model` for optimizer calls |
| `COPILOT_CHAT_TARGET_MODEL` | `copilot_chat` | Optional model ID passed as `--model` for target calls |
| `COPILOT_CHAT_TIMEOUT` | `copilot_chat` | Per-call timeout in seconds |
| `GOOGLE_GENAI_USE_VERTEXAI` | `gemini_chat` | `true` forces Vertex AI; `false` forces the Gemini Developer API. Unset: Vertex AI unless an API key is the only credential |
| `GOOGLE_CLOUD_PROJECT` | `gemini_chat` | Vertex AI project |
| `GOOGLE_CLOUD_LOCATION` | `gemini_chat` | Vertex AI location; defaults to `global` |
| `GEMINI_API_KEY` | `gemini_chat`, `agy_exec` | Gemini Developer API key (`GOOGLE_API_KEY` is accepted as an alias) |
| `GEMINI_MODEL` | `gemini_chat` | Model for direct library use (`OPTIMIZER_GEMINI_MODEL` / `TARGET_GEMINI_MODEL` per role); train/eval YAML role models take precedence |
| `GEMINI_THINKING_LEVEL` | `gemini_chat` | Pin `minimal`, `low`, `medium`, or `high` instead of deriving it from `reasoning_effort` |
| `GEMINI_MAX_TOKENS` / `GEMINI_TIMEOUT_SECONDS` / `GEMINI_TEMPERATURE` | `gemini_chat` | Output cap (default 65536, includes thinking), request timeout (default 600), and temperature (default: model default) |
| `AGY_EXEC_PATH` | `agy_exec` | Path to `agy`; defaults to `agy` on `PATH`, then `~/.local/bin/agy` |
| `AGY_ADC_AUTH` | `agy_exec` | Read by agy itself: `true` authenticates from Application Default Credentials |
| `AGY_EXEC_SKIP_PERMISSIONS` | `agy_exec` | Pass `--dangerously-skip-permissions` (default `true`) |
| `AGY_EXEC_SANDBOX` | `agy_exec` | Pass `--sandbox` (default `false`) |
| `AGY_EXEC_MODE` | `agy_exec` | Optional `--mode`: `accept-edits` or `plan` |
| `AGY_EXEC_EFFORT` | `agy_exec` | `--effort` for bare model slugs: `low`, `medium`, `high`, `max`, or blank |
| `AGY_CHAT_TIMEOUT` | `agy_exec` | Optimizer-call timeout in seconds (default 900) |

`OPTIMIZER_` and `TARGET_` prefixes provide per-role overrides for the
Azure, OpenAI-compatible, and Qwen variable families. See the
[Configuration Reference](../reference/config.md) for exact names.

`claude_chat` launches the installed Claude Code CLI with `claude -p`; install
and authenticate that CLI before use. Setting `ANTHROPIC_API_KEY` is one way
the CLI may authenticate, but SkillOpt does not call the Anthropic API
directly through this backend.

`cursor_exec` is a target-only benchmark harness. Install and authenticate
Cursor Agent first, then select it with `model.target_backend=cursor_exec`.
Read-only rollouts use Ask mode; artifact-producing rollouts add Cursor's
headless `--force` flag inside the benchmark workspace. SkillOpt enables the
Cursor sandbox by default and rejects file-edit rollouts if it is disabled;
read-only Ask-mode rollouts may explicitly disable it. SkillOpt does not approve
MCP servers automatically.

`copilot_chat` drives the GitHub Copilot CLI as a chat model for either role.
Because the CLI carries its own sign-in, selecting it for both roles
(`--backend copilot`) avoids separate provider API-key configuration:

```bash
copilot login
skillopt-train --config configs/searchqa/default.yaml --backend copilot
```

Chat calls use an empty tool allowlist and disable built-in MCP servers and
custom instructions, so the model sees only the prompt SkillOpt sends.
`COPILOT_ALLOW_ALL` is removed from child environments, and
`--allow-all-tools` is never passed for chat calls. `copilot_exec` is the
separate target-only harness that runs the CLI as an agent inside a benchmark
workspace; unlike the other exec harnesses it does not grant unattended tool
use unless `COPILOT_EXEC_ALLOW_ALL_TOOLS` is set.

### Three OpenAI-compatible paths

- Research, generic provider: select `openai_compatible` and use
  `OPENAI_COMPATIBLE_*`.
- Research, Azure-family compatibility mode: keep `openai_chat`, set
  `AZURE_OPENAI_AUTH_MODE=openai_compatible`, and use `AZURE_OPENAI_*`.
- SkillOpt-Sleep: run with `--backend azure_openai` and use the same
  compatibility-mode `AZURE_OPENAI_*` variables. Sleep does not read the
  research backend's role-specific variables.

## Full Reference

See [Configuration Reference](../reference/config.md) for the complete parameter list.
