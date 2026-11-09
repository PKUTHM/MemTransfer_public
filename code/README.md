# Code Architecture

This directory contains the core runtime code for MemTransfer.

## Directory Structure

```text
code/
├── test_run.py                 One-click experiment runner (auto server + agent)
├── agent.py                    Model invocation, memory handling, action execution
├── server.py                   Scene state, action feedback, task judgment
├── experiment.py               Experiment config loading and validation
├── scene_index.py              Scene indexing and path resolution
├── metrics.py                  Metric calculation (SR, Step Ratio, SPL)
├── report.py                   Result aggregation
├── result_paths.py             Result path management
├── step_resume.py              Step recovery
├── token_probe.py              Token probing
├── prompt/                     Prompt assembly
│   ├── system.py               System rules + action vocabulary
│   ├── mission.py              Task description
│   ├── current_image.py        Current view
│   ├── history.py              History text + interleaved images
│   ├── reference_traj.py       Expert trajectory (text)
│   ├── reference_traj_images.py Expert trajectory (images)
│   ├── episodic_memory.py      Episodic memory retrieval
│   ├── vit_matcher.py          ViT visual matching
│   └── _helpers.py             Image encoding utilities
├── tests/                      Unit tests
└── rq3/                        RQ3 cross-case memory module
    ├── build_bank.py           Build candidate bank
    ├── build_replacement.py    Build replacement bank
    ├── cross_case_memory.py    Cross-case memory loading
    ├── runtime.py              RQ3 runtime entry point
    └── artifacts/              RQ3 generated data files
```

## Core Modules

### agent.py
- Calls VLM model (InternVL)
- Handles working memory and long-term memory
- Parses model output and executes actions
- Supports multiple memory mechanisms (none/full_context/episodic/summary/semantic)

### server.py
- Flask server managing scene state
- Handles action execution and task judgment
- Provides REST API endpoints
- Supports multiple scenes (V0-V5)

### experiment.py
- Loads and validates YAML configurations
- Manages experiment fingerprints
- Loads test tasks and expert trajectories
- Validates memory file SHA256

### test_run.py
- **One-click experiment runner**: automatically starts server, runs preflight checks, and launches agent
- Manages server lifecycle (start, wait for ready, stop)
- Generates `RUN_INFO.json` and `preflight.json`
- Supports experiment mode (via `--experiment_config`) and legacy mode
- Redirects server logs to correct `server_logs/` directory
- Outputs to `result/<model>/<config>/r<round>_<timestamp>/`

### scene_index.py
- Scene indexing and path resolution
- Manages V0-V5 scene mappings
- Supports base64 image lookup
- Handles scene fallback logic

### prompt/
- Assembles prompts sent to the model
- Includes system rules, task description, history, memory modules
- Supports image encoding and base64 caching

## Command Line Arguments

### agent.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--server_url` | http://127.0.0.1:5000 | Server URL |
| `--api_excel` | api.xlsx | API key file |
| `--temperature` | 0.0 | Model temperature |
| `--max_history_steps` | 5 | History steps |
| `--lookback_image_count` | 5 | History image count |
| `--max_workers` | 8 | Parallel cases |
| `--reference_traj` | none | Reference trajectory mode (none/forward_only/all_actions) |
| `--episodic_memory` | off | Enable episodic memory |
| `--episodic_memory_matcher` | pose | Matching method (pose/vit) |
| `--episodic_memory_horizon` | 6 | Episodic memory subsequent actions |

### server.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--port` | 5000 | Server port |
| `--host` | 127.0.0.1 | Server address |
| `--experiment_config` | - | Experiment config (auto-loads v4 test set) |
| `--log_dir` | - | Override server log directory |

### test_run.py

**Experiment Mode** (recommended):

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment_config` | - | Config name or path (required) |
| `--case_ids` | all 100 | Comma-separated case IDs (e.g., `0,1,2`) |
| `--round_id` | 0 | Repeat index for multiple runs |
| `--preflight_only` | off | Run offline checks only, no model calls |
| `--case_count` | 100 | Number of cases per config |
| `--max_workers` | config | Parallel cases (default from config) |
| `--api_excel` | api.xlsx | API key file path |
| `--save_prompts` | off | Save first-decision prompts |
| `--revalidate_keys` | off | Re-probe API keys instead of cache |
| `--resume_dir` | - | Resume a previous run from directory |
| `--results_dir` | - | Output directory override |

**Legacy Mode** (parameter exploration):

| Argument | Default | Description |
|----------|---------|-------------|
| `--configs` | `h=2,i=2` | Semicolon-separated configs |
| `--parallel` | off | Run configs concurrently |
| `--parallel_max` | 4 | Max parallel configs |
| `--per_config_server` | off | Separate server per config |
| `--no_server` | off | Assume server already running |

**Legacy config tokens**: `h=history_steps`, `i=image_count`, `ref=reference_traj`, `refimg=reference_traj_images`, `peer=peer_reference`, `em=episodic_memory`, `sum=summary_prompt_path`, `sem=semantic_prompt_path`

**Examples**:

```bash
# Smoke test (single case)
python code/test_run.py --experiment_config rq1/A01_working_memory_only__test_v0 --case_ids 0

# Full experiment
python code/test_run.py --experiment_config rq1/A01_working_memory_only__test_v0

# Repeat experiment 3 times
python code/test_run.py --experiment_config rq1/B01_working_memory_only__test_v4 --round_id 0
python code/test_run.py --experiment_config rq1/B01_working_memory_only__test_v4 --round_id 1
python code/test_run.py --experiment_config rq1/B01_working_memory_only__test_v4 --round_id 2

# Preflight validation only
python code/test_run.py --experiment_config rq1/A01_working_memory_only__test_v0 --preflight_only

# Legacy mode: compare parameter settings
python code/test_run.py --configs "h=5,i=5;h=5,i=5,ref=FWD,refimg=smart"
```

### metrics.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--run_dir` | Latest run | Result directory (can specify multiple) |
| `--by_task_type` | off | Output by task type |
| `--cases GROUP` | - | Output per-case results for specified group |
| `--csv` | - | Export CSV file |
| `--json` | - | Export JSON file |
| `--verify` | off | Cross-validate with report.py |

## Testing

Run unit tests:

```bash
# RQ1/RQ2 tests
python -m unittest discover -s code/tests

# RQ3 tests
python -m unittest code.rq3.tests.test_bank
python -m unittest code.rq3.tests.test_conflict_selection
```

## Server API

### GET /get_server_time
Health check, returns server status.

### POST /init_task
Start a case.

Request:
```json
{"case_id": 0, "agent_id": "agent_001"}
```

Response:
```json
{
  "status": "ready",
  "task_description": "...",
  "origin": {"coord": [0, 0], "heading": "North"},
  "current_pose": {"x": 0, "y": 0, "heading": "North"},
  "visual_observations": {"main_view": "path/to/image.png"},
  "stages_total": 2
}
```

### POST /step
Execute an action.

Request:
```json
{"agent_id": "agent_001", "action": "Move_Forward"}
```

Response:
```json
{
  "status": "success",
  "current_pose": {"x": 1, "y": 0, "heading": "North"},
  "stages_completed": 0,
  "stages_total": 2,
  "visual_observations": {"main_view": "path/to/image.png"}
}
```

Action types: `Move_Forward`, `Turn_Left`, `Turn_Right`, `Move_Backward`, `Move_Left`, `Move_Right`, `Declare_Done`.

## Conventions

- All step indices in user prompts are 1-based
- Distance unit: 1 cell = 1 meter
- Image filename format: `Grid_X{x}_Y{y}_{East|North|West|South}.png`
- Episodic memory matching: always based on current case's `forward_only` trajectory, no other sources
- Episodic memory matcher:
  - `pose` (default): filename/pose-based matching, no extra dependencies
  - `vit`: ViT-B/16 cosine similarity, requires pre-computed embeddings file

## RQ3 Cross-Case Memory Module

RQ3 uses a separate runtime entry point (`code/rq3/`) because RQ1's prompt templates assume all memory comes from the same case, making direct calls to `code/agent.py` unsafe.

### Build and Validate

```bash
python -m code.rq3.build_bank --config config/rq3/build_bank.yaml
python -m code.rq3.build_replacement --all
python -m code.rq3.validate_configs
```

### Eight Configurations and Selection Rules

D01–D04 use `low_overlap` rule (selecting by `(c, source_case, variant)` ascending); D05–D08 use `high_conflict` rule (selecting by `c*m` descending). The two groups form strict pairs, differing only in interference quality.

| Config | Mechanism | Layout | Rule |
|--------|-----------|--------|------|
| D01 / D03 | FC / Episodic | `replace1`: own V0, V1, V2 + external A | low_overlap |
| D02 / D04 | FC / Episodic | `replace2`: own V0, V1 + external B, A | low_overlap |
| D05 / D07 | FC / Episodic | `replace1` | high_conflict |
| D06 / D08 | FC / Episodic | `replace2` | high_conflict |

### Runtime

Run `--mode preflight` first to validate fingerprints, then start server and agent in separate terminals:

```bash
python -m code.rq3.runtime --config config/rq3/D01_full_context_easy__test_v4.yaml --mode preflight
python -m code.rq3.runtime --config config/rq3/D01_full_context_easy__test_v4.yaml --mode serve --port 5000
python -m code.rq3.runtime --config config/rq3/D01_full_context_easy__test_v4.yaml --mode agent --port 5000
```
