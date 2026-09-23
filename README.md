# MemTransfer: Memory Mechanism Navigation Experiment

This project enables Vision-Language Models (VLM) to perform navigation tasks based on warehouse views and historical memory, comparing the effectiveness of different memory mechanisms. The current configuration includes 26 experiments for RQ1, 5 for RQ2, and 8 for RQ3, each with 100 test cases.

## Project Overview

### Experiment Scenario

The warehouse is a 15x27 grid environment with 225 reachable cells and 48 objects (17 YCB categories). The agent starts from a designated position and navigates to the target location based on task descriptions.

### Test Scene Versions

| Version | Description |
|---------|-------------|
| V0 | Original starting position and scene |
| V1 | Same position, heading rotated 180° |
| V2 | Starting position 2-4 steps away, same heading |
| V3 | Starting position ≥6 steps away, heading ~90° different, different path |
| V4 | New starting position, same scene (for testing memory transfer) |
| V5 | New obstacles added, testing adaptability to environmental changes |

### Memory Mechanisms

| Mechanism | Description |
|-----------|-------------|
| `none` | Working memory only (baseline) |
| `full_context` | Complete expert trajectories (action sequences + keyframes) |
| `episodic` | Episodic memory: retrieves historical segments most similar to current view |
| `summary` | Summary memory: text summaries generated from historical experience |
| `semantic` | Semantic memory: rules derived from historical experience |

### Research Questions

- **RQ1**: Comparison of different memory mechanisms across test environments
  - A01-A17: Test environment V0, comparing 5 memory mechanisms × 4 memory versions
  - B01-B09: Test environment V4, validating memory transfer capability
  
- **RQ2**: Impact of environmental changes (V5 obstacles) on memory mechanisms
  
- **RQ3**: Effect of cross-case memory
  - D01-D04: Low-overlap external task memory (easy/hard)
  - D05-D08: High-conflict external task memory (easy/hard)

## Project Structure

```text
MemTransfer/
├── code/              Runtime code (see code/README.md)
├── config/            Experiment configurations (see config/README.md)
├── dataset/           Test data (see dataset/README.md)
└── result_<model>/    Experiment results (see result/README.md)
```

## Installation

```bash
# Python 3.10+, virtual environment recommended
python3 -m venv .venv
source .venv/bin/activate
pip install flask requests pandas openpyxl pillow pyyaml
```

Prepare API key file `api.xlsx`, fill column C with InternVL keys (one per row, e.g., `sk-...`).

## Quick Start

### 1. Offline Check

```bash
python code/agent.py --experiment_config rq1/A01_working_memory_only__test_v0 --preflight_only
```

### 2. Run Experiment

```bash
# Run single case
python code/agent.py --experiment_config rq1/A01_working_memory_only__test_v0 --case_ids 0

# Run all 100 cases
python code/agent.py --experiment_config rq1/A01_working_memory_only__test_v0
```

### 3. View Results

```bash
# Calculate metrics
python code/metrics.py

# View per-case results
python code/metrics.py --cases A01
```

## Modifying Experiment Configurations

Edit the corresponding YAML in `config/rq1/`, `config/rq2/`, or `config/rq3/`. Each file is self-contained; modifying one group does not automatically modify others.

Main configuration fields:

| Field | Description |
|-------|-------------|
| `experiment_id`, `description` | Experiment identifier and description |
| `test_version` | Test version (v0/v4/v5) |
| `memory`, `versions` | Memory mechanism and historical versions |
| `working_memory` | History steps and image count |
| `model` | Model name, API endpoint, temperature, etc. |
| `execution` | Concurrency, timeout, retry settings |

See [config/README.md](config/README.md) for detailed configuration.

## Repeating Experiments

When the same configuration needs to be tested three times, specify `--round_id`:

```bash
python code/agent.py --experiment_config rq1/B01_working_memory_only__test_v4 --round_id 0
python code/agent.py --experiment_config rq1/B01_working_memory_only__test_v4 --round_id 1
python code/agent.py --experiment_config rq1/B01_working_memory_only__test_v4 --round_id 2
```

## RQ3 Execution

RQ3 uses a separate entry point:

```bash
# Build candidate bank
python -m code.rq3.build_bank --config config/rq3/build_bank.yaml

# Build replacement bank
python -m code.rq3.build_replacement --all

# Validate configurations
python -m code.rq3.validate_configs

# Run experiment
python -m code.rq3.runtime --config config/rq3/D01_full_context_easy__test_v4.yaml
```

## Troubleshooting

- **`ModuleNotFoundError: No module named 'pandas'`** — Run `pip install pandas openpyxl pillow pyyaml`
- **`port 5000 already in use`** — Another server is running. Use `pkill -9 -f "code/server.py"` or change port
- **All cases 0% with `failed_timeout`** — InternVL endpoint unreachable or keys invalid. Check server logs
- **`Connection refused`** — Server not started. Start manually: `python code/server.py --port 5000`
- **SHA256 mismatch** — Memory files modified. Update `memory_files.*.sha256` in config

## Detailed Documentation

- [code/README.md](code/README.md) - Code architecture, CLI arguments, server API
- [config/README.md](config/README.md) - Experiment configuration details
- [dataset/README.md](dataset/README.md) - Dataset structure
- [result/README.md](result/README.md) - Result directory structure and metrics
