<div align="center">

# MemTransfer: Benchmarking Memory Beyond Matched Experience in Embodied Decision-Making

[![arXiv](https://img.shields.io/badge/arXiv-2609.32313-b31b1b.svg)](https://arxiv.org/pdf/2609.32313)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE-CODE)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Haiming Tang<sup>1\*</sup>, Xianjie Dai<sup>2\*</sup>, Gujie Shao<sup>1</sup>, Zuyi Guo<sup>2</sup>, Jingguang Li<sup>2</sup>, Kailang Ma<sup>2</sup>, Yihong Tang<sup>3</sup>, Heye Huang<sup>2†</sup>

<sup>1</sup> National University of Singapore  
<sup>2</sup> Korea Advanced Institute of Science and Technology  
<sup>3</sup> McGill University  

*\*Equal contribution, †Corresponding author*

</div>

---

## Project Overview

This project enables Vision-Language Models (VLM) to perform navigation tasks based on warehouse views and historical memory, comparing the effectiveness of different memory mechanisms. The current configuration includes 26 experiments for RQ1, 5 for RQ2, and 8 for RQ3, each with 100 test cases.

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
└── result/            Experiment results (see result/README.md)
```

## Installation

```bash
# Python 3.10+
pip install flask requests pandas openpyxl pillow pyyaml
```

## API Key File

Place `api.xlsx` in the project root directory (`MemTransfer/api.xlsx`).

- **Column A**: API keys, one per row (e.g., `sk-xxx`)
- Other columns (B, C, ...) are optional (e.g., email, password)
- Missing `api.xlsx` → `RuntimeError: Failed to load API keys`

## Quick Start

### 0. Download Dataset

Download the dataset from [Google Drive](https://drive.google.com/file/d/1LRqrAgpea6tRJNkKRC_hJktxTXxv4Dmg/view?usp=sharing) and extract to `dataset/`:

```bash
# After downloading the archive
cd dataset
unzip <archive_name>.zip
mv dataset/* . && rmdir dataset
```

Verify: `dataset/v0-v5/` and `dataset/memory/` should exist.

### 1. Run Experiment

Use `test_run.py` — it automatically starts the server, runs preflight checks, and launches the agent:

```bash
# Run single case (smoke test)
python code/test_run.py --experiment_config rq1/A01_working_memory_only__test_v0 --case_ids 0

# Run all 100 cases
python code/test_run.py --experiment_config rq1/A01_working_memory_only__test_v0

# Repeat experiment 3 times (for statistical significance)
python code/test_run.py --experiment_config rq1/B01_working_memory_only__test_v4 --round_id 0
python code/test_run.py --experiment_config rq1/B01_working_memory_only__test_v4 --round_id 1
python code/test_run.py --experiment_config rq1/B01_working_memory_only__test_v4 --round_id 2

# Preflight only (no model calls, just data validation)
python code/test_run.py --experiment_config rq1/A01_working_memory_only__test_v0 --preflight_only
```

See [code/README.md](code/README.md#test_runpy) for all parameters.

### 2. Manual Start (Alternative)

If you need to start server and agent separately:

```bash
# Terminal 1: start server (MUST include --experiment_config)
python code/server.py --port 5000 --experiment_config rq1/A01_working_memory_only__test_v0

# Terminal 2: run agent
python code/agent.py --experiment_config rq1/A01_working_memory_only__test_v0 --case_ids 0
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

- **`FileNotFoundError: dataset/v0-v5/...`** — Dataset not downloaded or not extracted. Download from [Google Drive](https://drive.google.com/file/d/1LRqrAgpea6tRJNkKRC_hJktxTXxv4Dmg/view?usp=sharing) and extract to `dataset/`. The directory should contain `dataset/v0-v5/` and `dataset/memory/`
- **`FileNotFoundError: dataset/warehouse_01_01/metadata.json`** — Server started without `--experiment_config`. Use: `python code/server.py --port 5000 --experiment_config <config_name>`
- **`Connection refused`** — Server not started. Start in a separate terminal: `python code/server.py --port 5000 --experiment_config <config_name>`
- **`connection timed out` / `cannot reach server`** — Agent started before server was ready. Wait a few seconds and retry, or use the one-liner with `sleep 3`
- **`API keys validated keys` but no experiment output** — Keys are cached but server wasn't running. Check server terminal for errors
- **`port 5000 already in use`** — Another server is running. Use `pkill -9 -f "code/server.py"` or change port
- **All cases 0% with `failed_timeout`** — InternVL endpoint unreachable or keys invalid. Check server logs
- **`No sk- keys in column A`** — API keys must be in column A of `api.xlsx` (not column C)
- **SHA256 mismatch** — Memory files modified. Update `memory_files.*.sha256` in config

## Detailed Documentation

- [code/README.md](code/README.md) - Code architecture, CLI arguments, server API
- [config/README.md](config/README.md) - Experiment configuration details
- [dataset/README.md](dataset/README.md) - Dataset structure
- [result/README.md](result/README.md) - Result directory structure and metrics

## License

This project utilizes a dual-licensing model:

- **Source Code**: Licensed under the [MIT License](LICENSE-CODE).
- **Datasets, Documentation, and Other Media**: Licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC-BY 4.0).
