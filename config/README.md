# Experiment Configuration

This directory contains all experiment YAML configuration files.

## Directory Structure

```text
config/
├── rq1/                    # RQ1 experiments: 26 groups
│   ├── A01-A17             # Test environment V0
│   └── B01-B09             # Test environment V4
├── rq2/                    # RQ2 experiments: 5 groups (test V5)
└── rq3/                    # RQ3 experiments: 8 groups (cross-case memory)
```

## Configuration File Description

Each YAML file corresponds to one experiment group. Files are self-contained with all parameters.

### File Naming Convention

`<number>_<slug>__test_<test_version>.yaml`

- Number: Experiment group identifier (e.g., A01, B03)
- Slug: Experiment description
- Test version: v0/v4/v5

### Main Configuration Fields

| Field | Description |
|-------|-------------|
| `experiment_id` | Experiment identifier, format: `rq1_<filename>` |
| `description` | Experiment description |
| `test_version` | Test version (v0/v4/v5) |
| `dataset_root` | Dataset root directory |
| `case_map` | Case ID mapping file path |
| `scene` | Scene configuration (metadata, image_base64_dir, etc.) |
| `memory` | Memory mechanism (none/full_context/episodic/summary/semantic) |
| `versions` | List of memory versions used (e.g., [v0, v1, v2, v3]) |
| `working_memory` | Working memory settings (history_steps, history_images) |
| `max_steps` | Maximum steps per case (default 200) |
| `model` | Model configuration (name, api_url, temperature, etc.) |
| `execution` | Execution configuration (concurrency, timeout, retry, etc.) |
| `memory_files` | Memory file paths and SHA256 |

### Example Configuration

```yaml
schema_version: 1
experiment_id: rq1_A01_working_memory_only__test_v0
description: "RQ1 1A-1 | Working Memory only | memory: none | test: V0"
ref_model: Intern-S2-Preview-397B

test_version: v0

dataset_root: dataset/v0-v5
case_map: dataset/v0-v5/case_id_map.json

scene:
  image_base64_dir: dataset/v0-v5/scene_v0-v4/image_base64
  metadata: dataset/v0-v5/scene_v0-v4/metadata.json

working_memory:
  history_images: 6
  history_steps: 6

max_steps: 200
max_prompt_images: 50

model:
  name: intern-s2-preview-397b
  api_url: https://chat.intern-ai.org.cn/api/v1/chat/completions
  temperature: 0.0
  max_tokens: 30

execution:
  max_workers: 36
  max_api_attempts_per_step: 5

method: working_memory_only
memory: none
versions: []
```

## RQ1 Experiment Matrix

### Experiment 1A (Test Environment V0)

| ID | Mechanism | Memory Version |
|----|-----------|----------------|
| A01 | Working Memory only | — |
| A02-A05 | Full Context | V0 / V1 / V2 / V3 |
| A06-A09 | Episodic | V0 / V1 / V2 / V3 |
| A10-A13 | Summary | V0 / V1 / V2 / V3 |
| A14-A17 | Semantic | V0 / V1 / V2 / V3 |

### Experiment 1B (Test Environment V4)

| ID | Mechanism | Memory Version |
|----|-----------|----------------|
| B01 | Working Memory only | — |
| B02-B05 | Full Context | V0 / V0-V3 / V0-V3 / V0-V3 |
| B06-B09 | Episodic | V0 / V0-V3 / V0-V3 / V0-V3 |

## RQ2: V5 Route Blocking Experiment (5 groups)

RQ2 tests the impact of environmental changes (obstacles) on memory mechanisms. All five groups enable `prompt.obstacle_notice: true`, with task descriptions indicating new obstacles in the warehouse and that historical routes may be invalid. Except for the V5 environment notice, memory and runtime settings follow the corresponding RQ1 V4 configurations.

| ID | Method | Corresponding RQ1 Config | Historical Memory |
|----|--------|-------------------------|-------------------|
| R02 | Working Memory only | B01 | None |
| R05 | Budgeted Full Context | B03 | Current case's V0–V3 four trajectories, max 10 reference images each, total 40 |
| R08 | Episodic | B05 | Current case's V0–V3 trajectories, using original retrieval settings |
| R11 | Summary | B07 | Using V0–V3 respective generated Summary files |
| R14 | Semantic | B09 | Using V0–V3 respective generated Semantic files |

Summary / Semantic do not re-merge 400 trajectories to generate new memory, keeping corresponding V4 configuration files and SHA256 unchanged. Test scenes are specified by each `v5/task.json`, loader automatically loads corresponding obstacle scenes.

## RQ3: Cross-Case Memory Experiment (8 groups)

RQ3 tests the effect of external task memory. D01–D04 use low-overlap (`low_overlap`) interference trajectories, D05–D08 use high-conflict (`high_conflict`) interference trajectories.

| ID | Mechanism | Layout | Selection Rule |
|----|-----------|--------|----------------|
| D01 | Full Context | Own V0, V1, V2 + External A | low_overlap |
| D02 | Full Context | Own V0, V1 + External B, A | low_overlap |
| D03 | Episodic | Own V0, V1, V2 + External A | low_overlap |
| D04 | Episodic | Own V0, V1 + External B, A | low_overlap |
| D05 | Full Context | Own V0, V1, V2 + External A | high_conflict |
| D06 | Full Context | Own V0, V1 + External B, A | high_conflict |
| D07 | Episodic | Own V0, V1, V2 + External A | high_conflict |
| D08 | Episodic | Own V0, V1 + External B, A | high_conflict |

RQ3 configurations use a separate runtime entry point `code/rq3/runtime.py`, see [code/README.md](../code/README.md#rq3-cross-case-memory-module) for details.

## Modifying Configurations

### Self-Contained Principle

- Each YAML file contains all parameters, no `extends` used
- Modifying one group does not affect others
- RQ1's 26 configurations share some parameters (data, scene, model, etc.), keep consistent when modifying

### Common Modifications

1. **Change memory mechanism**
   ```yaml
   memory: episodic  # none / full_context / episodic / summary / semantic
   versions: [v0, v1, v2, v3]
   ```

2. **Change working memory**
   ```yaml
   working_memory:
     history_steps: 6
     history_images: 6
   ```

3. **Change model settings**
   ```yaml
   model:
     name: intern-s2-preview-397b
     temperature: 0.0
     max_tokens: 30
   ```

4. **Change execution parameters**
   ```yaml
   execution:
     max_workers: 36  # concurrent cases
     max_api_attempts_per_step: 5
   ```

### Validating Configurations

After modifying configurations, run offline check:

```bash
python code/agent.py --experiment_config <config_name> --preflight_only
```

## Configuration Fingerprint

Each configuration generates a fingerprint for:
- Determining if two runs are the same experiment
- Automatically skipping completed cases
- Preventing mixing results from different configurations

Fingerprint includes:
- experiment_id
- test_version
- model settings
- working_memory settings
- memory mechanism and versions
- memory file SHA256

Does not include:
- execution parameters (concurrency, timeout, etc.)

## Adding New Experiments

1. Copy a similar configuration file
2. Rename file (follow naming convention)
3. Update `experiment_id` and `description`
4. Modify memory settings
5. Run offline check to validate

```bash
python code/agent.py --experiment_config <new_config_name> --preflight_only
```

## Notes

- After modifying memory file content, must update SHA256 in configuration
- Model, working memory, and retry settings should remain consistent within the same round
- RQ1 configuration tests verify shared parameters across 26 groups are consistent
- V5 tests automatically load scene info from each case's `task.json`
