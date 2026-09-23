# Experiment Results

This directory stores MemTransfer experiment run results.

## Directory Structure

```text
result_<model_name>/
└── <experiment_config_name>/
    └── r<round>_<timestamp>/
        ├── RUN_INFO.json           Run configuration and fingerprint
        ├── preflight.json          Offline check results
        ├── client_*/               Case results, step-by-step responses, and action feedback
        ├── client_logs/            Client logs
        └── server_logs/            Server logs
```

## Result File Description

### RUN_INFO.json
Records run configuration information:
- `experiment_id`: Experiment identifier
- `config_fingerprint`: Configuration fingerprint (for determining if same experiment)
- `model`: Model information
- `test_version`: Test version (v0/v4/v5)
- `group`: Experiment group information

### preflight.json
Offline check results, including:
- Configuration validation results
- Memory file SHA256 verification
- Scene path checks

### client_*/
Per experiment group client result directory, containing:
- `SUMMARY.json`: Summary of group run
- `case_*_<case_id>_a<attempt>.json`: Detailed results per case
  - `success`: Whether task completed successfully
  - `actions_executed`: Number of actions executed
  - `final_status`: Final status
  - `steps`: Model responses and action feedback per step

### client_logs/
Client runtime logs, recording:
- API call status
- Error and retry information
- Performance statistics

### server_logs/
Server logs, containing:
- Scene state changes
- Action execution records
- Task judgment results

## Evaluation Metrics

Use `code/metrics.py` to calculate experiment metrics.

### Metric Definitions

For each case *i*:
- `S_i` = 1 if task completed (`task_completed`), otherwise 0
- `p_i` = number of actions agent actually executed (including rejected actions)
- `l_i` = optimal step count from expert trajectory `all_actions`

| Metric | Formula | Description |
|--------|---------|-------------|
| **SR** (Success Rate) | `(1/N) × Σ S_i` | Success rate, higher is better |
| **Step Ratio** | `mean(successful cases' p_i / l_i)` | Step efficiency, lower is better, 1 = optimal |
| **SPL** | `(1/N) × Σ (S_i × l_i / max(p_i, l_i))` | Combined metric, higher is better, max 1 |

### Notes

- `p_i` includes all actions accepted by server (including `blocked` and `out_of_range`)
- Step Ratio is averaged only on successful cases, be cautious comparing groups with large SR differences
- SPL comprehensively considers success rate and path efficiency, suitable for comparing groups with large SR differences

### Calculation Methods

```bash
# Calculate metrics for latest run
python code/metrics.py

# Specify run directory
python code/metrics.py --run_dir result_<model>/<config>/r<round>_<timestamp>

# Output by task type
python code/metrics.py --by_task_type

# Output per-case results for specified group
python code/metrics.py --cases A01

# Export CSV and JSON
python code/metrics.py --csv metrics.csv --json metrics.json

# Cross-validate with report.py
python code/metrics.py --verify
```

## Viewing Results

### View Summary Results

```bash
# View group's SUMMARY.json
cat result_<model>/<config>/r<round>_<timestamp>/client_*/SUMMARY.json | python -m json.tool

# View run information
cat result_<model>/<config>/r<round>_<timestamp>/RUN_INFO.json | python -m json.tool
```

### View Single Case Results

```bash
# View case 0 results
cat result_<model>/<config>/r<round>_<timestamp>/client_*/case_*_000_a*.json | python -m json.tool
```

### View Logs

```bash
# View client logs
tail -f result_<model>/<config>/r<round>_<timestamp>/client_logs/*.log

# View server logs
tail -f result_<model>/<config>/r<round>_<timestamp>/server_logs/*.log
```

## Resuming Runs

If experiment interrupted, use `--resume_dir` to continue:

```bash
python code/agent.py --experiment_config <config_name> --resume_dir "result_<model>/<config>/r<round>_<timestamp>"
```

System automatically skips completed cases, only runs incomplete ones.

## Result Comparison

Compare results from different experiment groups:

```bash
# Export metrics from multiple groups to CSV
python code/metrics.py --run_dir result_<model>/<config1>/r0_* --run_dir result_<model>/<config2>/r0_* --csv comparison.csv

# Export JSON format
python code/metrics.py --run_dir result_<model>/<config1>/r0_* --run_dir result_<model>/<config2>/r0_* --json comparison.json
```

## Troubleshooting

- **Result directory empty** — Check if experiment ran correctly, view log files
- **Metric calculation error** — Confirm result directory structure complete, RUN_INFO.json exists
- **SHA256 mismatch** — Memory files modified, need to update hash values in configuration or rerun experiment
