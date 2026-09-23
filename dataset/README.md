# Dataset Structure

This directory contains all test data required for MemTransfer experiments.

Dataset download: https://osf.io/2q943/overview?view_only=f167a152660247b098ad5b817409b778

## Directory Structure

```
dataset/
├── v0-v5/                    # Test tasks and expert trajectories
│   ├── T01-T10/              # 10 task types, 10 cases each
│   │   └── case01/           # Each case contains 6 versions (v0-v5)
│   │       ├── v0/           # Original starting position and scene
│   │       ├── v1/           # Same position, heading rotated 180°
│   │       ├── v2/           # Starting position 2-4 steps away, same heading
│   │       ├── v3/           # Starting position ≥6 steps away, heading ~90° different
│   │       ├── v4/           # New starting position, same scene (for memory transfer testing)
│   │       └── v5/           # New obstacles added (for environmental change adaptability testing)
│   ├── scene_v0-v4/          # Base warehouse scene shared by V0-V4
│   │   ├── image_base64/     # Pre-encoded base64 image cache
│   │   ├── image_embeddings.npz  # ViT visual features
│   │   ├── metadata.json     # Scene metadata (reachable points, object positions)
│   │   └── tasks.json        # Task definitions
│   ├── scenes_v5/            # V5 obstacle scenes (38 different obstacle configurations)
│   │   └── obs_*/            # Each obstacle scene contains metadata and image_base64
│   └── case_id_map.json      # Case ID mapping table
└── memory/                   # Long-term memory files
    ├── v0/                   # Original version memory
    │   ├── summary_memory/summary.md
    │   └── semantic_memory/memory.md
    ├── v1/                   # Same position, rotated heading version (same structure as v0)
    ├── v2/                   # Nearby starting position version (same structure as v0)
    └── v3/                   # Distant starting position version (same structure as v0)
```

## Test Version Description

| Version | Starting Position Change | Scene Change | Purpose |
|---------|-------------------------|--------------|---------|
| v0 | Original starting position | None | Baseline test |
| v1 | Same position, heading rotated 180° | None | Test heading change adaptability |
| v2 | 2-4 steps away, same heading | None | Test nearby starting position transfer |
| v3 | ≥6 steps away, heading ~90° different | None | Test distant starting position transfer |
| v4 | Completely new starting position | None | Test memory transfer capability |
| v5 | Same as v4 starting position | New obstacles added | Test environmental change adaptability |

## Memory Mechanisms

Each version (v0-v3) contains two types of memory:
- **summary_memory**: Summary memory, text summaries generated from historical experience
- **semantic_memory**: Semantic memory, rules derived from historical experience

## Task Types

10 task types (T01-T10), 10 cases each:
1. Single target retrieval
2. Collect all specified objects
3. Single category retrieval
4. Collect all category objects
5. Multiple object retrieval (any order)
6. Multiple object retrieval (specified order)
7. Retrieve then return to start
8. Collect then return
9. A→B→A loop
10. A→B→Start

## Expert Trajectories

Each case's trajectory.json contains two expert trajectories:
- **forward_only**: Only forward movement and turns, used for runtime prompts
- **all_actions**: Optimal path, allows movement in any direction, used for scoring metrics

## Scene Structure

Base scene is a 15x27 grid containing:
- 225 reachable cells
- 48 objects (17 YCB categories)
- 900 view images (4 headings per cell)

V5 scenes add obstacles to the base scene, each obstacle scene reduces reachable cells by 2 (223 cells).
