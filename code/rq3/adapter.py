"""Read real forward_only paths without changing the handed-off algorithm."""
import hashlib
import json
from pathlib import Path

from . import legacy_bank as legacy

HEADINGS = dict(zip(('North', 'East', 'South', 'West'), legacy.HEADINGS))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text(path):
    """sha256 of a text file with line endings normalised to LF.

    Used for hashes recorded inside version-controlled configs. A raw byte
    hash of a checked-out text file is not portable: the same commit yields
    CRLF on a Windows checkout and LF elsewhere, so the recorded value would
    only ever match on the machine that wrote it.
    """
    data = Path(path).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pose(raw):
    x, y, heading = raw
    require(float(x).is_integer() and float(y).is_integer(), f'Non-grid pose: {raw}')
    return int(x), int(y), HEADINGS.get(heading, heading)


def goal_sets(goals):
    required, categories = set(), set()
    for goal in goals:
        if goal.get('name') == 'Return to Origin':
            continue
        if goal.get('category'):
            categories.add(goal['category'])
        instances = list(goal.get('instances', []))
        if 'instance' in goal:
            instances.append(goal['instance'])
        for item in instances:
            x, y = item['position']
            required.add(f"{item['name']}@{float(x):g},{float(y):g}")
            if item.get('category'):
                categories.add(item['category'])
    return tuple(sorted(required)), tuple(sorted(categories))


def load_real(task_path, trajectory_path, key, version):
    task, raw = read_json(task_path), read_json(trajectory_path)
    prefix = str(trajectory_path)
    for field, value in [('task_id', key['task_id']), ('case_no', key['case_no']),
                         ('legacy_case_id', key['old_case_id']), ('variant', version)]:
        require(task.get(field) == value, f'{prefix}: identity mismatch {field}')
    require(raw.get('variant_id') == f"{key['new_id']}-{version}", f'{prefix}: variant_id')
    require(task.get('task_type') == key['task_type'], f'{prefix}: task_type')
    fo = raw['forward_only']
    require(fo.get('reachable') is True, f'{prefix}: unreachable')
    poses = [pose(p) for p in fo['poses']]
    actions = [a.removesuffix(' 1m').replace('Turn_180', 'Turn_Back') for a in fo['actions']]
    require(fo['length'] == len(actions) and len(poses) == len(actions) + 1,
            f'{prefix}: action/pose length mismatch')
    require(all(p[2] in legacy.HEADINGS for p in poses), f'{prefix}: heading')
    require(all(a in legacy.TURNS or a in legacy.MOVE_DIR for a in actions), f'{prefix}: action')
    origin = task['origin']
    require(poses[0] == pose([*origin['coord'], origin['heading']]), f'{prefix}: origin mismatch')
    walkable = {(x, y) for x, y, _ in poses}
    for i, a in enumerate(actions):
        require(legacy.transition(poses[i], a, walkable) == poses[i + 1],
                f'{prefix}: transition mismatch at {i}')
    events = fo.get('stage_events', [])
    milestones = set()
    for event in events:
        i = event['step']
        require(type(i) is int and 0 <= i < len(poses), f'{prefix}: stage index')
        require(pose(event['pose']) == poses[i], f'{prefix}: stage pose mismatch')
        milestones.add(i)
    require(bool(milestones), f'{prefix}: no milestones')
    notes = []
    if fo.get('stages_completed') != fo.get('stages_total') or len(events) != fo.get('stages_total'):
        notes.append('stage_count_mismatch: retain stored events; no inferred/repaired milestones')
    required, categories = goal_sets(task['goals'])
    require(bool(required or categories), f'{prefix}: empty goal sets')
    # Legacy toy data has one action per pose. The terminal placeholder is neutral
    # under its translation-only distance metric, and is NEVER a real path action.
    steps = [legacy.Step(p, actions[i] if i < len(actions) else 'Turn_Left', i in milestones)
             for i, p in enumerate(poses)]
    tr = legacy.Trajectory(f"{key['new_id']}-{version.upper()}", key['task_id'],
                           key['new_id'], version.upper(), required, categories, steps)
    return tr, {'real_action_count': len(actions), 'scoring_pose_count': len(poses),
                'terminal_synthetic_action': 'Turn_Left', 'terminal_index': len(actions),
                'milestone_indices': sorted(milestones), 'validation_notes': notes,
                'required': list(required), 'categories': list(categories)}


def load_dataset(root, dataset_path, case_map_path):
    keys = sorted(read_json(case_map_path)['cases'], key=lambda k: (k['task_id'], k['case_no']))
    require(len(keys) == 100 and len({k['new_id'] for k in keys}) == 100, 'Expected 100 unique cases')
    require(len({k['old_case_id'] for k in keys}) == 100, 'Duplicate legacy case identity')
    trajectories, sources = {}, {}
    for key in keys:
        for version in ('v0', 'v1', 'v2', 'v3', 'v4'):
            directory = dataset_path / key['new_dir'] / version
            task_path, trajectory_path = directory / 'task.json', directory / 'trajectory.json'
            tr, info = load_real(task_path, trajectory_path, key, version)
            trajectories[tr.episode_id] = tr
            sources[tr.episode_id] = dict(info, case_id=tr.case_id, variant=tr.variant,
                memory_eligible=version != 'v4', legacy_case_id=key['old_case_id'],
                task_path=task_path.relative_to(root).as_posix(),
                trajectory_path=trajectory_path.relative_to(root).as_posix(),
                task_sha256=sha256(task_path), trajectory_sha256=sha256(trajectory_path))
    legacy.validate(trajectories)
    return trajectories, sources
