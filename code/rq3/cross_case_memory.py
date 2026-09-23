"""RQ3 cross-case memory adapter around the unchanged RQ1 experiment code."""
import json
import sys
from pathlib import Path

import yaml

from .adapter import require, sha256
from .build_bank import ROOT
from .build_replacement import OUTPUT, SOURCES
from .validate_configs import RULE_BY_OUTPUT
from .validate_configs import validate

CODE = ROOT / 'code'


def imports():
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import experiment  # noqa: PLC0415
    return experiment


def config_from_rq3(config_path):
    """Parse RQ3 YAML using the existing strict ExperimentConfig type.

    RQ3-only fields are validated separately and removed before parsing.
    The bank/source hashes enter experiment_id, hence the server/client
    fingerprint and resume identity. This prevents reusing a result after
    changing the selected source trajectories.
    """
    ex = imports()
    path = Path(config_path).resolve()
    require(path.parent == (ROOT / 'config/rq3').resolve(), 'Config must be in config/rq3')
    validate()
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    allowed = set(ex.TOP_LEVEL_KEYS)
    filtered = {key: value for key, value in raw.items() if key in allowed}
    bank_path = raw.get('replacement_bank', {}).get('path', OUTPUT)
    require(bank_path in RULE_BY_OUTPUT, f'Unknown replacement bank: {bank_path}')
    identity_files = [ROOT / bank_path, ROOT / SOURCES]
    import hashlib
    h = hashlib.sha256()
    for file in identity_files:
        h.update(file.read_bytes())
    filtered['experiment_id'] += '__mem_' + h.hexdigest()[:16]
    return ex.ExperimentConfig(filtered, str(path))


def make_config_loader():
    """An experiment.load_config replacement scoped to the RQ3 process."""
    ex = imports()
    original = ex.load_config

    def load(path):
        p = Path(path).resolve()
        if p.parent == (ROOT / 'config/rq3').resolve():
            return config_from_rq3(p)
        return original(path)
    return load


def load_case_history(cfg, target_key, slots, sources, names=None):
    """Return four existing HistoricalTrajectory objects in declared order."""
    ex = imports()
    loaded, manifest = [], []
    for slot in slots:
        eid = slot['episode_id']
        source = sources[eid]
        source_key = cfg.case_map.by_legacy(source['legacy_case_id'])
        version = source['variant'].lower()
        require(source_key.task_id == source['case_id'].split('-')[0] and
                source_key.case_no == int(source['case_id'].split('case')[1]),
                f'{eid}: source case map mismatch')
        traj = ex.load_history_trajectory(cfg, source_key, version, base64_names=names)
        require(Path(traj.source_path).resolve() == (ROOT / source['trajectory_path']).resolve(),
                f'{eid}: trajectory path mismatch')
        require(traj.source_sha256 == source['trajectory_sha256'] and
                sha256(ROOT / source['task_path']) == source['task_sha256'],
                f'{eid}: source hash mismatch')
        require((source_key.legacy_case_id == target_key.legacy_case_id) ==
                (slot['role'] == 'own'), f'{eid}: own/foreign role mismatch')
        prompt = traj.as_prompt_traj()
        prompt['_rq3_episode_id'] = eid
        prompt['_rq3_role'] = slot['role']
        prompt['_rq3_slot'] = slot['slot']
        loaded.append(prompt)
        manifest.append({'target_legacy_case_id': target_key.legacy_case_id,
                         'slot': slot['slot'], 'role': slot['role'],
                         'episode_id': eid, 'source_legacy_case_id': source_key.legacy_case_id,
                         'source_task_id': source_key.task_id, 'version': version,
                         'trajectory_path': source['trajectory_path'],
                         'trajectory_sha256': traj.source_sha256,
                         'length': traj.length})
    require([p['_rq3_slot'] for p in loaded] == [1, 2, 3, 4], 'Slot order mismatch')
    return loaded, manifest


def preload(cfg, group, resolved=None, names=None):
    """Load 100 x four prompt-shaped histories and complete source manifest."""
    require(group.memory in ('full_context', 'episodic'), 'RQ3 requires trajectory memory')
    if resolved is None:
        resolved = validate()
    spec_id = cfg.raw['experiment_id'].split('__mem_')[0]
    require(spec_id in resolved, f'Unknown RQ3 experiment: {spec_id}')
    spec = resolved[spec_id]
    sources = json.loads((ROOT / SOURCES).read_text(encoding='utf-8'))['sources']
    if names is None:
        names = imports().scene_base64_names(cfg)
    history, entries = {}, []
    for key in cfg.case_map.all_keys():
        cid = f'{key.task_id}-case{key.case_no:02d}'
        loaded, manifest = load_case_history(cfg, key, spec['cases'][cid], sources, names)
        history[key.legacy_case_id] = loaded
        entries.extend(manifest)
    require(len(history) == 100 and len(entries) == 400, 'History coverage mismatch')
    bank_path = spec.get('bank', OUTPUT)
    manifest = {'memory': group.memory, 'versions': list(group.versions),
                'rq3_replacement_bank_path': bank_path,
                'rq3_replacement_bank_sha256': sha256(ROOT / bank_path),
                'rq3_source_manifest_sha256': sha256(ROOT / SOURCES),
                'trajectories': entries}
    return history, manifest


def mixed_scope_block(ctx, trajectories, test_version):
    task_id, case_no = ctx.case_label
    lines = ['\n[MEMORY SCOPE — MIXED CASES]',
             f'Current test target: {test_version.upper()} / {task_id} / case{int(case_no):02d}.',
             'The four memory slots include records from this case and from other tasks.',
             'Each historical record has its own task, case, origin and goal; read its label before using its actions.']
    for traj in trajectories:
        lines.append(f"Slot {traj['_rq3_slot']}: {traj['case_label']} / {traj['version'].upper()}"
                     f" / {'current case' if traj['_rq3_role'] == 'own' else 'other task'}.")
    return {'type': 'text', 'text': '\n'.join(lines) + '\n'}


def correct_prompt_blocks(front, back, ctx, trajectories, test_version):
    """Replace the RQ1 same-case claims while keeping actions/images intact."""
    if not back:
        # Episodic returns no memory blocks when no valid match exists.
        return front, back
    scope_found = False
    for i, block in enumerate(back):
        if block.get('type') != 'text':
            continue
        message = block.get('text', '')
        if '[MEMORY SCOPE — SAME CASE]' in message:
            back[i] = mixed_scope_block(ctx, trajectories, test_version)
            scope_found = True
            continue
        message = message.replace('one contiguous excerpt from this case\'s record',
                                  'one contiguous excerpt from the historical case named below')
        message = message.replace('1 historical trajectory of this same case',
                                  '1 trajectory from the historical case named below')
        message = message.replace('completed this same case starting from',
                                  'completed its own historical task starting from')
        block['text'] = message
    require(scope_found, 'RQ1 scope marker changed; cannot safely relabel prompt')
    require(not any('this same case' in b.get('text', '').lower() for b in back if b.get('type') == 'text'),
            'Uncorrected same-case claim in memory prompt')
    return front, back


def install_agent(agent_module):
    """Patch only the current RQ3 agent process, leaving code/ unchanged."""
    ex = imports()
    ex.load_config = make_config_loader()
    original_blocks = agent_module.experiment_memory_blocks

    def preload_hook(cfg, group):
        history, manifest = preload(cfg, group)
        agent_module.EXPERIMENT_MEMORY_TEXT = ''
        agent_module.EXPERIMENT_HISTORY = history
        agent_module.EXPERIMENT_MANIFEST = manifest
        return manifest

    def blocks_hook(group, trajectories, current_img_path, ctx, history_image_count, telemetry):
        front, back = original_blocks(group, trajectories, current_img_path,
                                      ctx, history_image_count, telemetry)
        telemetry['rq3_sources'] = [{'slot': t['_rq3_slot'], 'role': t['_rq3_role'],
                                     'episode_id': t['_rq3_episode_id']} for t in trajectories]
        match = telemetry.get('episodic_match')
        if match is not None:
            matched = trajectories[match['traj_pos']]
            match['rq3_episode_id'] = matched['_rq3_episode_id']
            match['rq3_source_role'] = matched['_rq3_role']
        return correct_prompt_blocks(front, back, ctx, trajectories,
                                     agent_module.EXPERIMENT.test_version)

    agent_module._preload_experiment_memory = preload_hook
    agent_module.experiment_memory_blocks = blocks_hook


def install_server():
    ex = imports()
    ex.load_config = make_config_loader()
