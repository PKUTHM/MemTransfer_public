"""Build and audit the offline bank. Exit 2 means delivered but quality gate failed."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile

import yaml

from . import legacy_bank as legacy
from .adapter import load_dataset, require, sha256

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def selftest():
    tmp = ROOT / 'code' / 'rq3' / '.tmp'
    tmp.mkdir(exist_ok=True)
    previous = tempfile.tempdir
    try:
        tempfile.tempdir = str(tmp)
        return legacy._selftest()
    finally:
        tempfile.tempdir = previous


def build(config_path):
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    require(cfg['kind'] == 'rq3_offline_bank' and cfg['schema_version'] == 1, 'Invalid config')
    require(cfg['algorithm'] == 'handoff_20260912_unchanged' and cfg['trajectory_kind'] == 'forward_only',
            'Only the unchanged handoff forward_only algorithm is supported')
    out = (ROOT / cfg['output_dir']).resolve()
    require(out.is_relative_to(ROOT / 'code' / 'rq3') and out != ROOT / 'code' / 'rq3', 'Output must be inside code/rq3/')
    trajectories, sources = load_dataset(ROOT, ROOT / cfg['dataset_root'], ROOT / cfg['case_map'])
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        result = legacy.build(trajectories, out)
    # Sorting output only (not the algorithm) removes set/hash-seed iteration order.
    for points in result['traps'].values():
        points.sort(key=lambda p: tuple(p['pose']))
    write_json(out / 'rq3_trap_points.json', result['traps'])
    write_json(out / 'rq3_sources.json', {'schema_version': 1, 'sources': sources,
        'case_map_sha256': sha256(ROOT / cfg['case_map']),
        'algorithm_sha256': sha256(Path(legacy.__file__)),
        'adapter_sha256': sha256(ROOT / 'code/rq3/adapter.py'),
        'build_config_sha256': sha256(config_path)})
    walkable = legacy.walkable_from_trajectories(trajectories.values())
    grid = legacy.Grid(walkable)
    candidates = [t for t in trajectories.values() if t.variant in legacy.MEMORY_ELIGIBLE]
    by_cv = {(t.case_id, t.variant): t for t in trajectories.values()}
    count = 0
    with (out / 'rq3_candidates.jsonl').open('w', encoding='utf-8', newline='\n') as f:
        for cid, bank in result['bank'].items():
            target, v0 = by_cv[cid, 'V4'], by_cv[cid, 'V0']
            ref = legacy.progress_reference(target)
            selected = {entry['episode_id']: {'role': role, 'rank': entry['rank']}
                        for role, entries in bank['roles'].items() for entry in entries}
            n = 0
            for candidate in candidates:
                if candidate.case_id == cid:
                    continue
                n += 1
                score = legacy.score_candidate(target, candidate, grid, walkable, ref)
                reasons = []
                if legacy.satisfies_target_goal(candidate, target):
                    reasons.append('required_or_category_overlap')
                ratio, window = candidate.length() / max(v0.length(), 1), result['window']
                if window is not None and not 1 - window <= ratio <= 1 + window:
                    reasons.append('length_window')
                record = {'target_case': cid, 'episode_id': candidate.episode_id,
                    'source_case': candidate.case_id, 'variant': candidate.variant,
                    'trajectory_path': sources[candidate.episode_id]['trajectory_path'],
                    'c': score.c, 'm': score.m, 'trap': score.trap, 'benign': score.benign,
                    'n_shared': score.n_shared, 'n_judged': score.n_judged,
                    'length': candidate.length(), 'length_ratio': ratio,
                    'eligible': not reasons, 'exclusion_reasons': reasons,
                    'selection': selected.get(candidate.episode_id)}
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
            require(n == 396, f'{cid}: expected 396 candidates, got {n}')
            count += n
    counts = {role: sum(len(b['roles'][role]) for b in result['bank'].values()) for role in ('H', 'E', 'L')}
    means = {role: {metric: sum(r[metric] for r in result['rows'] if r['role'] == role) / max(counts[role], 1)
                    for metric in ('c', 'm', 'trap')} for role in counts}
    traps_mean = sum(map(len, result['traps'].values())) / len(result['bank'])
    checks = {'100_cases': len(result['bank']) == 100, 'H400_E300': counts['H'] == 400 and counts['E'] == 300,
              'trap_points_mean_ge_3': traps_mean >= 3,
              'H_E_c_gap_ge_0.15': means['H']['c'] - means['E']['c'] >= 0.15,
              'H_trap_mean_ge_0.05': means['H']['trap'] >= 0.05}
    quality = all(checks.values())
    summary = {'schema_version': 1, 'status': 'usable' if quality else 'quality_gate_failed',
               'legacy_report_ok': result['ok'], 'checks': checks, 'counts': counts,
               'means': means, 'trap_points_mean': traps_mean, 'candidate_rows': count,
               'length_window': result['window'], 'walkable_cells': len(walkable),
               'stage_notes': {eid: s['validation_notes'] for eid, s in sources.items() if s['validation_notes']}}
    write_json(out / 'rq3_build_summary.json', summary)
    (out / 'rq3_health_report.txt').write_text(report.getvalue(), encoding='utf-8')
    lines = ['# RQ3 Bank Build Report', '', f"Status: **{summary['status']}** (model experiment not run)", '',
             f'Input: 500 V0-V4 trajectories; candidate scoring: {count}; inferred grid: {len(walkable)} cells.', '',
             f'H={counts["H"]}, E={counts["E"]}, backup L={counts["L"]}; mean trap points={traps_mean:.3f}.', '',
             '| Check | Result |', '|---|---|']
    lines += [f'| {name} | {"PASS" if passed else "FAIL"} |' for name, passed in checks.items()]
    lines += ['', '## Original Script Health Output', '', '```text', report.getvalue().strip(), '```', '',
        '## Methodology and Limitations', '',
        '- The original filtering algorithm is preserved verbatim in legacy_bank.py; '
        'no trajectories were manually swapped and no sorting was modified.',
        '- required uses target instance names and coordinates; categories aggregates target and instance '
        'categories. Any intersection triggers exclusion. This is a conservative simplification from the '
        'original script, not a full success-condition check, and does not rule out paths that happen to '
        'pass through other goals.',
        '- BFS uses the union of V0-V4 path cells; V4 participates in offline scoring but does not enter '
        'the bank in a candidate role. V5 is not read.',
        '- Duplicate poses take the first action/goal; the original script milestone >= current step '
        'convention is retained; no task progress state is introduced.',
        '- The terminal pose is retained with a synthetic neutral Turn_Left appended for original algorithm '
        'scoring only; it is not a real action. The original algorithm length is the pose count '
        '(real action count + 1), not the action count.',
        '- H/E/L share a dedup set, in order H4->E3->L3; length feasibility only guarantees H4+E3. '
        'L shortfall is reported separately.',
        '- c*m is retained as the original script ranking score; m excludes neutral actions, so it cannot '
        'be strictly interpreted as the error-rate across all goal poses.',
        '- Same pose implies same image is a design assumption; no images are read or visual retrieval '
        'verified at this stage.',
        '- The forward_only data contains only forward moves and turns: at the same pose, V0 and H move '
        'in the same direction, while turns are NEUTRAL. Therefore the original trap_points definition '
        'requiring V0 advancing / H retreating cannot simultaneously hold; a trap count of 0 is a '
        'structural conflict between the action space and the definition, not a data read error. '
        'H nonzero m does not guarantee the existence of V0-correctable traps.',
        '- The four YAML files are pending-integration configs and must not be handed to the existing '
        'experiment runner directly. Before formal run, verify baseline order, seed, and instruction labels.', '',
        '## Data Stage Event Notes', '']
    lines += [f'- {eid}: {notes}' for eid, notes in summary['stage_notes'].items()] or ['None.']
    (out / 'rq3_bank_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(report.getvalue())
    print(f'Extended quality gate: {summary["status"]}; artifacts: {out}')
    return 0 if quality else 2


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', default='config/rq3/build_bank.yaml')
    parser.add_argument('--selftest', action='store_true')
    args = parser.parse_args()
    try:
        return selftest() if args.selftest else build((ROOT / args.config).resolve())
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f'RQ3 input error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
