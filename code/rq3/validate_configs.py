"""Validate the RQ3 replacement specs and resolve their source paths.

Two things vary across the specs and both are checked here:

* **layout** - how many of the four slots are foreign.
  ``replace1`` keeps the case's own V0/V1/V2 and replaces V3 (foreign A);
  ``replace2`` keeps V0/V1 and replaces V2+V3 (foreign B, A).
  The legacy file names say ``_easy__`` / ``_hard__``; those are layout names,
  not distractor difficulty (see rq3/README.md).

* **rule** - how the foreign pair was chosen, taken from the bank the spec
  points at.  ``low_overlap`` is the original zero-overlap pair;
  ``high_conflict`` is the standard card's H (top ``c*m``).

A spec is only accepted if re-deriving its bank with its own rule reproduces
the bank file exactly, so the selection can never silently drift from the rule
a spec claims to use.
"""
import json
from collections import defaultdict

import yaml

from .adapter import read_json, require, sha256, sha256_text
from .build_bank import ROOT, write_json
from .build_replacement import AUDIT, RULES, SOURCES, load_candidates, pick_two

REPLACE1 = [{'slot': 1, 'own_version': 'v0'}, {'slot': 2, 'own_version': 'v1'},
            {'slot': 3, 'own_version': 'v2'}, {'slot': 4, 'foreign_role': 'A'}]
REPLACE2 = [{'slot': 1, 'own_version': 'v0'}, {'slot': 2, 'own_version': 'v1'},
            {'slot': 3, 'foreign_role': 'B'}, {'slot': 4, 'foreign_role': 'A'}]
LAYOUTS = {'replace1': REPLACE1, 'replace2': REPLACE2}
# legacy names kept so D01-D04 file names and returned metadata do not change
STEM_LAYOUT = {'_easy__': 'replace1', '_hard__': 'replace2',
               '_conflict1__': 'replace1', '_conflict2__': 'replace2'}
LEGACY_DIFFICULTY = {'replace1': 'easy', 'replace2': 'hard'}
BASELINES = {'full_context': 'B03_budgeted_full_context_v0-v3__test_v4.yaml',
             'episodic': 'B05_episodic_v0-v3__test_v4.yaml'}
CHECKS = ['replacement_selection_valid', 'baseline_slot_order_verified',
          'baseline_seed_verified', 'reference_instruction_labels_reviewed']
# 'execution' is explicitly outside ExperimentConfig.fingerprint() (concurrency,
# timeouts, save_prompts change nothing the model sees), so an RQ3 spec is allowed
# to differ from its baseline there and only there.
NON_EXPERIMENT_KEYS = ('experiment_id', 'description', 'execution')
RULE_BY_OUTPUT = {spec['output']: name for name, spec in RULES.items()}


def bank_block(rule):
    spec = RULES[rule]
    return {'path': spec['output'], 'audit': spec['audit'],
            'summary': spec['summary'], 'sources': SOURCES}


def check_bank(root, rule, grouped, sources):
    """Re-derive one bank under its own rule and check every entry."""
    spec = RULES[rule]
    selection = read_json(root / spec['output'])
    require(selection['schema_version'] == 1 and len(selection['cases']) == 100,
            f'{rule}: replacement bank coverage')
    require(selection['selection_rule'] == spec['selection_rule'], f'{rule}: selection rule text')
    require(selection['candidate_audit_path'] == AUDIT and
            selection['candidate_audit_sha256'] == sha256(root / AUDIT), 'Candidate audit changed')
    require(selection['sources_path'] == SOURCES and
            selection['sources_sha256'] == sha256(root / SOURCES), 'Source manifest changed')
    for cid, record in selection['cases'].items():
        expected = pick_two(grouped[cid], cid, rule)
        require([record[role]['episode_id'] for role in ('A', 'B')] ==
                [row['episode_id'] for row in expected],
                f'{cid}: selection does not match rule {rule}')
        for role in ('A', 'B'):
            entry = record[role]
            source = sources[entry['episode_id']]
            if rule == 'low_overlap':
                require(entry['c'] == 0, f'{cid}: expected zero-overlap {role}')
            require(entry['source_case'] == source['case_id'] and
                    entry['task_id'] != cid.split('-')[0], f'{cid}: invalid foreign {role}')
            require(source['memory_eligible'] and source['variant'] in ('V0', 'V1', 'V2', 'V3'),
                    f'{cid}: V4/V5 memory leakage')
        require(record['A']['source_case'] != record['B']['source_case'], f'{cid}: duplicate source case')
    return selection


def validate(root=ROOT):
    paths = sorted((root / 'config/rq3').glob('D*.yaml'))
    require(len(paths) >= 4, 'Expected at least the four original RQ3 experiment specs')
    sources = read_json(root / SOURCES)['sources']
    grouped = load_candidates(root)
    banks, resolved, seen = {}, {}, set()
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
        require(cfg['schema_version'] == 1 and cfg['kind'] == 'rq3_experiment_spec', f'{path}: schema')
        require(cfg['status'] == 'rq3_runtime_only', f'{path}: must use the RQ3 runtime')
        require(cfg['experiment_id'] == 'rq3_' + path.stem, f'{path}: experiment_id')
        require(cfg['test_version'] == 'v4' and cfg['case_count'] == 100 and cfg['repeats'] == 1,
                f'{path}: test settings')
        require('extends' not in cfg and 'bank' not in cfg, f'{path}: stale config schema')
        base = root / cfg['baseline_config']
        require(sha256_text(base) == cfg['baseline_sha256'], f'{path}: baseline hash changed')
        baseline = yaml.safe_load(base.read_text(encoding='utf-8'))
        for key, value in baseline.items():
            if key not in NON_EXPERIMENT_KEYS:
                require(cfg.get(key) == value, f'{path}: baseline parameter changed: {key}')
        method = cfg['method']
        require(method in BASELINES and method == baseline['method'], f'{path}: method')
        require(cfg['baseline_config'].endswith(BASELINES[method]), f'{path}: wrong baseline')

        marks = [tok for tok in STEM_LAYOUT if tok in path.stem]
        require(len(marks) == 1, f'{path}: file name must carry exactly one layout token')
        layout = STEM_LAYOUT[marks[0]]
        require(cfg['memory_slots'] == LAYOUTS[layout], f'{path}: memory slots')

        declared = cfg['replacement_bank']
        rule = RULE_BY_OUTPUT.get(declared.get('path'))
        require(rule is not None, f'{path}: unknown replacement bank {declared.get("path")}')
        require(declared == bank_block(rule), f'{path}: replacement bank block')
        require(cfg['pre_run_checks'] == CHECKS, f'{path}: pre-run checks')
        require(cfg['seed'] == {'policy': 'match_baseline_run', 'value': None}, f'{path}: seed')
        require((method, layout, rule) not in seen, f'{path}: duplicate group')
        seen.add((method, layout, rule))

        if rule not in banks:
            banks[rule] = check_bank(root, rule, grouped, sources)
        selection = banks[rule]

        cases = {}
        for cid, record in selection['cases'].items():
            slots = []
            for slot_spec in cfg['memory_slots']:
                slot = slot_spec['slot']
                role = slot_spec.get('foreign_role')
                eid = record[role]['episode_id'] if role else f"{cid}-{slot_spec['own_version'].upper()}"
                source = sources[eid]
                require(source['memory_eligible'] and source['variant'] in ('V0', 'V1', 'V2', 'V3'),
                        f'{cid}: V4/V5 memory leakage')
                require((source['case_id'] == cid) == (role is None), f'{cid}: slot {slot} case mismatch')
                if role:
                    require(source['case_id'].split('-')[0] != cid.split('-')[0],
                            f'{cid}: slot {slot} must come from different task_id')
                for kind in ('task', 'trajectory'):
                    require(sha256(root / source[kind + '_path']) == source[kind + '_sha256'],
                            f'{cid}: source file changed')
                slots.append({'slot': slot, 'role': role or 'own', 'episode_id': eid,
                              'task_path': source['task_path'],
                              'trajectory_path': source['trajectory_path']})
            require(len(slots) == 4 and slots[3]['episode_id'] == record['A']['episode_id'],
                    f'{cid}: A slot mismatch')
            cases[cid] = slots
        resolved[cfg['experiment_id']] = {'method': method, 'layout': layout, 'rule': rule,
                                          'difficulty': LEGACY_DIFFICULTY[layout],
                                          'bank': declared['path'],
                                          'status': cfg['status'], 'cases': cases}

    original = {(m, l) for (m, l, r) in seen if r == 'low_overlap'}
    require(original == {(m, l) for m in BASELINES for l in LAYOUTS},
            'The four original low_overlap groups must all be present')
    for rule in banks:
        for layout in LAYOUTS:
            matching = [item['cases'] for item in resolved.values()
                        if item['rule'] == rule and item['layout'] == layout]
            if matching:
                require(len(matching) == len(BASELINES) and
                        all(item == matching[0] for item in matching),
                        f'{rule}/{layout}: FC and Episodic must share one selection')
    return resolved


def main():
    resolved = validate()
    write_json(ROOT / 'code/rq3/artifacts/rq3_resolved_specs.json', resolved)
    by_rule = defaultdict(list)
    for name, item in sorted(resolved.items()):
        by_rule[item['rule']].append(f"{name.replace('rq3_', '')} [{item['layout']}]")
    print(f'PASS: {len(resolved)} specs, 100 cases each.')
    for rule, names in sorted(by_rule.items()):
        print(f'  {rule}: ' + ', '.join(names))
    print('replace1 = own V0,V1,V2 + foreign A; replace2 = own V0,V1 + foreign B,A.')
    print('A is fixed at slot 4; within one rule+layout both mechanisms share the selection.')
    print('Use code.rq3.runtime for cross-case loading; the stock RQ1 runner is not compatible.')


if __name__ == '__main__':
    main()
