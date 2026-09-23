"""Select fixed cross-task memories for 1/2-slot replacement.

Two selection rules are available; each writes its own bank so that runs made
under one rule stay reproducible when the other is added.

``low_overlap``    ascending ``(c, source_case, variant)``.  The foreign pair
                   least likely to be retrieved at all.  This is the original
                   rule and the default; it reproduces the existing
                   ``rq3_replacement_bank.json`` byte for byte, so D01-D04 keep
                   their fingerprints and their completed results.

``high_conflict``  descending ``c*m`` (ties broken by descending ``c``).  The
                   foreign pair most likely to be retrieved AND, once retrieved,
                   to move the agent away from the target's next milestone.
                    This is the "H hard distractor" rule of the RQ3 standard card
                    (Criterion 6): H = the highest ``c*m``.

``c`` is the fraction of the target V4 route's poses that the candidate also
visits (retrieval attractiveness); ``m`` is the fraction of those shared poses
where the candidate's action increases the grid distance to the target's next
milestone (conflict).  Both come from the existing candidate audit; nothing is
re-derived here.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from .adapter import require, sha256
from .build_bank import ROOT, write_json


AUDIT = 'code/rq3/artifacts/rq3_candidates.jsonl'
SOURCES = 'code/rq3/artifacts/rq3_sources.json'
OUTPUT = 'code/rq3/artifacts/rq3_replacement_bank.json'
CONFLICT_OUTPUT = 'code/rq3/artifacts/rq3_conflict_bank.json'

RULES = {
    'low_overlap': {
        'output': OUTPUT,
        'audit': 'code/rq3/artifacts/rq3_replacement_audit.jsonl',
        'summary': 'code/rq3/artifacts/rq3_replacement_summary.json',
        'selection_rule': 'eligible; different target task_id; distinct source cases; '
                          'ascending (c, source_case, variant)',
    },
    'high_conflict': {
        'output': CONFLICT_OUTPUT,
        'audit': 'code/rq3/artifacts/rq3_conflict_audit.jsonl',
        'summary': 'code/rq3/artifacts/rq3_conflict_summary.json',
        'selection_rule': 'eligible; different target task_id; distinct source cases; '
                          'descending (c*m, c) then ascending (source_case, variant)',
    },
}
DEFAULT_RULE = 'low_overlap'


def conflict(row):
    """c * m for one candidate; missing m counts as 0."""
    return row['c'] * (row.get('m') or 0.0)


def rank(row):
    return (row['c'], row['source_case'], row['variant'])


def rank_conflict(row):
    return (-conflict(row), -row['c'], row['source_case'], row['variant'])


RANKERS = {'low_overlap': rank, 'high_conflict': rank_conflict}


def pick_two(rows, target_case, rule=DEFAULT_RULE):
    """A is first, B second; exclusions come from the existing candidate audit."""
    require(rule in RANKERS, f'Unknown selection rule: {rule}')
    pool = [r for r in rows if r['target_case'] == target_case and r['eligible']
            and r['source_case'].split('-')[0] != target_case.split('-')[0]]
    pool.sort(key=RANKERS[rule])
    chosen, seen_cases = [], set()
    for row in pool:
        if row['source_case'] in seen_cases:
            continue
        chosen.append(row)
        seen_cases.add(row['source_case'])
        if len(chosen) == 2:
            break
    require(len(chosen) == 2, f'{target_case}: fewer than 2 distinct eligible foreign cases')
    return chosen


def load_candidates(root=ROOT):
    grouped = defaultdict(list)
    audit_count = 0
    with (root / AUDIT).open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            grouped[row['target_case']].append(row)
            audit_count += 1
    require(len(grouped) == 100 and audit_count == 39600, 'Candidate audit coverage mismatch')
    return grouped


def build(root=ROOT, rule=DEFAULT_RULE):
    require(rule in RULES, f'Unknown selection rule: {rule}')
    spec = RULES[rule]
    audit_path = root / AUDIT
    source_path = root / SOURCES
    sources = json.loads(source_path.read_text(encoding='utf-8'))['sources']
    grouped = load_candidates(root)
    bank, audit = {}, []
    for cid in sorted(grouped):
        pair = pick_two(grouped[cid], cid, rule)
        entries = {}
        for label, row in zip(('A', 'B'), pair):
            source = sources[row['episode_id']]
            require(source['case_id'] == row['source_case'] and source['variant'] == row['variant'],
                    f'{cid}: candidate/source mismatch')
            require(source['memory_eligible'] and row['variant'] in ('V0', 'V1', 'V2', 'V3'),
                    f'{cid}: V4/V5 memory leakage')
            require(source['case_id'].split('-')[0] != cid.split('-')[0], f'{cid}: same task_id')
            entry = {'episode_id': row['episode_id'], 'source_case': row['source_case'],
                     'task_id': row['source_case'].split('-')[0], 'variant': row['variant'],
                     'c': row['c'], 'trajectory_path': source['trajectory_path'],
                     'task_path': source['task_path']}
            entries[label] = entry
            item = {'target_case': cid, 'label': label,
                    'rank_key': list(RANKERS[rule](row)),
                    'episode_id': row['episode_id'], 'c': row['c'],
                    'candidate_eligible': row['eligible'],
                    'candidate_exclusion_reasons': row['exclusion_reasons'],
                    'source_case': row['source_case'], 'task_id': entry['task_id'],
                    'variant': row['variant']}
            if rule != DEFAULT_RULE:
                item['m'] = row.get('m')
                item['c_times_m'] = conflict(row)
            audit.append(item)
        bank[cid] = entries
    write_json(root / spec['output'], {'schema_version': 1,
        'selection_rule': spec['selection_rule'],
        'candidate_audit_path': AUDIT, 'candidate_audit_sha256': sha256(audit_path),
        'sources_path': SOURCES, 'sources_sha256': sha256(source_path), 'cases': bank})
    with (root / spec['audit']).open('w', encoding='utf-8', newline='\n') as f:
        for item in audit:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n')
    if rule == DEFAULT_RULE:
        summary = {'schema_version': 1, 'cases': len(bank), 'selected_trajectories': len(audit),
                   'A_zero_overlap_cases': sum(v['A']['c'] == 0 for v in bank.values()),
                   'B_zero_overlap_cases': sum(v['B']['c'] == 0 for v in bank.values()),
                   'max_selected_c': max(e['c'] for v in bank.values() for e in v.values()),
                   'note': 'E/H mean one/two replacement slots, not original c*m hard classification.'}
    else:
        picked = {(i['target_case'], i['label']): i for i in audit}
        cms = [i['c_times_m'] for i in audit]
        cvals = [i['c'] for i in audit]
        summary = {'schema_version': 1, 'rule': rule, 'cases': len(bank),
                   'selected_trajectories': len(audit),
                   'nonzero_overlap_selected': sum(1 for v in cvals if v > 0),
                   'nonzero_conflict_selected': sum(1 for v in cms if v > 0),
                   'mean_c': round(sum(cvals) / len(cvals), 6),
                   'max_c': max(cvals),
                   'mean_c_times_m': round(sum(cms) / len(cms), 6),
                   'max_c_times_m': max(cms),
                    'note': 'H hard distractor per the RQ3 standard card: top-2 by c*m among eligible '
                           'foreign candidates from distinct source cases.'}
    write_json(root / spec['summary'], summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--rule', choices=sorted(RULES), default=DEFAULT_RULE,
                        help='low_overlap (default, reproduces the original bank) or high_conflict')
    parser.add_argument('--all', action='store_true', help='build every rule in turn')
    args = parser.parse_args()
    rules = sorted(RULES) if args.all else [args.rule]
    for rule in rules:
        print(json.dumps(build(args.root, rule), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
