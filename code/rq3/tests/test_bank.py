import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from code.rq3 import legacy_bank as legacy
from code.rq3.adapter import goal_sets, load_real, read_json
from code.rq3.build_bank import ROOT
from code.rq3.build_replacement import pick_two
from code.rq3.validate_configs import validate


class AlgorithmTests(unittest.TestCase):
    def test_unmodified_handoff(self):
        handoff = list(ROOT.glob('MemTransfer_RQ3_*/rq3_build_bank.py'))
        if not handoff:
            self.skipTest("handoff directory not present")
        original = handoff[0]
        self.assertEqual(original.read_bytes(), Path(legacy.__file__).read_bytes())

    def test_conflict_and_neutral(self):
        walkable = {(0, 0), (0, 1), (0, 2)}
        grid = legacy.Grid(walkable)
        target = legacy.Trajectory('t-V4', 'T', 't', 'V4', steps=[
            legacy.Step((0, 0, 'N'), 'Move_Forward'),
            legacy.Step((0, 1, 'N'), 'Move_Forward'),
            legacy.Step((0, 2, 'N'), 'Turn_Left', True)])
        candidate = legacy.Trajectory('c-V0', 'T', 'c', 'V0', steps=[
            legacy.Step((0, 1, 'N'), 'Move_Backward'), legacy.Step((0, 2, 'N'), 'Turn_Left')])
        score = legacy.score_candidate(target, candidate, grid, walkable, legacy.progress_reference(target))
        self.assertEqual((score.c, score.m, score.n_judged), (2 / 3, 1, 1))
        self.assertEqual(len(legacy.trap_points(target, target, [candidate], grid, walkable,
                                               legacy.progress_reference(target))), 1)

    def test_forward_only_cannot_have_opposing_translation(self):
        grid = legacy.Grid({(0, 0), (0, 1), (0, 2)})
        for action in ('Turn_Left', 'Turn_Right', 'Turn_Back'):
            self.assertEqual(legacy.classify(grid, (0, 1, 'N'), action, (0, 2), grid.walkable), legacy.NEUTRAL)
        self.assertEqual(legacy.classify(grid, (0, 1, 'N'), 'Move_Forward', (0, 2), grid.walkable), legacy.PRESERVING)

    def test_selection_ties_and_cross_role_dedup(self):
        pool = [(legacy.Trajectory(f'c{i}-V{v}', 'T', f'c{i}', f'V{v}'), legacy.Score(0.5, 0.5, 1, 1))
                for i in range(10) for v in range(4)]
        roles = legacy.select_roles(pool)
        self.assertEqual([t.episode_id for t, s in roles['H']], [f'c{i}-V0' for i in range(4)])
        self.assertEqual([t.case_id for t, s in roles['E']], ['c4', 'c5', 'c6'])
        self.assertEqual([t.case_id for t, s in roles['L']], ['c7', 'c8', 'c9'])

    def test_global_length_window(self):
        target = legacy.Trajectory('t-V4', 'T', 't', 'V4', ('target',))
        candidates = [(legacy.Trajectory(f'c{i}-V0', 'T', f'c{i}', 'V0', (f'o{i}',),
                        steps=[legacy.Step((0, 0, 'N'), 'Turn_Left')] * 12), legacy.Score(0, 0, 0, 0))
                      for i in range(7)]
        self.assertEqual(legacy.choose_length_window([target], {'t': candidates}, {'t': 10}), 0.2)


class AdapterTests(unittest.TestCase):
    def test_task_families_and_terminal(self):
        keys = read_json(ROOT / 'dataset/v0-v5/case_id_map.json')['cases']
        for task in ('T01', 'T02', 'T03', 'T04', 'T06', 'T07', 'T09', 'T10'):
            key = next(k for k in keys if k['task_id'] == task and k['case_no'] == 1)
            directory = ROOT / 'dataset/v0-v5' / key['new_dir'] / 'v4'
            tr, info = load_real(directory / 'task.json', directory / 'trajectory.json', key, 'v4')
            raw = read_json(directory / 'trajectory.json')['forward_only']
            self.assertEqual(len(tr.steps), len(raw['actions']) + 1)
            self.assertEqual(info['milestone_indices'], sorted({e['step'] for e in raw['stage_events']}))
            self.assertEqual(tr.steps[-1].action, 'Turn_Left')

    def test_goal_mapping_and_legacy_conservative_filter(self):
        item = {'name': 'A', 'position': [1.0, 2], 'category': 'Food'}
        self.assertEqual(goal_sets([{'instance': item}, {'name': 'Return to Origin', 'coord': [0, 0]}]),
                         (('A@1,2',), ('Food',)))
        a = legacy.Trajectory('a', 'T', 'a', 'V0', ('a',), ('Food',))
        b = legacy.Trajectory('b', 'T', 'b', 'V4', ('b',), ('Food',))
        self.assertTrue(legacy.satisfies_target_goal(a, b))

    def test_invalid_transition_and_milestone_fail(self):
        key = read_json(ROOT / 'dataset/v0-v5/case_id_map.json')['cases'][0]
        directory = ROOT / 'dataset/v0-v5' / key['new_dir'] / 'v4'
        raw = read_json(directory / 'trajectory.json')
        with tempfile.TemporaryDirectory(dir=ROOT / 'code/rq3/.tmp') as tmp:
            p = Path(tmp) / 'trajectory.json'
            for mutation in ('pose', 'milestone'):
                changed = copy.deepcopy(raw)
                if mutation == 'pose':
                    changed['forward_only']['poses'][1][0] += 100
                else:
                    changed['forward_only']['stage_events'][0]['step'] = 9999
                p.write_text(json.dumps(changed), encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_real(directory / 'task.json', p, key, 'v4')


class ArtifactTests(unittest.TestCase):
    def test_full_candidate_audit_and_no_leakage(self):
        bank = read_json(ROOT / 'code/rq3/artifacts/rq3_bank.json')['cases']
        sources = read_json(ROOT / 'code/rq3/artifacts/rq3_sources.json')['sources']
        candidates = [json.loads(line) for line in (ROOT / 'code/rq3/artifacts/rq3_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(candidates), 39600)
        selected = {(r['target_case'], r['episode_id']): r for r in candidates if r['selection']}
        for cid, b in bank.items():
            ids = []
            for role in ('H', 'E', 'L'):
                for e in b['roles'][role]:
                    source = sources[e['episode_id']]
                    self.assertIn(source['variant'], ('V0', 'V1', 'V2', 'V3'))
                    self.assertNotEqual(source['case_id'], cid)
                    ids.append(source['case_id'])
                    row = selected[cid, e['episode_id']]
                    self.assertTrue(row['eligible'])
                    self.assertEqual(row['selection'], {'role': role, 'rank': e['rank']})
                    self.assertEqual(round(row['trap'], 4), e['trap'])
            self.assertEqual(len(ids), len(set(ids)))
        summary = read_json(ROOT / 'code/rq3/artifacts/rq3_build_summary.json')
        self.assertFalse(summary['checks']['trap_points_mean_ge_3'])
        self.assertEqual(summary['status'], 'quality_gate_failed')

    def test_four_specs(self):
        resolved = validate()
        self.assertEqual(len(resolved), 8)
        # Group by rule+layout+difficulty to check consistency
        from collections import defaultdict
        groups = defaultdict(list)
        for item in resolved.values():
            key = (item['rule'], item['layout'], item['difficulty'])
            groups[key].append(item)
        # Each group should have exactly 2 configs (FC and Episodic)
        for key, items in groups.items():
            self.assertEqual(len(items), 2, f"{key} has {len(items)} configs")
            # FC and Episodic should share the same case selections
            self.assertEqual(items[0]['cases'], items[1]['cases'], f"{key} cases differ")
        # Check that low_overlap easy configs have the expected slot layout
        for item in resolved.values():
            if item['rule'] == 'low_overlap' and item['difficulty'] == 'easy':
                for cid, slots in item['cases'].items():
                    self.assertEqual([s['role'] for s in slots], ['own', 'own', 'own', 'A'])
            elif item['rule'] == 'low_overlap' and item['difficulty'] == 'hard':
                for cid, slots in item['cases'].items():
                    self.assertEqual([s['role'] for s in slots], ['own', 'own', 'B', 'A'])

    def test_replacement_ties_exclude_same_case_and_task(self):
        def row(cid, source, variant, c, eligible=True):
            return {'target_case': cid, 'source_case': source,
                    'variant': variant, 'episode_id': source + '-' + variant,
                    'c': c, 'eligible': eligible}
        cid = 'T01-case01'
        pool = [row(cid, 'T01-case02', 'V0', 0),
                row(cid, 'T02-case01', 'V1', 0, False),
                row(cid, 'T02-case01', 'V0', 0),
                row(cid, 'T02-case01', 'V2', 0),
                row(cid, 'T03-case01', 'V0', 0.1)]
        self.assertEqual([r['episode_id'] for r in pick_two(pool, cid)],
                         ['T02-case01-V0', 'T03-case01-V0'])


if __name__ == '__main__':
    unittest.main()
