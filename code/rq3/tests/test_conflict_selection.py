"""The two selection rules must differ in the way the standard card intends."""
import json
import unittest

from code.rq3.build_bank import ROOT
from code.rq3.build_replacement import RULES, conflict, load_candidates, pick_two


class ConflictSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grouped = load_candidates(ROOT)
        cls.picks = {rule: {cid: pick_two(rows, cid, rule) for cid, rows in cls.grouped.items()}
                     for rule in RULES}

    def test_both_rules_cover_every_case_with_valid_foreign_pairs(self):
        for rule, picks in self.picks.items():
            self.assertEqual(len(picks), 100, rule)
            for cid, pair in picks.items():
                self.assertEqual(len(pair), 2, f'{rule}/{cid}')
                self.assertNotEqual(pair[0]['source_case'], pair[1]['source_case'], f'{rule}/{cid}')
                for row in pair:
                    self.assertTrue(row['eligible'], f'{rule}/{cid}')
                    self.assertIn(row['variant'], ('V0', 'V1', 'V2', 'V3'), f'{rule}/{cid}')
                    self.assertNotEqual(row['source_case'].split('-')[0], cid.split('-')[0],
                                        f'{rule}/{cid}')

    def test_low_overlap_is_the_zero_overlap_pair(self):
        for cid, pair in self.picks['low_overlap'].items():
            for row in pair:
                self.assertEqual(row['c'], 0, cid)

    def test_high_conflict_maximises_c_times_m(self):
        """Every pick must be the best still-available candidate under c*m."""
        better = 0
        for cid, pair in self.picks['high_conflict'].items():
            best = max(conflict(r) for r in self.grouped[cid]
                       if r['eligible'] and r['source_case'].split('-')[0] != cid.split('-')[0])
            self.assertAlmostEqual(conflict(pair[0]), best, msg=cid)
            if conflict(pair[0]) > 0:
                better += 1
        # the whole point: for almost every case a genuinely conflicting foreign
        # trajectory exists, and the original rule threw all of them away.
        self.assertGreater(better, 90)

    def test_high_conflict_dominates_low_overlap(self):
        def mean(rule, fn):
            rows = [r for pair in self.picks[rule].values() for r in pair]
            return sum(fn(r) for r in rows) / len(rows)
        self.assertEqual(mean('low_overlap', conflict), 0)
        self.assertGreater(mean('high_conflict', conflict), 0.05)
        self.assertGreater(mean('high_conflict', lambda r: r['c']), 0.3)

    def test_banks_on_disk_match_their_rules(self):
        for rule, spec in RULES.items():
            path = ROOT / spec['output']
            if not path.exists():
                continue
            bank = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(bank['selection_rule'], spec['selection_rule'], rule)
            for cid, record in bank['cases'].items():
                expected = [r['episode_id'] for r in self.picks[rule][cid]]
                self.assertEqual([record['A']['episode_id'], record['B']['episode_id']],
                                 expected, f'{rule}/{cid}')


if __name__ == '__main__':
    unittest.main()
