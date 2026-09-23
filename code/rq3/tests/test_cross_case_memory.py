"""Exercise the real RQ1 trajectory loader and prompt path through RQ3."""
import hashlib
from pathlib import Path
import unittest

from code.rq3.build_bank import ROOT
from code.rq3.cross_case_memory import config_from_rq3, imports, install_agent, preload


class CrossCaseMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ex = imports()
        import agent
        cls.agent = agent
        install_agent(agent)

    def test_all_four_groups_load_exact_sources(self):
        histories = {}
        for name in ('D01_full_context_easy__test_v4.yaml',
                     'D02_full_context_hard__test_v4.yaml',
                     'D03_episodic_easy__test_v4.yaml',
                     'D04_episodic_hard__test_v4.yaml'):
            cfg = config_from_rq3(ROOT / 'config/rq3' / name)
            loaded, manifest = preload(cfg, cfg.group)
            self.assertEqual((len(loaded), len(manifest['trajectories'])), (100, 400))
            bank_hash = hashlib.sha256((ROOT / 'code/rq3/artifacts/rq3_replacement_bank.json').read_bytes()).hexdigest()
            self.assertEqual(manifest['rq3_replacement_bank_sha256'], bank_hash)
            self.assertIn('__mem_', cfg.experiment_id)
            histories[name[:3]] = loaded
        # Check a subset of cases for consistency
        for cid in (0, 50, 99):
            easy_fc, hard_fc = histories['D01'][cid], histories['D02'][cid]
            easy_epi, hard_epi = histories['D03'][cid], histories['D04'][cid]
            self.assertEqual([x['_rq3_episode_id'] for x in easy_fc],
                             [x['_rq3_episode_id'] for x in easy_epi])
            self.assertEqual([x['_rq3_episode_id'] for x in hard_fc],
                             [x['_rq3_episode_id'] for x in hard_epi])
            self.assertEqual([x['_rq3_role'] for x in easy_fc], ['own', 'own', 'own', 'A'])
            self.assertEqual([x['_rq3_role'] for x in hard_fc], ['own', 'own', 'B', 'A'])
            self.assertEqual(easy_fc[3]['_rq3_episode_id'], hard_fc[3]['_rq3_episode_id'])
            self.assertEqual([x['_rq3_episode_id'] for x in easy_fc[:2]],
                             [x['_rq3_episode_id'] for x in hard_fc[:2]])

    def test_real_prompt_names_foreign_case_without_same_case_claim(self):
        from prompt import DecisionContext
        for name in ('D01_full_context_easy__test_v4.yaml',
                     'D03_episodic_easy__test_v4.yaml'):
            cfg = config_from_rq3(ROOT / 'config/rq3' / name)
            loaded, _ = preload(cfg, cfg.group)
            self.agent.EXPERIMENT = cfg
            self.agent.MAX_PROMPT_IMAGES = 50
            self.agent.REFERENCE_TRAJ_SMART_LAYERS = set(cfg.prompt['reference_traj_smart_layers'])
            self.agent.REFERENCE_TRAJ_SMART_SAMPLE_COUNT = cfg.prompt['reference_traj_smart_sample_count']
            key = cfg.case_map.by_legacy(0)
            ctx = DecisionContext(case_label=(key.task_id, key.case_no),
                                  initial_origin=loaded[0][0]['origin'],
                                  current_pose={'x': -5, 'y': 16, 'heading': 'North'},
                                  decision_index=0, stages_completed=0,
                                  steps_in_current_stage=0)
            telemetry = {}
            front, back = self.agent.experiment_memory_blocks(
                cfg.group, loaded[0], loaded[0][0]['image_paths'][0], ctx, 0, telemetry)
            message = '\n'.join(b.get('text', '') for b in back if b.get('type') == 'text')
            self.assertIn('[MEMORY SCOPE — MIXED CASES]', message)
            self.assertIn('T02/case01', message)
            self.assertNotIn('this same case', message.lower())
            self.assertEqual(len(telemetry['rq3_sources']), 4)
            self.assertEqual(telemetry['rq3_sources'][3]['episode_id'], 'T02-case01-V0')
            if cfg.group.memory == 'episodic' and telemetry.get('episodic_match'):
                self.assertIn('rq3_episode_id', telemetry['episodic_match'])
                self.assertIn('rq3_source_role', telemetry['episodic_match'])


if __name__ == '__main__':
    unittest.main()
