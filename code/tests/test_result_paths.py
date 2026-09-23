import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from result_paths import experiment_result_dir


class ResultPathsTests(unittest.TestCase):
    def test_model_yaml_and_round(self):
        cfg = SimpleNamespace(raw={'ref_model': 'Intern-S2-Preview-397B'}, model={'name': 'api-model'})
        path = experiment_result_dir(cfg, 'config/rq1/A09_example.yaml', 2, stamp='fixed')
        self.assertTrue(path.endswith(os.path.join('result_Intern-S2-Preview-397B', 'A09_example', 'r2_fixed')))

    def test_override_and_model_fallback(self):
        cfg = SimpleNamespace(raw={}, model={'name': 'provider/model'})
        path = experiment_result_dir(cfg, 'sample.yaml', root='custom', stamp='fixed')
        self.assertEqual(path, os.path.join('custom', 'sample', 'r0_fixed'))
        path = experiment_result_dir(cfg, 'sample.yaml', stamp='fixed')
        self.assertIn('result_provider_model', path)


if __name__ == '__main__':
    unittest.main()
