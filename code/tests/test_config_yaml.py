"""Checks for the config machinery in ``experiment.py``.

``config/`` holds the self-contained RQ1 / RQ2 configs, which
test_rq1_configs.py checks in detail. What stays here is the machinery every
config goes through, wherever it lives:

  * layering (``extends``, deep merge, cycles) -- still
    supported, even though no current config uses it;
  * name resolution (YAML only; the retired ``experiments/*.json`` format
    is rejected);
  * "every parameter is in the config": typos fail loudly, a value changed
    from its default shows up in the fingerprint and an untouched default
    does not.

Run from the repo root:
    python code/tests/test_config_yaml.py
"""
import glob
import io
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import experiment as ex                                    # noqa: E402

CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
A06 = os.path.join(CONFIG_DIR, "rq1", "A06_episodic_v0__test_v0.yaml")
B05 = os.path.join(CONFIG_DIR, "rq1", "B05_episodic_v0-v3__test_v4.yaml")


def _yaml_configs():
    """Return RQ1/RQ2 configs only; RQ3 has its own loader and validator."""
    all_yaml = sorted(glob.glob(os.path.join(CONFIG_DIR, "**", "*.yaml"), recursive=True))
    return [p for p in all_yaml if os.sep + "rq3" + os.sep not in p]


def _same(a, b):
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


class TestLayering(unittest.TestCase):
    """``extends`` machinery, exercised on throwaway files."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="cfg_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, name, text):
        p = os.path.join(self.dir, name)
        with io.open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return p

    def test_deep_merge_recurses_into_mappings(self):
        merged = ex._deep_merge({"a": {"x": 1, "y": 2}, "b": 1},
                                {"a": {"y": 9}, "c": 3})
        self.assertEqual(merged, {"a": {"x": 1, "y": 9}, "b": 1, "c": 3})

    def test_deep_merge_replaces_lists_rather_than_extending(self):
        """An override that lists two versions means exactly those two."""
        merged = ex._deep_merge({"versions": ["v0", "v1", "v2", "v3"]},
                                {"versions": ["v0"]})
        self.assertEqual(merged["versions"], ["v0"])

    def test_extends_pulls_in_the_parent(self):
        self._write("base.yaml", "schema_version: 1\nmax_steps: 200\n"
                                 "prompt:\n  image_quality: 80\n  image_max_size: 448\n")
        p = self._write("child.yaml", "extends: base.yaml\nexperiment_id: t\n"
                                      "prompt:\n  image_quality: 60\n")
        raw = ex._load_layered(p)
        self.assertEqual(raw["experiment_id"], "t")
        self.assertEqual(raw["max_steps"], 200)
        self.assertEqual(raw["prompt"], {"image_quality": 60, "image_max_size": 448})
        self.assertNotIn("extends", raw, "extends must not survive the merge")

    def test_extends_list_applies_left_to_right(self):
        self._write("a.yaml", "max_steps: 1\nrepeats: 1\n")
        self._write("b.yaml", "max_steps: 2\n")
        p = self._write("c.yaml", "extends: [a.yaml, b.yaml]\n")
        raw = ex._load_layered(p)
        self.assertEqual((raw["max_steps"], raw["repeats"]), (2, 1))

    def test_override_touches_one_field_only(self):
        self._write("base.yaml",
                    "memory: episodic\nversions: [v0]\nepisodic_horizon: 6\n")
        p = self._write("child.yaml", "extends: base.yaml\nepisodic_horizon: 3\n")
        raw = ex._load_layered(p)
        self.assertEqual(raw["episodic_horizon"], 3)
        self.assertEqual(raw["versions"], ["v0"],
                         "untouched fields must come from the parent")

    def test_group_wrappers_are_rejected(self):
        """Memory settings sit at the top level and the group id and slug
        come from the file name; an old wrapper must not be ignored."""
        for key, value in (("groups", {"A06": {}}), ("group", {}),
                           ("group_id", "A06"), ("slug", "x"), ("label", "x")):
            raw = ex._read_config_file(A06)
            raw[key] = value
            with self.assertRaises(ex.ExperimentError, msg=key):
                ex.ExperimentConfig(raw, A06)

    def test_unknown_top_level_key_is_rejected(self):
        raw = ex._read_config_file(A06)
        raw["memroy"] = "none"
        with self.assertRaises(ex.ExperimentError) as cm:
            ex.ExperimentConfig(raw, A06)
        self.assertIn("memroy", str(cm.exception))

    def test_group_id_and_slug_come_from_the_file_name(self):
        cfg = ex.load_config(A06)
        self.assertEqual((cfg.group_id, cfg.group.slug), ("A06", "episodic_v0"))
        raw = ex._read_config_file(A06)
        for bad in ("episodic_v0.yaml",                 # no code
                    "A06_episodic_v0__test_v4.yaml"):   # says v4, config is v0
            with self.assertRaises(ex.ExperimentError, msg=bad):
                ex.ExperimentConfig(dict(raw), os.path.join(self.dir, bad))

    def test_circular_extends_is_reported(self):
        self._write("a.yaml", "extends: b.yaml\n")
        p = self._write("b.yaml", "extends: a.yaml\n")
        with self.assertRaises(ex.ExperimentError) as cm:
            ex._load_layered(p)
        self.assertIn("circular", str(cm.exception))

    def test_missing_extends_target_is_reported(self):
        p = self._write("missing.yaml", "extends: nope.yaml\n")
        with self.assertRaises(ex.ExperimentError):
            ex._load_layered(p)


class TestNameResolution(unittest.TestCase):

    def test_bare_name_is_found_in_a_subdirectory(self):
        self.assertTrue(_same(ex.resolve_config_path("A06_episodic_v0__test_v0"), A06))

    def test_name_relative_to_the_config_dir(self):
        self.assertTrue(_same(ex.resolve_config_path("rq1/A06_episodic_v0__test_v0"), A06))

    def test_full_path_still_resolves(self):
        self.assertTrue(_same(ex.resolve_config_path(A06), A06))

    def test_json_configs_are_no_longer_read(self):
        """The retired experiments/*.json format was removed on 2026-09-11."""
        d = tempfile.mkdtemp(prefix="cfg_")
        try:
            p = os.path.join(d, "old.json")
            with io.open(p, "w", encoding="utf-8") as f:
                f.write("{}")
            with self.assertRaises(ex.ExperimentError):
                ex.load_config(p)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_unknown_name_is_reported(self):
        with self.assertRaises(ex.ExperimentError):
            ex.resolve_config_path("no_such_experiment")

    def test_ambiguous_bare_name_is_reported(self):
        dup_dir = tempfile.mkdtemp(prefix="_dup_", dir=CONFIG_DIR)
        try:
            shutil.copy(A06, dup_dir)
            with self.assertRaises(ex.ExperimentError) as cm:
                ex.resolve_config_path("A06_episodic_v0__test_v0")
            self.assertIn("ambiguous", str(cm.exception))
        finally:
            shutil.rmtree(dup_dir, ignore_errors=True)


class TestEveryConfigLoads(unittest.TestCase):

    def test_all_yaml_configs_parse_and_validate(self):
        found = _yaml_configs()
        self.assertTrue(found, "no config/**/*.yaml found")
        for p in found:
            cfg = ex.load_config(p)
            self.assertTrue(cfg.group_id, p)
            self.assertIn(cfg.test_version, ex.TEST_VERSIONS, p)

    def test_experiment_ids_are_unique(self):
        """Two configs sharing an id would write into one results directory."""
        ids = [ex.load_config(p).experiment_id for p in _yaml_configs()]
        self.assertEqual(len(ids), len(set(ids)), sorted(ids))

    def test_scene_image_base64_dir_is_a_real_directory(self):
        for p in _yaml_configs():
            self.assertTrue(os.path.isdir(ex.load_config(p).scene_image_base64_dir), p)


class TestGroupConfig(unittest.TestCase):

    def setUp(self):
        self.cfg = ex.load_config(B05)

    def test_unknown_group_key_is_rejected(self):
        """A typo must fail, not silently fall back to the default."""
        with self.assertRaises(ex.ExperimentError) as cm:
            ex.GroupConfig(group_id="X01", memroy="none")
        self.assertIn("memroy", str(cm.exception))

    def test_defaults_fill_what_a_group_omits(self):
        g = ex.GroupConfig(group_id="X", memory="episodic", versions=["v0"],
                           max_reference_images=7).apply_defaults(6)
        self.assertEqual((g.method, g.reference_traj_images, g.max_total_images,
                          g.max_smart_images_per_traj),
                         ("episodic", "none", 14, 0))
        self.assertEqual((g.episodic_horizon, g.episodic_matcher, g.episodic_content),
                         (6, "vit", "all"))
        g = ex.GroupConfig(group_id="Y", memory="summary",
                           versions=["v0"]).apply_defaults(6)
        self.assertEqual((g.max_total_images, g.episodic_horizon), (7, None),
                         "a non-episodic group must not gain episodic defaults")

    def test_slug_defaults_from_method_and_versions(self):
        g = ex.GroupConfig(group_id="G99", method="episodic",
                           versions=["v0", "v1", "v2", "v3"])
        self.assertEqual(g.slug, "episodic_v0-v3")
        g = ex.GroupConfig(group_id="G98", method="summary", versions=["v0"])
        self.assertEqual(g.slug, "summary_v0")
        g = ex.GroupConfig(group_id="G97", method="working_memory_only", versions=[])
        self.assertEqual(g.slug, "working_memory_only")

    def test_slug_is_not_part_of_the_fingerprint(self):
        """Renaming a group must not invalidate the runs already recorded."""
        before = self.cfg.fingerprint()
        self.cfg.group.slug = "renamed_entirely"
        self.assertEqual(self.cfg.fingerprint(), before)


class TestHyperparametersAreInTheConfig(unittest.TestCase):
    """A run must be described by its config, not config + code constants."""

    def setUp(self):
        self.cfg = ex.load_config(A06)

    def test_defaults_equal_the_constants_they_replaced(self):
        """If these drift, a config that omits a block silently starts
        behaving differently from the runs it was meant to reproduce."""
        self.assertEqual(ex.PROMPT_DEFAULTS["image_max_size"], 448)
        self.assertEqual(ex.PROMPT_DEFAULTS["image_quality"], 80)
        self.assertEqual(set(ex.PROMPT_DEFAULTS["reference_traj_smart_layers"]),
                         {"start", "stage", "end", "turn"})
        self.assertEqual(ex.PROMPT_DEFAULTS["vit_exact_cosine"], 0.97)
        self.assertEqual(ex.PROMPT_DEFAULTS["vit_near_cosine"], 0.75)
        self.assertIs(ex.PROMPT_DEFAULTS["same_origin_wording"], False)
        self.assertEqual(ex.SERVER_DEFAULTS["action_cache_limit"], 208)
        self.assertIs(ex.SERVER_DEFAULTS["notify_on_progress"], True)
        self.assertEqual(ex.RETRY_DEFAULTS, {
            "truncation_retries": 3, "max_tokens_ceiling": 32768,
            "max_consecutive_truncated": 5, "max_consecutive_model_failures": 3,
            "max_consecutive_technical": 3})

    def test_retry_defaults_match_the_agent_constants(self):
        import agent as ag
        self.assertEqual(
            (ag.MAX_TRUNCATION_RETRIES, ag.MAX_OUTPUT_TOKENS_CEILING,
             ag.MAX_CONSECUTIVE_TRUNCATED, ag.MAX_CONSECUTIVE_MODEL_FAIL,
             ag.MAX_CONSECUTIVE_TECHNICAL),
            tuple(ex.RETRY_DEFAULTS[k] for k in (
                "truncation_retries", "max_tokens_ceiling",
                "max_consecutive_truncated", "max_consecutive_model_failures",
                "max_consecutive_technical")))

    def test_config_carries_every_block_key(self):
        self.assertEqual(set(self.cfg.prompt), set(ex.PROMPT_DEFAULTS))
        self.assertEqual(set(self.cfg.server), set(ex.SERVER_DEFAULTS))
        self.assertEqual(set(self.cfg.retry), set(ex.RETRY_DEFAULTS))

    def test_unknown_key_is_rejected_rather_than_ignored(self):
        """A typo must fail loudly, not silently keep the default."""
        for block, key in (("prompt", "imag_quality"), ("server", "notify"),
                           ("retry", "truncation_retry")):
            raw = ex._load_layered(A06)
            raw[block] = dict(raw.get(block) or {}, **{key: 1})
            with self.assertRaises(ex.ExperimentError) as cm:
                ex.ExperimentConfig(raw, "t")
            self.assertIn(key, str(cm.exception))

    def test_untouched_defaults_stay_out_of_the_hash(self):
        """Only values that differ from the defaults enter the fingerprint.

        In RQ1 that is ``same_origin_wording`` alone; every other prompt,
        server and retry value is the default and hashes as if absent."""
        self.assertEqual(self.cfg._tuned_params(),
                         {"prompt.same_origin_wording": True})

    def test_changing_a_parameter_changes_the_hash(self):
        """...but a real change must be visible in the fingerprint."""
        before = self.cfg.fingerprint()
        for block, key, value in (("prompt", "image_quality", 60),
                                  ("prompt", "vit_exact_cosine", 0.99),
                                  ("prompt", "reference_traj_smart_layers", ["start", "end"]),
                                  ("prompt", "obstacle_notice", True),
                                  ("server", "notify_on_progress", False),
                                  ("retry", "truncation_retries", 0)):
            cfg = ex.load_config(A06)
            getattr(cfg, block)[key] = value
            self.assertNotEqual(cfg.fingerprint(), before, key)
            self.assertIn("%s.%s" % (block, key), cfg._tuned_params())

    def test_execution_only_server_knobs_stay_out_of_the_hash(self):
        before = self.cfg.fingerprint()
        cfg = ex.load_config(A06)
        cfg.server["action_cache_limit"] = 999
        self.assertEqual(cfg.fingerprint(), before,
                         "a replay-cache size does not change the experiment")

    def test_image_encoding_is_resolved_at_call_time(self):
        """Default arguments bind at import, so they must not be used here."""
        from prompt import _helpers
        old = (_helpers.DEFAULT_MAX_SIZE, _helpers.DEFAULT_QUALITY)
        try:
            _helpers.set_image_encoding(256, 50)
            self.assertEqual(_helpers.DEFAULT_MAX_SIZE, 256)
            self.assertEqual(_helpers.DEFAULT_QUALITY, 50)
            import inspect
            sig = inspect.signature(_helpers.encode_image_to_base64)
            self.assertIsNone(sig.parameters["max_size"].default)
            self.assertIsNone(sig.parameters["quality"].default)
        finally:
            _helpers.set_image_encoding(*old)

    def test_inverted_vit_bands_are_rejected(self):
        from prompt import vit_matcher
        old = (vit_matcher.EXACT_COSINE_THRESHOLD,
               vit_matcher.NEAR_COSINE_THRESHOLD,
               vit_matcher.COSINE_TIE_EPS)
        try:
            with self.assertRaises(ValueError):
                vit_matcher.set_thresholds(exact=0.5, near=0.9)
        finally:
            vit_matcher.set_thresholds(*old)


if __name__ == "__main__":
    unittest.main(verbosity=2)
