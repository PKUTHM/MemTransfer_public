"""Checks for the 26 RQ1 configs in ``config/rq1/``.

The configs promise four things, and each is asserted here rather than
assumed:

  * they form the RQ1 experiment matrix: 17 experiments on the V0 test
    (section 1A, A01-A17) and 9 on V4 (1B, B01-B09);
  * each file is self-contained -- no ``extends``, and every parameter the
    code reads is written out, so nothing comes from a code default;
  * the blocks that are not the experiment itself are identical in all 26,
    so two RQ1 runs differ only where the matrix says they do;
  * same-origin wording fires for V0 memory on the V0 test and nowhere
    else.

Run from the repo root:
    python code/tests/test_rq1_configs.py
"""
import glob
import json
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import experiment as ex                                    # noqa: E402

RQ1_DIR = os.path.join(PROJECT_ROOT, "config", "rq1")
ALL = ["v0", "v1", "v2", "v3"]

#: code -> (test version, memory, versions). Sheet RQ1, rows 2-18 (1A) and
#: 22-30 (1B); the code is the row's number within its section.
MATRIX = {
    "A01": ("v0", "none", []),
    "A02": ("v0", "full_context", ["v0"]),
    "A03": ("v0", "full_context", ["v1"]),
    "A04": ("v0", "full_context", ["v2"]),
    "A05": ("v0", "full_context", ["v3"]),
    "A06": ("v0", "episodic", ["v0"]),
    "A07": ("v0", "episodic", ["v1"]),
    "A08": ("v0", "episodic", ["v2"]),
    "A09": ("v0", "episodic", ["v3"]),
    "A10": ("v0", "summary", ["v0"]),
    "A11": ("v0", "summary", ["v1"]),
    "A12": ("v0", "summary", ["v2"]),
    "A13": ("v0", "summary", ["v3"]),
    "A14": ("v0", "semantic", ["v0"]),
    "A15": ("v0", "semantic", ["v1"]),
    "A16": ("v0", "semantic", ["v2"]),
    "A17": ("v0", "semantic", ["v3"]),
    "B01": ("v4", "none", []),
    "B02": ("v4", "full_context", ["v0"]),
    "B03": ("v4", "full_context", ALL),
    "B04": ("v4", "episodic", ["v0"]),
    "B05": ("v4", "episodic", ALL),
    "B06": ("v4", "summary", ["v0"]),
    "B07": ("v4", "summary", ALL),
    "B08": ("v4", "semantic", ["v0"]),
    "B09": ("v4", "semantic", ALL),
}

#: Every top-level key a file states -- nothing else, nothing less.
TOP_LEVEL = {"schema_version", "experiment_id", "description", "ref_model",
             "test_version", "dataset_root", "case_map", "scene",
             "working_memory", "max_steps", "max_prompt_images", "repeats",
             "prompt", "server", "retry", "memory_files", "model", "execution"
             } | set(ex.GROUP_FIELDS)
#: What one RQ1 experiment may state differently from another.
OWN = {"experiment_id", "description", "test_version"} | set(ex.GROUP_FIELDS)

MODEL_KEYS = {"name", "api_url", "provider", "temperature", "max_tokens",
              "extra_body", "action_extraction"}
EXECUTION_KEYS = {"api_excel", "max_workers", "max_api_attempts_per_step",
                  "step_wall_clock_seconds", "connect_timeout", "read_timeout",
                  "retry_interval_seconds", "max_case_attempts",
                  "server_retry_attempts", "circuit_breaker", "validate_keys",
                  "key_probe_workers", "api_key_cache", "min_valid_keys",
                  "save_prompts"}
#: Read by the code but deliberately not stated: inline keys are secrets,
#: and these configs take theirs from ``execution.api_excel``.
NOT_A_PARAMETER = {"api_keys"}

NEUTRAL = "from origins or initial headings that may differ from yours"
SAME = "from the SAME origins and initial headings"
SAME_V0_MEMORY = ("A02", "A06", "A10", "A14")


def _files():
    return sorted(glob.glob(os.path.join(RQ1_DIR, "*.yaml")))


def _code(path):
    return os.path.basename(path).split("_", 1)[0]


def _path(code):
    (p,) = [f for f in _files() if _code(f) == code]
    return p


def _raw(path):
    return ex._read_config_file(path)


def _origin(cfg, key, version):
    with open(cfg.task_path(key, version), encoding="utf-8") as f:
        o = json.load(f)["origin"]
    return int(float(o["coord"][0])), int(float(o["coord"][1])), o["heading"]


class TestMatrix(unittest.TestCase):

    def test_one_file_per_row_of_the_sheet(self):
        codes = [_code(p) for p in _files()]
        self.assertEqual(sorted(codes), sorted(MATRIX))
        self.assertEqual(len(codes), 26)

    def test_each_config_runs_its_row(self):
        for p in _files():
            code = _code(p)
            tv, memory, versions = MATRIX[code]
            cfg = ex.load_config(p)
            self.assertEqual(cfg.test_version, tv, p)
            self.assertEqual(cfg.group_id, code, p)
            g = cfg.group
            self.assertEqual(g.memory, memory, p)
            self.assertEqual(list(g.versions), versions, p)

    def test_file_name_and_experiment_id(self):
        """``<code>_<slug>__test_<version>.yaml``; the id is ``rq1_`` + that."""
        for p in _files():
            cfg = ex.load_config(p)
            g = cfg.group
            stem = os.path.splitext(os.path.basename(p))[0]
            self.assertEqual(stem, "%s_%s__test_%s"
                             % (g.group_id, g.slug, cfg.test_version))
            self.assertEqual(cfg.experiment_id, "rq1_" + stem)

    def test_ids_and_fingerprints_are_unique(self):
        """A shared id would write two experiments into one results tree."""
        cfgs = [ex.load_config(p) for p in _files()]
        self.assertEqual(len({c.experiment_id for c in cfgs}), 26)
        self.assertEqual(len({c.fingerprint() for c in cfgs}), 26)

    def test_image_budgets(self):
        """Full Context 40 (one trajectory may use all 40, four get 10 each),
        Episodic 7, text memories and the baseline 0."""
        for p in _files():
            cfg = ex.load_config(p)
            g = cfg.group
            ref, per = {"full_context": (40, 40 if len(g.versions) == 1 else 10),
                        "episodic": (7, 0)}.get(g.memory, (0, 0))
            self.assertEqual(g.max_reference_images, ref, p)
            self.assertEqual(g.max_smart_images_per_traj, per, p)
            self.assertEqual(g.max_total_images,
                             1 + cfg.history_images + g.max_reference_images, p)


class TestSelfContained(unittest.TestCase):
    """A file is the whole description of its run."""

    def test_no_extends(self):
        for p in _files():
            self.assertNotIn("extends", _raw(p), p)

    def test_every_top_level_key_is_stated(self):
        for p in _files():
            self.assertEqual(set(_raw(p)), TOP_LEVEL, p)

    def test_every_group_field_is_stated_and_no_default_applies(self):
        for p in _files():
            raw = _raw(p)
            gid = _code(p)
            fields = {k: raw[k] for k in ex.GROUP_FIELDS}
            parsed = ex.load_config(p).group
            for k, v in fields.items():
                self.assertEqual(getattr(parsed, k), v,
                                 "%s: %s.%s was changed by a default" % (p, gid, k))

    def test_every_prompt_and_server_key_is_stated(self):
        for p in _files():
            raw = _raw(p)
            self.assertEqual(set(raw["prompt"]), set(ex.PROMPT_DEFAULTS), p)
            self.assertEqual(set(raw["server"]), set(ex.SERVER_DEFAULTS), p)
            self.assertEqual(set(raw["retry"]), set(ex.RETRY_DEFAULTS), p)

    def test_every_model_and_execution_key_is_stated(self):
        for p in _files():
            raw = _raw(p)
            self.assertEqual(set(raw["model"]), MODEL_KEYS, p)
            self.assertEqual(set(raw["execution"]), EXECUTION_KEYS, p)
            self.assertEqual(set(raw["execution"]["circuit_breaker"]),
                             {"probe_cases", "max_technical_step_ratio"}, p)

    def test_code_reads_no_key_the_configs_leave_out(self):
        """A knob added to the code has to be added to these files too."""
        src = ""
        for name in ("agent.py", "token_probe.py"):
            with open(os.path.join(CODE_DIR, name), encoding="utf-8") as f:
                src += f.read()
        model = set(re.findall(r'cfg\.model\.get\("([a-z_]+)"', src))
        execution = set(re.findall(r'(?:ex_cfg|cfg\.execution)\.get\("([a-z_]+)"', src))
        self.assertTrue(model and execution, "the patterns no longer match the code")
        self.assertLessEqual(model, MODEL_KEYS)
        self.assertLessEqual(execution - NOT_A_PARAMETER, EXECUTION_KEYS)

class TestSharedBlocks(unittest.TestCase):

    def test_shared_blocks_are_identical_in_all_26(self):
        raws = [(p, _raw(p)) for p in _files()]
        first = raws[0][1]
        for key in sorted(TOP_LEVEL - OWN):
            for p, raw in raws:
                self.assertEqual(raw[key], first[key], "%s: %s differs" % (p, key))

class TestSectionB(unittest.TestCase):
    """1B: the nine memory groups on the V4 test, one config per group."""

    def test_no_v4_origin_is_a_history_origin(self):
        """Why same_origin_wording cannot fire on any 1B config."""
        cfg = ex.load_config(_path("B05"))
        for key in cfg.case_map.all_keys():
            test = _origin(cfg, key, "v4")
            for v in ALL:
                self.assertNotEqual(_origin(cfg, key, v), test, (key.label(), v))


class TestV0TestSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = ex.load_config(_path("A01"))
        cls.tasks, _ = ex.load_test_tasks(cls.cfg)
        cls.legacy = ex.resolve_path(cls.cfg.raw["scene"]["legacy_tasks"])

    def test_v0_is_a_test_version(self):
        self.assertIn("v0", ex.TEST_VERSIONS)
        self.assertEqual([t.version for t in self.tasks], ["v0"] * 100)

    def test_v0_tasks_are_the_legacy_task_set(self):
        rep = ex.assert_v0_matches_legacy_tasks(self.cfg, self.tasks, self.legacy)
        self.assertEqual(rep["cases"], 100)

    def test_the_v4_invariant_would_reject_them(self):
        """...which is why the preflight picks the invariant by test_version."""
        with self.assertRaises(ex.ExperimentError):
            ex.assert_v4_matches_v0_tasks(self.cfg, self.tasks, self.legacy)

    def test_scoring_uses_the_answer_key_of_the_version_under_test(self):
        import metrics as M
        import report as rp
        root = ex.resolve_path(os.path.join("dataset", "v0-v5"))
        opt0, _, planners0 = M.load_optimal(root, "v0")
        opt4, _, planners4 = M.load_optimal(root, "v4")
        opt5, _, _ = M.load_optimal(root, "v5")
        self.assertEqual(len(opt0), 100)
        self.assertEqual(planners0, ["original"])
        self.assertEqual(planners4, ["optimal"])
        self.assertNotEqual(opt0, opt4)
        # T01/case01 permits lateral moves; its forward-only lengths are
        # 18, 21 and 28 for v0, v4 and v5 respectively.
        self.assertEqual((opt0[0], opt4[0], opt5[0]), (16, 18, 22))
        self.assertEqual(rp.optimal_steps({"test_version": "v0"})[0], opt0)
        self.assertEqual(rp.optimal_steps({})[0], opt4, "v4 stays the default")


class TestConfigNames(unittest.TestCase):

    def test_subdirectory_and_bare_names_resolve(self):
        want = os.path.normcase(_path("A06"))
        for name in ("rq1/A06_episodic_v0__test_v0",
                     "A06_episodic_v0__test_v0",
                     "A06_episodic_v0__test_v0.yaml",
                     # test_run.py abspaths the argument before resolving it
                     os.path.join(PROJECT_ROOT, "rq1", "A06_episodic_v0__test_v0")):
            self.assertEqual(os.path.normcase(ex.resolve_config_path(name)), want, name)


class TestSameOriginWording(unittest.TestCase):
    """Only a record that starts exactly at the test spawn is "same origin"."""

    @classmethod
    def setUpClass(cls):
        import agent as ag
        cls.ag = ag
        cls._tasks = {}

    def tearDown(self):
        self.ag.EXPERIMENT = None
        self.ag.EXPERIMENT_GROUP = None
        self.ag.EXPERIMENT_MEMORY_TEXT = ""

    def _tasks_for(self, cfg):
        if cfg.test_version not in self._tasks:
            self._tasks[cfg.test_version] = ex.load_test_tasks(cfg)[0]
        return self._tasks[cfg.test_version]

    def _build(self, cfg, gid, legacy_id=0):
        """Decision 1 of one case, assembled by the agent's own code."""
        ag = self.ag
        g = cfg.group
        key = cfg.case_map.by_legacy(legacy_id)
        task = self._tasks_for(cfg)[legacy_id]
        origin = {"x": int(task.origin["coord"][0]),
                  "y": int(task.origin["coord"][1]),
                  "heading": task.origin["heading"]}
        img = task.scene.image_path(origin["x"], origin["y"], origin["heading"])
        ag.EXPERIMENT, ag.EXPERIMENT_GROUP = cfg, g
        ag.MAX_HISTORY_STEPS = cfg.history_steps
        ag.LOOKBACK_IMAGE_COUNT = cfg.history_images
        ag.MAX_PROMPT_IMAGES = cfg.max_prompt_images
        ag.ENABLE_HISTORY = True
        ag.EXPERIMENT_MEMORY_TEXT = ""
        trajs = []
        if g.uses_text_memory:
            mems = ex.load_text_memory_set(cfg, g.versions, g.memory)
            ag.EXPERIMENT_MEMORY_TEXT = ag.render_experiment_text_memory(cfg, g, mems)
        if g.uses_history_trajectories:
            loaded = ex.load_history_set(cfg, key, g.versions,
                                         base64_names=ex.scene_base64_names(cfg))
            trajs = [t.as_prompt_traj() for t in loaded]
        ctx = ag.DecisionContext(case_label=(key.task_id, key.case_no),
                                 initial_origin=dict(origin),
                                 current_pose=dict(origin), decision_index=1)
        return ag.build_experiment_user_content(
            task.description, origin, origin, [], img, [], trajs, ctx, {})

    def _text(self, code, cfg=None):
        cfg = cfg or ex.load_config(_path(code))
        return "".join(b.get("text", "") for b in self._build(cfg, code)
                       if b.get("type") == "text")

    def test_v0_memory_on_the_v0_test_is_introduced_as_same_origin(self):
        for code in SAME_V0_MEMORY:
            text = self._text(code)
            self.assertIn(SAME, text, code)
            self.assertNotIn(NEUTRAL, text, code)
            self.assertNotIn("NOT a route you should replay", text, code)

    def test_same_origin_reaches_every_marker(self):
        fc, ep = self._text("A02"), self._text("A06")
        # legacy 0 is T01/case01, whose v0 spawn is (2, 9) facing South
        self.assertIn("Historical origin: (2, 9) facing South (same as your origin)", fc)
        self.assertIn("Current test target: V0 / T01 / case01.", fc)
        self.assertIn("starting from the same origin and initial heading as yours", ep)
        self.assertIn("(same as your origin)", ep)

    def test_other_memory_on_the_v0_test_stays_neutral(self):
        """V1 shares V0's cell but not its heading: that is not the same origin."""
        for code in sorted(MATRIX):
            if not code.startswith("A") or code in SAME_V0_MEMORY or code == "A01":
                continue
            text = self._text(code)
            self.assertIn(NEUTRAL, text, code)
            self.assertNotIn(SAME, text, code)
            self.assertNotIn("(same as your origin)", text, code)
            if MATRIX[code][1] == "episodic":
                self.assertIn("starting from its own origin", text, code)

    def test_baselines_carry_no_memory_note(self):
        for code in ("A01", "B01"):
            self.assertNotIn("COMMON NOTE", self._text(code), code)

    def test_switching_it_off_restores_the_neutral_note(self):
        for code in SAME_V0_MEMORY:
            cfg = ex.load_config(_path(code))
            cfg.prompt["same_origin_wording"] = False
            text = self._text(code, cfg)
            self.assertIn(NEUTRAL, text, code)
            self.assertNotIn(SAME, text, code)


class TestRequestBody(unittest.TestCase):
    """What actually goes over the wire, captured at ``requests.post``."""

    NAMES = ("EXPERIMENT", "EXPERIMENT_GROUP", "EXPERIMENT_MEMORY_TEXT",
             "MODEL_NAME", "API_URL", "API_PROVIDER", "TEMPERATURE",
             "MAX_OUTPUT_TOKENS", "MODEL_EXTRA_BODY", "ACTION_EXTRACTION",
             "API_KEYS", "MAX_API_ATTEMPTS", "MAX_HISTORY_STEPS",
             "LOOKBACK_IMAGE_COUNT", "MAX_PROMPT_IMAGES", "ENABLE_HISTORY",
             "SAVE_PROMPTS")

    def setUp(self):
        import agent as ag
        self.ag = ag
        self.saved = {n: getattr(ag, n) for n in self.NAMES}
        self.saved_post = ag.requests.post

    def tearDown(self):
        for n, v in self.saved.items():
            setattr(self.ag, n, v)
        self.ag.requests.post = self.saved_post

    def _send(self, code, reply):
        """One decision of case 0 with the model wired as run_experiment_main
        wires it; returns (bodies posted, call_internvl's result)."""
        ag = self.ag
        cfg = ex.load_config(_path(code))
        ag.EXPERIMENT, ag.EXPERIMENT_GROUP = cfg, cfg.group
        ag.EXPERIMENT_MEMORY_TEXT = ""
        ag.MODEL_NAME, ag.API_URL = cfg.model["name"], cfg.model["api_url"]
        ag.API_PROVIDER = cfg.model["provider"]
        ag.TEMPERATURE = float(cfg.model["temperature"])
        ag.MAX_OUTPUT_TOKENS = int(cfg.model["max_tokens"])
        ag.MODEL_EXTRA_BODY = dict(cfg.model["extra_body"])
        ag.ACTION_EXTRACTION = cfg.model["action_extraction"]
        ag.API_KEYS, ag.MAX_API_ATTEMPTS = ["sk-test"], 1
        ag.MAX_HISTORY_STEPS = cfg.history_steps
        ag.LOOKBACK_IMAGE_COUNT = cfg.history_images
        ag.MAX_PROMPT_IMAGES = cfg.max_prompt_images
        ag.ENABLE_HISTORY = True

        sent = []

        class _Reply(object):
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": reply},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                  "total_tokens": 2}}

        def _post(url, headers=None, data=None, timeout=None):
            sent.append(json.loads(data))
            return _Reply()

        ag.requests.post = _post
        task = ex.load_test_tasks(cfg)[0][0]
        key = cfg.case_map.by_legacy(0)
        origin = {"x": int(task.origin["coord"][0]),
                  "y": int(task.origin["coord"][1]),
                  "heading": task.origin["heading"]}
        img = task.scene.image_path(origin["x"], origin["y"], origin["heading"])
        ctx = ag.DecisionContext(case_label=(key.task_id, key.case_no),
                                 initial_origin=dict(origin),
                                 current_pose=dict(origin), decision_index=1)
        out = ag.call_internvl(0, task.description, origin, [], img, [], None,
                               None, "probe", initial_origin=origin,
                               trajectories=[], ctx=ctx, telemetry={})
        return sent, out

    def test_no_reasoning_and_30_tokens_reach_the_wire(self):
        sent, out = self._send("A01", "Move_Forward 1m")
        self.assertEqual(len(sent), 1)
        body = sent[0]
        self.assertEqual(body["model"], "intern-s2-preview-397b")
        self.assertEqual(body["max_tokens"], 30)
        self.assertIs(body["thinking_mode"], False)
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual((out[0], out[-1]), ("Move_Forward 1m", "ok"))

    def test_saved_request_records_thinking_mode_too(self):
        _, out = self._send("A01", "Move_Forward 1m")
        saved = self.ag._serialize_payload_for_save(out[4])
        self.assertIs(saved["thinking_mode"], False)
        self.assertEqual(saved["max_tokens"], 30)

    def test_step_log_keeps_image_paths_with_save_prompts_on(self):
        """--save_prompts used to hand the step log an already-serialized
        payload, so every decision recorded ``image_paths: [""]``."""
        self.ag.SAVE_PROMPTS = True
        _, out = self._send("A01", "Move_Forward 1m")
        summary = self.ag._summarise_request(out[4], {})
        self.assertEqual(len(summary["image_paths"]), 1)
        self.assertTrue(summary["image_paths"][0].endswith("Grid_X2_Y9_South.png"),
                        summary["image_paths"])
        saved = self.ag._serialize_payload_for_save(out[4])
        imgs = [b for b in saved["messages"][1]["content"]
                if b.get("type") == "image_url"]
        self.assertEqual([b.get("image_path") for b in imgs], summary["image_paths"])

    def test_strict_rejects_a_reply_naming_two_actions(self):
        _, out = self._send("A01", "Turn_Left or Move_Forward 1m")
        self.assertEqual((out[0], out[-1]), (None, "model_output_failure"))
        _, out = self._send("A01", "I will Move_Forward 1m.")
        self.assertEqual((out[0], out[-1]), ("Move_Forward 1m", "ok"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
