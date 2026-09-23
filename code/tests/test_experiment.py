"""Offline checks for the v0-v3 -> v4 experiment wiring.

Standard-library ``unittest`` only — no new test dependency.

Run from the repo root:
    python code/tests/test_experiment.py
"""
import glob
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import experiment as ex                                    # noqa: E402
from prompt.reference_traj_images import _select_smart_indices  # noqa: E402
from prompt.episodic_memory import (find_best_match, DecisionContext,  # noqa: E402
                                    stage_progress_by_step)

RQ1_DIR = os.path.join(PROJECT_ROOT, "config", "rq1")
#: Section 1B: the nine memory groups on the V4 test, one config each.
B_CODES = ["B0%d" % i for i in range(1, 10)]


def _rq1_path(code):
    (p,) = glob.glob(os.path.join(RQ1_DIR, code + "_*.yaml"))
    return p


B01 = _rq1_path("B01")


class _Matrix1B(object):
    """The nine 1B configs viewed as one nine-group matrix.

    Each config runs one group, and B01..B09 differ only in that group
    (test_rq1_configs.py asserts the shared blocks are identical), so the
    shared settings come from B01 and ``group(code)`` from each file.
    """

    def __init__(self):
        self._cfgs = {c: ex.load_config(_rq1_path(c)) for c in B_CODES}

    def __getattr__(self, name):
        return getattr(self._cfgs["B01"], name)

    def group_ids(self):
        return list(B_CODES)

    def group(self, code):
        return self._cfgs[code].group


def _traj(version, poses, actions=None, stage_events=None):
    """Minimal prompt-shaped trajectory for retrieval tests."""
    actions = actions if actions is not None else ["Move_Forward 1m"] * (len(poses) - 1)
    return {"version": version, "case_label": "T01/case01 (legacy 0)",
            "description": "test", "origin": {"coord": poses[0][:2],
                                              "heading": poses[0][2]},
            "actions": actions, "poses": poses,
            "image_paths": ["Grid_X%d_Y%d_%s.png" % (p[0], p[1], p[2]) for p in poses],
            "stage_events": stage_events or [], "length": len(actions),
            "source": "forward_only"}


class TestConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _Matrix1B()

    def test_nine_groups(self):
        self.assertEqual(self.cfg.group_ids(),
                         ["B0%d" % i for i in range(1, 10)])

    def test_hyperparameters_match_experiment_table(self):
        """Sheet1 of experiment.xls, encoded as assertions."""
        cfg = self.cfg
        self.assertEqual(cfg.test_version, "v4")
        self.assertEqual(cfg.history_steps, 6)      # h=6
        self.assertEqual(cfg.history_images, 6)     # i=6
        self.assertEqual(cfg.max_steps, 200)
        self.assertEqual(cfg.model["temperature"], 0.0)
        self.assertEqual(cfg.model["name"], "intern-s2-preview-397b")

        expected = {
            # group: (memory, versions, per-traj images, ref cap, total cap)
            "B01": ("none",         [],                             0,  0,  7),
            "B02": ("full_context", ["v0"],                        40, 40, 47),
            "B03": ("full_context", ["v0", "v1", "v2", "v3"],      10, 40, 47),
            "B04": ("episodic",     ["v0"],                         0,  7, 14),
            "B05": ("episodic",     ["v0", "v1", "v2", "v3"],       0,  7, 14),
            "B06": ("summary",      ["v0"],                         0,  0,  7),
            "B07": ("summary",      ["v0", "v1", "v2", "v3"],       0,  0,  7),
            "B08": ("semantic",     ["v0"],                         0,  0,  7),
            "B09": ("semantic",     ["v0", "v1", "v2", "v3"],       0,  0,  7),
        }
        for gid, (mem, versions, per_traj, ref_cap, total_cap) in expected.items():
            g = cfg.group(gid)
            self.assertEqual(g.memory, mem, gid)
            self.assertEqual(list(g.versions), versions, gid)
            self.assertEqual(g.max_smart_images_per_traj, per_traj, gid)
            self.assertEqual(g.max_reference_images, ref_cap, gid)
            self.assertEqual(g.max_total_images, total_cap, gid)

    def test_episodic_groups_use_vit_horizon_six_content_all(self):
        """B04 and B05 must differ only in pool size.

        Both retrieve with ViT; the whole point of the pair is that the
        candidate pool is the single variable between them, so horizon,
        matcher and content have to stay identical.
        """
        for gid in ("B04", "B05"):
            g = self.cfg.group(gid)
            self.assertEqual(g.episodic_horizon, 6, gid)
            self.assertEqual(g.episodic_matcher, "vit", gid)
            self.assertEqual(g.episodic_content, "all", gid)
        g4, g5 = self.cfg.group("B04"), self.cfg.group("B05")
        self.assertEqual(list(g4.versions), ["v0"])
        self.assertEqual(list(g5.versions), ["v0", "v1", "v2", "v3"])
        for field in ("episodic_horizon", "episodic_matcher", "episodic_content",
                      "max_reference_images", "max_total_images"):
            self.assertEqual(getattr(g4, field), getattr(g5, field), field)

    def test_image_budget_arithmetic(self):
        """1 current + <=6 history + memory images must stay under each cap."""
        cfg = self.cfg
        for gid in cfg.group_ids():
            g = cfg.group(gid)
            worst = 1 + cfg.history_images + g.max_reference_images
            self.assertEqual(worst, g.max_total_images,
                             "%s: 1+%d+%d != %d" % (gid, cfg.history_images,
                                                    g.max_reference_images,
                                                    g.max_total_images))
            self.assertLessEqual(g.max_total_images, cfg.max_prompt_images, gid)

    def test_multi_version_per_traj_budget_sums_to_cap(self):
        g = self.cfg.group("B03")
        self.assertEqual(g.max_smart_images_per_traj * len(g.versions),
                         g.max_reference_images)

    def test_fingerprint_is_stable_and_content_sensitive(self):
        cfg = ex.load_config(B01)
        first = cfg.fingerprint()
        self.assertEqual(first, ex.load_config(B01).fingerprint())
        raw = json.loads(json.dumps(cfg.raw))
        raw["working_memory"]["history_steps"] = 5
        self.assertNotEqual(first, ex.ExperimentConfig(raw, B01).fingerprint())


class TestCaseIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _Matrix1B()

    def test_hundred_unique_cases(self):
        keys = self.cfg.case_map.all_keys()
        self.assertEqual(len(keys), 100)
        self.assertEqual(sorted(k.legacy_case_id for k in keys), list(range(100)))

    def test_t02_and_t03_task_types_are_swapped(self):
        """T02 is task_type 3 and T03 is task_type 2 — directory order lies."""
        self.assertEqual(self.cfg.case_map.by_logical("T02", 1).task_type, 3)
        self.assertEqual(self.cfg.case_map.by_logical("T03", 1).task_type, 2)
        self.assertEqual(self.cfg.case_map.by_logical("T02", 1).legacy_case_id, 20)
        self.assertEqual(self.cfg.case_map.by_logical("T03", 1).legacy_case_id, 10)

    def test_same_case_no_across_tasks_is_distinct(self):
        a = self.cfg.case_map.by_logical("T01", 1)
        b = self.cfg.case_map.by_logical("T05", 1)
        self.assertNotEqual(a.legacy_case_id, b.legacy_case_id)


class TestDataLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _Matrix1B()
        cls.names = ex.scene_base64_names(cls.cfg)

    def test_v4_matches_v0_except_origin(self):
        tasks, _ = ex.load_test_tasks(self.cfg)
        legacy = ex.resolve_path(self.cfg.raw["scene"]["legacy_tasks"])
        inv = ex.assert_v4_matches_v0_tasks(self.cfg, tasks, legacy)
        self.assertEqual(inv["cases"], 100)
        self.assertEqual(inv["goals_identical"], 70)
        self.assertEqual(len(inv["return_to_origin_coord_updated"]), 30)

    def test_all_four_versions_load_for_every_case(self):
        for key in self.cfg.case_map.all_keys():
            trajs = ex.load_history_set(self.cfg, key, ["v0", "v1", "v2", "v3"],
                                        base64_names=self.names)
            self.assertEqual([t.version for t in trajs], ["v0", "v1", "v2", "v3"])
            self.assertEqual(len({t.source_path for t in trajs}), 4,
                             "%s: versions overwrote each other" % key.label())

    def test_missing_version_is_an_error_not_a_fallback(self):
        key = self.cfg.case_map.by_legacy(0)
        with self.assertRaises(ex.ExperimentError):
            ex.load_history_trajectory(self.cfg, key, "v9")

    def test_v4_is_not_a_loadable_history_version(self):
        """The test set's expert route must be unreachable from the loader."""
        key = self.cfg.case_map.by_legacy(0)
        with self.assertRaises(ex.ExperimentError):
            ex.load_history_trajectory(self.cfg, key, "v4")

    def test_only_known_stage_anomaly_is_tolerated(self):
        noted = []
        for key in self.cfg.case_map.all_keys():
            for t in ex.load_history_set(self.cfg, key, ["v0", "v1", "v2", "v3"],
                                         base64_names=self.names):
                if t.validation_notes:
                    noted.append((key.task_id, key.case_no, t.version))
        self.assertEqual(noted, [("T09", 10, "v0")])

    def test_text_memory_hash_mismatch_is_fatal(self):
        raw = json.loads(json.dumps(self.cfg.raw))
        raw["memory_files"]["summary"]["v0"]["sha256"] = "0" * 64
        cfg = ex.ExperimentConfig(raw, B01)
        with self.assertRaises(ex.ExperimentError):
            ex.load_text_memory(cfg, "v0", "summary")

    def test_all_eight_memories_load_and_cover_case_99(self):
        for row in ex.memory_profile(self.cfg):
            self.assertGreater(row["chars"], 0, row)
            self.assertEqual(row["max_case_mentioned"], 99, row)


class TestSmartImageBudget(unittest.TestCase):
    def test_budget_zero_returns_nothing(self):
        actions = ["Turn_Left"] * 10
        self.assertEqual(_select_smart_indices(actions, [], 0), [])

    def test_hard_cap_holds_when_priority_layers_overflow(self):
        # 15 turns + 4 stage events: the priority layers alone want 21
        # frames, far more than a budget of 10.
        actions = ["Turn_Left"] * 15 + ["Move_Forward 1m"] * 5
        events = [{"step": s, "label": "stage"} for s in (3, 6, 9, 12)]
        for budget in (1, 2, 5, 10, 20, 40):
            picked = _select_smart_indices(actions, events, budget)
            self.assertLessEqual(len(picked), budget, budget)
            idx = [i for i, _ in picked]
            self.assertEqual(idx, sorted(idx), "not in time order")
            self.assertEqual(len(set(idx)), len(idx), "duplicate index")

    def test_duplicate_index_takes_one_slot_and_keeps_both_labels(self):
        # Single Turn action: index 0 is start, index 1 is the final frame.
        picked = dict(_select_smart_indices(["Turn_Left"], [], 5))
        self.assertEqual(picked[0], "start")
        self.assertEqual(picked[1], "task_complete")

    def test_short_trajectory_is_not_padded(self):
        actions = ["Move_Forward 1m"] * 3          # 4 frames available
        picked = _select_smart_indices(actions, [], 10)
        self.assertEqual(len(picked), 4)


class TestEpisodicRetrieval(unittest.TestCase):
    def setUp(self):
        self.ctx = DecisionContext()

    def test_exact_beats_cell_beats_nearest(self):
        near = _traj("v0", [[5, 5, "East"], [6, 5, "East"]])
        cell = _traj("v1", [[1, 1, "West"], [2, 2, "North"]])
        exact = _traj("v2", [[9, 9, "South"], [1, 1, "East"]])
        target = {"x": 1, "y": 1, "heading": "East"}
        m = find_best_match([near, cell, exact], target, self.ctx)
        self.assertEqual((m.version, m.match_kind), ("v2", "exact"))
        m = find_best_match([near, cell], target, self.ctx)
        self.assertEqual((m.version, m.match_kind), ("v1", "cell"))
        m = find_best_match([near], target, self.ctx)
        self.assertEqual((m.version, m.match_kind), ("v0", "nearest"))

    def test_version_priority_breaks_exact_ties(self):
        poses = [[1, 1, "East"], [2, 1, "East"]]
        trajs = [_traj(v, list(poses)) for v in ("v3", "v2", "v1", "v0")]
        m = find_best_match(trajs, {"x": 1, "y": 1, "heading": "East"}, self.ctx)
        self.assertEqual(m.version, "v0")

    def test_stage_difference_outranks_version_priority(self):
        # v0 offers the pose at stage 0; v1 offers it at stage 1, which is
        # where the agent actually is.
        v0 = _traj("v0", [[1, 1, "East"], [2, 1, "East"], [3, 1, "East"]])
        v1 = _traj("v1", [[9, 9, "East"], [1, 1, "East"], [3, 1, "East"]],
                   stage_events=[{"step": 1, "label": "hit"}])
        ctx = DecisionContext(stages_completed=1, steps_in_current_stage=0)
        m = find_best_match([v0, v1], {"x": 1, "y": 1, "heading": "East"}, ctx)
        self.assertEqual(m.version, "v1")
        self.assertEqual(m.historical_stage_progress, 1)

    def test_nearest_tier_compares_distance_first(self):
        far = _traj("v0", [[10, 10, "East"], [11, 10, "East"]])
        close = _traj("v1", [[2, 1, "East"], [3, 1, "East"]])
        m = find_best_match([far, close], {"x": 1, "y": 1, "heading": "North"},
                            self.ctx)
        self.assertEqual(m.version, "v1")
        self.assertAlmostEqual(m.distance, 1.0)

    def test_single_trajectory_degenerates_to_single_version(self):
        v0 = _traj("v0", [[1, 1, "East"], [2, 1, "East"]])
        m = find_best_match([v0], {"x": 2, "y": 1, "heading": "East"}, self.ctx)
        self.assertEqual((m.version, m.matched_index), ("v0", 1))

    def test_lowest_step_index_is_the_final_tiebreak(self):
        v0 = _traj("v0", [[1, 1, "East"], [5, 5, "North"], [1, 1, "East"]])
        m = find_best_match([v0], {"x": 1, "y": 1, "heading": "East"}, self.ctx)
        self.assertEqual(m.matched_index, 0)

    def test_stage_progress_helper(self):
        t = _traj("v0", [[0, 0, "East"]] * 5,
                  stage_events=[{"step": 2, "label": "a"}, {"step": 4, "label": "b"}])
        self.assertEqual(stage_progress_by_step(t), [0, 0, 1, 1, 2])


class TestPromptAssembly(unittest.TestCase):
    """Builds every group's request offline and checks the invariants."""

    @classmethod
    def setUpClass(cls):
        import agent as ag
        cls.ag = ag
        cls.cfg = _Matrix1B()
        cls.names = ex.scene_base64_names(cls.cfg)
        cls.tasks, _ = ex.load_test_tasks(cls.cfg)

    def tearDown(self):
        self.ag.EXPERIMENT = None
        self.ag.EXPERIMENT_GROUP = None
        self.ag.EXPERIMENT_MEMORY_TEXT = ""

    def _build(self, gid, legacy_id=0, history_text=None, history_images=None):
        ag, cfg = self.ag, self.cfg
        g = cfg.group(gid)
        key = cfg.case_map.by_legacy(legacy_id)
        task = self.tasks[legacy_id]
        origin = {"x": int(task.origin["coord"][0]),
                  "y": int(task.origin["coord"][1]),
                  "heading": task.origin["heading"]}
        img = os.path.join(cfg.scene_image_dir, "Grid_X%d_Y%d_%s.png"
                           % (origin["x"], origin["y"], origin["heading"]))
        ag.EXPERIMENT, ag.EXPERIMENT_GROUP = cfg, g
        ag.MAX_HISTORY_STEPS = cfg.history_steps
        ag.LOOKBACK_IMAGE_COUNT = cfg.history_images
        ag.MAX_PROMPT_IMAGES = cfg.max_prompt_images
        ag.ENABLE_HISTORY = True
        ag.EXPERIMENT_MEMORY_TEXT = ""
        trajs = []
        if g.uses_text_memory:
            from prompt import wrap_text_memory
            mems = ex.load_text_memory_set(cfg, g.versions, g.memory)
            ag.EXPERIMENT_MEMORY_TEXT = wrap_text_memory(
                mems, g.memory.capitalize() + " memory")
        if g.uses_history_trajectories:
            loaded = ex.load_history_set(cfg, key, g.versions, base64_names=self.names)
            trajs = [t.as_prompt_traj(cfg.scene_image_dir) for t in loaded]
        ctx = ag.DecisionContext(case_label=(key.task_id, key.case_no),
                                 initial_origin=dict(origin), current_pose=dict(origin),
                                 decision_index=1)
        telemetry = {}
        parts = ag.build_experiment_user_content(
            task.description, origin, origin, history_text or [], img,
            history_images or [], trajs, ctx, telemetry)
        return parts, telemetry

    @staticmethod
    def _text(parts):
        return "".join(b.get("text", "") for b in parts if b.get("type") == "text")

    @staticmethod
    def _images(parts):
        return sum(1 for b in parts if b.get("type") == "image_url")

    def test_every_group_builds_and_respects_its_image_cap(self):
        for gid in self.cfg.group_ids():
            parts, tel = self._build(gid)
            n = self._images(parts)
            self.assertLessEqual(n, self.cfg.group(gid).max_total_images, gid)
            self.assertEqual(n, tel["images"]["total"], gid)

    def test_non_memory_body_is_identical_across_all_nine_groups(self):
        bodies = set()
        for gid in self.cfg.group_ids():
            parts, _ = self._build(gid)
            keep = [(b.get("type"), b.get("text"), b.get("_image_path"))
                    for b in parts if b.get("_section") != "memory"]
            bodies.add(json.dumps(keep, ensure_ascii=False))
        self.assertEqual(len(bodies), 1,
                         "groups differ outside their memory sections")

    def test_baseline_carries_no_memory_at_all(self):
        parts, _ = self._build("B01")
        text = self._text(parts)
        for marker in ("HISTORICAL", "MEMORY RECALL", "EPISODIC"):
            self.assertNotIn(marker, text)
        self.assertEqual(self._images(parts), 1)

    def test_mission_block_shows_the_fixed_v4_origin(self):
        # legacy 0 is T01/case01, whose v4 spawn is (-5, 16) facing North.
        parts, _ = self._build("B01")
        self.assertIn("Your initial spawn: (-5, 16) facing North",
                      self._text(parts))

    def test_initial_spawn_stays_fixed_while_the_pose_moves(self):
        ag, cfg = self.ag, self.cfg
        g = cfg.group("B01")
        ag.EXPERIMENT, ag.EXPERIMENT_GROUP = cfg, g
        ag.MAX_HISTORY_STEPS, ag.LOOKBACK_IMAGE_COUNT = 6, 6
        ag.MAX_PROMPT_IMAGES, ag.ENABLE_HISTORY = 50, True
        initial = {"x": -5, "y": 16, "heading": "North"}
        moved = {"x": -5, "y": 14, "heading": "South"}
        img = os.path.join(cfg.scene_image_dir, "Grid_X-5_Y14_South.png")
        ctx = ag.DecisionContext(case_label=("T01", 1), initial_origin=initial,
                                 current_pose=moved, decision_index=3)
        parts = ag.build_experiment_user_content(
            "Find any instance of Tomato Soup Can.", initial, moved,
            ["Step 1: Turn_180 -> success"], img, [], [], ctx, {})
        text = self._text(parts)
        self.assertIn("Your initial spawn: (-5, 16) facing North", text)
        self.assertIn("Current view: you are at (-5, 14) facing South", text)

    def test_full_context_multi_lists_all_four_versions_once(self):
        parts, tel = self._build("B03")
        text = self._text(parts)
        for v in ("V0", "V1", "V2", "V3"):
            self.assertEqual(text.count("[HISTORICAL MEMORY %s — BEGIN]" % v), 1, v)
            self.assertEqual(text.count("[HISTORICAL MEMORY %s — END]" % v), 1, v)
        self.assertEqual([s["version"] for s in tel["full_context"]],
                         ["v0", "v1", "v2", "v3"])
        self.assertLessEqual(tel["reference_images"],
                             self.cfg.group("B03").max_reference_images)
        for s in tel["full_context"]:
            self.assertLessEqual(s["images_sent"], 10)

    def test_full_context_single_lists_only_v0(self):
        text = self._text(self._build("B02")[0])
        self.assertIn("[HISTORICAL MEMORY V0 — BEGIN]", text)
        for v in ("V1", "V2", "V3"):
            self.assertNotIn("[HISTORICAL MEMORY %s — BEGIN]" % v, text)

    def test_full_context_actions_match_the_source_verbatim(self):
        cfg = self.cfg
        key = cfg.case_map.by_legacy(0)
        traj = ex.load_history_trajectory(cfg, key, "v0", base64_names=self.names)
        text = self._text(self._build("B02")[0])
        self.assertIn("Trajectory Sequence: [%s]" % ", ".join(traj.actions), text)
        self.assertIn("Historical trajectory length: %d steps" % traj.length, text)

    def test_episodic_returns_one_excerpt_not_four(self):
        parts, tel = self._build("B05")
        text = self._text(parts)
        self.assertEqual(text.count("[EPISODIC MEMORY - RETRIEVED HISTORICAL EXCERPT]"), 1)
        self.assertEqual(text.count("[matched view |"), 1)
        self.assertEqual(tel["episodic_candidates"], 4)
        begins = sum(text.count("[HISTORICAL MEMORY %s — BEGIN]" % v.upper())
                     for v in ("v0", "v1", "v2", "v3"))
        self.assertEqual(begins, 1)

    def test_episodic_does_not_dump_whole_trajectories(self):
        text = self._text(self._build("B05")[0])
        self.assertNotIn("Trajectory Sequence:", text)

    def test_episodic_drops_the_old_prior_successful_run_wording(self):
        for gid in ("B04", "B05"):
            text = self._text(self._build(gid)[0])
            self.assertNotIn("prior successful run", text, gid)
            self.assertNotIn("prior-successful", text, gid)
            self.assertNotIn("THIS same case successfully", text, gid)

    def test_text_memory_groups_include_each_version_once(self):
        for gid, kind in (("B07", "summary"), ("B09", "semantic")):
            text = self._text(self._build(gid)[0])
            for v in ("V0", "V1", "V2", "V3"):
                self.assertEqual(text.count("[HISTORICAL MEMORY %s — BEGIN]" % v), 1,
                                 "%s %s" % (gid, v))
            for version in ("v0", "v1", "v2", "v3"):
                mem = ex.load_text_memory(self.cfg, version, kind)
                self.assertIn(mem.text.strip(), text, "%s %s" % (gid, version))

    def test_summary_and_semantic_never_mix(self):
        sem_v0 = ex.load_text_memory(self.cfg, "v0", "semantic").text.strip()
        sum_v0 = ex.load_text_memory(self.cfg, "v0", "summary").text.strip()
        summary_text = self._text(self._build("B07")[0])
        semantic_text = self._text(self._build("B09")[0])
        self.assertNotIn(sem_v0, summary_text)
        self.assertNotIn(sum_v0, semantic_text)

    def test_text_memory_groups_send_no_expert_trajectory(self):
        for gid in ("B06", "B07", "B08", "B09"):
            text = self._text(self._build(gid)[0])
            self.assertNotIn("Trajectory Sequence:", text, gid)
            self.assertNotIn("[EPISODIC MEMORY", text, gid)

    def test_single_version_group_also_gets_the_common_note(self):
        """Sheet2: the annotation must not be an extra hint given only to
        the multi-version arm."""
        for single, multi in (("B02", "B03"), ("B04", "B05"),
                              ("B06", "B07"), ("B08", "B09")):
            a = self._text(self._build(single)[0])
            b = self._text(self._build(multi)[0])
            self.assertIn("[HISTORICAL EXPERIENCE — COMMON NOTE]", a, single)
            self.assertIn("[HISTORICAL EXPERIENCE — COMMON NOTE]", b, multi)

    def test_memory_blocks_are_tagged_and_stripped_from_the_wire(self):
        parts, _ = self._build("B03")
        self.assertTrue(any(b.get("_section") == "memory" for b in parts))
        payload = {"model": "m", "temperature": 0, "max_tokens": 30,
                   "messages": [{"role": "system", "content": "sys"},
                                {"role": "user", "content": parts}]}
        wire = self.ag._strip_private_blocks(payload)
        for blk in wire["messages"][1]["content"]:
            self.assertFalse([k for k in blk if k.startswith("_")], blk.keys())

    def test_history_images_are_capped_at_six(self):
        logs = ["Step %d: Move_Forward 1m -> success at (0, %d) North" % (i, i)
                for i in range(1, 11)]
        imgs = [{"action": "Move_Forward 1m", "status": "moved",
                 "path": os.path.join(self.cfg.scene_image_dir,
                                      "Grid_X-5_Y%d_North.png" % (16 - i)),
                 "pose": {"x": -5, "y": 16 - i, "heading": "North"}}
                for i in range(10)]
        parts, tel = self._build("B01", history_text=logs, history_images=imgs)
        self.assertLessEqual(tel["images"]["history"], 6)
        self.assertLessEqual(tel["images"]["total"], 7)


class TestActionParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import agent as ag
        cls.parse = staticmethod(ag.parse_model_action)

    def test_exact_and_tolerated_forms(self):
        self.assertEqual(self.parse("Move_Forward 1m")[0], "Move_Forward 1m")
        self.assertEqual(self.parse("  Turn_Left \n")[0], "Turn_Left")
        self.assertEqual(self.parse("```Turn_180```")[0], "Turn_180")
        self.assertEqual(self.parse('"Move_Left 1m"')[0], "Move_Left 1m")
        self.assertEqual(self.parse("Move_Forward")[0], "Move_Forward 1m")
        self.assertEqual(self.parse("I will Move_Forward 1m")[0], "Move_Forward 1m")

    def test_no_turn_left_fallback(self):
        """v0 substituted Turn_Left for anything unparseable."""
        for bad in ("", "   ", "banana", "I am not sure", None):
            action, reason = self.parse(bad)
            self.assertIsNone(action, bad)
            self.assertTrue(reason)

    def test_ambiguous_output_is_a_failure_not_a_guess(self):
        action, reason = self.parse("Turn_Left then Move_Forward")
        self.assertIsNone(action)
        self.assertIn("ambiguous", reason)


class TestProviders(unittest.TestCase):
    """Two API dialects: intern-ai (errors as HTTP 200 + an envelope) and
    OpenAI-compatible gateways (real HTTP status codes)."""

    @classmethod
    def setUpClass(cls):
        import agent as ag
        cls.ag = ag

    def tearDown(self):
        self.ag.API_PROVIDER = "internai"
        self.ag.ACTION_EXTRACTION = "strict"

    def test_internai_200_error_envelope_is_a_key_failure(self):
        """This endpoint reports dead keys as HTTP 200 + {"code","msg"}."""
        err = self.ag._response_error(
            {"code": -20035, "msg": "please bind a phone number", "data": {}})
        self.assertIsNotNone(err)
        self.assertIn("-20035", err)
        self.assertIsNone(self.ag._response_error(
            {"choices": [{"message": {"content": "Turn_Left"}}]}))

    def test_thinking_mode_is_never_sent(self):
        """It suppresses reasoning on one gateway and has no counterpart on
        the other, so sending it would make the same model behave
        differently depending on which key served the request.

        Checked against the parsed AST rather than the source text, so a
        docstring that merely explains the decision does not pass or fail
        the test by accident."""
        import ast as _ast
        import inspect
        tree = _ast.parse(inspect.getsource(self.ag.call_internvl).lstrip())
        docstrings = set()
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef,
                                 _ast.ClassDef, _ast.Module)):
                d = _ast.get_docstring(node, clean=False)
                if d:
                    docstrings.add(d)
        live = [n.value for n in _ast.walk(tree)
                if isinstance(n, _ast.Constant) and isinstance(n.value, str)
                and n.value not in docstrings]
        self.assertNotIn("thinking_mode", live,
                         "thinking_mode reached the request body")

    def test_last_extraction_recovers_a_reasoning_model_answer(self):
        """Intern-S2-Preview on blsc narrates, then answers."""
        reply = ("Thinking Process:\n1. I could Turn_Left to face the aisle.\n"
                 "2. But the target is straight ahead.\n\nMove_Forward 1m")
        self.ag.ACTION_EXTRACTION = "strict"
        action, reason = self.ag.parse_model_action(reply)
        self.assertIsNone(action)
        self.assertIn("ambiguous", reason)
        self.ag.ACTION_EXTRACTION = "last"
        action, reason = self.ag.parse_model_action(reply)
        self.assertEqual(action, "Move_Forward 1m")

    def test_last_extraction_does_not_change_unambiguous_replies(self):
        for mode in ("strict", "last"):
            self.ag.ACTION_EXTRACTION = mode
            self.assertEqual(self.ag.parse_model_action("Turn_180")[0], "Turn_180", mode)
            self.assertEqual(self.ag.parse_model_action("Move_Forward")[0],
                             "Move_Forward 1m", mode)
            self.assertIsNone(self.ag.parse_model_action("banana")[0], mode)

    def test_inline_key_loading(self):
        pool = self.ag.load_api_keys_from_list(
            ["sk-aaa", " sk-bbb, ", "not-a-key"],
            default_url="https://x/y", default_model="M", default_provider="openai")
        self.assertEqual(sorted(e.key for e in pool), ["sk-aaa", "sk-bbb"])
        self.assertTrue(all(e.model == "M" and e.api_url == "https://x/y"
                            for e in pool))
        with self.assertRaises(SystemExit):
            self.ag.load_api_keys_from_list(["nope"], default_url="u", default_model="m")

    def test_per_key_endpoint_binding(self):
        """A key issued by one gateway is rejected by the other, so each key
        carries its own url/model/provider."""
        kb = "sk-blscAAAAAAAAAAAAAAAAAAAAAAAA123456"
        ki = "sk-iaBBBBBBBBBBBBBBBBBBBBBBBBBB987654"
        pool = self.ag.load_api_keys_from_list([
            {"key": kb, "api_url": "https://llmapi.blsc.cn/v1/chat/completions",
             "model": "Intern-S2-Preview", "provider": "openai", "label": "blsc"},
            {"key": ki, "api_url": "https://chat.intern-ai.org.cn/api/v1/chat/completions",
             "model": "intern-s2-preview-35b", "provider": "internai", "label": "internai"},
        ], default_url="https://unused/", default_model="unused")
        by_key = {e.key: e for e in pool}
        self.assertEqual(by_key[kb].model, "Intern-S2-Preview")
        self.assertEqual(by_key[kb].provider, "openai")
        self.assertEqual(by_key[ki].model, "intern-s2-preview-35b")
        self.assertEqual(by_key[ki].provider, "internai")
        self.assertNotEqual(by_key[kb].api_url, by_key[ki].api_url)
        # The manifest keeps only the last few characters of a key.
        dumped = json.dumps(by_key[kb].as_dict())
        self.assertNotIn(kb, dumped)
        self.assertIn(kb[-6:], dumped)


class TestRateLimitBackoff(unittest.TestCase):
    """intern-ai reports throttling as HTTP 400 + code -20048, not 429."""

    @classmethod
    def setUpClass(cls):
        import agent as ag
        cls.ag = ag

    def setUp(self):
        self.ag._KEY_COOLDOWN.clear()

    tearDown = setUp

    def test_detects_the_intern_ai_throttle_shape(self):
        body = ('{"error":{"type":"invalid_request_error","code":"-20048",'
                '"message":"请求过于频繁"}}')
        self.assertTrue(self.ag._is_rate_limited(400, body))
        self.assertTrue(self.ag._is_rate_limited(429, ""))
        self.assertTrue(self.ag._is_rate_limited(400, "Rate limit exceeded"))

    def test_does_not_mistake_a_real_error_for_throttling(self):
        self.assertFalse(self.ag._is_rate_limited(400, '{"error":"bad model id"}'))
        self.assertFalse(self.ag._is_rate_limited(401, "Authentication Error"))
        self.assertFalse(self.ag._is_rate_limited(500, "internal error"))

    def test_backoff_grows_then_caps(self):
        waits = [self.ag._cool_down(0) for _ in range(8)]
        self.assertEqual(waits[0], self.ag.RATE_LIMIT_COOLDOWN)
        self.assertTrue(all(b >= a for a, b in zip(waits, waits[1:])),
                        "backoff must not shrink: %s" % waits)
        self.assertLessEqual(max(waits), self.ag.RATE_LIMIT_COOLDOWN_MAX)

    def test_a_throttled_key_is_skipped_while_others_stay_ready(self):
        self.ag._cool_down(1)
        ready, earliest = self.ag._available_keys([0, 1, 2])
        self.assertEqual(sorted(ready), [0, 2])
        self.assertIsNone(earliest)

    def test_when_every_key_is_cooling_the_caller_is_told_when_to_retry(self):
        for i in (0, 1, 2):
            self.ag._cool_down(i)
        ready, earliest = self.ag._available_keys([0, 1, 2])
        self.assertEqual(ready, [])
        self.assertIsNotNone(earliest)

    def test_success_clears_the_backoff(self):
        self.ag._cool_down(0)
        self.assertEqual(self.ag._available_keys([0])[0], [])
        self.ag._clear_cooldown(0)
        self.assertEqual(self.ag._available_keys([0])[0], [0])

    def test_concurrent_backoff_updates_do_not_corrupt_state(self):
        import threading
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            for _ in range(50):
                self.ag._cool_down(7)
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        until, hits = self.ag._KEY_COOLDOWN[7]
        self.assertEqual(hits, 400, "lost updates under concurrency")


if __name__ == "__main__":
    unittest.main(verbosity=2)
