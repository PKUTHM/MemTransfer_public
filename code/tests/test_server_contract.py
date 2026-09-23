"""Server contract checks: v4 wiring, action_seq idempotency, concurrency.

No model calls — the "agent" here is a scripted action sequence, so this
exercises the parts of the loop that decide whether 100 cases can safely
run in parallel against one server.

Run from the repo root:
    python code/tests/test_server_contract.py
"""
import os
import random
import sys
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import server                                    # noqa: E402
import experiment as ex                          # noqa: E402

CONFIG = os.path.join(PROJECT_ROOT, "config", "rq1",
                      "B01_working_memory_only__test_v4.yaml")


def setUpModule():
    server.load_experiment(CONFIG)
    server.load_environment()
    server.app.config["TESTING"] = True


class TestExperimentWiring(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def test_server_advertises_the_experiment(self):
        info = self.client.get("/get_server_time").get_json()
        exp = info["experiment"]
        cfg = ex.load_config(CONFIG)
        self.assertEqual(exp["test_version"], "v4")
        self.assertEqual(exp["experiment_id"], cfg.experiment_id)
        self.assertEqual(exp["config_fingerprint"], cfg.fingerprint())
        self.assertEqual(len(exp["case_ids"]), 100)
        self.assertEqual(info["config"]["max_steps"], 200)

    def test_init_returns_the_v4_origin_and_logical_identity(self):
        r = self.client.post("/init_task",
                             json={"case_id": 0, "agent_id": "t_identity"}).get_json()
        self.assertEqual(r["origin"], {"coord": [-5, 16], "heading": "North"})
        self.assertEqual(r["case_identity"],
                         {"task_id": "T01", "case_no": 1,
                          "legacy_case_id": 0, "task_type": 1})
        self.assertEqual(r["test_version"], "v4")
        self.client.post("/end_case", json={"agent_id": "t_identity"})

    def test_t02_t03_case_ids_keep_their_swapped_task_types(self):
        for case_id, task_id, task_type in ((20, "T02", 3), (10, "T03", 2)):
            aid = "t_swap_%d" % case_id
            r = self.client.post("/init_task",
                                 json={"case_id": case_id, "agent_id": aid}).get_json()
            self.assertEqual(r["case_identity"]["task_id"], task_id)
            self.assertEqual(r["task_type"], task_type)
            self.client.post("/end_case", json={"agent_id": aid})

    def test_return_to_origin_target_follows_the_v4_spawn(self):
        """T07 case01: the goal must be the v4 spawn, not the v0 one."""
        aid = "t_rto"
        r = self.client.post("/init_task",
                             json={"case_id": 60, "agent_id": aid}).get_json()
        self.assertEqual(r["origin"]["coord"], [-3, 13])   # v4, not v0's [-6, 14]
        state = server.AGENT_SESSIONS[aid]
        self.assertEqual(state["origin"], (-3, 13))
        self.client.post("/end_case", json={"agent_id": aid})


class TestActionSeqIdempotency(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        self.aid = "t_seq_%d" % random.randint(0, 10 ** 9)
        self.client.post("/init_task", json={"case_id": 0, "agent_id": self.aid})

    def tearDown(self):
        self.client.post("/end_case", json={"agent_id": self.aid})

    def _step(self, action, seq):
        return self.client.post("/step", json={"action": action,
                                               "agent_id": self.aid,
                                               "action_seq": seq})

    def test_replaying_a_sequence_number_does_not_move_twice(self):
        first = self._step("Move_Forward 1m", 1).get_json()
        state_after = dict(server.AGENT_SESSIONS[self.aid])
        replay = self._step("Move_Forward 1m", 1).get_json()
        self.assertEqual(replay["current_pose"], first["current_pose"])
        self.assertEqual(replay["step"], first["step"])
        self.assertTrue(replay["replayed"])
        self.assertEqual(server.AGENT_SESSIONS[self.aid]["step_count"],
                         state_after["step_count"])

    def test_same_sequence_with_a_different_action_is_rejected(self):
        self._step("Turn_Left", 1)
        resp = self._step("Turn_Right", 1)
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(server.AGENT_SESSIONS[self.aid]["step_count"], 1)

    def test_gaps_are_rejected(self):
        self._step("Turn_Left", 1)
        self.assertEqual(self._step("Turn_Left", 3).status_code, 409)
        self.assertEqual(server.AGENT_SESSIONS[self.aid]["step_count"], 1)

    def test_step_count_tracks_action_seq_exactly(self):
        for i in range(1, 6):
            self._step("Turn_Left", i)
            self.assertEqual(server.AGENT_SESSIONS[self.aid]["step_count"], i)

    def test_replaying_the_final_completing_step_returns_the_same_result(self):
        """A session that already ended must still answer its last retry."""
        aid = "t_final_%d" % random.randint(0, 10 ** 9)
        client = server.app.test_client()
        client.post("/init_task", json={"case_id": 0, "agent_id": aid})
        state = server.AGENT_SESSIONS[aid]
        seq = 0
        done = None
        # Walk onto a goal reachable point directly through the state.
        target = tuple(int(c) for c in state["goals"][0]["instances"][0]["reachable_points"][0])
        state["x"], state["y"] = target[0], target[1] + 1
        state["d"] = "South"
        seq += 1
        done = client.post("/step", json={"action": "Move_Forward 1m",
                                          "agent_id": aid, "action_seq": seq}).get_json()
        self.assertEqual(done["status"], "task_completed")
        replay = client.post("/step", json={"action": "Move_Forward 1m",
                                            "agent_id": aid,
                                            "action_seq": seq}).get_json()
        self.assertEqual(replay["status"], "task_completed")
        self.assertEqual(replay["current_pose"], done["current_pose"])
        client.post("/end_case", json={"agent_id": aid})

    def test_end_case_releases_the_session_and_its_cache(self):
        self._step("Turn_Left", 1)
        self.assertIn(self.aid, server.AGENT_SESSIONS)
        r = self.client.post("/end_case", json={"agent_id": self.aid}).get_json()
        self.assertTrue(r["released"])
        self.assertNotIn(self.aid, server.AGENT_SESSIONS)
        # Idempotent.
        self.assertFalse(self.client.post("/end_case",
                                          json={"agent_id": self.aid}).get_json()["released"])


class TestConcurrentCases(unittest.TestCase):
    """The runner drives up to max_workers cases against one server."""

    def test_sessions_stay_independent_under_concurrency(self):
        n_cases, n_steps = 16, 25
        case_ids = list(range(n_cases))
        errors = []
        finals = {}
        barrier = threading.Barrier(n_cases)

        def drive(case_id):
            client = server.app.test_client()
            aid = "conc_%03d" % case_id
            try:
                init = client.post("/init_task",
                                   json={"case_id": case_id, "agent_id": aid}).get_json()
                origin = init["origin"]["coord"]
                barrier.wait()  # maximise interleaving
                rng = random.Random(case_id)
                seq = 0
                pose = init["current_pose"]
                for _ in range(n_steps):
                    action = rng.choice(["Turn_Left", "Turn_Right", "Turn_180"])
                    seq += 1
                    r = client.post("/step", json={"action": action, "agent_id": aid,
                                                   "action_seq": seq}).get_json()
                    if r.get("status") == "error":
                        errors.append((case_id, r))
                        return
                    pose = r["current_pose"]
                    if r["step"] != seq:
                        errors.append((case_id, "step %s != seq %s" % (r["step"], seq)))
                        return
                # Turning never moves: the position must equal the v4 origin.
                if [pose["x"], pose["y"]] != list(origin):
                    errors.append((case_id, "drifted from %s to %s" % (origin, pose)))
                finals[case_id] = (origin, pose, seq)
            except Exception as e:
                errors.append((case_id, "%s: %s" % (type(e).__name__, e)))
            finally:
                client.post("/end_case", json={"agent_id": aid})

        threads = [threading.Thread(target=drive, args=(c,)) for c in case_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

        self.assertEqual(errors, [], "concurrent cases interfered with each other")
        self.assertEqual(len(finals), n_cases)
        # Every case kept its own distinct origin.
        origins = {c: tuple(v[0]) for c, v in finals.items()}
        self.assertEqual(len(origins), n_cases)
        for c, (origin, pose, seq) in finals.items():
            self.assertEqual(seq, n_steps, c)
        self.assertEqual(len(server.AGENT_SESSIONS), 0,
                         "sessions leaked after end_case")

    def test_concurrent_retries_of_the_same_action_do_not_double_move(self):
        """Two threads submitting the same action_seq must apply it once."""
        client = server.app.test_client()
        aid = "conc_replay"
        client.post("/init_task", json={"case_id": 0, "agent_id": aid})
        start = dict(server.AGENT_SESSIONS[aid])
        results, lock = [], threading.Lock()
        barrier = threading.Barrier(8)

        def submit():
            c = server.app.test_client()
            barrier.wait()
            r = c.post("/step", json={"action": "Turn_Left", "agent_id": aid,
                                      "action_seq": 1})
            with lock:
                results.append((r.status_code, r.get_json()))

        threads = [threading.Thread(target=submit) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        ok = [r for code, r in results if code == 200]
        self.assertEqual(len(ok), 8, "all 8 should get the same 200 answer")
        self.assertEqual({r["step"] for r in ok}, {1})
        self.assertEqual(server.AGENT_SESSIONS[aid]["step_count"], 1,
                         "the action was applied more than once")
        self.assertEqual(len(server.AGENT_SESSIONS[aid]["actions"]), 1)
        self.assertNotEqual(server.AGENT_SESSIONS[aid]["d"], start["d"])
        client.post("/end_case", json={"agent_id": aid})


class TestAgentThreadSafety(unittest.TestCase):
    """Retrieval must not leak stage state between concurrent cases."""

    def test_decision_context_is_per_call_not_module_global(self):
        from prompt.episodic_memory import find_best_match, DecisionContext

        poses = [[0, 0, "East"], [1, 0, "East"], [2, 0, "East"], [3, 0, "East"]]
        traj = {"version": "v0", "case_label": "T01/case01", "description": "d",
                "origin": {"coord": [0, 0], "heading": "East"},
                "actions": ["Move_Forward 1m"] * 3, "poses": poses,
                "image_paths": ["Grid_X%d_Y0_East.png" % p[0] for p in poses],
                "stage_events": [{"step": 2, "label": "s"}], "length": 3}
        target = {"x": 1, "y": 0, "heading": "East"}

        seen, lock = [], threading.Lock()
        barrier = threading.Barrier(12)

        def worker(stage):
            barrier.wait()
            for _ in range(60):
                m = find_best_match([traj], target,
                                    DecisionContext(stages_completed=stage))
                with lock:
                    seen.append((stage, m.stage_difference, m.historical_stage_progress))

        threads = [threading.Thread(target=worker, args=(i % 3,)) for i in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        # Index 1 sits at stage progress 0, so the difference is exactly
        # the caller's own stage count. Any cross-thread leak shows up as
        # a mismatch here.
        for stage, diff, prog in seen:
            self.assertEqual(prog, 0)
            self.assertEqual(diff, stage)

    def test_agent_module_globals_are_read_only_during_a_run(self):
        """Guard against a new mutable global sneaking back in."""
        import agent as ag
        mutable_shared = {
            name for name in dir(ag)
            if name.isupper() and isinstance(getattr(ag, name), (list, dict, set))
        }
        # Populated once in main() and only read afterwards, or constant
        # lookup tables that are never written at all.
        allowed = {"API_KEYS", "TRAJECTORY_INDEX", "EXPERIMENT_HISTORY",
                   "EXPERIMENT_MANIFEST", "REFERENCE_TRAJ_SMART_LAYERS",
                   "VALID_ACTIONS", "DIRS_CCW", "DELTAS",
                   "_BASE_MOVE_ACTIONS", "_AGENT_DELTAS",
                   "_AGENT_LEFT_OF", "_AGENT_RIGHT_OF",
                   "MODEL_EXTRA_BODY",
                   "EXPERIMENT_TEXT_BY_KIND",
                   # Shared on purpose: per-key rate-limit backoff must be
                   # visible across case threads so one throttled key is
                   # skipped by all of them. Every read and write goes
                   # through agent._COOLDOWN_LOCK.
                   "_KEY_COOLDOWN"}
        self.assertEqual(mutable_shared - allowed, set(),
                         "new mutable module global — check thread safety")
        # The one intentionally-shared dict must have its lock.
        self.assertTrue(hasattr(ag, "_COOLDOWN_LOCK"),
                        "_KEY_COOLDOWN is exempted on the grounds that it is "
                        "lock-guarded; the lock is missing")


class TestV5PerCaseScenes(unittest.TestCase):
    """v5 puts every case on its own two-cell obstacle scene.

    Geometry and renders both move: the reachable set drops from 225 cells
    to 223, and the frames that see the obstacle are re-rendered. Serving
    base-scene images while enforcing obstacle-scene movement (or the
    reverse) would be silent, so both are asserted together.
    """

    V5_CONFIG = os.path.join(PROJECT_ROOT, "config", "rq2",
                             "R02_working_memory_only__test_v5.yaml")

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(cls.V5_CONFIG):
            raise unittest.SkipTest("v5 config not present")
        server.load_experiment(cls.V5_CONFIG)
        server.load_environment()

    @classmethod
    def tearDownClass(cls):
        # Restore the module-wide v4 wiring the other classes rely on.
        server.load_experiment(CONFIG)
        server.load_environment()

    def setUp(self):
        self.client = server.app.test_client()

    def test_one_scene_env_per_obstacle_scene(self):
        self.assertEqual(len(server.SCENE_ENVS), 38)
        self.assertEqual(len(server.CASE_SCENE), 100)
        self.assertEqual({len(e.reachable) for e in server.SCENE_ENVS.values()},
                         {223})

    def test_init_reports_the_case_scene_and_serves_its_render(self):
        r = self.client.post("/init_task",
                             json={"case_id": 0, "agent_id": "t_v5_init"}).get_json()
        info = r["scene_info"]
        self.assertTrue(info["scene_id"].startswith("obs_"))
        self.assertEqual(info["reachable_count"], 223)
        self.assertEqual(r["test_version"], "v5")
        img = r["visual_observations"]["main_view"]
        self.assertIn(info["scene_id"], img,
                      "served a frame from another scene")
        # PNG directory is optional; base64 is sufficient for running experiments
        self.client.post("/end_case", json={"agent_id": "t_v5_init"})

    def test_v5_keeps_the_v4_origin(self):
        r = self.client.post("/init_task",
                             json={"case_id": 0, "agent_id": "t_v5_origin"}).get_json()
        self.assertEqual(r["origin"], {"coord": [-5, 16], "heading": "North"})
        self.client.post("/end_case", json={"agent_id": "t_v5_origin"})

    def test_obstacle_cells_are_blocked(self):
        """The whole point of v5: those two cells must stop a move."""
        import scene_index as si
        base = server.SceneEnv.from_scene(si.base_scene())
        checked = 0
        for scene_id, env in server.SCENE_ENVS.items():
            blocked = base.reachable - env.reachable
            self.assertEqual(len(blocked), 2,
                             "%s should block exactly 2 cells" % scene_id)
            for cell in blocked:
                self.assertNotIn(cell, env.reachable)
                self.assertIn(cell, base.reachable)
            checked += 1
        self.assertEqual(checked, 38)

    def test_stepping_into_the_obstacle_is_refused(self):
        """Drive a real session into a blocked cell and read the status."""
        import scene_index as si
        # find a case whose obstacle sits next to a reachable cell we can
        # stand on facing it
        deltas = server.DELTAS
        for case_id in range(100):
            scene_id = server.CASE_SCENE[case_id]
            env = server.SCENE_ENVS[scene_id]
            blocked = server.SceneEnv.from_scene(si.base_scene()).reachable - env.reachable
            for bx, by in blocked:
                for heading, (dx, dy) in deltas.items():
                    stand = (bx - dx, by - dy)
                    if stand not in env.reachable:
                        continue
                    aid = "t_v5_block"
                    self.client.post("/init_task",
                                     json={"case_id": case_id, "agent_id": aid})
                    st = server.AGENT_SESSIONS[aid]
                    st["x"], st["y"], st["d"] = int(stand[0]), int(stand[1]), heading
                    r = self.client.post("/step", json={
                        "agent_id": aid, "action": "Move_Forward"}).get_json()
                    self.client.post("/end_case", json={"agent_id": aid})
                    self.assertEqual(r["status"], "blocked",
                                     "case %d: moving into %s from %s facing %s "
                                     "should be blocked" % (case_id, (bx, by),
                                                            stand, heading))
                    return
        self.skipTest("no reachable cell adjacent to an obstacle was found")

    def test_the_same_move_succeeds_on_the_base_scene(self):
        """Control for the previous test: v4 lets the agent through."""
        import scene_index as si
        deltas = server.DELTAS
        base = server.SceneEnv.from_scene(si.base_scene())
        for case_id in range(100):
            env = server.SCENE_ENVS[server.CASE_SCENE[case_id]]
            for bx, by in (base.reachable - env.reachable):
                for heading, (dx, dy) in deltas.items():
                    stand = (bx - dx, by - dy)
                    if stand not in env.reachable:
                        continue
                    state = {"x": int(stand[0]), "y": int(stand[1]), "d": heading,
                             "scene_id": None}
                    benv = server.env_for(state)
                    self.assertIn((bx, by), benv.reachable,
                                  "the base scene should allow this cell")
                    return
        self.skipTest("no reachable cell adjacent to an obstacle was found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
