"""Offline checks for the scene-scoped image index (v4 / v5).

Standard-library ``unittest`` only, and no torch: everything here reads the
tables that ``scripts/build_scene_embeddings.py`` already produced.

Run from the repo root:
    python code/tests/test_scene_index.py
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import scene_index as si                                    # noqa: E402
import experiment as ex                                     # noqa: E402
from prompt import vit_matcher                              # noqa: E402
from prompt import episodic_memory as em                    # noqa: E402
from prompt import _helpers                                 # noqa: E402

V4_CONFIG = os.path.join(PROJECT_ROOT, "config", "rq1",
                         "B05_episodic_v0-v3__test_v4.yaml")
V5_CONFIG = os.path.join(PROJECT_ROOT, "config", "rq2",
                         "R05_budgeted_full_context_v0-v3__test_v5.yaml")

#: Every scene renders one image per (cell, heading). The base scene has 225
#: reachable cells, an obstacle scene 223.
BASE_IMAGES = 900
OBSTACLE_IMAGES = 892


class TestSceneResolution(unittest.TestCase):

    def test_all_scenes_present_and_valid(self):
        scenes = si.all_scenes()
        self.assertEqual(len(scenes), 39, "1 base + 38 obstacle scenes")
        self.assertEqual(scenes[0].scene_id, si.BASE_SCENE_ID)
        for s in scenes:
            n = BASE_IMAGES if s.is_base else OBSTACLE_IMAGES
            self.assertEqual(si.validate(s, expect_images=n), [],
                             "%s has layout problems" % s.scene_id)

    def test_scene_id_encodes_the_obstacle(self):
        for s in si.obstacle_scenes():
            self.assertEqual(len(s.obstacle_cells), 2, s.scene_id)
            cells = sorted(tuple(c) for c in s.obstacle_cells)
            rebuilt = "obs_" + "__".join(
                "x%s_y%s" % (str(x).replace("-", "n"), str(y).replace("-", "n"))
                for x, y in cells)
            self.assertEqual(rebuilt, s.scene_id)

    def test_base64_mirror_resolves_to_the_render_tree(self):
        """A scene resolved out of dataset/v0-v5/scenes_v5/ still finds PNGs."""
        mirror = os.path.join(PROJECT_ROOT, si.MIRROR_SCENES_REL)
        if not os.path.isdir(mirror):
            self.skipTest("base64 mirror not present")
        name = sorted(n for n in os.listdir(mirror) if n.startswith("obs_"))[0]
        scene = si.resolve(os.path.join(mirror, name, "metadata.json"))
        self.assertEqual(scene.scene_id, name)
        # PNG directory is optional; base64 is sufficient for running experiments
        in_mirror = os.path.normcase(os.path.join(mirror, name))
        # base64 comes from the mirror it was resolved from ...
        self.assertTrue(os.path.normcase(scene.image_base64_dir).startswith(in_mirror))
        # ... the ViT table sits with the base64, in the tree that is.
        self.assertTrue(os.path.normcase(scene.embeddings_path).startswith(in_mirror))
        self.assertTrue(scene.has_embeddings())

    def test_both_trees_name_the_same_embedding_table(self):
        """One table per scene, reachable from either tree, never copied."""
        render_root = os.path.join(PROJECT_ROOT, si.OBSTACLE_SCENES_REL)
        if not os.path.isdir(render_root):
            self.skipTest("render tree not present")
        for scene in si.obstacle_scenes():
            render = si.resolve(os.path.join(render_root,
                                             scene.scene_id, "metadata.json"))
            mirror_meta = os.path.join(PROJECT_ROOT, si.MIRROR_SCENES_REL,
                                       render.scene_id, "metadata.json")
            if not os.path.isfile(mirror_meta):
                continue
            mirror = si.resolve(mirror_meta)
            self.assertEqual(os.path.normcase(render.embeddings_path),
                             os.path.normcase(mirror.embeddings_path),
                             render.scene_id)

    def test_missing_metadata_field_is_fatal(self):
        """Guessing a scene would silently put v5 on the base scene."""
        with self.assertRaises(si.SceneIndexError):
            si.resolve_for_task({"id": "T01-case01-v5"}, "fake/task.json")

    def test_legacy_layout_aliases_keep_scene_identity(self):
        self.assertEqual(si.resolve(si.LEGACY_BASE_METADATA_REL).as_dict(),
                         si.base_scene().as_dict())
        for scene in si.obstacle_scenes():
            old = os.path.join('dataset_v1-v5_base64', 'scenes', scene.scene_id, 'metadata.json')
            self.assertEqual(si.resolve(old).as_dict(), scene.as_dict())
        with self.assertRaises(si.SceneIndexError):
            si.resolve('dataset_v1-v5_base64/scenes/obs_missing/metadata.json')

    def test_reverse_lookup_by_image_path(self):
        si.reset_registry()
        scene = si.obstacle_scenes()[0]
        found = si.for_image_path(scene.image_path(-5, 16, "North"))
        self.assertIsNotNone(found)
        self.assertEqual(found.scene_id, scene.scene_id)
        # The configs hand out paths under a base image dir that does not
        # exist; those must still resolve to the base scene.
        for cand in si.BASE_IMAGE_DIR_CANDIDATES:
            p = os.path.join(PROJECT_ROOT, cand, "Grid_X-5_Y16_North.png")
            self.assertEqual(si.for_image_path(p).scene_id, si.BASE_SCENE_ID,
                             "%s did not resolve to the base scene" % cand)


class TestEmbeddingTables(unittest.TestCase):

    def test_every_scene_has_a_table_in_v0_schema(self):
        for s in si.all_scenes():
            self.assertTrue(s.has_embeddings(), "%s has no table" % s.scene_id)
            z = np.load(s.embeddings_path, allow_pickle=True)
            self.assertEqual(sorted(z.keys()), ["embeddings", "filenames"])
            names = [str(n) for n in z["filenames"]]
            embs = z["embeddings"]
            n = BASE_IMAGES if s.is_base else OBSTACLE_IMAGES
            self.assertEqual(len(names), n, s.scene_id)
            self.assertEqual(embs.shape, (n, 768), s.scene_id)
            self.assertEqual(embs.dtype, np.float32, s.scene_id)
            self.assertEqual(names, sorted(names), "%s not sorted" % s.scene_id)
            norms = np.linalg.norm(embs, axis=1)
            self.assertTrue(np.allclose(norms, 1.0, atol=1e-5),
                            "%s rows are not L2-normalised" % s.scene_id)

    def test_table_covers_exactly_the_scene_png_set(self):
        for s in si.all_scenes():
            if not os.path.isdir(s.image_dir):
                continue
            z = np.load(s.embeddings_path, allow_pickle=True)
            on_disk = sorted(f for f in os.listdir(s.image_dir)
                             if f.endswith(".png"))
            self.assertEqual([str(n) for n in z["filenames"]], on_disk,
                             "%s table and image_dir disagree" % s.scene_id)

    def test_identical_pixels_get_identical_vectors(self):
        """Copied frames share one vector, bit for bit.

        Most obstacle-scene frames are byte-for-byte copies of the base
        render. Deduplicating the forward pass is what guarantees they get
        the *same* row rather than two rows that merely round the same way
        -- and it is also why cosine can never recover the scene for them.
        """
        a, b = si.obstacle_scenes()[0], si.obstacle_scenes()[1]
        if not os.path.isdir(a.image_dir) or not os.path.isdir(b.image_dir):
            self.skipTest("PNG directories not present")
        ia, ea = _table(a)
        ib, eb = _table(b)
        shared = [n for n in ia if n in ib
                  and _same_bytes(os.path.join(a.image_dir, n),
                                   os.path.join(b.image_dir, n))]
        self.assertGreater(len(shared), 100,
                           "expected the two scenes to share many copied frames")
        for n in shared:
            self.assertTrue(np.array_equal(ea[ia[n]], eb[ib[n]]),
                            "%s: identical pixels, different vectors" % n)

    def test_reproduces_the_legacy_base_table(self):
        """The rebuilt vectors agree with the table v0 shipped.

        Not bit-identical: ``dataset/warehouse_01_01/image_embeddings.npz``
        was produced by ``compute_image_embeddings.py`` on a different
        machine, and float32 accumulation order differs across BLAS builds.
        Measured deviation is ~1.4e-06 per component (cosine > 0.9999997),
        which is numeric noise, not a different model or transform.
        """
        base = si.base_scene()
        scene = si.obstacle_scenes()[0]
        if not os.path.isdir(base.image_dir) or not os.path.isdir(scene.image_dir):
            self.skipTest("PNG directories not present")
        bidx, bemb = _table(base)
        idx, emb = _table(scene)
        copies = [n for n in idx
                  if _same_bytes(os.path.join(base.image_dir, n),
                                 os.path.join(scene.image_dir, n))]
        self.assertGreater(len(copies), 500,
                           "expected most frames to be copies of the base render")
        old = bemb[[bidx[n] for n in copies]]
        new = emb[[idx[n] for n in copies]]
        self.assertTrue(np.allclose(old, new, atol=1e-5),
                        "rebuilt vectors drifted from the legacy base table")
        self.assertGreater(float((old * new).sum(axis=1).min()), 0.99999)


class TestVitMatcher(unittest.TestCase):

    def setUp(self):
        vit_matcher.reset_cache()
        si.reset_registry()

    def test_resolves_each_path_to_its_own_scene_table(self):
        base = si.base_scene()
        scene = si.obstacle_scenes()[0]
        self.assertEqual(
            vit_matcher._table_path_for(base.image_path(-5, 16, "North")),
            base.embeddings_path)
        self.assertEqual(
            vit_matcher._table_path_for(scene.image_path(-5, 16, "North")),
            scene.embeddings_path)

    def test_exact_self_match_scores_one(self):
        scene = si.obstacle_scenes()[0]
        img = scene.image_path(-5, 16, "North")
        i, score = vit_matcher.find_similar_index(img, [img], return_score=True)
        self.assertEqual(i, 0)
        self.assertAlmostEqual(score, 1.0, places=4)
        self.assertEqual(vit_matcher.classify_cosine(score), "exact")

    def test_matches_across_scenes(self):
        """A v5 current view has to match base-scene history frames."""
        base = si.base_scene()
        scene = si.obstacle_scenes()[0]
        cur = scene.image_path(-5, 16, "North")
        candidates = [base.image_path(-5, 16, "South"),
                      base.image_path(-5, 16, "North"),
                      base.image_path(0, 0, "East")]
        i, score = vit_matcher.find_similar_index(cur, candidates,
                                                  return_score=True)
        self.assertEqual(i, 1, "should pick the same pose in the base scene")
        self.assertGreater(score, vit_matcher.NEAR_COSINE_THRESHOLD)

    def test_unknown_image_falls_back_to_none(self):
        self.assertIsNone(vit_matcher.find_similar_index(
            "nowhere/Grid_X999_Y999_North.png", ["also/nowhere.png"]))

    def test_scenes_ready_reports_every_scene(self):
        ready = vit_matcher.scenes_ready()
        self.assertEqual(len(ready), 39)
        self.assertTrue(all(ready.values()), "some scene has no ViT table")


class TestBase64IsSceneScoped(unittest.TestCase):
    """The bug this index exists to prevent.

    ``_get_precomputed_b64`` used to look a basename up in one global
    directory. Under v5 that hands back the *base* scene's JPEG for every
    pose -- with no error, because the cache is designed to work when the
    PNG is absent -- so the model would never see the obstacle.
    """

    def setUp(self):
        si.reset_registry()

    def test_obstacle_scene_gets_its_own_jpeg(self):
        scene = None
        for s in si.obstacle_scenes():
            # pick a pose whose render actually differs from the base scene
            base = si.base_scene()
            for cell in s.obstacle_cells:
                for heading in ("North", "South", "East", "West"):
                    for dx in (-1, 1):
                        x, y = cell[0] + dx, cell[1]
                        a = base.base64_path(x, y, heading)
                        b = s.base64_path(x, y, heading)
                        if not (os.path.isfile(a) and os.path.isfile(b)):
                            continue
                        if open(a).read() != open(b).read():
                            scene = (s, x, y, heading)
                            break
                    if scene:
                        break
                if scene:
                    break
            if scene:
                break
        self.assertIsNotNone(scene, "no differing frame found to test with")
        s, x, y, heading = scene
        got = _helpers.encode_image_to_base64(s.image_path(x, y, heading))
        want = open(s.base64_path(x, y, heading)).read().strip()
        base_b64 = open(si.base_scene().base64_path(x, y, heading)).read().strip()
        self.assertEqual(got, want, "served the wrong scene's JPEG")
        self.assertNotEqual(got, base_b64, "served the base scene's JPEG")


class TestV5LookupOrder(unittest.TestCase):
    """V5 frames come from ``scenes_v5`` first, then from the base scene.

    Inside the agent the registry is only partly filled when the first v5
    frame arrives (loading the history trajectories registers the base
    scene). A partial registry used to be taken as complete: the obstacle
    frame then resolved to no scene and the model was served the base
    render, with ViT silently degrading to the pose matcher.
    """

    def setUp(self):
        si.reset_registry()
        vit_matcher.reset_cache()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        si.reset_registry()
        vit_matcher.reset_cache()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _differing_frame(self):
        base = si.base_scene()
        for s in si.obstacle_scenes():
            for name in sorted(os.listdir(s.image_base64_dir)):
                b = os.path.join(base.image_base64_dir, name)
                if os.path.isfile(b) and _read(b) != _read(os.path.join(s.image_base64_dir, name)):
                    return s, os.path.splitext(name)[0]
        self.fail("no obstacle frame differs from the base render")

    def test_partial_registry_still_finds_the_obstacle_scene(self):
        s, stem = self._differing_frame()
        si.reset_registry()
        si.register(si.base_scene())          # what history preloading does
        path = os.path.join(s.image_dir, stem + ".png")
        self.assertEqual(si.for_image_path(path).scene_id, s.scene_id)
        self.assertEqual(_helpers.encode_image_to_base64(path),
                         _read(os.path.join(s.image_base64_dir, stem + ".b64")),
                         "served the base render for an obstacle frame")
        idx, emb = _table(s)
        self.assertTrue(np.allclose(vit_matcher._vector(path), emb[idx[stem + ".png"]],
                                    atol=1e-6))

    def test_lookup_chain_is_scene_then_base(self):
        s = si.obstacle_scenes()[0]
        base = si.base_scene()
        self.assertEqual([c.scene_id for c in si.lookup_chain(s.image_path(-5, 16, "North"))],
                         [s.scene_id, si.BASE_SCENE_ID])
        self.assertEqual([c.scene_id for c in si.lookup_chain(base.image_path(-5, 16, "North"))],
                         [si.BASE_SCENE_ID])

    def test_frames_the_scene_lacks_come_from_the_base_scene(self):
        """An obstacle scene holding only its re-rendered frames still works."""
        b64 = os.path.join(self.tmp, "image_base64")
        img = os.path.join(self.tmp, "image")
        os.makedirs(b64)
        os.makedirs(img)
        with open(os.path.join(b64, "Grid_X-5_Y16_North.b64"), "w") as f:
            f.write("OWN")
        fake = si.register(si.SceneRef(
            "obs_fake", os.path.join(self.tmp, "metadata.json"), img, b64,
            os.path.join(self.tmp, "image_embeddings.npz")))
        base = si.base_scene()
        # its own frame wins ...
        self.assertEqual(_helpers.encode_image_to_base64(fake.image_path(-5, 16, "North")),
                         "OWN")
        # ... a frame it lacks is the base render, for the JPEG and the ViT row
        other = fake.image_path(-5, 16, "South")
        self.assertEqual(_helpers.encode_image_to_base64(other),
                         _read(base.base64_path(-5, 16, "South")))
        idx, emb = _table(base)
        self.assertTrue(np.allclose(vit_matcher._vector(other),
                                    emb[idx["Grid_X-5_Y16_South.png"]], atol=1e-6))


class TestEpisodicVitRetrieval(unittest.TestCase):
    """G04/G05 retrieve by ViT cosine over the per-scene tables.

    The pooling contract is unchanged from the pose matcher: every frame
    of every candidate trajectory competes and exactly one contiguous
    excerpt comes back, from the trajectory that owns the winning frame.
    """

    @classmethod
    def setUpClass(cls):
        cls.cfg = ex.load_config(V4_CONFIG)
        cls.tasks, _ = ex.load_test_tasks(cls.cfg)
        cls.key = cls.cfg.case_map.by_legacy(0)
        cls.trajs = [ex.load_history_trajectory(cls.cfg, cls.key, v).as_prompt_traj()
                     for v in ("v0", "v1", "v2", "v3")]
        t = cls.tasks[0]
        cls.origin = {"x": int(t.origin["coord"][0]),
                      "y": int(t.origin["coord"][1]),
                      "heading": t.origin["heading"]}
        cls.img = t.scene.image_path(cls.origin["x"], cls.origin["y"],
                                     cls.origin["heading"])

    def setUp(self):
        vit_matcher.reset_cache()
        self.ctx = em.DecisionContext(current_pose=self.origin)

    def test_returns_a_vit_labelled_match(self):
        m = em.find_best_match_vit(self.trajs, self.img, self.ctx)
        self.assertIsNotNone(m)
        self.assertEqual(m.matcher, "vit")
        self.assertIsNotNone(m.cosine)
        self.assertIsNone(m.distance, "distance is meaningless for a ViT match")
        self.assertIn(m.match_kind, ("exact", "near", "far"))

    def test_pools_every_frame_of_every_trajectory(self):
        one = em.find_best_match_vit(self.trajs[:1], self.img, self.ctx)
        four = em.find_best_match_vit(self.trajs, self.img, self.ctx)
        self.assertEqual(one.candidate_pose_count,
                         len(self.trajs[0]["image_paths"]))
        self.assertEqual(four.candidate_pose_count,
                         sum(len(t["image_paths"]) for t in self.trajs))

    def test_winner_really_has_the_highest_cosine(self):
        m = em.find_best_match_vit(self.trajs, self.img, self.ctx)
        cur = vit_matcher._vector(self.img)
        best = max(float(vit_matcher._vector(f) @ cur)
                   for t in self.trajs for f in t["image_paths"]
                   if vit_matcher._vector(f) is not None)
        self.assertAlmostEqual(m.cosine, best, places=5)

    def test_self_match_is_exact(self):
        """Feeding a trajectory's own frame back must score ~1.0."""
        frame = self.trajs[0]["image_paths"][5]
        m = em.find_best_match_vit(self.trajs[:1], frame, self.ctx)
        self.assertEqual(m.matched_index, 5)
        self.assertAlmostEqual(m.cosine, 1.0, places=4)
        self.assertEqual(m.match_kind, "exact")

    def test_excerpt_comes_from_one_trajectory(self):
        sink = []
        blocks = em.episodic_memory_multi_blocks(
            self.img, self.trajs, max_actions=6, ctx=self.ctx,
            match_sink=sink, matcher="vit")
        self.assertTrue(blocks)
        self.assertEqual(len(sink), 1)
        self.assertEqual(sink[0].matcher, "vit")

    def test_falls_back_to_pose_and_says_so(self):
        """An unindexed image must degrade visibly, not silently."""
        sink = []
        blocks = em.episodic_memory_multi_blocks(
            "nowhere/Grid_X-5_Y16_North.png", self.trajs, max_actions=6,
            ctx=self.ctx, match_sink=sink, matcher="vit")
        self.assertTrue(blocks, "should still produce an excerpt via pose")
        self.assertEqual(sink[0].matcher, "pose",
                         "a pose fallback must not be reported as a ViT match")

    def test_vit_label_never_claims_a_cell_match(self):
        """ViT's 'exact' means visually identical, not same cell+heading."""
        frame = self.trajs[0]["image_paths"][5]
        m = em.find_best_match_vit(self.trajs[:1], frame, self.ctx)
        label = em._historical_match_label(m, self.trajs[0]["poses"], None)
        self.assertIn("cosine", label)
        self.assertNotIn("cell+heading", label)

    def test_pose_and_vit_are_independent_rankings(self):
        """Sanity: the two matchers are actually different mechanisms."""
        p = em.find_best_match(self.trajs, self.origin, self.ctx)
        v = em.find_best_match_vit(self.trajs, self.img, self.ctx)
        self.assertEqual(p.matcher, "pose")
        self.assertEqual(v.matcher, "vit")
        self.assertIsNone(p.cosine)
        self.assertIsNotNone(v.cosine)


class TestExperimentWiring(unittest.TestCase):

    def test_v4_config_resolves_every_case_to_the_base_scene(self):
        cfg = ex.load_config(V4_CONFIG)
        tasks, report = ex.load_test_tasks(cfg)
        self.assertEqual(len(tasks), 100)
        self.assertEqual({t.scene.scene_id for t in tasks}, {si.BASE_SCENE_ID})
        self.assertTrue(all(r["scene"] == si.BASE_SCENE_ID for r in report))

    def test_v5_config_gives_each_case_its_own_obstacle_scene(self):
        if not os.path.isfile(V5_CONFIG):
            self.skipTest("v5 config not present")
        cfg = ex.load_config(V5_CONFIG)
        self.assertEqual(cfg.test_version, "v5")
        tasks, _ = ex.load_test_tasks(cfg)
        self.assertEqual(len(tasks), 100)
        scenes = {t.scene.scene_id for t in tasks}
        self.assertNotIn(si.BASE_SCENE_ID, scenes)
        self.assertEqual(len(scenes), 38)
        for t in tasks:
            self.assertEqual(len(t.scene.obstacle_cells), 2)
            self.assertTrue(t.scene.has_embeddings())

    def test_v4_and_v5_share_origins_but_not_scenes(self):
        if not os.path.isfile(V5_CONFIG):
            self.skipTest("v5 config not present")
        v4 = ex.load_test_tasks(ex.load_config(V4_CONFIG))[0]
        v5 = ex.load_test_tasks(ex.load_config(V5_CONFIG))[0]
        for a, b in zip(v4, v5):
            self.assertEqual(a.origin, b.origin, "v5 reuses the v4 origin")
            self.assertNotEqual(a.scene.scene_id, b.scene.scene_id)

    def test_history_trajectories_carry_their_own_scene(self):
        cfg = ex.load_config(V4_CONFIG)
        key = cfg.case_map.by_legacy(0)
        for version in ("v0", "v1", "v2", "v3"):
            t = ex.load_history_trajectory(cfg, key, version)
            self.assertEqual(t.scene.scene_id, si.BASE_SCENE_ID)
            prompt = t.as_prompt_traj()
            # PNG directory is optional; base64 is sufficient for running experiments
            self.assertTrue(prompt["image_paths"][0].endswith(".png"),
                            "%s frame 0 is not a PNG path" % version)

    def test_rejects_an_unknown_test_version(self):
        cfg = ex._read_config_file(V4_CONFIG)
        cfg["test_version"] = "v6"
        with self.assertRaises(ex.ExperimentError):
            ex.ExperimentConfig(cfg, V4_CONFIG)


def _table(scene):
    """``({basename: row}, embeddings)`` for one scene."""
    with np.load(scene.embeddings_path, allow_pickle=True) as z:
        names = [str(n) for n in z["filenames"]]
        return {n: i for i, n in enumerate(names)}, z["embeddings"]


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _same_bytes(a, b):
    if os.path.getsize(a) != os.path.getsize(b):
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()


if __name__ == "__main__":
    unittest.main(verbosity=2)
