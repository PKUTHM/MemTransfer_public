"""Runtime ViT-similarity matcher for the episodic-memory section.

The agent's runtime env deliberately stays **free of torch / timm**: the
heavy ViT-B/16 forward pass is run once, offline, by
``scripts/build_scene_embeddings.py`` (and, for the base scene alone, the
older ``scripts/compute_image_embeddings.py``). Those scripts dump one
``.npz`` per scene, keyed by filename, holding a 768-dim L2-normalised
feature vector per reference image.

Scene scoping
-------------
There is no single table any more. Every scene renders the same filename
set, so ``Grid_X-5_Y16_North.png`` names one image in the base scene and a
*different* image in each of the 38 ``obs_*`` scenes -- and for most poses
those two files are byte-identical, which is precisely why a filename cannot
carry the scene and why cosine similarity cannot recover it either. Each
image path is therefore resolved to its own scene's table via
``scene_index.for_image_path``.

Vectors from different scenes still live in one space, so a v5 current view
compares correctly against v0-v3 history frames drawn from the base scene;
only the *table lookup* is per-scene, not the arithmetic.

This module:
  1. Lazy-loads each scene's .npz once, caching them by path.
  2. Given the agent's current image filename and a list of candidate
     reference trajectory image paths, returns the index with the highest
     cosine similarity.

A v5 obstacle scene was re-rendered only where the obstacle is visible, so a
filename missing from its table is looked up in the base scene's table next
(``scene_index.lookup_chain``). If no table in that chain holds it, the
matcher returns ``None`` and the caller is expected to fall back to the
pose-based matcher.
"""
import os
import threading

import numpy as np

try:                                    # code/ is on sys.path when the agent runs
    import scene_index
except ImportError:                     # standalone import of the prompt package
    scene_index = None


_DEFAULT_EMB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset", "warehouse_01_01", "image_embeddings.npz",
)


_cache_lock = threading.Lock()
#: ``emb_path -> (filenames, {basename: row}, (N, 768) float32)``; ``None``
#: marks a path that failed to load, so a missing table is not re-read on
#: every step.
_tables = {}


def _load(emb_path):
    """(Re)load one scene's embedding table. Idempotent within the process."""
    with _cache_lock:
        if emb_path in _tables:
            return _tables[emb_path] is not None
        table = None
        if os.path.exists(emb_path):
            try:
                data = np.load(emb_path, allow_pickle=True)
                names = [str(n) for n in data["filenames"]]
                embs = data["embeddings"].astype(np.float32)
                # Re-normalise just in case the file was generated with a
                # non-normalised pipeline.
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1.0)
                embs = embs / norms
                table = (names, {n: i for i, n in enumerate(names)}, embs)
            except Exception:
                table = None
        _tables[emb_path] = table
        return table is not None


def _table_path_for(image_path, override=None):
    """Which .npz holds ``image_path``'s vector, or ``None`` if unknown.

    A path that belongs to no registered scene gets **no** table. Falling
    back to the base scene here would look up the basename in the wrong
    scene's table and return a confident vector for the wrong image --
    the exact cross-scene confusion this index exists to prevent, and
    invisible because every scene uses the same filenames.

    The base-scene default applies only when :mod:`scene_index` could not
    be imported at all (the prompt package used standalone), where there
    is no scene information to be wrong about.
    """
    if override:
        return override
    if scene_index is None:
        return _DEFAULT_EMB_PATH
    scene = scene_index.for_image_path(image_path)
    return scene.embeddings_path if scene is not None else None


def _table_paths_for(image_path, override=None):
    """Tables to consult for ``image_path``, in order: its own scene's, then
    its fallback scene's (a v5 obstacle scene falls back to the base scene
    for poses it never re-rendered). ``override`` forces that one table."""
    if override:
        return [override]
    if scene_index is None:
        return [_DEFAULT_EMB_PATH]
    return [s.embeddings_path for s in scene_index.lookup_chain(image_path)]


def _vector(image_path, override=None):
    """The L2-normalised vector for one image path, or ``None``."""
    if not image_path:
        return None
    name = os.path.basename(image_path)
    for path in _table_paths_for(image_path, override):
        if not _load(path):
            continue
        _, index, embs = _tables[path]
        row = index.get(name)
        if row is not None:
            return embs[row]
    return None


def is_ready(emb_path=None):
    """True if the table for ``emb_path`` (default: the base scene) loads."""
    return _load(emb_path or _DEFAULT_EMB_PATH)


def scenes_ready():
    """``{scene_id: bool}`` for every scene the index knows about.

    Lets a caller report which scenes are indexed without forcing a load of
    all 39 tables' worth of vectors up front.
    """
    if scene_index is None:
        return {"base": _load(_DEFAULT_EMB_PATH)}
    return {s.scene_id: s.has_embeddings() for s in scene_index.all_scenes()}


def reset_cache():
    """Drop the in-process cache (used by tests; not strictly necessary)."""
    with _cache_lock:
        _tables.clear()


# Cosine thresholds that mirror the pose matcher's three-tier vocabulary:
#   score > EXACT_COSINE_THRESHOLD  → "exact"  (essentially the same image)
#   NEAR_COSINE_THRESHOLD < score ≤ → "near"   (same scene, distinct view)
#   score ≤ NEAR_COSINE_THRESHOLD  → "far"    (closest view, low similarity)
#: Overridden from the experiment config's ``prompt`` block via
#: :func:`set_thresholds`. They change the wording the model reads, so they
#: are part of the experiment rather than an implementation detail.
EXACT_COSINE_THRESHOLD = 0.97
NEAR_COSINE_THRESHOLD = 0.75

#: Candidates within this cosine of the best one are treated as tied and go
#: to :func:`episodic_memory._pick_from_ties`.
COSINE_TIE_EPS = 1e-4


def set_thresholds(exact=None, near=None, tie_eps=None):
    """Point the matcher at the config's cosine bands."""
    global EXACT_COSINE_THRESHOLD, NEAR_COSINE_THRESHOLD, COSINE_TIE_EPS
    if exact is not None:
        EXACT_COSINE_THRESHOLD = float(exact)
    if near is not None:
        NEAR_COSINE_THRESHOLD = float(near)
    if tie_eps is not None:
        COSINE_TIE_EPS = float(tie_eps)
    if NEAR_COSINE_THRESHOLD > EXACT_COSINE_THRESHOLD:
        raise ValueError(
            "prompt.vit_near_cosine (%s) must not exceed prompt.vit_exact_cosine "
            "(%s); the bands would be inverted"
            % (NEAR_COSINE_THRESHOLD, EXACT_COSINE_THRESHOLD))


def classify_cosine(score):
    """Map a cosine similarity to one of ``"exact" | "near" | "far"``."""
    if score is None:
        return "far"
    if score > EXACT_COSINE_THRESHOLD:
        return "exact"
    if score > NEAR_COSINE_THRESHOLD:
        return "near"
    return "far"


def find_similar_index(current_img_path, candidate_image_paths,
                       emb_path=None, return_score=False,
                       stage_progress=None, agent_stages=0,
                       agent_steps_in_stage=0):
    """Among ``candidate_image_paths``, return the index with the highest
    cosine similarity to the current image. When multiple candidates tie
    on the highest cosine (within :data:`COSINE_TIE_EPS`), the tiebreaker
    :func:`episodic_memory._pick_from_ties` is invoked using
    ``stage_progress`` and the agent's current stage progress.

    Each path is looked up in its own scene's table, so a v5 current view
    matches correctly against base-scene history frames. Passing
    ``emb_path`` forces every lookup into that one table instead.

    Returns ``None`` if no table can be loaded, the current image's
    basename is not in its table, or none of the candidate basenames are.
    """
    miss = (None, None) if return_score else None

    cur_vec = _vector(current_img_path, emb_path)
    if cur_vec is None:
        return miss

    vecs = []
    valid = []
    for i, p in enumerate(candidate_image_paths):
        v = _vector(p, emb_path)
        if v is not None:
            vecs.append(v)
            valid.append(i)
    if not valid:
        return miss

    sims = np.stack(vecs) @ cur_vec  # (M,)
    best_score = float(sims.max())
    tied_local = [k for k, s in enumerate(sims) if s >= best_score - COSINE_TIE_EPS]
    if len(tied_local) == 1:
        best_i = valid[tied_local[0]]
    else:
        from .episodic_memory import _pick_from_ties
        candidates_orig = [valid[k] for k in tied_local]
        best_i = _pick_from_ties(
            candidates_orig,
            stage_progress=stage_progress,
            agent_stages=agent_stages,
            agent_steps_in_stage=agent_steps_in_stage,
        )
    if return_score:
        return best_i, best_score
    return best_i
