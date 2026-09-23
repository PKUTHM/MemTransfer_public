"""Scene-scoped image index for the v1-v5 dataset.

Why this module exists
----------------------
Every scene renders the *same* filename set -- ``Grid_X{x}_Y{y}_{Heading}.png``,
one image per (cell, heading). A v4 frame and the v5 frame for the same pose
therefore carry byte-identical *names*, and for 29,234 of the 33,896 v5 frames
they are byte-identical *files* as well: only 4,662 frames were actually
re-rendered with the obstacle in view (see ``render_manifest.json`` in each
obstacle scene, fields ``rendered`` / ``copied_from_v0``).

Two consequences drive the whole design:

  * A bare filename is **not** an identity. Resolving one requires knowing
    which scene it belongs to, and the scene comes from ``task.json``'s
    ``metadata`` field -- v1-v4 point at the base scene, v5 at its own
    ``dataset-v1-v5/scenes/obs_*/metadata.json``.

  * A **content**-addressed index cannot recover the scene either, because
    for those 29,234 frames the content is identical. ViT similarity is only
    meaningful *within* one scene; it can never replace the scene hop.

So the index is scene-scoped: each scene owns a self-contained
``image_embeddings.npz`` in the v0 schema (a ``filenames`` object array plus
an aligned, L2-normalised ``embeddings`` matrix), and every lookup is keyed
by ``(scene_id, filename)``.
"""
import json
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Scene id of the obstacle-free warehouse used by v0-v4.
BASE_SCENE_ID = "base"

#: ``metadata.json`` of the base scene, relative to the repo root.
BASE_METADATA_REL = os.path.join("dataset", "v0-v5", "scene_v0-v4", "metadata.json")
LEGACY_BASE_METADATA_REL = os.path.join("dataset", "warehouse_01_01", "metadata.json")

#: Root holding the per-obstacle v5 scenes, with their rendered PNGs.
OBSTACLE_SCENES_REL = os.path.join("dataset-v1-v5", "scenes")

#: The base64-only mirror the experiment configs actually point at.
#:
#: V5's mirror carries ``metadata.json`` and ``image_base64/``; its PNGs
#: remain under ``dataset-v1-v5/``. A V5 scene therefore has to
#: reach across to the render tree for its pixels, which is what the offline
#: ViT indexer reads.
MIRROR_SCENES_REL = os.path.join("dataset", "v0-v5", "scenes_v5")

#: Filename of the per-scene embedding table (v0 schema, one table per scene).
EMBEDDINGS_BASENAME = "image_embeddings.npz"

#: Canonical base PNGs followed by historical aliases retained for old runs.
BASE_IMAGE_DIR_CANDIDATES = (
    os.path.join("dataset", "v0-v5", "scene_v0-v4", "image"),
    os.path.join("dataset_v1-v5_base64", "scene_v0-v4", "image"),
    os.path.join("dataset", "warehouse_01_01", "image"),
    os.path.join("warehouse_v0", "warehouse_01_01", "image"),
)


class SceneIndexError(RuntimeError):
    """Raised when a scene cannot be resolved or its layout is incomplete."""


def image_name(x, y, heading):
    """The canonical render filename for one pose. Mirrors ``variants/core.py``."""
    return "Grid_X%d_Y%d_%s.png" % (int(x), int(y), heading)


def _abs(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def canonical_metadata_path(path):
    """Map only known historical layout aliases; never guess a scene."""
    p = os.path.normpath(_abs(path))
    previous_root = os.path.normpath(_abs('dataset_v1-v5_base64'))
    if os.path.normcase(p).startswith(os.path.normcase(previous_root) + os.sep):
        p = os.path.join(_abs('dataset/v0-v5'), os.path.relpath(p, previous_root))
    if os.path.normcase(p) == os.path.normcase(_abs(LEGACY_BASE_METADATA_REL)):
        return _abs(BASE_METADATA_REL)
    old_root = os.path.normpath(_abs(os.path.join('dataset/v0-v5', 'scenes')))
    if os.path.normcase(p).startswith(os.path.normcase(old_root) + os.sep):
        return os.path.join(_abs(MIRROR_SCENES_REL), os.path.relpath(p, old_root))
    return p


def _rel(path):
    try:
        return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
    except ValueError:                      # a different drive on Windows
        return path


class SceneRef(object):
    """One scene: its id, its metadata, and the three per-scene asset dirs.

    ``image_dir`` (PNGs) feeds the offline embedder, ``image_base64_dir``
    (downscaled JPEGs) is what actually reaches the model, and
    ``embeddings_path`` is the ViT table built from ``image_dir``.
    """

    __slots__ = ("scene_id", "metadata_path", "image_dir", "image_base64_dir",
                 "embeddings_path", "obstacle_cells")

    def __init__(self, scene_id, metadata_path, image_dir, image_base64_dir,
                 embeddings_path, obstacle_cells=None):
        self.scene_id = scene_id
        self.metadata_path = metadata_path
        self.image_dir = image_dir
        self.image_base64_dir = image_base64_dir
        self.embeddings_path = embeddings_path
        self.obstacle_cells = obstacle_cells or []

    @property
    def is_base(self):
        return self.scene_id == BASE_SCENE_ID

    def image_path(self, x, y, heading):
        return os.path.join(self.image_dir, image_name(x, y, heading))

    def base64_path(self, x, y, heading):
        stem = os.path.splitext(image_name(x, y, heading))[0]
        return os.path.join(self.image_base64_dir, stem + ".b64")

    def has_embeddings(self):
        return os.path.isfile(self.embeddings_path)

    def as_dict(self):
        return {
            "scene_id": self.scene_id,
            "metadata": _rel(self.metadata_path),
            "image_dir": _rel(self.image_dir),
            "image_base64_dir": _rel(self.image_base64_dir),
            "embeddings": _rel(self.embeddings_path),
            "obstacle_cells": [list(c) for c in self.obstacle_cells],
        }

    def __repr__(self):
        return "SceneRef(%s)" % self.scene_id


def _obstacle_cells(metadata_path):
    """Read ``obstacle_cells`` from a scene's sibling ``members.json``."""
    members = os.path.join(os.path.dirname(metadata_path), "members.json")
    if not os.path.isfile(members):
        return []
    try:
        with open(members, "r", encoding="utf-8") as f:
            return [list(c) for c in (json.load(f).get("obstacle_cells") or [])]
    except (ValueError, KeyError, TypeError):
        return []


def _pick_dir(candidates, what):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return os.path.abspath(candidates[0])


def _base_image_dir_candidates():
    return [_abs(c) for c in BASE_IMAGE_DIR_CANDIDATES]


def _obstacle_image_dir_candidates(scene_id, scene_dir):
    """Where an ``obs_*`` scene's PNGs might be, in probe order."""
    return [os.path.join(scene_dir, "image"),
            os.path.join(_abs(OBSTACLE_SCENES_REL), scene_id, "image")]


def _embeddings_path(scene_id, scene_dir):
    """Where this scene's ViT table lives: next to ``image_base64/``.

    One table per scene, never one per tree. The table ships with the repo
    while the PNGs do not (they are ~45 GB and go via Drive), so it belongs
    in the tree that is actually committed -- the base64 mirror -- exactly
    as the base scene's table sits in ``dataset/warehouse_01_01/`` beside
    its own ``image_base64/`` while ``image/`` is gitignored.

    A scene resolved from the render tree points at that same mirror file
    rather than getting a second copy that could drift from it.
    """
    local = os.path.join(scene_dir, EMBEDDINGS_BASENAME)
    if scene_id == BASE_SCENE_ID:
        return local
    canonical = os.path.join(_abs(MIRROR_SCENES_REL), scene_id,
                             EMBEDDINGS_BASENAME)
    if os.path.isdir(os.path.dirname(canonical)):
        return canonical
    return local


def scene_id_for_metadata(metadata_path):
    """Derive the scene id from a ``metadata.json`` path.

    The base scene is ``"base"``; an obstacle scene is named by its own
    directory (``obs_xn4_y2__xn3_y2``), which already encodes the two
    blocked cells.
    """
    p = canonical_metadata_path(metadata_path)
    if os.path.normcase(p) == os.path.normcase(_abs(BASE_METADATA_REL)):
        return BASE_SCENE_ID
    parent = os.path.basename(os.path.dirname(p))
    if parent.startswith("obs_"):
        return parent
    raise SceneIndexError(
        "cannot derive a scene id from %s: expected the base scene metadata "
        "or a dataset-v1-v5/scenes/obs_*/metadata.json" % metadata_path)


def resolve(metadata_path):
    """Build the :class:`SceneRef` for one ``metadata.json`` path.

    This is the only place that turns ``task.json``'s ``metadata`` field into
    concrete asset directories.
    """
    p = canonical_metadata_path(metadata_path)
    if not os.path.isfile(p):
        raise SceneIndexError("scene metadata not found: %s" % p)
    scene_id = scene_id_for_metadata(p)
    scene_dir = os.path.dirname(p)
    if scene_id == BASE_SCENE_ID:
        image_dir = _pick_dir(_base_image_dir_candidates(), "base scene")
    else:
        image_dir = _pick_dir(
            _obstacle_image_dir_candidates(scene_id, scene_dir), scene_id)
    return SceneRef(
        scene_id=scene_id,
        metadata_path=p,
        image_dir=image_dir,
        image_base64_dir=os.path.join(scene_dir, "image_base64"),
        embeddings_path=_embeddings_path(scene_id, scene_dir),
        obstacle_cells=_obstacle_cells(p),
    )


def resolve_for_task(task_raw, task_path=""):
    """:class:`SceneRef` for a loaded ``task.json`` dict.

    ``metadata`` is required: it is the only field that says which scene a
    variant runs on, and guessing it would silently put v5 on the base scene
    -- exactly the failure this module exists to prevent.
    """
    rel = (task_raw or {}).get("metadata")
    if not rel:
        raise SceneIndexError(
            "%s: task.json has no 'metadata' field, so its scene is unknown"
            % (task_path or "<task>"))
    return resolve(rel)


def base_scene():
    """The obstacle-free scene shared by v0-v4."""
    return resolve(BASE_METADATA_REL)


def obstacle_scenes():
    """Every ``obs_*`` scene on disk, ordered by scene id."""
    root = _abs(MIRROR_SCENES_REL)
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        meta = os.path.join(root, name, "metadata.json")
        if name.startswith("obs_") and os.path.isfile(meta):
            out.append(resolve(meta))
    return out


def all_scenes():
    """The base scene followed by every obstacle scene."""
    return [base_scene()] + obstacle_scenes()


# --------------------------------------------------------------------------
# reverse lookup: image path -> scene
# --------------------------------------------------------------------------
#: ``normcase(image_dir) -> SceneRef``, filled lazily by :func:`for_image_path`.
_by_image_dir = {}

#: Whether :func:`register_all` has run since the last reset. Only a full
#: scan may conclude that an image path belongs to no scene.
_scanned = [False]

#: The base scene as a fallback target, resolved once per registry.
_fallback = {}


def register(scene):
    """Make ``scene`` findable from any image path inside it.

    The base scene registers every path in :data:`BASE_IMAGE_DIR_CANDIDATES`,
    not just the one that exists: the experiment configs hand out paths built
    from ``dataset/warehouse_01_01/image``, which is not where the PNGs are,
    and those paths still have to resolve to the base scene.
    """
    dirs = [scene.image_dir]
    if scene.scene_id == BASE_SCENE_ID:
        dirs += [_abs(c) for c in BASE_IMAGE_DIR_CANDIDATES]
    for d in dirs:
        _by_image_dir[os.path.normcase(os.path.abspath(d))] = scene
    return scene


def register_all():
    """Register every scene on disk. Idempotent."""
    try:
        for s in all_scenes():
            register(s)
    finally:
        _scanned[0] = True
    return _by_image_dir


def reset_registry():
    """Drop the reverse lookup (used by tests)."""
    _by_image_dir.clear()
    _fallback.clear()
    _scanned[0] = False


def for_image_path(image_path):
    """The :class:`SceneRef` owning ``image_path``, or ``None``.

    Trajectories and server responses carry image paths, not scene ids, so
    this is what lets the base64 cache and the ViT table pick the right
    per-scene assets instead of a process-wide default.

    The registry is filled lazily, and a *partial* registry is the normal
    state inside the agent: loading the history trajectories registers the
    base scene before the first v5 frame arrives from the server. A miss
    therefore triggers one full scan rather than being taken as "no scene".
    Treating any non-empty registry as complete is what once served v5 runs
    the base scene's JPEGs.
    """
    if not image_path:
        return None
    d = os.path.normcase(os.path.abspath(os.path.dirname(image_path)))
    scene = _by_image_dir.get(d)
    if scene is None and not _scanned[0]:
        try:
            register_all()
        except SceneIndexError:
            return None
        scene = _by_image_dir.get(d)
    return scene


def fallback_scene(scene):
    """The scene a frame is taken from when ``scene`` has no copy of it.

    V5 obstacle scenes were re-rendered only at poses that can see the
    obstacle; every other pose is, by construction, the base (v0-v4) render.
    So a lookup tries the obstacle scene (``dataset/v0-v5/scenes_v5/obs_*``)
    first and falls back to the base scene (``dataset/v0-v5/scene_v0-v4``).
    The base scene has no fallback.
    """
    if scene is None or scene.is_base:
        return None
    if "base" not in _fallback:
        _fallback["base"] = register(base_scene())
    return _fallback["base"]


def lookup_chain(image_path):
    """Scenes to consult for ``image_path``, in order: its own, then its
    fallback. Empty when the path belongs to no known scene."""
    scene = for_image_path(image_path)
    if scene is None:
        return []
    fb = fallback_scene(scene)
    return [scene] if fb is None else [scene, fb]


def validate(scene, require_embeddings=False, expect_images=None):
    """Return a list of human-readable problems with one scene's layout."""
    problems = []
    for label, path in (("metadata", scene.metadata_path),
                        ("image_base64_dir", scene.image_base64_dir)):
        if not os.path.exists(path):
            problems.append("%s missing: %s" % (label, path))
    if require_embeddings and not scene.has_embeddings():
        problems.append("embeddings missing: %s" % scene.embeddings_path)
    if expect_images and os.path.isdir(scene.image_dir):
        n = len([f for f in os.listdir(scene.image_dir) if f.endswith(".png")])
        if n != expect_images:
            problems.append("image_dir holds %d PNGs, expected %d"
                            % (n, expect_images))
    return problems
