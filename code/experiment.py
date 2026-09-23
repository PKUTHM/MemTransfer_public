"""EXPERIMENT CONFIG + DATA LOADER for the v0–v3 → v4 memory-transfer study.

This module is the single source of truth for:

  * parsing ``config/**/*.yaml`` (the RQ1 / RQ2 experiments, one
    self-contained file each),
  * resolving logical case identity (``task_id`` + ``case_no``) to the
    legacy flat ``case_id`` used by the existing server/agent HTTP API,
  * resolving each case's **scene** (metadata, PNGs, base64 cache, ViT
    table) from its own ``task.json``,
  * loading the v4 / v5 **test** tasks (server side),
  * loading the v0–v3 **historical** trajectories (agent side),
  * loading the v0–v3 Summary / Semantic text memories (agent side),
  * hashing every input so a run manifest can prove which bytes were used.

Nothing here talks to the network or to the model. Everything is offline
and importable from ``server.py``, ``agent.py``, ``test_run.py`` and the
unit tests.

Hard rules encoded here (see docs/v0_v3_memory_v4_evaluation_plan.md):

  * A missing / malformed historical trajectory is a **hard error**.
    It must never silently degrade into "no reference trajectory", which
    would turn a G02–G05 run into an unlabelled G01 run.
  * ``v4/trajectory.json`` is never opened by any runtime loader. Only
    ``v4/task.json`` is, and only its task-defining fields are projected
    out — ``planner`` / ``checks`` (which carry the expert length and the
    candidate statistics) are dropped before the data can reach a prompt.
  * Case identity always goes through ``case_id_map.json``. T02 is
    ``task_type=3`` and T03 is ``task_type=2``; directory order is not a
    valid substitute for the map.
  * A test case's scene comes from its own ``task.json``'s ``metadata``
    field, never from a process-wide default. v1–v4 share the base scene;
    each v5 case has its own two-cell obstacle scene with its own
    reachable set, PNGs, base64 cache and ViT table. Defaulting here would
    put a v5 run on the obstacle-free scene without raising anything —
    the agent would simply never see the obstacle.
"""
import hashlib
import json
import os
import re

import scene_index


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCHEMA_VERSION = 1
HISTORY_VERSIONS = ("v0", "v1", "v2", "v3")

#: Versions that may be put under test. ``v4`` is the obstacle-free test set;
#: ``v5`` reuses v4's origins but runs on a per-case obstacle scene, so a v5
#: run resolves its scene per case rather than once per config. ``v0`` is the
#: original task set itself (RQ1 experiment 1A): same tasks, same origins as
#: the legacy ``dataset/warehouse_01_01/tasks.json``, which the preflight
#: asserts case by case.
TEST_VERSIONS = ("v0", "v4", "v5")
TEST_VERSION = "v4"                     #: default when a config omits it

#: Roles accepted per version in ``<version>/task.json``.
EXPECTED_ROLES = {
    "v0": "original",
    "v1": "memory_candidate",
    "v2": "memory_candidate",
    "v3": "memory_candidate",
    "v4": "test",
    "v5": "test",
}

#: Parameters that shape what the model actually sees, with the values they
#: had while they were hardcoded. Surfacing them here makes a run fully
#: described by its config instead of by config + a handful of constants
#: spread across three modules.
#:
#: The defaults are the historical constants exactly, so an unchanged config
#: behaves identically -- see :meth:`ExperimentConfig.fingerprint` for how a
#: *changed* one becomes visible in the hash.
PROMPT_DEFAULTS = {
    # JPEG the agent encodes a PNG to when no precomputed .b64 exists.
    # The committed .b64 cache was generated at these values.
    "image_max_size": 448,
    "image_quality": 80,
    # Which frames the full-context groups sample out of a trajectory.
    "reference_traj_smart_layers": ["start", "stage", "end", "turn"],
    # None = no extra uniform sampling on top of the layers above.
    "reference_traj_smart_sample_count": None,
    # Cosine bands that decide how an episodic match is described to the
    # model ("essentially identical" / "closest similar" / "low similarity").
    "vit_exact_cosine": 0.97,
    "vit_near_cosine": 0.75,
    # Candidates within this cosine of the best are treated as tied and go
    # to the stage-aware tiebreaker.
    "vit_cosine_tie_eps": 1e-4,
    # How a memory segment that started from exactly the test origin (same
    # cell AND heading) is introduced. False: the neutral note everywhere
    # ("origins may differ ... NOT a route to replay"). True: those segments
    # say "same origin" instead. In this dataset only V0 memory on the V0 test
    # qualifies; no v4/v5 origin equals any v0-v3 one, so it never fires there.
    "same_origin_wording": False,
    # Explicit notice that historical routes may be blocked in the test scene.
    "obstacle_notice": False,
}

#: Server-side knobs. ``notify_on_progress`` reaches the model (it is the
#: per-step note in the observation), the other two do not.
SERVER_DEFAULTS = {
    "notify_on_progress": True,
    "require_explicit_declare": False,
    "action_cache_limit": 208,
}

#: Keys of :data:`PROMPT_DEFAULTS` / :data:`SERVER_DEFAULTS` that change what
#: the model is shown, and therefore belong in the fingerprint when set to
#: anything other than the default.
FINGERPRINTED_PROMPT_KEYS = tuple(sorted(PROMPT_DEFAULTS))
FINGERPRINTED_SERVER_KEYS = ("notify_on_progress",)

#: How one decision is retried and when an attempt gives up. These were
#: module constants in agent.py; the defaults are those constants, so a
#: config that omits the block behaves -- and hashes -- exactly as before.
RETRY_DEFAULTS = {
    # A reply cut off (finish_reason=length) before it named an action is
    # re-asked with double the decoding budget, at most this many times and
    # never above the ceiling. 0 makes model.max_tokens a hard cap.
    "truncation_retries": 3,
    "max_tokens_ceiling": 32768,
    # Consecutive failures of one kind after which the attempt ends:
    # truncated / technical end it unsettled (the case gets a fresh attempt,
    # up to execution.max_case_attempts); model-output failures settle the
    # case as failed.
    "max_consecutive_truncated": 5,
    "max_consecutive_model_failures": 3,
    "max_consecutive_technical": 3,
}
#: All of them can change a case's outcome, so all enter the fingerprint
#: once they deviate from the defaults.
FINGERPRINTED_RETRY_KEYS = tuple(sorted(RETRY_DEFAULTS))


def _with_defaults(block, defaults, label, path):
    """Merge one optional config block over its defaults, rejecting typos."""
    block = dict(block or {})
    unknown = sorted(set(block) - set(defaults))
    if unknown:
        raise ExperimentError(
            "%s: unknown %s key(s) %s; known keys are %s"
            % (path, label, unknown, sorted(defaults)))
    merged = dict(defaults)
    merged.update(block)
    return merged


#: Stage bookkeeping anomalies that are known, accepted and must be
#: reported rather than silently repaired. Key = (task_id, case_no, version).
KNOWN_STAGE_ANOMALIES = {
    ("T09", 10, "v0"): (
        "stages_completed=1 but stages_total=3 with 2 stage_events; the "
        "trajectory still replays to task_completed. Kept verbatim; episodic "
        "stage progress is derived from stage_events, not from the stored "
        "stages_completed field."
    ),
}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
class ExperimentError(RuntimeError):
    """Raised for any configuration / data-integrity problem.

    Always fatal: the caller must stop the affected configuration rather
    than fall back to another case, another version or no memory at all.
    """


def resolve_path(path):
    """Resolve ``path`` against the repo root unless it is already absolute."""
    if not path:
        return ""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------
class CaseKey(object):
    """Logical identity of one case: ``task_id`` + ``case_no`` (+ legacy id)."""

    __slots__ = ("task_id", "case_no", "legacy_case_id", "task_type", "rel_dir")

    def __init__(self, task_id, case_no, legacy_case_id, task_type, rel_dir):
        self.task_id = task_id
        self.case_no = int(case_no)
        self.legacy_case_id = int(legacy_case_id)
        self.task_type = int(task_type)
        self.rel_dir = rel_dir

    def __repr__(self):
        return "CaseKey(%s/case%02d, legacy=%d, type=%d)" % (
            self.task_id, self.case_no, self.legacy_case_id, self.task_type)

    def label(self):
        """Short human/prompt-facing label, e.g. ``T01/case01 (legacy 0)``."""
        return "%s/case%02d (legacy %d)" % (self.task_id, self.case_no, self.legacy_case_id)

    def as_dict(self):
        return {"task_id": self.task_id, "case_no": self.case_no,
                "legacy_case_id": self.legacy_case_id, "task_type": self.task_type}


class CaseMap(object):
    """``case_id_map.json`` wrapper. The only authority on case identity."""

    def __init__(self, map_path):
        self.source_path = map_path
        self.source_sha256 = sha256_file(map_path)
        raw = _read_json(map_path)
        entries = raw.get("cases") or []
        if len(entries) != 100:
            raise ExperimentError("case map must hold 100 cases, found %d in %s"
                                  % (len(entries), map_path))
        self._by_legacy = {}
        self._by_logical = {}
        for e in entries:
            key = CaseKey(e["task_id"], e["case_no"], e["old_case_id"],
                          e["task_type"], e["new_dir"])
            if key.legacy_case_id in self._by_legacy:
                raise ExperimentError("duplicate legacy case id %d in %s"
                                      % (key.legacy_case_id, map_path))
            self._by_legacy[key.legacy_case_id] = key
            self._by_logical[(key.task_id, key.case_no)] = key
        missing = sorted(set(range(100)) - set(self._by_legacy))
        if missing:
            raise ExperimentError("case map is missing legacy ids: %s" % missing)

    def __len__(self):
        return len(self._by_legacy)

    def by_legacy(self, legacy_case_id):
        try:
            return self._by_legacy[int(legacy_case_id)]
        except (KeyError, TypeError, ValueError):
            raise ExperimentError("unknown legacy case id: %r" % (legacy_case_id,))

    def by_logical(self, task_id, case_no):
        try:
            return self._by_logical[(task_id, int(case_no))]
        except (KeyError, TypeError, ValueError):
            raise ExperimentError("unknown logical case: %s/case%s" % (task_id, case_no))

    def all_keys(self):
        """All 100 case keys, ordered by legacy case id."""
        return [self._by_legacy[i] for i in sorted(self._by_legacy)]


# --------------------------------------------------------------------------
# loaded data objects
# --------------------------------------------------------------------------
class TestTask(object):
    """A v4 task, projected down to the fields the server actually needs.

    ``planner``, ``checks`` (expert length, candidate counts, detour band,
    …) and every other bookkeeping field are dropped here so they cannot
    leak into a decision prompt through the server response.
    """

    __slots__ = ("case_key", "version", "task_type", "description", "origin",
                 "goals", "scene", "source_path", "source_sha256")

    def __init__(self, case_key, version, task_type, description, origin, goals,
                 scene, source_path, source_sha256):
        self.case_key = case_key
        self.version = version
        self.task_type = task_type
        self.description = description
        self.origin = origin
        self.goals = goals
        self.scene = scene              #: :class:`scene_index.SceneRef`
        self.source_path = source_path
        self.source_sha256 = source_sha256

    def as_server_task(self):
        """The dict shape ``server.build_state()`` consumes."""
        return {"task_type": self.task_type, "origin": self.origin,
                "description": self.description, "goals": self.goals}


class HistoricalTrajectory(object):
    """One version's ``forward_only`` expert trajectory for one case."""

    __slots__ = ("case_key", "version", "source_path", "source_sha256",
                 "description", "origin", "actions", "poses", "image_paths",
                 "stage_events", "length", "validation_notes", "scene")

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name))

    def as_prompt_traj(self, image_dir=None):
        """Adapt to the dict shape the existing prompt modules consume.

        ``image_dir`` defaults to this trajectory's own scene, so history
        frames are addressed against the scene they were rendered in rather
        than against whatever scene the caller happens to be testing.
        """
        if image_dir is None:
            image_dir = self.scene.image_dir
        return {
            "source": "forward_only",
            "version": self.version,
            "case_label": self.case_key.label(),
            "description": self.description,
            "origin": self.origin,
            "actions": list(self.actions),
            "image_paths": [os.path.join(image_dir, p) for p in self.image_paths],
            "poses": [list(p) for p in self.poses],
            "stage_events": list(self.stage_events),
            "length": self.length,
        }


class TextMemory(object):
    """One version's Summary or Semantic final-state memory (full text)."""

    __slots__ = ("version", "kind", "source_path", "source_sha256",
                 "scope_case_count", "text")

    def __init__(self, version, kind, source_path, source_sha256,
                 scope_case_count, text):
        self.version = version
        self.kind = kind
        self.source_path = source_path
        self.source_sha256 = source_sha256
        self.scope_case_count = scope_case_count
        self.text = text


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
class GroupConfig(object):
    """The experiment's group: the memory it carries and its image budgets."""

    __slots__ = ("group_id", "slug", "method", "memory", "versions",
                 "reference_traj_images", "max_smart_images_per_traj",
                 "max_reference_images", "max_total_images",
                 "episodic_horizon", "episodic_matcher", "episodic_content")

    #: Presentation-only fields, excluded from :meth:`ExperimentConfig.fingerprint`.
    #: A fingerprint answers "is this the same experiment?", and naming a
    #: file does not make it a different one.
    COSMETIC = ("slug",)

    def __init__(self, **kw):
        unknown = sorted(set(kw) - set(self.__slots__))
        if unknown:
            # A misspelt field used to be dropped here and silently replaced
            # by its default -- a run the config does not describe.
            raise ExperimentError(
                "group %s: unknown key(s) %s; known keys are %s"
                % (kw.get("group_id"), unknown, sorted(self.__slots__)))
        for name in self.__slots__:
            setattr(self, name, kw.get(name))
        if not self.slug:
            self.slug = self.default_slug()

    def apply_defaults(self, history_images):
        """Fill in everything a group does not need to state for itself.

        Each default reproduces the value the configs used to spell out, so
        deleting the line changes nothing that reaches the model -- and, just
        as importantly, nothing that reaches :meth:`ExperimentConfig.
        fingerprint`. A hash change would make every finished run look like a
        different experiment, and ``agent.py`` only skips an already-settled
        case when its recorded fingerprint still matches, so it would also
        re-run and re-bill work that is already done.
        """
        if self.method is None:
            # Only G01 differs from its memory kind, and only because "none"
            # is the absence of memory rather than a mechanism.
            self.method = self.memory if self.memory != "none" else "working_memory_only"
        if self.max_reference_images is None:
            self.max_reference_images = 0
        if self.max_smart_images_per_traj is None:
            self.max_smart_images_per_traj = 0
        if self.reference_traj_images is None:
            # Only the full-context groups interleave trajectory images.
            self.reference_traj_images = ("smart" if self.memory == "full_context"
                                          else "none")
        if self.max_total_images is None:
            # 1 current view + the working-memory history + this group's
            # reference budget. Stating it again would just be a second copy
            # that could disagree; the preflight asserts this arithmetic.
            self.max_total_images = 1 + int(history_images) + self.max_reference_images
        if self.memory == "episodic":
            if self.episodic_horizon is None:
                self.episodic_horizon = 6
            if self.episodic_matcher is None:
                self.episodic_matcher = "vit"
            if self.episodic_content is None:
                self.episodic_content = "all"
        if self.memory == "hybrid":
            if self.method != "hybrid":
                raise ExperimentError("Hybrid memory requires method=hybrid")
            if (not self.versions or len(set(self.versions)) != len(self.versions)
                    or any(v not in HISTORY_VERSIONS for v in self.versions)):
                raise ExperimentError("Hybrid needs distinct historical versions V0-V3")
            if (self.reference_traj_images != "none" or self.max_reference_images != 0
                    or self.max_smart_images_per_traj != 0
                    or any(v is not None for v in (self.episodic_horizon,
                                                   self.episodic_matcher,
                                                   self.episodic_content))):
                raise ExperimentError("Hybrid is Summary + Semantic only; trajectory settings are forbidden")
            if self.max_total_images != 1 + int(history_images):
                raise ExperimentError("Hybrid image budget must contain only current and working-memory images")
        return self

    def default_slug(self):
        """``<method>_<versions>`` — used when a group omits ``slug``."""
        method = self.method or (self.group_id or "group").lower()
        versions = list(self.versions or [])
        if not versions:
            return method
        span = versions[0] if len(versions) == 1 else "%s-%s" % (versions[0],
                                                                 versions[-1])
        return "%s_%s" % (method, span)

    def identity(self):
        """The fields that actually define what this group sends."""
        identity = {n: getattr(self, n) for n in self.__slots__
                    if n not in self.COSMETIC}
        if self.memory == "hybrid":
            identity["hybrid_components"] = ["summary", "semantic"]
            identity["hybrid_contract_version"] = 1
        return identity

    def as_dict(self):
        return {n: getattr(self, n) for n in self.__slots__}

    @property
    def uses_history_trajectories(self):
        return self.memory in ("full_context", "episodic")

    @property
    def uses_text_memory(self):
        return self.memory in ("summary", "semantic", "hybrid")

    @property
    def text_memory_kinds(self):
        # Fixed ordering matches the legacy combined-memory entry point.
        if self.memory == "hybrid":
            return ("semantic", "summary")
        return (self.memory,) if self.uses_text_memory else ()


#: ``<code>_<slug>__test_<version>``. The code (``A06``) is the group id and
#: names every result file; the slug (``episodic_v0``) names the mechanism.
#: Neither is written inside the config -- the file name is the one place.
CONFIG_NAME = re.compile(r"^([A-Z]\d{2})_(.+)__test_(v\d)$")

#: Memory settings a config states at its top level.
GROUP_FIELDS = tuple(n for n in GroupConfig.__slots__
                     if n not in ("group_id", "slug"))

#: Every top-level key a config may state. Anything else is a typo, and a
#: typo must fail rather than silently leave a default in place.
TOP_LEVEL_KEYS = ("schema_version", "experiment_id", "description", "ref_model",
                  "test_version", "dataset_root", "case_map", "scene",
                  "working_memory", "max_steps", "max_prompt_images", "repeats",
                  "prompt", "server", "retry", "memory_files", "model",
                  "execution") + GROUP_FIELDS


class ExperimentConfig(object):
    """Parsed + validated experiment configuration."""

    def __init__(self, cfg, config_path):
        self.config_path = config_path
        self.raw = cfg
        self.schema_version = cfg.get("schema_version")
        if self.schema_version != SCHEMA_VERSION:
            raise ExperimentError("unsupported schema_version %r (expected %d)"
                                  % (self.schema_version, SCHEMA_VERSION))
        self.experiment_id = cfg["experiment_id"]
        self.dataset_root = resolve_path(cfg["dataset_root"])
        self.test_version = cfg.get("test_version", TEST_VERSION)
        if self.test_version not in TEST_VERSIONS:
            raise ExperimentError("test_version must be one of %s, got %r"
                                  % (", ".join(TEST_VERSIONS), self.test_version))
        self.case_map_path = resolve_path(cfg["case_map"])

        # The ``scene`` block describes the *base* scene only. It stays the
        # authority for v1–v4 (and for the history trajectories, which are
        # always base-scene), but a v5 case overrides it from its own
        # task.json — see :meth:`scene_for`.
        scene = cfg["scene"]
        self.scene_metadata = resolve_path(scene["metadata"])
        self.scene_image_dir = resolve_path(scene["image_dir"])
        self.scene_image_base64_dir = resolve_path(scene["image_base64_dir"])
        self._scene_cache = {}

        self.model = dict(cfg["model"])
        self.prompt = _with_defaults(cfg.get("prompt"), PROMPT_DEFAULTS,
                                     "prompt", config_path)
        self.server = _with_defaults(cfg.get("server"), SERVER_DEFAULTS,
                                     "server", config_path)
        self.retry = _with_defaults(cfg.get("retry"), RETRY_DEFAULTS,
                                    "retry", config_path)
        wm = cfg["working_memory"]
        self.history_steps = int(wm["history_steps"])
        self.history_images = int(wm["history_images"])
        self.max_steps = int(cfg["max_steps"])
        self.repeats = int(cfg.get("repeats", 1))
        self.max_prompt_images = int(cfg.get("max_prompt_images", 50))

        self.execution = dict(cfg.get("execution", {}))

        self.memory_files = cfg["memory_files"]

        stale = sorted(k for k in ("groups", "group", "group_id", "slug", "label")
                       if k in cfg)
        if stale:
            raise ExperimentError(
                "%s: %s no longer belong in a config: the memory settings sit "
                "at the top level and the group id and slug come from the file "
                "name" % (config_path, ", ".join(stale)))
        unknown = sorted(set(cfg) - set(TOP_LEVEL_KEYS))
        if unknown:
            raise ExperimentError("%s: unknown top-level key(s) %s"
                                  % (config_path, unknown))
        m = CONFIG_NAME.match(os.path.splitext(os.path.basename(config_path))[0])
        if not m:
            raise ExperimentError(
                "%s: a config file is named <code>_<slug>__test_<version>.yaml "
                "(e.g. A06_episodic_v0__test_v0.yaml); the code and slug name "
                "its group" % config_path)
        code, slug, named_version = m.groups()
        if named_version != self.test_version:
            raise ExperimentError("%s: the file name says test_%s but "
                                  "test_version is %s"
                                  % (config_path, named_version, self.test_version))
        #: The one group this experiment runs.
        self.group = GroupConfig(
            group_id=code, slug=slug,
            **{k: cfg[k] for k in GROUP_FIELDS if k in cfg}
        ).apply_defaults(self.history_images)

        self.case_map = CaseMap(self.case_map_path)
        self._validate_paths()

    # -- validation ------------------------------------------------------
    def _validate_paths(self):
        for label, path in (("dataset_root", self.dataset_root),
                            ("scene.metadata", self.scene_metadata),
                            ("scene.image_base64_dir", self.scene_image_base64_dir)):
            if not os.path.exists(path):
                raise ExperimentError("%s does not exist: %s" % (label, path))
    # ``scene.image_dir`` is not trusted as a literal string: when it
    # does not exist it is resolved through the scene index, and the
    # config is then held to the result. PNG directory is optional
    # because base64 cache is sufficient for running experiments.
        base = scene_index.resolve(self.scene_metadata)
        if not os.path.isdir(self.scene_image_dir):
            self.scene_image_dir = base.image_dir
        for kind in ("summary", "semantic"):
            for version in HISTORY_VERSIONS:
                entry = self.memory_files.get(kind, {}).get(version)
                if not entry:
                    raise ExperimentError("memory_files.%s.%s missing" % (kind, version))
                path = resolve_path(entry["path"])
                if not os.path.isfile(path):
                    raise ExperimentError("memory file missing: %s" % path)

    @property
    def group_id(self):
        """The group's id (``A06``); names its result files and directories."""
        return self.group.group_id

    # -- paths -----------------------------------------------------------
    def case_dir(self, case_key, version):
        return os.path.join(self.dataset_root, case_key.task_id,
                            "case%02d" % case_key.case_no, version)

    def task_path(self, case_key, version):
        return os.path.join(self.case_dir(case_key, version), "task.json")

    def scene_for(self, case_key, version):
        """The :class:`scene_index.SceneRef` one (case, version) runs on.

        Read from that variant's own ``task.json``. v1–v4 all resolve to the
        base scene; each v5 case resolves to its own ``obs_*`` scene, which
        has its own reachable set, PNGs, base64 cache and ViT table.
        """
        key = (case_key.task_id, case_key.case_no, version)
        if key not in self._scene_cache:
            path = self.task_path(case_key, version)
            if not os.path.isfile(path):
                raise ExperimentError("task file missing: %s" % path)
            try:
                scene = scene_index.resolve_for_task(_read_json(path), path)
            except scene_index.SceneIndexError as exc:
                raise ExperimentError("%s: %s" % (path, exc))
            self._scene_cache[key] = scene_index.register(scene)
        return self._scene_cache[key]

    def trajectory_path(self, case_key, version):
        return os.path.join(self.case_dir(case_key, version), "trajectory.json")

    # -- fingerprint -----------------------------------------------------
    def fingerprint(self):
        """Stable hash over everything that defines what gets sent to the model.

        Deliberately excludes execution knobs (workers, retries, log dir)
        so that resuming a run with a different concurrency does not look
        like a different experiment.
        """
        payload = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "test_version": self.test_version,
            "case_map_sha256": self.case_map.source_sha256,
            "model": self.model,
            "working_memory": {"history_steps": self.history_steps,
                               "history_images": self.history_images},
            "max_steps": self.max_steps,
            "max_prompt_images": self.max_prompt_images,
            # identity(), not as_dict(): the slug is a name, not behaviour.
            "group": self.group.identity(),
            "memory_files": {
                kind: {v: self.memory_files[kind][v]["sha256"] for v in HISTORY_VERSIONS}
                for kind in ("summary", "semantic")
            },
        }
        # Prompt-shaping parameters enter the hash only where they differ from
        # the historical defaults. Those defaults are the values that were
        # hardcoded when the finished runs were recorded, so a config that
        # leaves them alone hashes exactly as before and stays comparable with
        # those runs; the moment one is actually changed it shows up here.
        tuned = self._tuned_params()
        if tuned:
            payload["tuned"] = tuned
        return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def _tuned_params(self):
        """Prompt-affecting parameters that deviate from their defaults."""
        out = {}
        for key in FINGERPRINTED_PROMPT_KEYS:
            if self.prompt[key] != PROMPT_DEFAULTS[key]:
                out["prompt.%s" % key] = self.prompt[key]
        for key in FINGERPRINTED_SERVER_KEYS:
            if self.server[key] != SERVER_DEFAULTS[key]:
                out["server.%s" % key] = self.server[key]
        for key in FINGERPRINTED_RETRY_KEYS:
            if self.retry[key] != RETRY_DEFAULTS[key]:
                out["retry.%s" % key] = self.retry[key]
        return out


# --------------------------------------------------------------------------
# config files: config/**/*.yaml
# --------------------------------------------------------------------------
#: Where YAML configs live, relative to the repo root.
CONFIG_DIR = "config"

#: Tried in order when a config is named without a path or extension.
_CONFIG_SUFFIXES = (".yaml", ".yml")


def _read_yaml(path):
    try:
        import yaml
    except ImportError:
        raise ExperimentError(
            "PyYAML is required to read %s (pip install pyyaml)" % path)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ExperimentError("%s: top level must be a mapping" % path)
    return data


def _read_config_file(path):
    if os.path.splitext(path)[1].lower() not in (".yaml", ".yml"):
        raise ExperimentError(
            "%s: experiment configs are YAML (config/**/*.yaml); the retired "
            "experiments/*.json format is no longer read" % path)
    return _read_yaml(path)


def _deep_merge(base, over):
    """Recursive dict merge. Mappings merge; everything else replaces.

    Lists replace rather than concatenate: an override that lists three
    versions means exactly those three, never "the base's four plus these".
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_layered(path, _seen=None):
    """Read one config, applying its ``extends`` chain first.

    ``extends`` names one file (or a list, applied left to right) resolved
    relative to the including file, then to :data:`CONFIG_DIR`. It exists
    so the parts every experiment shares -- the memory file digests, the
    scene -- can be written once instead of copied into every config.
    """
    path = os.path.abspath(path)
    _seen = list(_seen or [])
    if path in _seen:
        raise ExperimentError("circular extends: %s"
                              % " -> ".join(_seen + [path]))
    _seen.append(path)

    raw = _read_config_file(path)
    parents = raw.pop("extends", None)
    if not parents:
        return raw
    if isinstance(parents, str):
        parents = [parents]

    merged = {}
    for parent in parents:
        ppath = _resolve_sibling(parent, os.path.dirname(path))
        merged = _deep_merge(merged, _load_layered(ppath, _seen))
    return _deep_merge(merged, raw)


def _resolve_sibling(name, base_dir):
    for cand in (os.path.join(base_dir, name),
                 os.path.join(PROJECT_ROOT, CONFIG_DIR, name),
                 resolve_path(name)):
        if os.path.isfile(cand):
            return cand
        for suffix in _CONFIG_SUFFIXES:
            if os.path.isfile(cand + suffix):
                return cand + suffix
    raise ExperimentError("extends target not found: %s (from %s)"
                          % (name, base_dir))


def resolve_config_path(name):
    """Turn a config name or path into a file path.

    Accepts a full path, a path relative to the repo root
    (``config/rq2/R05_budgeted_full_context_v0-v3__test_v5.yaml``), a name relative to
    :data:`CONFIG_DIR` (``rq1/A06_episodic_v0__test_v0``) or a bare file
    name found anywhere under it (``A06_episodic_v0__test_v0``), each with
    the usual suffixes.
    """
    direct = resolve_path(name)
    if os.path.isfile(direct):
        return direct
    config_dir = os.path.join(PROJECT_ROOT, CONFIG_DIR)
    # Callers differ in how much they pre-resolve: test_run.py hands over an
    # abspath, so a bare name arrives here already joined to the repo root.
    # Undo that before trying config/, and fall back on the basename too, or
    # `--experiment_config <name>` would only work from the tools that pass
    # it through untouched.
    rel = name
    if os.path.isabs(name):
        try:
            rel = os.path.relpath(name, PROJECT_ROOT)
        except ValueError:          # another drive on Windows
            rel = os.path.basename(name)
    stems = [os.path.join(config_dir, rel),
             os.path.join(config_dir, os.path.basename(name))]
    cands = [direct + s for s in _CONFIG_SUFFIXES]
    for stem in stems:
        cands += [stem] + [stem + s for s in _CONFIG_SUFFIXES]
    for cand in cands:
        if os.path.isfile(cand):
            return cand
    # Last resort: the bare file name in any subdirectory (config/rq1/...).
    base = os.path.basename(name)
    wanted = {base} | {base + s for s in _CONFIG_SUFFIXES}
    found = []
    for dirpath, _dirs, files in os.walk(config_dir):
        found += [os.path.join(dirpath, f) for f in files if f in wanted]
    if len(found) == 1:
        return found[0]
    if found:
        raise ExperimentError("config name %r is ambiguous: %s"
                              % (name, sorted(found)))
    raise ExperimentError("experiment config not found: %s" % name)


def load_config(path):
    path = resolve_config_path(path)
    raw = _load_layered(path)
    return ExperimentConfig(raw, path)


# --------------------------------------------------------------------------
# test tasks (server side)
# --------------------------------------------------------------------------
def load_test_tasks(cfg):
    """Return ``(tasks, report)``.

    ``tasks`` is a list of :class:`TestTask` **indexed by legacy case id**,
    so ``tasks[i]`` keeps the meaning the existing HTTP ``case_id`` field
    has always had.
    """
    tasks = [None] * len(cfg.case_map)
    report = []
    for key in cfg.case_map.all_keys():
        path = cfg.task_path(key, cfg.test_version)
        if not os.path.isfile(path):
            raise ExperimentError("%s task file missing: %s"
                                  % (cfg.test_version, path))
        raw = _read_json(path)
        _validate_identity(raw, key, cfg.test_version, path)
        for field in ("task_type", "description", "origin", "goals"):
            if field not in raw:
                raise ExperimentError("%s: missing required field %r" % (path, field))
        if int(raw["task_type"]) != key.task_type:
            raise ExperimentError("%s: task_type %s != case map %s"
                                  % (path, raw["task_type"], key.task_type))
        origin = raw["origin"]
        if "coord" not in origin or "heading" not in origin:
            raise ExperimentError("%s: malformed origin %r" % (path, origin))
        scene = cfg.scene_for(key, cfg.test_version)
        task = TestTask(
            case_key=key, version=cfg.test_version, task_type=int(raw["task_type"]),
            description=raw["description"], origin=origin, goals=raw["goals"],
            scene=scene, source_path=path, source_sha256=sha256_file(path),
        )
        tasks[key.legacy_case_id] = task
        report.append({"legacy_case_id": key.legacy_case_id,
                       "task_id": key.task_id, "case_no": key.case_no,
                       "task_type": task.task_type,
                       "origin": [int(c) for c in origin["coord"]],
                       "heading": origin["heading"],
                       "scene": scene.scene_id,
                       "sha256": task.source_sha256})
    if any(t is None for t in tasks):
        raise ExperimentError("%s task set is incomplete" % cfg.test_version)
    return tasks, report


def assert_v4_matches_v0_tasks(cfg, tasks, legacy_tasks_path):
    """Preflight invariant: v4 ≡ v0 except origin (and the derived goal coord).

    The v4 test set was produced by re-planning the *same* 100 tasks from
    new start poses. Verified offline: ``task_type`` and ``description``
    are identical for all 100 cases and ``goals`` differ only for task
    types 7/8/10, where the ``Return to Origin`` goal carries the new
    origin. Any drift beyond that means the wrong data source is wired up.

    Returns a report dict; raises :class:`ExperimentError` on violation.
    """
    legacy = _read_json(legacy_tasks_path)
    if len(legacy) != len(tasks):
        raise ExperimentError("legacy tasks.json has %d entries, v4 has %d"
                              % (len(legacy), len(tasks)))
    goal_coord_diffs = []
    same_origin = []
    for i, old in enumerate(legacy):
        new = tasks[i]
        if int(old["task_type"]) != new.task_type:
            raise ExperimentError("case %d: task_type drift %s -> %s"
                                  % (i, old["task_type"], new.task_type))
        if old["description"] != new.description:
            raise ExperimentError("case %d: description drift" % i)
        diff = _goal_diff(old["goals"], new.goals)
        if diff == "return_to_origin_coord":
            goal_coord_diffs.append(i)
        elif diff:
            raise ExperimentError("case %d: unexpected goal drift (%s)" % (i, diff))
        if _int_coord(old["origin"]["coord"]) == _int_coord(new.origin["coord"]) \
                and old["origin"]["heading"] == new.origin["heading"]:
            same_origin.append(i)
    if same_origin:
        raise ExperimentError(
            "v4 origin identical to v0 for cases %s — the test set must move "
            "the start pose" % same_origin[:10])
    return {"cases": len(tasks),
            "goals_identical": len(tasks) - len(goal_coord_diffs),
            "return_to_origin_coord_updated": goal_coord_diffs}


def assert_v0_matches_legacy_tasks(cfg, tasks, legacy_tasks_path):
    """Preflight invariant for a v0 test: v0 task files ≡ legacy tasks.json.

    ``v0/task.json`` is the legacy ``dataset/warehouse_01_01/tasks.json``
    re-filed per case, so a v0 run must see exactly the task set the legacy
    v0 evaluation used: task type, description, goals and origin (cell and
    heading) all identical. Unlike the v4 check, nothing may move.

    Returns a report dict; raises :class:`ExperimentError` on violation.
    """
    legacy = _read_json(legacy_tasks_path)
    if len(legacy) != len(tasks):
        raise ExperimentError("legacy tasks.json has %d entries, v0 has %d"
                              % (len(legacy), len(tasks)))
    for i, old in enumerate(legacy):
        new = tasks[i]
        if int(old["task_type"]) != new.task_type:
            raise ExperimentError("case %d: task_type drift %s -> %s"
                                  % (i, old["task_type"], new.task_type))
        if old["description"] != new.description:
            raise ExperimentError("case %d: description drift" % i)
        if _normalise(old["goals"]) != _normalise(new.goals):
            raise ExperimentError("case %d: goals differ from legacy tasks.json" % i)
        if _int_coord(old["origin"]["coord"]) != _int_coord(new.origin["coord"]) \
                or old["origin"]["heading"] != new.origin["heading"]:
            raise ExperimentError(
                "case %d: v0 origin %r/%r != legacy %r/%r"
                % (i, new.origin["coord"], new.origin["heading"],
                   old["origin"]["coord"], old["origin"]["heading"]))
    return {"cases": len(tasks), "identical": len(tasks)}


def _int_coord(coord):
    return [int(float(c)) for c in coord]


def _goal_diff(old_goals, new_goals):
    """Return "" when equal, "return_to_origin_coord" for the accepted
    difference, or a description of the unexpected difference."""
    if len(old_goals) != len(new_goals):
        return "goal count %d != %d" % (len(old_goals), len(new_goals))
    only_origin_coord = False
    for a, b in zip(old_goals, new_goals):
        if _normalise(a) == _normalise(b):
            continue
        if a.get("name") == b.get("name") == "Return to Origin" \
                and set(a) == set(b) == {"name", "coord"}:
            only_origin_coord = True
            continue
        return "goal %r differs beyond the origin coord" % a.get("name")
    return "return_to_origin_coord" if only_origin_coord else ""


def _normalise(obj):
    """Compare numerically: the legacy file stores coords as floats."""
    if isinstance(obj, dict):
        return {k: _normalise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalise(v) for v in obj]
    if isinstance(obj, float) and obj == int(obj):
        return int(obj)
    return obj


# --------------------------------------------------------------------------
# historical trajectories (agent side)
# --------------------------------------------------------------------------
def _validate_identity(raw, key, version, path):
    for field, expected in (("task_id", key.task_id),
                            ("case_no", key.case_no),
                            ("legacy_case_id", key.legacy_case_id)):
        if field in raw and raw[field] != expected:
            raise ExperimentError("%s: %s=%r does not match case map %r"
                                  % (path, field, raw[field], expected))
    if "variant" in raw and raw["variant"] != version:
        raise ExperimentError("%s: variant=%r, expected %r"
                              % (path, raw["variant"], version))
    role = raw.get("role")
    expected_role = EXPECTED_ROLES.get(version)
    if role is not None and expected_role is not None and role != expected_role:
        raise ExperimentError("%s: role=%r, expected %r" % (path, role, expected_role))


def load_history_trajectory(cfg, case_key, version, base64_names=None):
    """Load + fully validate one historical ``forward_only`` trajectory.

    Raises :class:`ExperimentError` for anything wrong. Never returns
    ``None``: a caller that asked for this trajectory must not proceed
    without it.
    """
    if version not in HISTORY_VERSIONS:
        raise ExperimentError("%r is not a historical version" % version)
    tpath = cfg.trajectory_path(case_key, version)
    kpath = cfg.task_path(case_key, version)
    for p in (tpath, kpath):
        if not os.path.isfile(p):
            raise ExperimentError("historical file missing: %s" % p)

    task_raw = _read_json(kpath)
    _validate_identity(task_raw, case_key, version, kpath)
    traj_raw = _read_json(tpath)
    _validate_identity(traj_raw, case_key, version, tpath)

    fo = traj_raw.get("forward_only")
    if not fo:
        raise ExperimentError("%s: no forward_only trajectory" % tpath)
    if not fo.get("reachable"):
        raise ExperimentError(
            "%s: forward_only.reachable is false. Refusing to run — the "
            "legacy loader would have silently dropped this trajectory and "
            "turned the run into an unlabelled working-memory baseline."
            % tpath)

    actions = fo.get("actions") or []
    poses = fo.get("poses") or []
    images = fo.get("image_paths") or []
    length = fo.get("length")
    notes = []

    if length != len(actions):
        raise ExperimentError("%s: length=%r != len(actions)=%d"
                              % (tpath, length, len(actions)))
    if len(poses) != len(actions) + 1:
        raise ExperimentError("%s: len(poses)=%d != len(actions)+1=%d"
                              % (tpath, len(poses), len(actions) + 1))
    if len(images) != len(poses):
        raise ExperimentError("%s: len(image_paths)=%d != len(poses)=%d"
                              % (tpath, len(images), len(poses)))

    origin = task_raw.get("origin") or {}
    ocoord = _int_coord(origin.get("coord", []))
    oheading = origin.get("heading")
    p0 = poses[0]
    if [int(p0[0]), int(p0[1])] != ocoord or p0[2] != oheading:
        raise ExperimentError("%s: first pose %r != task origin %r/%r"
                              % (tpath, p0, ocoord, oheading))

    for i, (pose, name) in enumerate(zip(poses, images)):
        expected = "Grid_X%d_Y%d_%s.png" % (int(pose[0]), int(pose[1]), pose[2])
        if os.path.basename(name) != expected:
            raise ExperimentError("%s: image[%d]=%r does not match pose %r "
                                  "(expected %s)" % (tpath, i, name, pose, expected))
        if base64_names is not None:
            stem = os.path.splitext(os.path.basename(name))[0]
            if stem not in base64_names:
                raise ExperimentError("%s: image[%d] %s has no precomputed "
                                      "base64 in the scene" % (tpath, i, stem))

    events = fo.get("stage_events") or []
    for ev in events:
        step = ev.get("step")
        if not isinstance(step, int) or not (0 <= step <= len(actions)):
            raise ExperimentError("%s: stage event step %r out of range"
                                  % (tpath, step))

    anomaly = KNOWN_STAGE_ANOMALIES.get((case_key.task_id, case_key.case_no, version))
    stages_completed = fo.get("stages_completed")
    stages_total = fo.get("stages_total")
    if stages_total is not None and (stages_completed != stages_total
                                     or len(events) != stages_total):
        if anomaly is None:
            raise ExperimentError(
                "%s: stage bookkeeping inconsistent (completed=%r total=%r "
                "events=%d) and this case is not a registered anomaly"
                % (tpath, stages_completed, stages_total, len(events)))
        notes.append("known stage anomaly: %s" % anomaly)

    scene = traj_raw.get("scene") or task_raw.get("scene") or {}
    if scene and scene.get("same_as_base") is False:
        raise ExperimentError("%s: trajectory does not use the base scene" % tpath)

    return HistoricalTrajectory(
        case_key=case_key, version=version, source_path=tpath,
        source_sha256=sha256_file(tpath),
        description=task_raw.get("description", ""),
        origin={"coord": ocoord, "heading": oheading},
        actions=actions, poses=poses, image_paths=images,
        stage_events=events, length=length, validation_notes=notes,
        scene=cfg.scene_for(case_key, version),
    )


def load_history_set(cfg, case_key, versions, base64_names=None):
    """Load several versions for one case, in the given order."""
    out = []
    for v in versions:
        out.append(load_history_trajectory(cfg, case_key, v, base64_names=base64_names))
    return out


def scene_base64_names(cfg, scene=None):
    """Set of ``Grid_X*_Y*_<Heading>`` stems that have a precomputed .b64.

    Defaults to the base scene, which is what the v0–v3 history frames are
    checked against. Pass ``scene`` to check a v5 case's own cache.
    """
    d = scene.image_base64_dir if scene is not None else cfg.scene_image_base64_dir
    return {os.path.splitext(n)[0] for n in os.listdir(d) if n.endswith(".b64")}


# --------------------------------------------------------------------------
# text memories (agent side)
# --------------------------------------------------------------------------
def load_text_memory(cfg, version, kind):
    """Load one Summary / Semantic final state and verify its hash."""
    if kind not in ("summary", "semantic"):
        raise ExperimentError("unknown text-memory kind %r" % kind)
    entry = cfg.memory_files.get(kind, {}).get(version)
    if not entry:
        raise ExperimentError("no %s memory configured for %s" % (kind, version))
    path = resolve_path(entry["path"])
    if not os.path.isfile(path):
        raise ExperimentError("%s memory for %s missing: %s" % (kind, version, path))
    actual = sha256_file(path)
    expected = entry.get("sha256")
    if expected and actual != expected:
        raise ExperimentError(
            "%s memory for %s changed on disk.\n  path:     %s\n  expected: %s\n"
            "  actual:   %s\nRefusing to mix different artefacts into one "
            "experiment id." % (kind, version, path, expected, actual))
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise ExperimentError("%s memory for %s is empty: %s" % (kind, version, path))
    return TextMemory(version=version, kind=kind, source_path=path,
                      source_sha256=actual,
                      scope_case_count=int(entry.get("scope_case_count", 100)),
                      text=text)


def load_text_memory_set(cfg, versions, kind):
    return [load_text_memory(cfg, v, kind) for v in versions]


def memory_profile(cfg):
    """Per-file byte/char profile for the preflight report (see plan §3.6)."""
    rows = []
    for kind in ("summary", "semantic"):
        for version in HISTORY_VERSIONS:
            mem = load_text_memory(cfg, version, kind)
            rows.append({
                "kind": kind, "version": version, "path": mem.source_path,
                "sha256": mem.source_sha256,
                "bytes": os.path.getsize(mem.source_path),
                "chars": len(mem.text),
                "max_case_mentioned": _max_case_mentioned(mem.text),
            })
    return rows


def _max_case_mentioned(text):
    """Largest ``Case N`` referenced — a cheap coverage check."""
    import re
    nums = [int(m) for m in re.findall(r"Case (\d+)", text)]
    return max(nums) if nums else None
