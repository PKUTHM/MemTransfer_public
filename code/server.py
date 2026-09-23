import os
import sys
import json
import threading
from datetime import datetime
from flask import Flask, request, jsonify

# Windows consoles default to the ANSI code page (GBK on this machine),
# which makes the emoji in the progress prints raise UnicodeEncodeError
# and kill the process. Force UTF-8 with replacement instead.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
SCENE_DIR = os.path.join(DATASET_DIR, "warehouse_01_01")
IMAGE_DIR = os.path.join(SCENE_DIR, "image")
METADATA_PATH = os.path.join(SCENE_DIR, "metadata.json")
TASKS_PATH = os.path.join(SCENE_DIR, "tasks.json")
RESULT_ROOT = os.path.join(PROJECT_ROOT, "result")

MAX_STEPS = 200
NOTIFY_ON_PROGRESS = True
REQUIRE_EXPLICIT_DECLARE = False

START_TIME_SUFFIX = datetime.now().strftime("%Y%m%d_%H%M%S")
# Default nests per-case logs under a subfolder (so direct `python3
# code/server.py` doesn't pollute result/ root). test_run.py always
# passes --log_dir explicitly to route logs into the test-run folder.
LOG_DIR = os.path.join(RESULT_ROOT, "server_run", f"evaluation_logs_{START_TIME_SUFFIX}")
# test_run.py can pass --log_dir to redirect the per-case log directory
# into its own output folder (so everything ends up under one test_run_<ts>/).

DIRS_CCW = ["East", "North", "West", "South"]
DELTAS = {"East": (1, 0), "North": (0, 1), "West": (-1, 0), "South": (0, -1)}

AGENT_SESSIONS = {}
TASKS_DB = []
ITEMS_POOL = []
REACHABLE = set()
BOUNDS_X = (-7, 7)
BOUNDS_Y = (-10, 16)

# --- experiment mode (v0-v3 -> v4/v5 study) -------------------------------
# When EXPERIMENT is set, TASKS_DB is built from the per-case v4/v5 task.json
# files instead of the legacy v0 tasks.json, and the agent is required to
# handshake on the experiment fingerprint before its first model call.
EXPERIMENT = None            # experiment.ExperimentConfig or None
EXPERIMENT_INFO = {}         # small dict echoed by /get_server_time
CASE_IDENTITY = {}           # legacy_case_id -> {task_id, case_no, ...}
SCENE_ENVS = {}              # scene_id -> SceneEnv
CASE_SCENE = {}              # legacy_case_id -> scene_id


class SceneEnv(object):
    """The world one case is stepped through: geometry plus where its
    renders live.

    v1-v4 all share the base scene, so a single set of module globals used
    to be enough. A v5 case blocks two cells, which changes ``reachable``
    (223 cells instead of 225) and re-renders the frames that see the
    obstacle, so movement legality and image lookup both become per-case.
    Holding them in one object keyed by scene id keeps the two from drifting
    apart -- serving base-scene images while enforcing obstacle-scene
    movement would be silent and very hard to spot in a log.
    """

    __slots__ = ("scene_id", "image_dir", "items", "reachable",
                 "bounds_x", "bounds_y")

    def __init__(self, scene_id, image_dir, items, reachable, bounds_x, bounds_y):
        self.scene_id = scene_id
        self.image_dir = image_dir
        self.items = items
        self.reachable = reachable
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y

    @classmethod
    def from_scene(cls, scene):
        """Build from a :class:`scene_index.SceneRef` by reading its metadata."""
        with open(scene.metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        sb = meta.get("scene_bounds", {})
        bx = (int(sb["x"][0]), int(sb["x"][1])) if len(sb.get("x", [])) == 2 else BOUNDS_X
        by = (int(sb["y"][0]), int(sb["y"][1])) if len(sb.get("y", [])) == 2 else BOUNDS_Y
        return cls(scene_id=scene.scene_id, image_dir=scene.image_dir,
                   items=meta["items"],
                   reachable=set(tuple(p) for p in meta["all_reachable_points"]),
                   bounds_x=bx, bounds_y=by)

    def as_info(self):
        return {"scene_id": self.scene_id,
                "reachable_count": len(self.reachable),
                "item_count": len(self.items),
                "scene_bounds": {"x": list(self.bounds_x), "y": list(self.bounds_y)},
                "directions": DIRS_CCW}


def _legacy_env():
    """The module-global scene, for runs that never called load_experiment."""
    return SceneEnv(scene_id="legacy", image_dir=IMAGE_DIR, items=ITEMS_POOL,
                    reachable=REACHABLE, bounds_x=BOUNDS_X, bounds_y=BOUNDS_Y)


def env_for(state):
    """The :class:`SceneEnv` a session is running in."""
    sid = (state or {}).get("scene_id")
    if sid is not None and sid in SCENE_ENVS:
        return SCENE_ENVS[sid]
    return _legacy_env()

#: Guards session mutation. Flask runs threaded and the agent drives up to
#: --max_workers cases against one server, so /step must not interleave.
SESSION_LOCK = threading.Lock()

#: Cap on the per-session action_seq replay cache (see /step). One entry
#: per executed action; MAX_STEPS is the natural bound, +8 for slack.
ACTION_CACHE_LIMIT = 208


def turn_left(d):
    return DIRS_CCW[(DIRS_CCW.index(d) + 1) % 4]


def turn_right(d):
    return DIRS_CCW[(DIRS_CCW.index(d) - 1) % 4]


def turn_180(d):
    return DIRS_CCW[(DIRS_CCW.index(d) + 2) % 4]


def step_delta(d, action):
    base = DELTAS[d]
    if action == "Move_Forward":
        return base
    if action == "Move_Backward":
        return (-base[0], -base[1])
    if action == "Move_Left":
        return DELTAS[turn_left(d)]
    if action == "Move_Right":
        return DELTAS[turn_right(d)]
    return (0, 0)


def image_path(x, y, d, env=None):
    """Path of the render for one pose **in that session's scene**.

    The filename is the same in every scene, so the directory is the only
    thing that distinguishes a v4 frame from its v5 counterpart.
    """
    name = f"Grid_X{x}_Y{y}_{d}.png"
    return os.path.join((env or _legacy_env()).image_dir, name)


def load_experiment(config_path):
    """Switch the server onto the v4 test set defined by an experiment config.

    ``TASKS_DB`` stays a flat list indexed by ``legacy_case_id``, so the
    numeric ``case_id`` in the HTTP API keeps exactly the meaning it has
    always had. Only ``v4/task.json`` is read; ``v4/trajectory.json`` (the
    expert answer) is never opened by the server.
    """
    global EXPERIMENT, EXPERIMENT_INFO, CASE_IDENTITY, TASKS_DB, MAX_STEPS
    global METADATA_PATH, IMAGE_DIR, SCENE_ENVS, CASE_SCENE
    global NOTIFY_ON_PROGRESS, REQUIRE_EXPLICIT_DECLARE, ACTION_CACHE_LIMIT
    import experiment as ex

    EXPERIMENT = ex.load_config(config_path)
    METADATA_PATH = EXPERIMENT.scene_metadata
    IMAGE_DIR = EXPERIMENT.scene_image_dir
    MAX_STEPS = EXPERIMENT.max_steps

    # Server-side knobs come from the config too. notify_on_progress is the
    # one the model can see: it decides whether each observation carries a
    # per-step note, so leaving it as a module constant meant a run was not
    # fully described by its config.
    srv = EXPERIMENT.server
    NOTIFY_ON_PROGRESS = bool(srv["notify_on_progress"])
    REQUIRE_EXPLICIT_DECLARE = bool(srv["require_explicit_declare"])
    ACTION_CACHE_LIMIT = int(srv["action_cache_limit"])

    tasks, report = ex.load_test_tasks(EXPERIMENT)
    TASKS_DB = [t.as_server_task() for t in tasks]
    CASE_IDENTITY = {t.case_key.legacy_case_id: t.case_key.as_dict() for t in tasks}

    # One SceneEnv per distinct scene in the test set: 1 for a v4 run, up to
    # 38 for a v5 run. Built here so /step never touches the filesystem.
    SCENE_ENVS = {}
    CASE_SCENE = {}
    for t in tasks:
        if t.scene.scene_id not in SCENE_ENVS:
            SCENE_ENVS[t.scene.scene_id] = SceneEnv.from_scene(t.scene)
        CASE_SCENE[t.case_key.legacy_case_id] = t.scene.scene_id

    EXPERIMENT_INFO = {
        "experiment_id": EXPERIMENT.experiment_id,
        "test_version": EXPERIMENT.test_version,
        "config_fingerprint": EXPERIMENT.fingerprint(),
        "config_path": EXPERIMENT.config_path,
        "case_ids": sorted(CASE_IDENTITY),
        "max_steps": MAX_STEPS,
        "scenes": sorted(SCENE_ENVS),
    }
    print(f"🧪 Experiment mode: {EXPERIMENT.experiment_id} "
          f"({EXPERIMENT.test_version}, {len(TASKS_DB)} cases, "
          f"{len(SCENE_ENVS)} scene(s), "
          f"fingerprint {EXPERIMENT_INFO['config_fingerprint'][:12]})")
    return report


def load_environment():
    global TASKS_DB, ITEMS_POOL, REACHABLE, BOUNDS_X, BOUNDS_Y
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    REACHABLE = set(tuple(p) for p in meta["all_reachable_points"])
    ITEMS_POOL = meta["items"]
    sb = meta.get("scene_bounds", {})
    if "x" in sb and len(sb["x"]) == 2:
        BOUNDS_X = (int(sb["x"][0]), int(sb["x"][1]))
    if "y" in sb and len(sb["y"]) == 2:
        BOUNDS_Y = (int(sb["y"][0]), int(sb["y"][1]))

    if EXPERIMENT is None:
        with open(TASKS_PATH, "r", encoding="utf-8") as f:
            TASKS_DB = json.load(f)
    # In experiment mode TASKS_DB was already built from the v4 task files
    # by load_experiment(); the legacy v0 tasks.json must not clobber it.

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"🌟 Server Core Loaded.")
    print(f"   tasks loaded     : {len(TASKS_DB)}")
    print(f"   reachable points : {len(REACHABLE)}")
    print(f"   items            : {len(ITEMS_POOL)}")
    print(f"   scene bounds     : X={BOUNDS_X}, Y={BOUNDS_Y}")
    print(f"   image dir        : {IMAGE_DIR}")
    if SCENE_ENVS:
        sizes = sorted({len(e.reachable) for e in SCENE_ENVS.values()})
        print(f"   per-case scenes  : {len(SCENE_ENVS)} "
              f"(reachable {sizes[0]}-{sizes[-1]} cells)")
    print(f"   log dir          : {LOG_DIR}")


def items_here(state):
    pos = (state["x"], state["y"])
    out = []
    for it in env_for(state).items:
        for rp in it.get("reachable_points", []):
            if (rp[0], rp[1]) == pos:
                out.append(it)
                break
    return out


def instance_key(it):
    return (it["position"][0], it["position"][1])


def _jsonify_progress(progress):
    if isinstance(progress, set):
        return sorted(list(progress))
    if isinstance(progress, dict):
        out = {}
        for k, v in progress.items():
            key = f"{k[0]},{k[1]}" if isinstance(k, tuple) else str(k)
            out[key] = list(v) if isinstance(v, set) else v
        return out
    return progress


def save_log(agent_id):
    s = AGENT_SESSIONS[agent_id]
    entry = {
        "agent_id": agent_id,
        "case_id": s["case_id"],
        "task_type": s["task_type"],
        "description": s["description"],
        "origin": {"coord": list(s["origin"]), "heading": s["origin_heading"]},
        "final_status": s["status"],
        "total_steps": s["step_count"],
        "progress": _jsonify_progress(s["progress"]),
        "stage_steps": {k: v for k, v in s["stage_steps"].items()},
        "l1": [[name, list(pos)] for (name, pos) in s["l1"]],
        "l2": [[name, list(pos)] for (name, pos) in s["l2"]],
        "stages_completed": s.get("stages_completed", 0),
        "stages_total": s.get("stages_total", 0),
        "action_history": s["actions"],
    }
    fp = os.path.join(LOG_DIR, f"case_{s['case_id']}_{agent_id}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=2, ensure_ascii=False)


def check_completion(state):
    """Return (done: bool, note: str). Mutates state.progress / state.stage_steps / state.a_visited / state.b_visited."""
    t = state["task_type"]
    goals = state["goals"]
    prog = state["progress"]
    here = items_here(state)
    here_keys = {instance_key(it): it for it in here}
    here_names = set(it["name"] for it in here)
    note = ""

    if t == 1:
        target = goals[0]["name"]
        if target in here_names:
            state["stage_steps"][f"hit_{target}"] = state["step_count"]
            _bump_stage(state)
            return True, f"Reached '{target}'."

    elif t == 2:
        target = goals[0]["name"]
        inst = goals[0]["instances"]
        for k, it in here_keys.items():
            if it["name"] == target and k not in prog:
                prog.add(k)
                state["stage_steps"][f"hit_{target}_{k}"] = state["step_count"]
                _bump_stage(state)
                note = f"Collected '{target}' at {k} ({len(prog)}/{len(inst)})."
        if len(prog) == len(inst):
            return True, f"All {len(inst)} '{target}' instances collected."

    elif t == 3:
        cat = goals[0]["category"]
        for it in here:
            if it["category"] == cat:
                state["stage_steps"][f"hit_{cat}"] = state["step_count"]
                _bump_stage(state)
                return True, f"Found category '{cat}' via '{it['name']}'."

    elif t == 4:
        cat = goals[0]["category"]
        inst = goals[0]["instances"]
        for k, it in here_keys.items():
            if it["category"] == cat and k not in prog:
                prog.add(k)
                state["stage_steps"][f"hit_{cat}_{k}"] = state["step_count"]
                _bump_stage(state)
                note = f"Collected '{it['name']}' from '{cat}' ({len(prog)}/{len(inst)})."
        if len(prog) == len(inst):
            return True, f"All {len(inst)} items in category '{cat}' collected."

    elif t == 5:
        names = [g["name"] for g in goals]
        for n in names:
            if n in here_names and n not in prog:
                prog.add(n)
                state["stage_steps"][f"hit_{n}"] = state["step_count"]
                _bump_stage(state)
                note = f"Found '{n}' ({len(prog)}/{len(names)})."
        if prog == set(names):
            return True, f"Found all: {sorted(prog)}."

    elif t == 6:
        names = [g["name"] for g in goals]
        expected = names[len(prog)] if len(prog) < len(names) else None
        if expected and expected in here_names:
            prog.add(expected)
            state["stage_steps"][f"step_{len(prog)}_{expected}"] = state["step_count"]
            _bump_stage(state)
            note = f"Found '{expected}' ({len(prog)}/{len(names)})."
        elif expected and here_names:
            others = here_names - {expected}
            if others:
                note = f"Out-of-order: found {sorted(others)}, expected '{expected}'."
        if len(prog) == len(names):
            return True, f"Sequence complete: {names}."

    elif t == 7:
        target = goals[0]["name"]
        origin = state["origin"]
        if prog.get("found_target") is None:
            if target in here_names:
                prog["found_target"] = True
                state["stage_steps"]["found_target"] = state["step_count"]
                _bump_stage(state)
                note = f"Reached '{target}'. Now return to origin {list(origin)}."
        else:
            if (state["x"], state["y"]) == origin:
                prog["returned"] = True
                state["stage_steps"]["returned"] = state["step_count"]
                _bump_stage(state)
                return True, "Returned to origin. Task complete."
            note = f"Returning: at ({state['x']},{state['y']}), origin {list(origin)}."

    elif t == 8:
        target = goals[0]["name"]
        inst = goals[0]["instances"]
        origin = state["origin"]
        for k, it in here_keys.items():
            if it["name"] == target and k not in prog:
                prog[k] = True
                state["stage_steps"][f"collected_{k}"] = state["step_count"]
                _bump_stage(state)
                note = f"Collected '{target}' at {k}."
        total = len(inst)
        got = sum(1 for k in inst if (k["position"][0], k["position"][1]) in prog and prog[(k["position"][0], k["position"][1])] is True)
        if got == total and prog.get("returned") is None:
            note = f"All {total} '{target}' collected. Return to origin {list(origin)}."
        if got == total and (state["x"], state["y"]) == origin:
            prog["returned"] = True
            state["stage_steps"]["returned"] = state["step_count"]
            _bump_stage(state)
            return True, "All collected and returned. Task complete."

    elif t == 9:
        target_a_pos = tuple(int(c) for c in goals[0]["instance"]["position"])
        target_b_pos = tuple(int(c) for c in goals[1]["instance"]["position"])
        name_b = goals[1]["name"]
        l1 = state["l1"]
        l2 = state["l2"]
        for it in here:
            it_pos = tuple(int(c) for c in it["position"])
            inst = (it["name"], it_pos)
            if it_pos == target_b_pos:
                l1.extend(l2)
                l2.clear()
                if not state.get("b_reached"):
                    state["b_reached"] = True
                    state["stage_steps"]["found_B"] = state["step_count"]
                    if l1:
                        _bump_stage(state)
                    note = f"Hit B '{name_b}'. Now return to an A visited before B."
            elif it_pos == target_a_pos:
                if state.get("b_reached"):
                    if inst in l1:
                        state["stage_steps"]["loop_complete"] = state["step_count"]
                        _bump_stage(state)
                        return True, f"A-B-A loop closed for '{inst[0]}' at {inst[1]}."
                    elif inst not in l2:
                        l2.append(inst)
                        note = f"New A '{inst[0]}' at {inst[1]} added to post-B list."
                else:
                    if inst not in l1:
                        was_empty = len(l1) == 0
                        l1.append(inst)
                        state["stage_steps"][f"found_A_{inst[1]}"] = state["step_count"]
                        if was_empty:
                            _bump_stage(state)
                        note = f"A '{inst[0]}' at {inst[1]} recorded before B."

    elif t == 10:
        name_a = goals[0]["name"]
        name_b = goals[1]["name"]
        origin = state["origin"]
        # Type 10 also uses fixed instances from goals, check position to avoid
        # matching other items with the same name.
        target_a_pos = tuple(int(c) for c in goals[0]["instance"]["position"])
        target_b_pos = tuple(int(c) for c in goals[1]["instance"]["position"])
        if prog.get("found_a") is None:
            for it in here:
                it_pos = tuple(int(c) for c in it["position"])
                if it_pos == target_a_pos:
                    prog["found_a"] = True
                    state["stage_steps"]["found_A"] = state["step_count"]
                    _bump_stage(state)
                    note = f"Hit A '{name_a}'."
                    return False, note
        elif prog.get("found_b") is None:
            for it in here:
                it_pos = tuple(int(c) for c in it["position"])
                if it_pos == target_b_pos:
                    prog["found_b"] = True
                    state["stage_steps"]["found_B"] = state["step_count"]
                    _bump_stage(state)
                    note = f"Hit B '{name_b}'. Now return to origin {list(origin)}."
                    return False, note
        else:
            if (state["x"], state["y"]) == origin:
                state["stage_steps"]["returned"] = state["step_count"]
                _bump_stage(state)
                return True, "A-B-origin sequence complete."
            note = f"Returning: at ({state['x']},{state['y']}), origin {list(origin)}."

    return False, note


def stages_total_for(task):
    """How many discrete stages a task has, indexed by task_type."""
    t = task["task_type"]
    if t in (1, 3):
        return 1
    if t in (2, 4):
        return len(task["goals"][0]["instances"])
    if t in (5, 6):
        return 2
    if t == 7:
        return 2
    if t == 8:
        return len(task["goals"][0]["instances"]) + 1
    if t == 9:
        return 3
    if t == 10:
        return 3
    return 0


def build_state(task, case_id):
    origin = task["origin"]
    t = task["task_type"]
    if t in (2, 4, 5, 6):
        progress = set()
    else:
        progress = {}
    return {
        "case_id": case_id,
        # Which scene this session steps through. Looked up per case so a v5
        # run gets its own obstacle scene instead of the base one.
        "scene_id": CASE_SCENE.get(case_id),
        "task_type": t,
        "description": task["description"],
        "goals": task["goals"],
        "origin": tuple(int(c) for c in origin["coord"]),
        "origin_heading": origin["heading"],
        "x": int(origin["coord"][0]),
        "y": int(origin["coord"][1]),
        "d": origin["heading"],
        "step_count": 0,
        "actions": [],
        "status": "running",
        "progress": progress,
        "l1": [],
        "l2": [],
        "stage_steps": {},
        "stages_completed": 0,
        "stages_total": stages_total_for(task),
    }


def _bump_stage(state, label=None):
    """Increment stages_completed by 1, optionally recording which label triggered it."""
    state["stages_completed"] = min(
        state.get("stages_completed", 0) + 1,
        state.get("stages_total", 0),
    )


app = Flask(__name__)


@app.route("/get_server_time", methods=["GET"])
def get_server_time():
    return jsonify({
        "status": "success",
        "server_start_time": START_TIME_SUFFIX,
        "total_tasks": len(TASKS_DB),
        "server_paths": {
            "metadata_path": METADATA_PATH,
            "tasks_path": TASKS_PATH,
            "image_dir": IMAGE_DIR,
            "log_dir": LOG_DIR,
        },
        "config": {
            "max_steps": MAX_STEPS,
            "notify_on_progress": NOTIFY_ON_PROGRESS,
            "scene_bounds": {"x": list(BOUNDS_X), "y": list(BOUNDS_Y)},
            "directions": DIRS_CCW,
        },
        # Empty dict when running the legacy v0 entry point. The agent
        # refuses to start an experiment group against such a server.
        "experiment": EXPERIMENT_INFO,
    })


@app.route("/init_task", methods=["POST"])
def init_task():
    req = request.json or {}
    case_id = req.get("case_id", 0)
    agent_id = req.get("agent_id")
    if agent_id is None:
        return jsonify({"status": "error", "message": "missing agent_id"}), 400
    if not isinstance(case_id, int) or case_id < 0 or case_id >= len(TASKS_DB):
        return jsonify({"status": "error", "message": f"case_id out of range [0, {len(TASKS_DB) - 1}]"}), 400

    task = TASKS_DB[case_id]
    state = build_state(task, case_id)
    # Replay cache for /step idempotency (see the action_seq contract).
    state["action_seq"] = 0
    state["action_cache"] = {}
    with SESSION_LOCK:
        AGENT_SESSIONS[agent_id] = state

    env = env_for(state)
    img = image_path(state["x"], state["y"], state["d"], env)
    init_msg = (
        f"Task {case_id} ready (type {state['task_type']}).\n"
        f"Origin: ({state['x']}, {state['y']}) facing {state['d']}.\n"
        f"View mode: single main view."
    )
    state["latest_notification"] = init_msg
    print(f"🚩 [Spawn {agent_id}] case {case_id} type {state['task_type']} "
          f"@({state['x']},{state['y']}) {state['d']} scene={env.scene_id}")
    return jsonify({
        "status": "ready",
        "agent_id": agent_id,
        "case_id": case_id,
        "task_type": state["task_type"],
        "task_description": state["description"],
        "origin": {"coord": [state["x"], state["y"]], "heading": state["d"]},
        "current_pose": {"x": state["x"], "y": state["y"], "heading": state["d"]},
        "stages_total": state["stages_total"],
        "visual_observations": {"main_view": img},
        # Logical identity + which version is under test. Empty in legacy
        # mode; the agent asserts on these in experiment mode.
        "case_identity": CASE_IDENTITY.get(case_id, {}),
        "test_version": EXPERIMENT_INFO.get("test_version", ""),
        "experiment_id": EXPERIMENT_INFO.get("experiment_id", ""),
        "max_steps": MAX_STEPS,
        "scene_info": env.as_info(),
    })


VALID_ACTIONS = {"Move_Forward", "Move_Backward", "Move_Left", "Move_Right",
                 "Turn_Left", "Turn_Right", "Turn_180"}


def normalize_action(raw):
    """Accept 'Move_Forward' or 'Move_Forward 1m'; strip the unit suffix.
    Returns the canonical base action, or None if invalid."""
    if not isinstance(raw, str):
        return None
    a = raw.strip()
    for base in VALID_ACTIONS:
        if a == base or a == f"{base} 1m":
            return base
    return None


@app.route("/end_case", methods=["POST"])
def end_case():
    """Release a finished session and its replay cache.

    Without this, ``AGENT_SESSIONS`` grows for the whole life of the
    process and every session keeps up to ``MAX_STEPS`` cached responses.
    Idempotent: ending an unknown session is not an error.
    """
    req = request.json or {}
    agent_id = req.get("agent_id")
    with SESSION_LOCK:
        existed = AGENT_SESSIONS.pop(agent_id, None) is not None
    return jsonify({"status": "success", "released": existed,
                    "active_sessions": len(AGENT_SESSIONS)})


@app.route("/step", methods=["POST"])
def step():
    req = request.json or {}
    agent_id = req.get("agent_id")
    raw_action = req.get("action")
    action_seq = req.get("action_seq")

    # One /step at a time. Flask is threaded and the agent drives many
    # cases concurrently against one server; without this, two requests
    # for the same session could interleave their read-modify-write of
    # step_count / position.
    with SESSION_LOCK:
        if not agent_id or agent_id not in AGENT_SESSIONS:
            return jsonify({"status": "error", "message": "unknown agent_id"}), 400
        state = AGENT_SESSIONS[agent_id]

        # --- action_seq replay protection -----------------------------
        # The client retries /step on connection errors. If the server
        # already applied the action but the response was lost, a naive
        # retry would move the agent twice. With action_seq, a retry of an
        # already-applied sequence number replays the cached response.
        # Checked BEFORE the "session ended" guard, so retrying the final
        # (task-completing) step still returns its original result.
        if action_seq is not None:
            if not isinstance(action_seq, int) or action_seq < 1:
                return jsonify({"status": "error",
                                "message": f"invalid action_seq: {action_seq!r}"}), 400
            cached = state["action_cache"].get(action_seq)
            if cached is not None:
                if cached["raw_action"] != raw_action:
                    return jsonify({
                        "status": "error",
                        "message": (f"action_seq {action_seq} was already applied with "
                                    f"action {cached['raw_action']!r}, cannot re-submit "
                                    f"{raw_action!r}")}), 409
                return jsonify(dict(cached["response"], replayed=True))
            expected = state["action_seq"] + 1
            if action_seq != expected:
                return jsonify({
                    "status": "error",
                    "message": (f"action_seq gap: expected {expected}, got {action_seq}")
                }), 409

        if state["status"] != "running":
            return jsonify({"status": state["status"],
                            "message": "session ended, call /init_task"}), 400
        action = normalize_action(raw_action)
        if action is None:
            return jsonify({"status": "error", "message": f"invalid action: {raw_action}"}), 400

        return _apply_step(agent_id, state, raw_action, action, action_seq)


def _apply_step(agent_id, state, raw_action, action, action_seq):
    """Execute one validated action. Caller holds ``SESSION_LOCK``."""
    state["step_count"] += 1
    state["actions"].append(raw_action if isinstance(raw_action, str) else action)
    move_status = "success"

    if action == "Turn_Left":
        state["d"] = turn_left(state["d"])
    elif action == "Turn_Right":
        state["d"] = turn_right(state["d"])
    elif action == "Turn_180":
        state["d"] = turn_180(state["d"])
    else:
        env = env_for(state)
        dx, dy = step_delta(state["d"], action)
        nx, ny = state["x"] + dx, state["y"] + dy
        if not (env.bounds_x[0] <= nx <= env.bounds_x[1]
                and env.bounds_y[0] <= ny <= env.bounds_y[1]):
            move_status = "out_of_range"
        elif (nx, ny) in env.reachable:
            state["x"], state["y"] = nx, ny
        else:
            move_status = "blocked"

    done, note = check_completion(state)
    final_status = move_status
    if done:
        state["status"] = "task_completed"
        final_status = "task_completed"
    elif state["step_count"] >= MAX_STEPS:
        state["status"] = "failed_timeout"
        final_status = "failed_timeout"

    if not note and NOTIFY_ON_PROGRESS and final_status == "success":
        note = f"Step {state['step_count']}: at ({state['x']},{state['y']}) {state['d']}."

    save_log(agent_id)
    img = image_path(state["x"], state["y"], state["d"], env_for(state))
    print(f"🎬 [{agent_id}][Step {state['step_count']}] {action:<15} -> ({state['x']},{state['y']}) {state['d']}  | {final_status}")
    payload = {
        "status": final_status,
        "message": note if NOTIFY_ON_PROGRESS else "",
        "step": state["step_count"],
        "current_pose": {"x": state["x"], "y": state["y"], "heading": state["d"]},
        "stages_completed": state.get("stages_completed", 0),
        "stages_total": state.get("stages_total", 0),
        "visual_observations": {"main_view": img},
    }
    if action_seq is not None:
        state["action_seq"] = action_seq
        cache = state["action_cache"]
        cache[action_seq] = {"raw_action": raw_action, "response": payload}
        if len(cache) > ACTION_CACHE_LIMIT:
            for old in sorted(cache)[:len(cache) - ACTION_CACHE_LIMIT]:
                del cache[old]
        payload = dict(payload, action_seq=action_seq)
    return jsonify(payload)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000,
                        help="port to bind the Flask app to (default 5000)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="host to bind to (default 127.0.0.1)")
    parser.add_argument("--log_dir", default=None,
                        help="override the per-case log directory (default "
                             "result/evaluation_logs_<start_ts>/). Pass this to "
                             "nest server logs under a parent folder (e.g. a "
                             "test_run_<ts>/ subtree).")
    parser.add_argument("--experiment_config", default=None,
                        help="path to a config/**/*.yaml, or just its name. Switches the "
                             "server onto that config's v4 test set (task.json per "
                             "case) instead of the legacy v0 tasks.json, and takes "
                             "max_steps from the config.")
    args = parser.parse_args()
    if args.log_dir:
        # At module level, plain assignment reaches the global variable.
        LOG_DIR = args.log_dir
        os.makedirs(LOG_DIR, exist_ok=True)
    if args.experiment_config:
        load_experiment(args.experiment_config)
    load_environment()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
