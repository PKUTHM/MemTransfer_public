import os
import io
import sys
import json
import time
import base64
import copy
import random
import argparse
import requests
import pandas as pd
from PIL import Image
from threading import Lock

# Windows consoles default to the ANSI code page (GBK on this machine),
# which makes the emoji in the progress prints raise UnicodeEncodeError
# and kill the worker. Force UTF-8 with replacement instead.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from prompt import (
    build_system_prompt,
    mission_block,
    history_with_images_blocks,
    history_text_block,
    history_images_blocks,
    grid_memory_block,
    current_image_blocks,
    reference_traj_block,
    peer_reference_traj_block,
    reference_traj_images_blocks,
    placeholders_block,
    episodic_memory_blocks,
    episodic_memory_multi_blocks,
    common_note_block,
    case_scope_declaration,
    version_begin,
    version_end,
    wrap_text_memory,
    origin_matches,
    DecisionContext,
    set_use_precomputed,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
SCENE_DIR = os.path.join(DATASET_DIR, "warehouse_01_01")

# Default paths for the optional memory-recall flags. Empty by default
# (feature off). When the user passes a path, relative paths are resolved
# against PROJECT_ROOT. Typical use: pass the running-memory output
# file produced by scripts/semantic_memory_consolidation.py /
# scripts/summary_memory_writer.py, e.g.
#   dataset_v1-v3_memory/v0/semantic_memory/memory.md
#   dataset_v1-v3_memory/v0/summary_memory/summary.md
DEFAULT_SEMANTIC_PROMPT_PATH = ""
DEFAULT_SUMMARY_PROMPT_PATH = ""
METADATA_PATH = os.path.join(SCENE_DIR, "metadata.json")
TASKS_PATH = os.path.join(SCENE_DIR, "tasks.json")
DEFAULT_SERVER_URL = "http://127.0.0.1:5000"
DEFAULT_API_EXCEL = os.path.join(PROJECT_ROOT, "api.xlsx")
# When invoked directly (without test_run.py), write to a timestamped
# subfolder so we never pollute the result/ root. test_run.py always
# passes --results_dir explicitly, so its own folder takes precedence.
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "result", f"agent_run")
DEFAULT_TRAJECTORY_JSON = os.path.join(SCENE_DIR, "trajectories.json")

API_URL = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
MODEL_NAME = "intern-s2-preview-397b"
#: Which API dialect the endpoint speaks.
#:   "internai" — chat.intern-ai.org.cn: reports errors as HTTP 200 with a
#:                {"code": ..., "msg": ...} envelope, accepts ``thinking_mode``.
#:   "openai"   — standard OpenAI-compatible gateway (e.g. llmapi.blsc.cn):
#:                errors use real HTTP status codes.
API_PROVIDER = "internai"
MODEL_EXTRA_BODY = {}
#: "strict" — the reply must be one action (v0 behaviour).
#: "last"   — reasoning models narrate first and put the action at the end;
#:            take the LAST action mentioned instead of calling it ambiguous.
ACTION_EXTRACTION = "strict"

MAX_HISTORY_STEPS = 5
LOOKBACK_IMAGE_COUNT = 5
TEMPERATURE = 0.0
MAX_WORKERS = 8
MAX_OUTPUT_TOKENS = 30
CONNECT_TIMEOUT = 30
RETRY_INTERVAL = 1.0
API_TIMEOUT = 300
MAX_API_RETRIES = 999999  # actually controlled by the tried set (stops after all keys are tried)
IMAGE_MAX_SIZE = 448
JPEG_QUALITY = 80
SAVE_PROMPTS = False

MAX_PROMPT_IMAGES = 50

ENABLE_GRID_MEMORY = False
ENABLE_HISTORY = True
ENABLE_PLACEHOLDERS = False
ENABLE_EPISODIC_MEMORY = False
EPISODIC_MEMORY_HORIZON = 6
EPISODIC_MEMORY_MATCHER = "pose"   # "pose" (default) | "vit" (ViT-B/16 cosine sim over a precomputed .npz)
EPISODIC_MEMORY_CONTENT = "all"    # "all" (default) | "actions" | "images"

# Per-call tiebreaker context for the episodic-memory matcher. Refreshed
# just before each call_internvl() invocation by run_case() so that
# absolute-pose ties can be broken in favour of the reference visit
# whose stage progress matches the agent's actual stage progress.
AGENT_STAGES_COMPLETED = 0
AGENT_STEPS_IN_CURRENT_STAGE = 0

# --- experiment mode (v0-v3 -> v4 study) ----------------------------------
# All of these stay None/empty in legacy mode, and every experiment-mode
# code path is gated on EXPERIMENT being set. Read-only after main() has
# wired them up, so they are safe to share across the case threads; all
# per-case mutable state lives in run_case()'s locals instead.
EXPERIMENT = None            # experiment.ExperimentConfig
EXPERIMENT_GROUP = None      # experiment.GroupConfig
EXPERIMENT_MEMORY_TEXT = ""  # pre-rendered Summary/Semantic block (all versions)
EXPERIMENT_TEXT_BY_KIND = {}  # immutable after preload, shared across workers
EXPERIMENT_HISTORY = {}      # legacy_case_id -> [prompt-shaped trajectory dicts]
EXPERIMENT_MANIFEST = {}     # provenance written into the run manifest
STEP_WALL_CLOCK = 900        # per-decision wall-clock ceiling, seconds
MAX_API_ATTEMPTS = 10        # distinct API keys tried per decision
SERVER_RETRIES = 3           # /init_task and /step transport retries

#: Per-key cooldown after a rate-limit response, in seconds. Gateways
#: differ: llmapi.blsc.cn has shown no per-key limit, while
#: chat.intern-ai.org.cn answers HTTP 400 with code -20048
#: ("requests too frequent") once a key is pushed hard. Backing that one
#: key off — instead of lowering concurrency for everybody — lets each
#: independent account run at its own ceiling.
RATE_LIMIT_COOLDOWN = 2.0
RATE_LIMIT_COOLDOWN_MAX = 30.0
#: key index -> (cooldown_until, consecutive_hits). Mutated from every
#: case thread, so every access goes through _COOLDOWN_LOCK.
_KEY_COOLDOWN = {}
_COOLDOWN_LOCK = Lock()


def _is_rate_limited(status_code, text):
    """Does this response mean 'slow down' rather than 'this key is bad'?"""
    if status_code == 429:
        return True
    if status_code == 400 and text:
        # intern-ai reports throttling as a 400 with its own error code.
        if "-20048" in text or "请求过于频繁" in text:
            return True
        low = text.lower()
        if "rate limit" in low or "too many requests" in low or "too frequent" in low:
            return True
    return False


def _cool_down(idx):
    """Back one key off with exponential growth, capped."""
    with _COOLDOWN_LOCK:
        until, hits = _KEY_COOLDOWN.get(idx, (0.0, 0))
        hits += 1
        wait = min(RATE_LIMIT_COOLDOWN * (2 ** (hits - 1)), RATE_LIMIT_COOLDOWN_MAX)
        _KEY_COOLDOWN[idx] = (time.time() + wait, hits)
        return wait


def _clear_cooldown(idx):
    with _COOLDOWN_LOCK:
        if idx in _KEY_COOLDOWN:
            _KEY_COOLDOWN.pop(idx, None)


def _available_keys(candidates):
    """Split candidate key indices into (ready, earliest_ready_time)."""
    now = time.time()
    with _COOLDOWN_LOCK:
        ready = [i for i in candidates if _KEY_COOLDOWN.get(i, (0.0, 0))[0] <= now]
        if ready:
            return ready, None
        waits = [_KEY_COOLDOWN[i][0] for i in candidates if i in _KEY_COOLDOWN]
    return [], (min(waits) if waits else None)
MAX_CONSECUTIVE_TECHNICAL = 3  # give up the attempt after this many in a row
#: How many times one decision may be re-issued with a larger decoding
#: budget after the reply was cut off mid-reasoning, and how far the budget
#: may grow.
MAX_TRUNCATION_RETRIES = 3
MAX_OUTPUT_TOKENS_CEILING = 32768
#: Truncation is stochastic — the same prompt often answers in a few tokens
#: and occasionally runs long. Retry the decision rather than discarding a
#: case that may be 150 steps in; only a persistent run of truncations
#: means the decoding budget is genuinely too small.
MAX_CONSECUTIVE_TRUNCATED = 5
#: One unparseable reply is not a verdict on the case either.
MAX_CONSECUTIVE_MODEL_FAIL = 3

REFERENCE_TRAJ_MODE = "none"
REFERENCE_TRAJ_IMAGES_MODE = "none"
REFERENCE_TRAJ_INTERVAL = 3
REFERENCE_TRAJ_MAX_SMART_IMAGES = 50
REFERENCE_TRAJ_SMART_LAYERS = {"start", "stage", "end", "turn"}
REFERENCE_TRAJ_SMART_SAMPLE_COUNT = None
TRAJECTORY_JSON_PATH = DEFAULT_TRAJECTORY_JSON
TRAJECTORY_INDEX = {}

PEER_REFERENCE = False

# Memory-recall prompt text (loaded once in main() from
# --summary_prompt_path / --semantic_prompt_path .md files). Read by
# build_user_content() / call_internvl() / run_case(). None = feature off.
SUMMARY_PROMPT_TEXT = None
SEMANTIC_PROMPT_TEXT = None

VALID_ACTIONS = [
    "Move_Forward 1m", "Move_Backward 1m", "Move_Left 1m", "Move_Right 1m",
    "Turn_Left", "Turn_Right", "Turn_180"
]

DIRS_CCW = ["East", "North", "West", "South"]
DELTAS = {"East": (1, 0), "North": (0, 1), "West": (-1, 0), "South": (0, -1)}
DIR_DELTAS_HINT = ", ".join(f"{d}→{DELTAS[d]}" for d in DIRS_CCW)
TURN_HINT = "Turn_Left: +1 in DIRS_CCW (E→N→W→S→E). Turn_Right: -1. Turn_180: +2."

CASE_START = 0
RUN_NAME = "agent_run"
API_KEYS = []


class Endpoint(object):
    """One API key bound to the gateway that accepts it.

    The same underlying model can be reachable through several gateways
    under different ids (Intern-S2-Preview-35B is ``Intern-S2-Preview`` on
    llmapi.blsc.cn and ``intern-s2-preview-35b`` on chat.intern-ai.org.cn),
    and a key issued by one gateway is rejected by the other. Binding the
    key to its own url/model/provider lets one run pool keys from several
    accounts while every request still hits the same model.
    """

    __slots__ = ("key", "api_url", "model", "provider", "label")

    def __init__(self, key, api_url, model, provider, label=None):
        self.key = key
        self.api_url = api_url
        self.model = model
        self.provider = provider
        self.label = label or api_url.split("/")[2]

    def as_dict(self, redact=True):
        return {"api_url": self.api_url, "model": self.model,
                "provider": self.provider, "label": self.label,
                "key": ("...%s" % self.key[-6:]) if redact else self.key}


def load_api_keys_from_list(keys, default_url=None, default_model=None,
                            default_provider=None):
    """Build the endpoint pool from ``execution.api_keys``.

    Entries may be a bare ``"sk-..."`` string (inherits the config's
    ``model`` block) or an object carrying its own ``api_url`` / ``model``
    / ``provider``.
    """
    out = []
    for entry in keys or []:
        if isinstance(entry, dict):
            k = str(entry.get("key", "")).strip().rstrip(",").strip()
            url = entry.get("api_url") or default_url
            model = entry.get("model") or default_model
            prov = entry.get("provider") or default_provider or "internai"
            label = entry.get("label")
        else:
            k = str(entry).strip().rstrip(",").strip()
            url, model, prov, label = default_url, default_model, default_provider, None
        if not k.startswith("sk-"):
            continue
        if not url or not model:
            raise SystemExit("api_keys entry %r has no api_url/model and the "
                             "config's model block does not supply one" % (k[-6:],))
        out.append(Endpoint(k, url, model, prov, label))
    if not out:
        raise SystemExit("execution.api_keys contained no sk- keys")
    random.shuffle(out)
    by_gw = {}
    for e in out:
        by_gw.setdefault("%s [%s]" % (e.label, e.model), 0)
        by_gw["%s [%s]" % (e.label, e.model)] += 1
    print("[keys] loaded %d key(s) from the experiment config: %s"
          % (len(out), ", ".join("%s x%d" % (k, v) for k, v in sorted(by_gw.items()))))
    return out


def load_api_keys(excel_path):
    try:
        df = pd.read_excel(excel_path)
        if df.shape[1] < 1:
            raise ValueError("Excel has no columns.")
        col = df.iloc[:, 0].dropna().astype(str)
        keys = []
        for v in col:
            v = v.strip().rstrip(",").strip()
            if v.startswith("sk-"):
                keys.append(v)
        if not keys:
            raise ValueError("No sk- keys in column A.")
        random.shuffle(keys)
        print(f"🔑 Loaded {len(keys)} API keys from {excel_path} (col A, shuffled).")
        return keys
    except Exception as e:
        print(f"⚠️ Failed to load API keys from {excel_path}: {e}")
        raise RuntimeError("Failed to load API keys from %s" % excel_path) from e


def encode_image_to_base64(image_path, max_size=IMAGE_MAX_SIZE, quality=JPEG_QUALITY):
    from prompt import encode_image_to_base64 as _enc
    return _enc(image_path, max_size=max_size, quality=quality)


_BASE_MOVE_ACTIONS = {"Move_Forward", "Move_Backward", "Move_Left", "Move_Right"}
_AGENT_DELTAS = {"East": (1, 0), "North": (0, 1), "West": (-1, 0), "South": (0, -1)}
_AGENT_LEFT_OF = {"East": "North", "North": "West", "West": "South", "South": "East"}
_AGENT_RIGHT_OF = {"East": "South", "South": "West", "West": "North", "North": "East"}


def _base_action(action):
    """Strip the optional ' 1m' suffix and return the base verb."""
    if not isinstance(action, str):
        return None
    a = action.strip()
    for v in ("Move_Forward", "Move_Backward", "Move_Left", "Move_Right",
              "Turn_Left", "Turn_Right", "Turn_180"):
        if a == v or a == f"{v} 1m":
            return v
    return None


def _attempted_position(pose, action):
    """Return the (x, y) the agent tried to move into, or None for turns."""
    base = _base_action(action)
    if base is None or base not in _BASE_MOVE_ACTIONS:
        return None
    x, y, d = pose["x"], pose["y"], pose["heading"]
    if base == "Move_Forward":
        dx, dy = _AGENT_DELTAS[d]
    elif base == "Move_Backward":
        dx, dy = (-_AGENT_DELTAS[d][0], -_AGENT_DELTAS[d][1])
    elif base == "Move_Left":
        dx, dy = _AGENT_DELTAS[_AGENT_LEFT_OF[d]]
    else:
        dx, dy = _AGENT_DELTAS[_AGENT_RIGHT_OF[d]]
    return (x + dx, y + dy)


def _serialize_payload_for_save(payload):
    """Return a JSON-serializable copy of the request payload with base64
    image data replaced by the source image's path. Text blocks pass
    through unchanged.

    The agent's image blocks all carry an ``_image_path`` tag (see
    ``prompt._helpers.image_block``), so we can rebuild the persisted
    form without needing the original image list.

    Handles both the system-message case (``content`` is a plain string)
    and the user-message case (``content`` is a list of text/image blocks).
    """
    # Every non-message field as sent -- including whatever model.extra_body
    # merged in (thinking_mode), so a saved request says what was really sent.
    out = {k: v for k, v in payload.items() if k != "messages"}
    out.setdefault("max_tokens", 30)
    out["messages"] = []
    for msg in payload["messages"]:
        if isinstance(msg["content"], str):
            # System message: keep the string as-is.
            new_content = msg["content"]
        else:
            new_content = []
            for blk in msg["content"]:
                if isinstance(blk, dict) and blk.get("type") == "image_url":
                    path = blk.get("_image_path", "")
                    if path:
                        new_content.append({
                            "type": "image_url",
                            "image_path": path,
                            "note": "base64 omitted on save",
                        })
                    else:
                        new_content.append({
                            "type": "image_url",
                            "note": "image source path unknown (base64 omitted)",
                        })
                else:
                    new_content.append(blk)
        out["messages"].append({"role": msg["role"], "content": new_content})
    return out


def _save_step_prompt(case_id, agent_name, step_idx, payload, action, raw_res,
                      server_res, run_name, results_dir):
    """Append one step's full prompt + response to the per-case prompt log."""
    if not SAVE_PROMPTS:
        return
    # Must live under the same results_dir that holds client_<run_name>/,
    # otherwise prompts would leak to result/ root even when test_run.py
    # nests the run under a test_run_<ts>/ folder.
    prompts_dir = os.path.join(results_dir, f"client_{run_name}", "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    path = os.path.join(prompts_dir, f"case_{case_id:03d}_{agent_name}.json")
    record = {
        "case_id": case_id,
        "agent_name": agent_name,
        "step": step_idx,
        "request": _serialize_payload_for_save(payload),
        "action_sent": action,
        "llm_response": raw_res,
        "server_response": server_res,
    }
    # Append-or-create: keep one file per (case, agent) as a JSON array.
    history = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not isinstance(history, list):
                history = []
        except Exception:
            history = []
    history.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def _format_step_log(step_idx, action, status, pose_before, pose_after, server_msg):
    """Return (log_entry, display) for the history prompt.

    The entry spells out the feedback (success / blocked / out_of_range)
    together with the resulting pose and, for failures, the cell that
    was attempted so the agent can learn from the wall it hit.
    """
    p_before = f"({pose_before['x']}, {pose_before['y']}) {pose_before['heading']}"
    p_after = f"({pose_after['x']}, {pose_after['y']}) {pose_after['heading']}"
    if status == "success":
        kind = "heading changed" if _base_action(action) and _base_action(action).startswith("Turn") \
            else "moved 1m"
        entry = f"Step {step_idx}: {action} → success at {p_after} ({kind})"
        return entry, "moved" if kind == "moved 1m" else "turned"
    if status == "blocked":
        attempted = _attempted_position(pose_before, action)
        tgt = f"({attempted[0]}, {attempted[1]})" if attempted else "target"
        suffix = f" — server: {server_msg}" if server_msg else ""
        entry = f"Step {step_idx}: {action} → blocked at {p_before} ({tgt} is shelf, no move){suffix}"
        return entry, "blocked"
    if status == "out_of_range":
        attempted = _attempted_position(pose_before, action)
        tgt = f"({attempted[0]}, {attempted[1]})" if attempted else "target"
        suffix = f" — server: {server_msg}" if server_msg else ""
        entry = f"Step {step_idx}: {action} → out_of_range at {p_before} ({tgt} outside scene bounds, no move){suffix}"
        return entry, "out_of_range"
    suffix = f" ({server_msg})" if server_msg else ""
    entry = f"Step {step_idx}: {action} → {status}{suffix}"
    return entry, status


def load_trajectories(json_path):
    """Load the trajectory JSON once; return {case_id: {forward_only: {...}, all_actions: {...}}}."""
    if not json_path or not os.path.exists(json_path):
        print(f"⚠️ Trajectory JSON not found: {json_path}")
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {t["task_id"]: t for t in data.get("trajectories", [])}


def get_trajectory_for_case(case_id, traj_kind):
    """Return the trajectory dict for a given case + kind ('forward_only' / 'all_actions')."""
    if traj_kind not in ("forward_only", "all_actions"):
        return None
    rec = TRAJECTORY_INDEX.get(case_id)
    if not rec:
        return None
    t = rec.get(traj_kind)
    if not t or not t.get("reachable"):
        return None
    image_dir = os.path.join(PROJECT_ROOT, "dataset", "warehouse_01_01", "image")
    return {
        "source": traj_kind,
        "actions": t["actions"],
        "image_paths": [os.path.join(image_dir, p) for p in t.get("image_paths", [])],
        "poses": t.get("poses") or [],
        "stage_events": t.get("stage_events") or [],
        "length": t["length"],
    }


def get_peer_trajectory(case_id, traj_kind):
    """Return ONE trajectory from another case of the SAME task_type.

    Parameters
    ----------
    case_id : int
        The current case ID (will be excluded from results).
    traj_kind : str
        'forward_only' or 'all_actions'.

    Returns
    -------
    dict or None
        A trajectory dict with keys: source_case_id, source, actions,
        image_paths, poses, stage_events, length, description, origin.
        Returns None if no valid peer exists.
    """
    if traj_kind not in ("forward_only", "all_actions"):
        return None
    current_rec = TRAJECTORY_INDEX.get(case_id)
    if not current_rec:
        return None
    current_task_type = current_rec.get("task_type")
    if current_task_type is None:
        return None

    image_dir = os.path.join(PROJECT_ROOT, "dataset", "warehouse_01_01", "image")
    peers = []
    for other_id, other_rec in TRAJECTORY_INDEX.items():
        if other_id == case_id:
            continue
        if other_rec.get("task_type") != current_task_type:
            continue
        t = other_rec.get(traj_kind)
        if not t or not t.get("reachable"):
            continue
        peers.append({
            "source_case_id": other_id,
            "source": traj_kind,
            "actions": t["actions"],
            "image_paths": [os.path.join(image_dir, p) for p in t.get("image_paths", [])],
            "poses": t.get("poses") or [],
            "stage_events": t.get("stage_events") or [],
            "length": t["length"],
            "description": other_rec.get("description", ""),
            "origin": other_rec.get("origin", {}),
        })

    if not peers:
        return None
    return random.choice(peers)


def _strip_private_blocks(payload):
    """Drop the ``_``-prefixed bookkeeping keys before the request goes out.

    ``image_block()`` tags every image with ``_image_path`` so the log can
    record which frame was sent without keeping the base64. That tag has
    no business being in the HTTP body.
    """
    out = {k: v for k, v in payload.items() if k != "messages"}
    msgs = []
    for msg in payload["messages"]:
        content = msg["content"]
        if isinstance(content, str):
            msgs.append({"role": msg["role"], "content": content})
            continue
        clean = []
        for blk in content:
            if isinstance(blk, dict):
                clean.append({k: v for k, v in blk.items() if not k.startswith("_")})
            else:
                clean.append(blk)
        msgs.append({"role": msg["role"], "content": clean})
    out["messages"] = msgs
    return out


def _count_images(blocks):
    return sum(1 for b in blocks if isinstance(b, dict) and b.get("type") == "image_url")


class ExperimentPromptError(RuntimeError):
    """Image budget / assembly invariant broken. Always a code defect."""


def _same_origin_wording_on():
    """``prompt.same_origin_wording`` of the running config (off outside one)."""
    return bool(EXPERIMENT is not None
                and EXPERIMENT.prompt.get("same_origin_wording"))


def render_experiment_text_memory(cfg, group, memories):
    """Summary / Semantic text exactly as the agent injects it.

    Memory built from the version under test (V0 memory on the V0 test) was
    collected from the very origins being tested, so with
    ``prompt.same_origin_wording`` on it gets the same-origin note. The
    agent, the preflight, export_prompts and token_probe all render through
    here so none of them can disagree about which note a group gets.
    """
    same = (bool(cfg.prompt.get("same_origin_wording"))
            and all(m.version == cfg.test_version for m in memories))
    return wrap_text_memory(memories, memories[0].kind.capitalize() + " memory",
                            same_origin=same)


def prepare_experiment_text_memory(cfg, group):
    """One loading path for live requests, preflight and prompt exports."""
    global EXPERIMENT_MEMORY_TEXT, EXPERIMENT_TEXT_BY_KIND
    import experiment as ex
    rendered, sources = {}, []
    for kind in group.text_memory_kinds:
        memories = ex.load_text_memory_set(cfg, group.versions, kind)
        rendered[kind] = render_experiment_text_memory(cfg, group, memories)
        sources.extend(memories)
    EXPERIMENT_TEXT_BY_KIND = rendered
    EXPERIMENT_MEMORY_TEXT = rendered.get(group.memory, "")
    return sources


def experiment_memory_blocks(group, trajectories, current_img_path, ctx,
                             history_image_count, telemetry):
    """Build the long-term-memory sections for one decision.

    Returns ``(front_blocks, back_blocks)``: ``front_blocks`` go at the
    very front of the user message (Summary / Semantic, matching the v0
    placement), ``back_blocks`` go after the history section (Episodic /
    Full Context, also matching v0).

    Every group that carries memory gets the same common note and the
    same per-version BEGIN/END markers, so single- and multi-version
    groups differ only in how many segments appear.
    """
    front, back = [], []
    if group.memory == "none":
        return front, back

    for kind in group.text_memory_kinds:
        memory_text = (EXPERIMENT_TEXT_BY_KIND.get(kind, "")
                       if group.memory == "hybrid" else EXPERIMENT_MEMORY_TEXT)
        if not memory_text:
            raise ExperimentPromptError(
                "group %s needs %s memory but none was loaded"
                % (group.group_id, kind))
        tag = kind.upper()
        # Same outer markers the v0 evaluation used for these two
        # sections; the version segmentation lives inside them.
        front.append({
            "type": "text",
            "text": (f"[MEMORY RECALL — {tag} INSTRUCTIONS]\n"
                     f"{memory_text.strip()}\n"
                     f"[END {tag} INSTRUCTIONS]\n"),
            "_memory_ref": {"kind": kind,
                            "versions": list(group.versions),
                            "chars": len(memory_text)},
        })
        telemetry["memory_versions"] = list(group.versions)
    if group.memory == "hybrid":
        telemetry["hybrid_components"] = ["summary", "semantic"]
        telemetry["text_memory_chars"] = {
            kind: len(EXPERIMENT_TEXT_BY_KIND[kind]) for kind in group.text_memory_kinds}
    if group.uses_text_memory and not group.uses_history_trajectories:
        return front, back

    if not trajectories:
        raise ExperimentPromptError(
            "group %s needs historical trajectories but none were loaded"
            % group.group_id)

    task_id, case_no = ctx.case_label
    # Which segments started exactly at this run's spawn. Consulted only when
    # the config opts into same-origin wording; otherwise every segment gets
    # the neutral note. The common note switches only when all of them do.
    wording = _same_origin_wording_on()
    same_origin = [wording and origin_matches(t.get("origin"), ctx.initial_origin)
                   for t in trajectories]
    if wording:
        telemetry["same_origin_versions"] = [
            t["version"] for t, s in zip(trajectories, same_origin) if s]
    back.append(common_note_block(same_origin=all(same_origin)))
    back.append(case_scope_declaration(task_id, case_no,
                                       [t["version"] for t in trajectories],
                                       test_version=EXPERIMENT.test_version))

    if group.memory == "episodic":
        sink = []
        blocks = episodic_memory_multi_blocks(
            current_img_path, trajectories,
            max_actions=group.episodic_horizon or 6,
            ctx=ctx, content=group.episodic_content or "all",
            match_sink=sink,
            matcher=group.episodic_matcher or "pose",
            same_origin_wording=wording,
        )
        if not blocks or not sink:
            # No usable pose anywhere in the candidate pool. Drop the
            # scope preamble too rather than emit a dangling header.
            telemetry["episodic_match"] = None
            return front, []
        match = sink[0]
        chosen = trajectories[match.traj_pos]
        back.append(version_begin(
            match.version, "retrieved episodic excerpt",
            "one contiguous excerpt from this case's record in version %s"
            % match.version.upper(),
            case_label=chosen["case_label"],
            task_description=chosen["description"],
            origin=chosen["origin"],
            same_origin=same_origin[match.traj_pos],
        ))
        back.extend(blocks)
        back.append(version_end(match.version))
        telemetry["episodic_match"] = match.as_dict()
        telemetry["episodic_candidates"] = len(trajectories)
        return front, back

    if group.memory == "full_context":
        # Reference-image budget: never exceed the group's own cap, and
        # never exceed what is left of the global per-request image cap.
        # Computed up front so no post-hoc trimming is ever needed.
        remaining = min(group.max_reference_images,
                        MAX_PROMPT_IMAGES - 1 - history_image_count)
        per_traj_cap = group.max_smart_images_per_traj
        selections = []
        for traj, same in zip(trajectories, same_origin):
            back.append(version_begin(
                traj["version"], "reference trajectory (full expert action sequence)",
                "1 historical trajectory of this same case",
                case_label=traj["case_label"],
                task_description=traj["description"],
                origin=traj["origin"],
                same_origin=same,
            ))
            budget = max(0, min(per_traj_cap, remaining))
            sink = []
            img_blocks = reference_traj_images_blocks(
                traj, image_mode="smart", max_smart_images=budget,
                smart_layers=REFERENCE_TRAJ_SMART_LAYERS,
                smart_sample_count=REFERENCE_TRAJ_SMART_SAMPLE_COUNT,
                selection_sink=sink,
            )
            n_imgs = _count_images(img_blocks)
            if n_imgs > budget:
                raise ExperimentPromptError(
                    "smart selection returned %d images for a budget of %d "
                    "(%s %s)" % (n_imgs, budget, group.group_id, traj["version"]))
            back.append(reference_traj_block(traj, has_images=bool(img_blocks),
                                             historical=True))
            back.extend(img_blocks)
            back.append(version_end(traj["version"]))
            remaining -= n_imgs
            selections.append({"version": traj["version"],
                               "length": traj["length"],
                               "images_available": traj["length"] + 1,
                               "images_budget": budget,
                               "images_sent": n_imgs,
                               "indices": [i for i, _ in sink]})
        telemetry["full_context"] = selections
        telemetry["reference_images"] = sum(s["images_sent"] for s in selections)
        return front, back

    raise ExperimentPromptError("unknown memory type %r" % group.memory)


def build_experiment_user_content(mission, initial_origin, current_pose,
                                  history_text, current_img_path, history_images,
                                  trajectories, ctx, telemetry):
    """Assemble the user message for an experiment group.

    Section order is exactly the v0 order — memory-recall text at the
    front, then mission, current view, history, episodic, reference
    trajectory, closing reminder — so that stripping the memory sections
    from any two groups leaves byte-identical content.

    The one deliberate difference from v0: ``mission_block`` now receives
    the run's fixed initial spawn instead of the current pose. v0 passed
    the live pose into a slot labelled "Your initial spawn", which
    contradicts both the system prompt's return-to-origin rule and the
    T07/T08/T10 task definitions.
    """
    group = EXPERIMENT_GROUP
    parts = []

    # History first, because the reference-image budget depends on how
    # many history images actually made it in.
    hist_blocks = []
    if ENABLE_HISTORY and MAX_HISTORY_STEPS > 0:
        hist_blocks = history_with_images_blocks(
            history_text, history_images,
            max_history_steps=MAX_HISTORY_STEPS,
            max_images=LOOKBACK_IMAGE_COUNT,
        )
    n_history_imgs = _count_images(hist_blocks)

    front, back = experiment_memory_blocks(
        group, trajectories, current_img_path, ctx, n_history_imgs, telemetry)
    # Tag every memory block so the preflight can strip exactly the
    # memory sections and prove the remaining request body is identical
    # across all nine groups. The tag never reaches the wire.
    for blk in list(front) + list(back):
        if isinstance(blk, dict):
            blk["_section"] = "memory"

    parts.extend(front)                                                    # §0
    parts.append(mission_block(
        mission, initial_origin,
        obstacle_notice=EXPERIMENT.prompt.get("obstacle_notice", False)))  # §1
    parts.extend(current_image_blocks(current_img_path, pose=current_pose))  # §2
    parts.extend(hist_blocks)                                              # §4
    parts.extend(back)                                                     # §5/§6
    parts.append({"type": "text",
                  "text": f"\n[REMINDER] Choose ONLY from: {', '.join(VALID_ACTIONS)}. No markdown, no prose."})

    total_imgs = _count_images(parts)
    telemetry["images"] = {"current": 1, "history": n_history_imgs,
                           "memory": total_imgs - 1 - n_history_imgs,
                           "total": total_imgs}
    if total_imgs > group.max_total_images:
        raise ExperimentPromptError(
            "%s built a request with %d images, above its cap of %d"
            % (group.group_id, total_imgs, group.max_total_images))
    if total_imgs > MAX_PROMPT_IMAGES:
        raise ExperimentPromptError(
            "%s built a request with %d images, above the global cap of %d"
            % (group.group_id, total_imgs, MAX_PROMPT_IMAGES))
    return parts


def build_user_content(mission, origin, history_text, current_img_path, history_images,
                        grid_state, reference_traj, peer_trajectory=None,
                        episodic_only_trajectory=None):
    """Assemble the user message content from the prompt-package blocks.

    Always-on sections (cannot be turned off):
      1. mission        — task description + initial spawn
      2. current image  — current first-person view (always exactly 1 image)
      7. closing line   — reminder of the exact action vocabulary

    Optional sections (gated by the corresponding ENABLE_* global, CLI
    flags ``--no_<section>`` turn them off):
      0a. semantic mem  — optional semantic-memory instructions
                           (injected at the FRONT of the user message,
                           before §0b / §1)
      0b. summary mem   — optional summary-memory instructions
                           (injected at the FRONT of the user message,
                           AFTER §0a / BEFORE §1)
      3. grid memory    — 2-D ASCII map of visited cells + walls
      4. history        — per-step action + feedback, with each step's
                           view-at-start interleaved (newer steps get the
                           images). ``MAX_HISTORY_STEPS`` controls the
                           number of text lines, and ``LOOKBACK_IMAGE_COUNT``
                           controls the number of images; the two counts
                           are independent.
      5. episodic mem   — ALWAYS anchored on the case's own
                           ``forward_only`` trajectory, independent of §6:
                           closest matched view in that trajectory + the
                           next ``--episodic_memory_horizon`` expert
                           actions with their resulting views. Rendered
                           AFTER history so the LLM sees its own past
                           steps first, then a reference-grounded recall
                           slice. Works standalone (no §6 trajectory
                           needed) and also coexists with §6 (which keeps
                           its own source).
      6. reference traj — expert trajectory (text + optional images)
                           OR peer trajectory (mutually exclusive)
      7. placeholders   — hook for future memory mechanisms

    Image budget: total images across sections 2 + 2b + 4 + 5 cannot
    exceed ``MAX_PROMPT_IMAGES`` (default 50). The smart selector for
    the reference trajectory is given the remaining budget so it can
    pick "as many as possible" up to the cap.
    """
    parts = []
    # §0a / §0b — optional memory-recall instructions. Injected at the
    # very front so the model sees them before the mission block. Order
    # is fixed: semantic first, then summary. Text is loaded once in
    # main() from --semantic_prompt_path / --summary_prompt_path into
    # the module globals SEMANTIC_PROMPT_TEXT / SUMMARY_PROMPT_TEXT.
    if SEMANTIC_PROMPT_TEXT:
        parts.append({"type": "text",
                      "text": f"[MEMORY RECALL — SEMANTIC INSTRUCTIONS]\n{SEMANTIC_PROMPT_TEXT.strip()}\n[END SEMANTIC INSTRUCTIONS]\n"})
    if SUMMARY_PROMPT_TEXT:
        parts.append({"type": "text",
                      "text": f"[MEMORY RECALL — SUMMARY INSTRUCTIONS]\n{SUMMARY_PROMPT_TEXT.strip()}\n[END SUMMARY INSTRUCTIONS]\n"})
    parts.append(mission_block(mission, origin))                              # §1
    parts.extend(current_image_blocks(current_img_path, pose=origin))         # §2

    # Resolve which trajectory to use for §6 only. The episodic-memory
    # slice (§5) is INDEPENDENT — it always anchors on
    # ``episodic_only_trajectory`` (the case's own forward_only traj,
    # loaded by run_case whenever --episodic_memory is set). Peer takes
    # precedence over the full reference traj in §6.
    if PEER_REFERENCE and peer_trajectory:
        active_traj = peer_trajectory
        is_peer = True
    elif REFERENCE_TRAJ_MODE != "none" and reference_traj:
        active_traj = reference_traj
        is_peer = False
    else:
        active_traj = None
        is_peer = False

    if ENABLE_GRID_MEMORY:
        parts.append(grid_memory_block(grid_state))                          # §3
    n_history_imgs = 0
    if ENABLE_HISTORY and MAX_HISTORY_STEPS > 0:
        hist_blocks = history_with_images_blocks(
            history_text, history_images,
            max_history_steps=MAX_HISTORY_STEPS,
            max_images=LOOKBACK_IMAGE_COUNT,
        )
        n_history_imgs = sum(1 for b in hist_blocks if b['type'] == 'image_url')
        parts.extend(hist_blocks)                                            # §4

    if ENABLE_EPISODIC_MEMORY and episodic_only_trajectory:                 # §5
        parts.extend(episodic_memory_blocks(
            current_img_path, episodic_only_trajectory,
            max_actions=EPISODIC_MEMORY_HORIZON,
            matcher=EPISODIC_MEMORY_MATCHER,
            agent_stages=AGENT_STAGES_COMPLETED,
            agent_steps_in_current_stage=AGENT_STEPS_IN_CURRENT_STAGE,
            content=EPISODIC_MEMORY_CONTENT,
        ))

    # §6: Reference trajectory (own OR peer, mutually exclusive).
    # Skipped when only --episodic_memory is enabled (no full traj wanted).
    if is_peer:
        # Peer reference mode: use trajectory from another case of same type
        has_images = REFERENCE_TRAJ_IMAGES_MODE != "none"
        parts.append(peer_reference_traj_block(active_traj, has_images=has_images))
    elif REFERENCE_TRAJ_MODE in ("forward_only", "all_actions"):
        # Own reference mode: use this case's trajectory
        has_images = REFERENCE_TRAJ_IMAGES_MODE != "none"
        parts.append(reference_traj_block(reference_traj, has_images=has_images))

    if REFERENCE_TRAJ_IMAGES_MODE != "none" and active_traj:
        # Budget: total cap (50) minus the 1 current image minus history.
        budget = max(0, MAX_PROMPT_IMAGES - 1 - n_history_imgs)
        if REFERENCE_TRAJ_IMAGES_MODE == "smart":
            effective_max = min(budget, REFERENCE_TRAJ_MAX_SMART_IMAGES)
        else:
            effective_max = budget
        rt_img_blocks = reference_traj_images_blocks(
            active_traj,
            image_mode=REFERENCE_TRAJ_IMAGES_MODE,
            interval=REFERENCE_TRAJ_INTERVAL,
            max_smart_images=effective_max,
            smart_layers=REFERENCE_TRAJ_SMART_LAYERS,
            smart_sample_count=REFERENCE_TRAJ_SMART_SAMPLE_COUNT,
        )
        # Post-cap: hard guarantee the total never exceeds the budget.
        rt_img_count = sum(1 for b in rt_img_blocks if b['type'] == 'image_url')
        if rt_img_count > budget:
            kept, kept_imgs = [], 0
            for b in rt_img_blocks:
                if b['type'] == 'image_url':
                    if kept_imgs >= budget:
                        continue
                    kept_imgs += 1
                kept.append(b)
            rt_img_blocks = kept
        parts.extend(rt_img_blocks)                                          # §6 images

    if ENABLE_PLACEHOLDERS:
        parts.append(placeholders_block())                                   # §7
    parts.append({"type": "text",
                  "text": f"\n[REMINDER] Choose ONLY from: {', '.join(VALID_ACTIONS)}. No markdown, no prose."})
    return parts


def _response_error(res_json):
    """Classify a HTTP-200 body from the chat endpoint.

    This service reports most failures as **HTTP 200 with an error
    envelope** (``{"code": -20035, "msg": "...", "data": {}}``) rather
    than a non-200 status. A caller that only checks ``status_code``
    treats a dead key as a successful call, so every 200 has to be
    inspected for ``choices`` before it counts.

    Returns ``None`` when the body is a usable completion, else a short
    description of the problem.
    """
    if not isinstance(res_json, dict):
        return "non-dict response"
    if "choices" in res_json:
        try:
            res_json["choices"][0]["message"]["content"]
        except Exception:
            return "malformed choices"
        return None
    code = res_json.get("code")
    msg = res_json.get("msg") or res_json.get("message") or ""
    if code is not None:
        return "api code %s: %s" % (code, str(msg)[:120])
    return "no choices in response: %s" % (str(res_json)[:120])


def probe_api_key(key, api_url, model, timeout=(15, 60)):
    """One-token probe. Returns (ok, detail).

    ``key`` may be an :class:`Endpoint`, in which case its own url/model
    are used and the positional arguments are ignored.
    """
    if isinstance(key, Endpoint):
        api_url, model, key = key.api_url, key.model, key.key
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 1, "temperature": 0})
    headers = {"Content-Type": "application/json", "Authorization": "Bearer %s" % key}
    try:
        r = requests.post(api_url, headers=headers, data=body, timeout=timeout)
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)
    if r.status_code != 200:
        return False, "http %d" % r.status_code
    try:
        err = _response_error(r.json())
    except Exception as e:
        return False, "unparseable body: %s" % e
    return (err is None), (err or "ok")


def validate_api_keys(keys, api_url, model, workers=16, cache_path=None,
                      source_sha=None, min_keys=1, force=False):
    """Return the subset of ``keys`` that can actually complete a request.

    Most keys in a large pool are dead (phone-binding required, quota
    exhausted, revoked). Without this filter, the per-step attempt cap
    burns its budget on dead keys and every decision fails technically
    even though the service is healthy.

    The result is cached against the key file's hash so the nine group
    processes do not each re-probe the whole pool.
    """
    from concurrent.futures import ThreadPoolExecutor

    endpoint_mode = bool(keys) and isinstance(keys[0], Endpoint)
    if cache_path and not force and os.path.exists(cache_path):
        try:
            with io.open(cache_path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("source_sha256") == source_sha \
                    and cached.get("model") == model \
                    and cached.get("valid_keys"):
                good = set(cached["valid_keys"])
                valid = [k for k in keys
                         if (k.key if endpoint_mode else k) in good]
                if len(valid) >= min_keys:
                    print("[keys] reusing %d validated keys from %s"
                          % (len(valid), cache_path))
                    return valid
        except Exception as e:
            print("[keys] ignoring unreadable cache %s: %s" % (cache_path, e))

    print("[keys] probing %d keys with %d workers..." % (len(keys), workers))
    t0 = time.time()
    reasons = {}
    valid = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, (ok, detail) in zip(keys, pool.map(
                lambda k: probe_api_key(k, api_url, model), keys)):
            if ok:
                valid.append(key)
            else:
                head = detail.split(":")[0]
                if endpoint_mode:
                    head = "%s %s" % (key.label, head)
                reasons[head] = reasons.get(head, 0) + 1
    print("[keys] %d/%d usable in %.1fs" % (len(valid), len(keys), time.time() - t0))
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:5]:
        print("[keys]   %5d rejected: %s" % (n, reason))
    if len(valid) < min_keys:
        raise SystemExit(
            "only %d usable API key(s), need at least %d. Check the key "
            "spreadsheet before starting a run." % (len(valid), min_keys))
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with io.open(cache_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"source_sha256": source_sha, "model": model,
                           "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "probed": len(keys),
                           "endpoints": ([e.as_dict() for e in valid]
                                         if endpoint_mode else None),
                           "valid_keys": [(k.key if endpoint_mode else k)
                                          for k in valid]}, f, indent=2)
            print("[keys] cached %d validated keys -> %s" % (len(valid), cache_path))
        except Exception as e:
            print("[keys] could not write cache: %s" % e)
    return valid


def parse_model_action(text):
    """Map a raw model response to one canonical action.

    Returns ``(action, note)`` on success and ``(None, reason)`` on
    failure. There is deliberately no fallback action: the v0 code
    silently substituted ``Turn_Left`` for anything it could not parse,
    which turned a model-side failure into a real environment step and
    contaminated both the step count and the trajectory.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "empty response"
    cleaned = text.strip()
    for ch in ("`", '"', "'"):
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.strip()
    if cleaned in VALID_ACTIONS:
        return cleaned, "exact"
    for canon in VALID_ACTIONS:
        base = canon[:-3] if canon.endswith(" 1m") else canon
        if cleaned == base:
            return canon, "base verb without unit"
    low = cleaned.lower()
    # Every (position, action) the text mentions, in order of appearance.
    hits = []
    for canon in VALID_ACTIONS:
        base = canon[:-3] if canon.endswith(" 1m") else canon
        start = 0
        while True:
            i = low.find(base.lower(), start)
            if i < 0:
                break
            hits.append((i, canon))
            start = i + 1
    if not hits:
        return None, "no valid action in response"
    hits.sort()
    distinct = {c for _, c in hits}
    if len(distinct) == 1:
        return hits[0][1], "extracted from prose"
    if ACTION_EXTRACTION == "last":
        # Reasoning models narrate their options ("I could Turn_Left, but
        # Move_Forward is better") and state the decision last. Taking the
        # final mention is the only reading that recovers the answer; the
        # alternative is discarding every reply such a model produces.
        return hits[-1][1], ("last of %d actions mentioned (%s)"
                             % (len(hits), sorted(distinct)))
    return None, "ambiguous, %d actions present: %s" % (len(distinct), sorted(distinct))


def call_internvl(case_id, mission, origin, history_text, current_img_path,
                  history_images, grid_state, reference_traj, agent_name,
                  peer_trajectory=None, episodic_only_trajectory=None,
                  initial_origin=None, trajectories=None, ctx=None,
                  telemetry=None, max_tokens_override=None):
    """One decision: build the request, call the model, parse one action.

    Returns ``(action, clean_log, raw_res, key_idx, payload, token_info,
    outcome)`` where ``outcome`` is one of:

      ``"ok"``                    - a valid action was parsed
      ``"model_output_failure"``  - HTTP 200 but no single valid action
      ``"technical"``             - transport/service failure, no usable reply

    ``action`` is ``None`` for both failure outcomes; the caller decides
    whether to retry the decision or settle the case.
    """
    if telemetry is None:
        telemetry = {}
    if EXPERIMENT is not None:
        user_content = build_experiment_user_content(
            mission, initial_origin if initial_origin is not None else origin,
            origin, history_text, current_img_path, history_images,
            trajectories, ctx, telemetry,
        )
    else:
        user_content = build_user_content(mission, origin, history_text, current_img_path,
                                          history_images, grid_state, reference_traj,
                                          peer_trajectory, episodic_only_trajectory)
    def _payload_for(ep):
        """Request body for one gateway.

        The same model is reachable under different ids per gateway, so the
        id comes from the endpoint rather than a global. ``thinking_mode``
        is deliberately NOT sent: it suppresses the model's reasoning on
        intern-ai but has no counterpart on the other gateway, so sending
        it would make the two serve different behaviour for the same model.
        """
        body = {
            "model": ep.model if ep is not None else MODEL_NAME,
            "messages": [
                {"role": "system", "content": build_system_prompt()},
                {"role": "user", "content": user_content}
            ],
            "temperature": TEMPERATURE,
            "max_tokens": max_tokens_override or MAX_OUTPUT_TOKENS,
        }
        if MODEL_EXTRA_BODY:
            body.update(MODEL_EXTRA_BODY)
        return body

    payload = _payload_for(API_KEYS[0] if API_KEYS and isinstance(API_KEYS[0], Endpoint) else None)
    clean_log = copy.deepcopy(payload)
    last_img_block = None
    for blk in reversed(clean_log["messages"][1]["content"]):
        if blk.get("type") == "image_url":
            last_img_block = blk
            break
    if last_img_block and current_img_path:
        last_img_block["image_url"]["url"] = f"file:///{current_img_path}"
    # The payload handed back keeps its private tags (``_image_path``,
    # ``_memory_ref``): the step log reads the image paths from them, and
    # whoever persists a prompt serializes it at write time. Handing back an
    # already-serialized copy used to leave every step log with
    # ``image_paths: [""]`` whenever --save_prompts was on.

    # The private bookkeeping tags (_image_path, _memory_ref) exist for
    # the log, not for the endpoint. Bodies are cached per gateway so the
    # (large) content is serialised once per distinct model id.
    _body_cache = {}

    def _body_for(ep):
        mid = ep.model if ep is not None else MODEL_NAME
        if mid not in _body_cache:
            _body_cache[mid] = json.dumps(_strip_private_blocks(_payload_for(ep)))
        return _body_cache[mid]

    tried = set()
    attempts = []
    last_err = ""
    total = len(API_KEYS)
    max_attempts = min(MAX_API_ATTEMPTS, total) if total else 0
    deadline = time.time() + STEP_WALL_CLOCK
    timeouts = (CONNECT_TIMEOUT, API_TIMEOUT)

    while len(tried) < max_attempts:
        if time.time() >= deadline:
            last_err = last_err or "step wall-clock exceeded"
            attempts.append({"key_index": -1, "status": "wall_clock_exceeded",
                             "elapsed": 0.0})
            break
        pool = [i for i in range(total) if i not in tried]
        if not pool:
            break
        ready, earliest = _available_keys(pool)
        if not ready:
            # Every untried key is cooling off. Wait for the first one
            # rather than burning an attempt on a throttled key.
            nap = max(0.05, min((earliest or 0) - time.time(), 5.0))
            if time.time() + nap >= deadline:
                last_err = last_err or "all keys rate-limited until the step deadline"
                break
            time.sleep(nap)
            continue
        idx = random.choice(ready)
        tried.add(idx)
        entry = API_KEYS[idx]
        if isinstance(entry, Endpoint):
            ep, key, url, provider = entry, entry.key, entry.api_url, entry.provider
        else:
            ep, key, url, provider = None, entry, API_URL, API_PROVIDER
        body = _body_for(ep)
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        t0 = time.time()
        try:
            r = requests.post(url, headers=headers, data=body, timeout=timeouts)
            elapsed = time.time() - t0
            if r.status_code == 200:
                try:
                    res_json = r.json()
                except Exception as e:
                    res_json, envelope_err = None, "unparseable body: %s" % e
                else:
                    envelope_err = _response_error(res_json)
                if envelope_err is not None:
                    # HTTP 200 carrying an error envelope (dead key, quota,
                    # phone-binding). A key problem, not a model problem.
                    last_err = envelope_err
                    attempts.append({"key_index": idx, "status": "api_error_200",
                                     "detail": envelope_err,
                                     "elapsed": round(elapsed, 3)})
                    time.sleep(RETRY_INTERVAL)
                    continue
                choice0 = res_json["choices"][0]
                content = choice0["message"]["content"]
                finish = choice0.get("finish_reason")
                usage = res_json.get("usage") or {}
                token_info = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "usage_reported": bool(usage),
                }
                _clear_cooldown(idx)
                attempts.append({"key_index": idx, "status": "http_200",
                                 "gateway": (ep.label if ep else None),
                                 "model": (ep.model if ep else MODEL_NAME),
                                 "finish_reason": finish,
                                 "completion_tokens": usage.get("completion_tokens"),
                                 "elapsed": round(elapsed, 3)})
                telemetry["api_attempts"] = attempts
                telemetry["finish_reason"] = finish
                action, note = parse_model_action(content)
                if action is None and finish == "length":
                    # The reply was cut off mid-reasoning, so it never
                    # reached the action. That is a decoding-budget problem,
                    # not the model declining to answer: retrying the same
                    # request with the same budget cannot help, and it must
                    # not be reported as a model-output failure.
                    note = ("truncated at max_tokens=%d before producing an "
                            "action (%s completion tokens)"
                            % (max_tokens_override or MAX_OUTPUT_TOKENS,
                               usage.get("completion_tokens")))
                    telemetry["parse_note"] = note
                    return (None, clean_log, res_json, idx,
                            payload, token_info,
                            "truncated")
                telemetry["parse_note"] = note
                if action is None:
                    return (None, clean_log, res_json, idx,
                            payload, token_info,
                            "model_output_failure")
                return (action, clean_log, res_json, idx,
                        payload, token_info, "ok")
            rate_limited = _is_rate_limited(r.status_code, r.text)
            if rate_limited:
                waited = _cool_down(idx)
                tried.discard(idx)      # throttling is not the key's fault
                last_err = f"rate limited on {ep.label if ep else 'endpoint'}"
            else:
                last_err = f"http {r.status_code}: {r.text[:200]}"
            attempts.append({"key_index": idx,
                             "status": ("rate_limited" if rate_limited
                                        else f"http_{r.status_code}"),
                             "gateway": (ep.label if ep else None),
                             "cooldown_s": (round(waited, 2) if rate_limited else None),
                             "elapsed": round(elapsed, 3)})
            if rate_limited:
                continue
        except requests.exceptions.Timeout as e:
            # Gateway saturation may make concurrent requests exceed their
            # connect/read timeout.  A timeout is not a model decision, so it
            # must not consume a key attempt or settle the case as a technical
            # failure.  Refresh the per-step window and retry until a request
            # receives a response (or the process is explicitly stopped).
            elapsed = time.time() - t0
            last_err = f"timeout: {type(e).__name__}: {e}"
            attempts.append({"key_index": idx, "status": type(e).__name__,
                             "gateway": (ep.label if ep else None),
                             "elapsed": round(elapsed, 3),
                             "retry": "unlimited"})
            tried.discard(idx)
            deadline = time.time() + STEP_WALL_CLOCK
            print(f"[{agent_name}] key[{key[-8:]}] timed out; retrying indefinitely")
            time.sleep(RETRY_INTERVAL)
            continue
        except Exception as e:
            elapsed = time.time() - t0
            last_err = f"exception: {type(e).__name__}: {e}"
            attempts.append({"key_index": idx, "status": type(e).__name__,
                             "gateway": (ep.label if ep else None),
                             "elapsed": round(elapsed, 3)})
        print(f"[{agent_name}] key[{key[-8:]}] failed: {last_err}")
        time.sleep(RETRY_INTERVAL)

    telemetry["api_attempts"] = attempts
    return (None, clean_log,
            {"error": last_err or "no API attempt succeeded",
             "attempts": len(attempts)},
            -1, payload, None, "technical")


def stages_total_for(init_data):
    """Mirror of the server's stages_total logic (used as a fallback)."""
    t = init_data.get("task_type")
    if t in (1, 3):
        return 1
    if t in (2, 4):
        # The agent only knows the task type here; for an exact count we'd
        # need to parse the task JSON. Use a conservative default.
        return 1
    if t in (5, 6):
        return 2
    if t == 7:
        return 2
    if t == 8:
        return 1
    if t in (9, 10):
        return 3
    return 0


def _safe_run_tag():
    """Sanitize RUN_NAME into a short per-run tag, so concurrent or repeated
    test_run invocations don't clobber each other's server-side logs."""
    safe = "".join(c if (c.isalnum() or c in "-_") else "" for c in RUN_NAME)[:16]
    return safe or "run"


def _post_with_retry(url, json_data, timeout=30, retries=3, return_last=False):
    """Post to the local server, retrying timeouts without a limit.

    ``return_last=True`` hands back the final non-200 response instead of
    ``None``, so the caller can tell a rejected request (a 409 action_seq
    conflict, say) from a transport failure. Retrying a 409 cannot help;
    collapsing both to ``None`` hides that. Connection failures and non-200
    responses retain the configured retry limit. A timeout consumes no
    attempt and is retried until the server responds or the process is stopped.
    """
    last = None
    attempt = 0
    while attempt < retries:
        try:
            r = requests.post(url, json=json_data, timeout=timeout)
            if r.status_code == 200:
                return r
            last = r
            attempt += 1
            if attempt < retries:
                print(f"⚠️ Server returned {r.status_code}, retrying ({attempt+1}/{retries})...")
                time.sleep(1)
        except requests.exceptions.Timeout as e:
            print("Server request timed out (%s); retrying indefinitely..."
                  % type(e).__name__)
            time.sleep(1)
            continue
        except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
            attempt += 1
            if attempt < retries:
                print(f"️ Connection error ({type(e).__name__}), retrying ({attempt+1}/{retries})...")
                time.sleep(1)
            else:
                raise e
    return last if return_last else None


def _append_step_record(path, record):
    """Append one decision to a JSONL step log.

    Line-append rather than the legacy read-modify-write of a whole JSON
    array: with the four-version text memories a per-case file reaches
    tens of MB, and rewriting it every step turned logging into the
    dominant cost of a run.
    """
    try:
        with io.open(path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[log] failed to append step record: {e}")


def _summarise_request(payload, telemetry):
    """Compact request description for the step log.

    Long-term memory is recorded by reference (kind + versions + size),
    never by copying its text into every step. Images are recorded by
    path; the base64 never reaches the log.
    """
    text_chars = 0
    images = []
    memory_refs = []
    for msg in payload["messages"]:
        content = msg["content"]
        if isinstance(content, str):
            text_chars += len(content)
            continue
        for blk in content:
            if not isinstance(blk, dict):
                continue
            if blk.get("type") == "image_url":
                images.append(blk.get("_image_path", ""))
            elif blk.get("type") == "text":
                ref = blk.get("_memory_ref")
                if ref:
                    memory_refs.append(ref)
                else:
                    text_chars += len(blk.get("text", ""))
    return {"text_chars_excluding_memory": text_chars,
            "memory_refs": memory_refs,
            "image_paths": images,
            "image_count": len(images),
            "telemetry": telemetry}


def _experiment_trajectories(case_key):
    """Prompt-shaped historical trajectories for one case, from the cache."""
    return EXPERIMENT_HISTORY.get(case_key.legacy_case_id, [])


def run_experiment_case(case_key, attempt_id, server_url, results_dir, lock,
                        counters, round_id=0, resume_steps=False):
    """Run one (group, case, attempt) and return a structured result.

    Never raises for an expected failure: every terminal condition comes
    back as a result dict so the runner can settle the case, and so an
    init failure still leaves a record instead of vanishing from the
    denominator.
    """
    group = EXPERIMENT_GROUP
    legacy_id = case_key.legacy_case_id
    agent_name = "%s_%s_c%03d_a%d" % (EXPERIMENT.experiment_id, group.group_id,
                                      legacy_id, attempt_id)
    step_log = os.path.join(results_dir,
                            "steps_%s_case%03d_a%d.jsonl"
                            % (group.group_id, legacy_id, attempt_id))
    started = time.time()

    # First line of every step file identifies the run that produced it, so
    # a stray .jsonl is never ambiguous about which experiment, group,
    # model or config fingerprint it came from.
    run_identity = {
        "experiment_id": EXPERIMENT.experiment_id,
        "config_fingerprint": EXPERIMENT.fingerprint(),
        "config_path": EXPERIMENT.config_path,
        "group_id": group.group_id,
        "group_slug": group.slug,
        "memory": group.memory,
        "memory_versions": list(group.versions),
        "test_version": EXPERIMENT.test_version,
        "round_id": round_id,
        "attempt_id": attempt_id,
        "legacy_case_id": legacy_id,
        "task_id": case_key.task_id,
        "case_no": case_key.case_no,
        "task_type": case_key.task_type,
        "model": MODEL_NAME,
        "api_url": API_URL,
        "provider": API_PROVIDER,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
        "action_extraction": ACTION_EXTRACTION,
        "max_steps": EXPERIMENT.max_steps,
        "history_steps": MAX_HISTORY_STEPS,
        "history_images": LOOKBACK_IMAGE_COUNT,
        "agent_name": agent_name,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    checkpoint = None
    elapsed_before = 0.0
    if resume_steps:
        from step_resume import load_checkpoint
        checkpoint = load_checkpoint(step_log, run_identity)
    if checkpoint is None:
        _append_step_record(step_log, {"record_type": "run_header", **run_identity})
    # Compact identity repeated on every decision line, so a single line
    # lifted out of the file still says what it belongs to.
    step_tag = {"experiment_id": EXPERIMENT.experiment_id,
                "group_id": group.group_id, "round_id": round_id,
                "attempt_id": attempt_id, "legacy_case_id": legacy_id,
                "task_id": case_key.task_id, "case_no": case_key.case_no}

    result = {
        "experiment_id": EXPERIMENT.experiment_id,
        "config_fingerprint": EXPERIMENT.fingerprint(),
        "group_id": group.group_id,
        "round_id": round_id,
        "attempt_id": attempt_id,
        "legacy_case_id": legacy_id,
        "task_id": case_key.task_id,
        "case_no": case_key.case_no,
        "task_type": case_key.task_type,
        "test_version": EXPERIMENT.test_version,
        "agent_name": agent_name,
        "memory": group.memory,
        "memory_versions": list(group.versions),
        "model": MODEL_NAME,
        "api_url": API_URL,
        "provider": API_PROVIDER,
        "config_path": EXPERIMENT.config_path,
        "final_status": "not_started",
        "settled": False,
        "success": False,
        "decisions": 0,
        "actions_executed": 0,
        "technical_failures": 0,
        "model_output_failures": 0,
        "truncated_decisions": 0,
        "stages_completed": 0,
        "stages_total": 0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "steps_with_unknown_usage": 0,
        "error": None,
        "step_log": os.path.basename(step_log),
    }

    def _finish(status, settled, error=None):
        result["final_status"] = status
        result["settled"] = settled
        result["success"] = (status == "task_completed")
        result["elapsed"] = round(elapsed_before + time.time() - started, 2)
        if error:
            result["error"] = error
        _write_case_result(results_dir, result)
        with lock:
            counters["attempts"] += 1
            counters["technical_steps"] += result.get("technical_failures", 0)
            if settled:
                counters["done"] += 1
                if result["success"]:
                    counters["success"] += 1
            print("[%s] %s case %d (%s/case%02d) -> %s  (settled %d, success %d, "
                  "attempts %d)"
                  % (group.group_id, "OK " if result["success"] else "END",
                     legacy_id, case_key.task_id, case_key.case_no, status,
                     counters["done"], counters["success"], counters["attempts"]))
        return result

    # ---- init -----------------------------------------------------------
    try:
        r = _post_with_retry(f"{server_url}/init_task",
                             {"case_id": legacy_id, "agent_id": agent_name},
                             timeout=CONNECT_TIMEOUT, retries=SERVER_RETRIES)
    except Exception as e:
        return _finish("failed_init", False, "init exception: %s" % e)
    if r is None or r.status_code != 200:
        return _finish("failed_init", False, "init failed after retries")

    data = r.json()
    identity = data.get("case_identity") or {}
    if data.get("test_version") != EXPERIMENT.test_version:
        return _finish("failed_init", False,
                       "server test_version=%r, expected %r"
                       % (data.get("test_version"), EXPERIMENT.test_version))
    if identity.get("task_id") != case_key.task_id or \
            identity.get("case_no") != case_key.case_no:
        return _finish("failed_init", False,
                       "server case identity %r does not match %r"
                       % (identity, case_key.as_dict()))

    mission = data["task_description"]
    pose = data["current_pose"]
    img_path = data["visual_observations"].get("main_view")
    stages_total = data.get("stages_total", 0)
    stages_completed = 0
    # The run's fixed spawn. Never overwritten; the live pose is tracked
    # separately in `pose`.
    initial_origin = {"x": pose["x"], "y": pose["y"], "heading": pose["heading"]}
    result["initial_origin"] = dict(initial_origin)
    result["description"] = mission
    result["stages_total"] = stages_total

    # The current view must resolve to the scene the server steps this case
    # through. Otherwise the image cache and the ViT table read another
    # scene's frames without raising -- a v5 case would be shown the
    # obstacle-free warehouse while the server enforces the obstacle.
    import scene_index
    served = (data.get("scene_info") or {}).get("scene_id")
    seen = scene_index.for_image_path(img_path)
    result["scene_id"] = seen.scene_id if seen is not None else None
    if seen is None or seen.scene_id != served:
        _release_session(server_url, agent_name)
        return _finish("failed_config", False,
                       "current view %s resolves to scene %r but the server "
                       "runs %r" % (img_path, result["scene_id"], served))

    trajectories = _experiment_trajectories(case_key)
    if group.uses_history_trajectories and not trajectories:
        return _finish("failed_config", False,
                       "group %s requires historical trajectories for %s but "
                       "none were loaded" % (group.group_id, case_key.label()))

    log_path, img_history = [], []
    token_usage = result["token_usage"]
    step_tokens = []
    decision_idx = 0          # counts model calls, including retried ones
    action_seq = 0            # counts actions actually applied by the server
    last_stage_completion_step = 0
    consecutive_technical = 0
    consecutive_truncated = 0
    consecutive_model_fail = 0

    if checkpoint is not None:
        from step_resume import replay_checkpoint

        def replay_step(action, sequence):
            response = _post_with_retry(
                f"{server_url}/step",
                {"action": action, "agent_id": agent_name, "action_seq": sequence},
                timeout=CONNECT_TIMEOUT, retries=SERVER_RETRIES, return_last=True)
            if response is None or response.status_code != 200:
                raise ValueError("local server rejected checkpoint replay")
            return response.json()

        try:
            restored = replay_checkpoint(checkpoint["rows"], data, replay_step, _format_step_log)
        except Exception as exc:
            _release_session(server_url, agent_name)
            return _finish("failed_config", False, "step resume: %s" % exc)
        elapsed_before = checkpoint["elapsed_before"]
        log_path, img_history = restored["log_path"], restored["img_history"]
        pose, img_path = restored["pose"], restored["img_path"]
        action_seq, decision_idx = restored["action_seq"], restored["decision_idx"]
        stages_completed, stages_total = restored["stages_completed"], restored["stages_total"]
        last_stage_completion_step = restored["last_stage_completion_step"]
        consecutive_technical = restored["consecutive_technical"]
        consecutive_truncated = restored["consecutive_truncated"]
        consecutive_model_fail = restored["consecutive_model_fail"]
        for key in ("technical_failures", "model_output_failures", "truncated_decisions",
                    "steps_with_unknown_usage", "stages_completed", "stages_total"):
            result[key] = restored[key]
        token_usage.update(restored["token_usage"])
        step_tokens = restored["step_tokens"]
        result.update(actions_executed=action_seq, decisions=decision_idx,
                      checkpoint_restored_actions=action_seq,
                      checkpoint_usage_unknown_decisions=restored["steps_with_unknown_usage"])
        _append_step_record(step_log, {"record_type": "step_resume",
            "elapsed_before": elapsed_before + time.time() - started,
            "resumed_at_epoch": time.time(), "restored_actions": action_seq})
        print("[%s] RESTORED case %d at action %d / decision %d (no model replay)"
              % (group.group_id, legacy_id, action_seq, decision_idx))
        terminal = restored["terminal"]
        if terminal or action_seq >= EXPERIMENT.max_steps:
            _release_session(server_url, agent_name)
            result["step_tokens"] = step_tokens
            return _finish(terminal or "failed_timeout", True)
        if (consecutive_technical >= MAX_CONSECUTIVE_TECHNICAL or
                consecutive_truncated >= MAX_CONSECUTIVE_TRUNCATED or
                consecutive_model_fail >= MAX_CONSECUTIVE_MODEL_FAIL):
            _release_session(server_url, agent_name)
            terminal = ("failed_technical" if consecutive_technical >= MAX_CONSECUTIVE_TECHNICAL
                        else "failed_truncated" if consecutive_truncated >= MAX_CONSECUTIVE_TRUNCATED
                        else "failed_model_output")
            return _finish(terminal, terminal == "failed_model_output")

    while True:
        decision_idx += 1
        ctx = DecisionContext(
            case_label=(case_key.task_id, case_key.case_no),
            initial_origin=dict(initial_origin),
            current_pose=dict(pose),
            decision_index=decision_idx,
            stages_completed=stages_completed,
            steps_in_current_stage=action_seq - last_stage_completion_step,
        )
        telemetry = {}
        budget = None            # None = the configured max_tokens
        trunc_retries = 0
        try:
            while True:
                action, clean_req, raw_res, key_idx, sent_payload, token_info, outcome = \
                    call_internvl(legacy_id, mission, pose, log_path, img_path,
                                  img_history, None, None, agent_name,
                                  initial_origin=initial_origin,
                                  trajectories=trajectories, ctx=ctx,
                                  telemetry=telemetry,
                                  max_tokens_override=budget)
                if outcome != "truncated":
                    break
                if trunc_retries >= MAX_TRUNCATION_RETRIES or \
                        (budget or MAX_OUTPUT_TOKENS) >= MAX_OUTPUT_TOKENS_CEILING:
                    break
                # Reasoning length has a long tail, and no action was
                # produced, so no environment step was taken. Re-issuing the
                # same decision with more room is safe and leaves the action
                # sequence untouched. Ending the case here instead would kill
                # almost every long case: at a 1.5% truncation rate,
                # surviving 200 steps has probability 0.985**200.
                trunc_retries += 1
                result["truncated_decisions"] += 1
                budget = min((budget or MAX_OUTPUT_TOKENS) * 2,
                             MAX_OUTPUT_TOKENS_CEILING)
                telemetry.setdefault("truncation_retries", []).append(budget)
        except ExperimentPromptError as e:
            # An assembly invariant broke (image budget, missing memory).
            # That is a code defect, not something a retry can fix.
            _release_session(server_url, agent_name)
            return _finish("failed_config", False, "prompt assembly: %s" % e)

        if token_info:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = token_info.get(k)
                if isinstance(v, int):
                    token_usage[k] += v
            if not token_info.get("usage_reported"):
                result["steps_with_unknown_usage"] += 1
            step_tokens.append({"decision": decision_idx,
                                "prompt_tokens": token_info.get("prompt_tokens"),
                                "completion_tokens": token_info.get("completion_tokens"),
                                "total_tokens": token_info.get("total_tokens")})

        if SAVE_PROMPTS and decision_idx == 1:
            _dump_full_request(results_dir, group.group_id, legacy_id,
                               attempt_id, sent_payload)
        record = {
            "record_type": "decision",
            **step_tag,
            "decision_index": decision_idx,
            "action_seq_before": action_seq,
            "initial_origin": dict(initial_origin),
            "pose_before": dict(pose),
            "outcome": outcome,
            "key_index": key_idx,
            "request": _summarise_request(sent_payload, telemetry),
            "model_response": _extract_response_text(raw_res),
            "token_info": token_info,
            "action": action,
        }

        if outcome == "technical":
            result["technical_failures"] += 1
            consecutive_technical += 1
            record["error"] = raw_res.get("error") if isinstance(raw_res, dict) else None
            result["decisions"] = decision_idx
            _append_step_record(step_log, record)
            # The decision counter advances; action_seq does not, so a
            # technical retry never consumes an environment step.
            if consecutive_technical >= MAX_CONSECUTIVE_TECHNICAL:
                _release_session(server_url, agent_name)
                return _finish("failed_technical", False,
                               "%d consecutive technical failures"
                               % consecutive_technical)
            continue

        if outcome == "truncated":
            # The reply ran out of decoding budget mid-reasoning, so it
            # never reached an action. That is a configuration fault, not
            # the model failing the task, and it is reported separately so
            # it cannot be mistaken for a low success rate.
            result["truncated_decisions"] += 1
            consecutive_truncated += 1
            result["decisions"] = decision_idx
            _append_step_record(step_log, record)
            if consecutive_truncated < MAX_CONSECUTIVE_TRUNCATED:
                continue          # same decision, fresh attempt
            _release_session(server_url, agent_name)
            return _finish("failed_truncated", False,
                           "%d consecutive truncated replies; raise "
                           "model.max_tokens (currently %d) for this model"
                           % (consecutive_truncated, MAX_OUTPUT_TOKENS))

        if outcome == "model_output_failure":
            result["model_output_failures"] += 1
            consecutive_model_fail += 1
            result["decisions"] = decision_idx
            _append_step_record(step_log, record)
            if consecutive_model_fail < MAX_CONSECUTIVE_MODEL_FAIL:
                continue          # give the model another go at this step
            _release_session(server_url, agent_name)
            return _finish("failed_model_output", True,
                           telemetry.get("parse_note"))

        consecutive_technical = 0
        consecutive_truncated = 0
        consecutive_model_fail = 0
        action_seq += 1
        result["decisions"] = decision_idx
        try:
            resp = _post_with_retry(f"{server_url}/step",
                                    {"action": action, "agent_id": agent_name,
                                     "action_seq": action_seq},
                                    timeout=CONNECT_TIMEOUT, retries=SERVER_RETRIES,
                                    return_last=True)
        except Exception as e:
            record["server_error"] = str(e)
            _append_step_record(step_log, record)
            _release_session(server_url, agent_name)
            return _finish("failed_technical", False, "step post exception: %s" % e)
        if resp is None:
            record["server_error"] = "step post failed after retries"
            _append_step_record(step_log, record)
            _release_session(server_url, agent_name)
            return _finish("failed_technical", False, "step post failed after retries")
        if resp.status_code != 200:
            # A well-formed rejection (409 action_seq conflict, 400 bad
            # request) is a defect in the client/server contract, not
            # something another attempt can fix.
            try:
                detail = resp.json().get("message")
            except Exception:
                detail = resp.text[:200]
            record["server_error"] = "HTTP %d: %s" % (resp.status_code, detail)
            _append_step_record(step_log, record)
            _release_session(server_url, agent_name)
            return _finish("failed_config", False, record["server_error"])
        r2 = resp.json()

        status = r2.get("status", "error")
        msg = r2.get("message", "")
        new_pose = r2.get("current_pose", pose)
        new_img = r2.get("visual_observations", {}).get("main_view", img_path)

        log_entry, display = _format_step_log(action_seq, action, status, pose,
                                              new_pose, msg)
        log_path.append(log_entry)
        img_history.append({"action": action, "status": display,
                            "path": img_path, "pose": pose})

        record["action_seq"] = action_seq
        record["server_status"] = status
        record["server_message"] = msg
        record["pose_after"] = dict(new_pose)
        record["replayed"] = bool(r2.get("replayed"))
        _append_step_record(step_log, record)

        pose = new_pose
        img_path = new_img
        result["actions_executed"] = action_seq

        prev_stages = stages_completed
        stages_completed = r2.get("stages_completed", 0)
        stages_total = r2.get("stages_total", 0) or stages_total
        if stages_completed > prev_stages:
            last_stage_completion_step = action_seq
        result["stages_completed"] = stages_completed
        result["stages_total"] = stages_total
        result["decisions"] = decision_idx

        if status == "task_completed":
            _release_session(server_url, agent_name)
            result["step_tokens"] = step_tokens
            return _finish("task_completed", True)
        if status == "failed_timeout" or action_seq >= EXPERIMENT.max_steps:
            _release_session(server_url, agent_name)
            result["step_tokens"] = step_tokens
            return _finish("failed_timeout", True)


def _extract_response_text(raw_res):
    """Pull the model's raw text out of a response, or describe the error."""
    if not isinstance(raw_res, dict):
        return None
    if "error" in raw_res:
        return {"error": raw_res.get("error")}
    try:
        return raw_res["choices"][0]["message"]["content"]
    except Exception:
        return {"unparsed": str(raw_res)[:500]}


def _release_session(server_url, agent_name):
    """Best-effort session release so the server does not accumulate state."""
    try:
        requests.post(f"{server_url}/end_case", json={"agent_id": agent_name},
                      timeout=10)
    except Exception:
        pass


def _dump_full_request(results_dir, group_id, legacy_id, attempt_id, payload):
    """Write one complete request per case for auditing.

    ``--save_prompts`` in experiment mode saves the first decision only.
    Saving every step would mean re-writing the four-version memory text
    200 times per case; one full sample plus the per-step JSONL (which
    records memory by reference) covers the same audit need at a bounded
    cost.
    """
    if payload is None:
        return
    d = os.path.join(results_dir, "request_samples")
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "request_%s_case%03d_a%d.json"
                            % (group_id, legacy_id, attempt_id))
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(_serialize_payload_for_save(payload), f, indent=2,
                      ensure_ascii=False)
    except Exception as e:
        print("[log] failed to write request sample: %s" % e)


def _write_case_result(results_dir, result):
    path = os.path.join(results_dir,
                        "case_%s_%03d_a%d.json"
                        % (result["group_id"], result["legacy_case_id"],
                           result["attempt_id"]))
    try:
        with io.open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[log] failed to write case result: {e}")


def run_case(case_id, case_count, server_url, results_dir, lock, counters):
    """Run a single case. Writes case log at key moments (init, API fail, server fail, end)."""
    run_tag = _safe_run_tag()
    agent_name = f"Agent_{run_tag}_{str(case_id).zfill(3)}"
    case_log_path = os.path.join(results_dir, f"client_case_{case_id}_{agent_name}.json")

    def _flush_case_log():
        """Write current case state to log file (real-time update at key moments)."""
        report = {
            "agent_name": agent_name,
            "case_id": case_id,
            "task_type": task_type,
            "description": mission,
            "origin": {"coord": [origin_pose["x"], origin_pose["y"]], "heading": origin_pose["heading"]},
            "final_status": final_status,
            "total_steps": step_idx,
            "steps_consumed": step_idx,
            "stages_completed": stages_completed,
            "stages_total": stages_total,
            "token_usage": token_usage,
            "brief_execution_log": log_path[-20:],  # last 20 steps
        }
        try:
            with open(case_log_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f" [{agent_name}] Failed to write case log: {e}")

    # Init task with retry
    try:
        r = _post_with_retry(f"{server_url}/init_task",
                             {"case_id": case_id, "agent_id": agent_name},
                             timeout=30, retries=3)
        if r is None or r.status_code != 200:
            print(f"❌ [{agent_name}] init failed after retries")
            return
    except Exception as e:
        print(f"❌ [{agent_name}] cannot reach server: {e}")
        return

    data = r.json()
    mission = data["task_description"]
    pose = data["current_pose"]
    img_path = data["visual_observations"].get("main_view")
    task_type = data.get("task_type")
    stages_total = data.get("stages_total", stages_total_for(data))
    stages_completed = 0

    origin_pose = {"x": pose["x"], "y": pose["y"], "heading": pose["heading"]}

    log_path = []
    img_history = []
    step_idx = 0
    final_status = "running"  # Initial status
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    step_tokens = []

    grid_state = {
        "visited": {(pose["x"], pose["y"])},
        "items_seen": {},
        "walls": set(),
        "bounds": {"x": (-7, 7), "y": (-10, 16)},
    }
    # Determine which trajectory to use for §6 (the optional full-reference
    # section) and for §5 (the episodic-memory slice). These are
    # INDEPENDENT: §6 honours --peer_reference / --reference_traj; §5
    # is ALWAYS anchored on the current case's own forward_only trajectory
    # — that is the "what did the expert do at the place the agent is
    # looking at?" recall, regardless of which trajectory (if any) was
    # chosen for §6.
    peer_trajectory = None
    reference_traj = None
    episodic_only_trajectory = None
    if PEER_REFERENCE:
        # Peer mode: get trajectory from another case of the same task_type
        peer_trajectory = get_peer_trajectory(case_id, REFERENCE_TRAJ_MODE)
        if peer_trajectory:
            print(f"📋 [{agent_name}] Using peer trajectory from case {peer_trajectory['source_case_id']}")
    elif REFERENCE_TRAJ_MODE != "none":
        # Own mode: get this case's trajectory (kind chosen via --reference_traj)
        reference_traj = get_trajectory_for_case(case_id, REFERENCE_TRAJ_MODE)

    # Episodic memory always pulls from the case's own forward_only
    # trajectory — independent of §6's choice above.
    if ENABLE_EPISODIC_MEMORY:
        episodic_only_trajectory = get_trajectory_for_case(case_id, "forward_only")
        if episodic_only_trajectory:
            print(f"🧠 [{agent_name}] Anchoring episodic memory on own forward_only trajectory "
                  f"({len(episodic_only_trajectory['actions'])} actions)")
    consecutive_api_failures = 0  # consecutive API failure count; give up the case after threshold

    # Stage-progress tracking for the em §5 tiebreaker. Refreshed before
    # each call_internvl() so build_user_content sees the agent's CURRENT
    # stage position, not stale state from the previous iteration.
    last_stage_completion_step = 0  # 1-based; 0 means "no stage completed yet"

    # Write initial log
    _flush_case_log()

    while True:
        step_idx += 1
        # Refresh per-call tiebreaker context from the LAST KNOWN state
        # (this iteration's previous /step response). At step 1 this is
        # all zeros — i.e., no tiebreaker context for the very first call.
        global AGENT_STAGES_COMPLETED, AGENT_STEPS_IN_CURRENT_STAGE
        AGENT_STAGES_COMPLETED = stages_completed
        AGENT_STEPS_IN_CURRENT_STAGE = step_idx - last_stage_completion_step
        # Pass the FULL log and the FULL img_history to the prompt module;
        # the rendering function trims log to MAX_HISTORY_STEPS and picks
        # the last LOOKBACK_IMAGE_COUNT images, while still being able to
        # look up the pose for every step in the log.
        recent_text = log_path
        recent_imgs = img_history

        action, clean_req, raw_res, last_key_idx, sent_payload, token_info, outcome = call_internvl(
            case_id, mission, pose, recent_text, img_path, recent_imgs,
            grid_state, reference_traj, agent_name, peer_trajectory,
            episodic_only_trajectory,
        )
        if token_info:
            token_usage["prompt_tokens"] += token_info.get("prompt_tokens", 0)
            token_usage["completion_tokens"] += token_info.get("completion_tokens", 0)
            token_usage["total_tokens"] += token_info.get("total_tokens", 0)
            step_tokens.append({
                "step": step_idx,
                "prompt_tokens": token_info.get("prompt_tokens", 0),
                "completion_tokens": token_info.get("completion_tokens", 0),
                "total_tokens": token_info.get("total_tokens", 0),
            })
        if outcome == "technical":
            consecutive_api_failures += 1
            print(f" [{agent_name}] API call failed ({consecutive_api_failures}/3), retrying without consuming step {step_idx}.")
            _save_step_prompt(case_id, agent_name, step_idx, sent_payload,
                              "Error_Stop", raw_res, {"error": "all attempts failed"},
                              RUN_NAME, results_dir)
            step_idx -= 1  # roll back, don't consume a step
            if consecutive_api_failures >= 3:
                print(f"❌ [{agent_name}] API call failed 3 consecutive times, giving up.")
                final_status = "failed_api"
                _flush_case_log()  # Update log
                break
            continue

        if outcome == "model_output_failure":
            # v0 silently substituted Turn_Left here, which executed a real
            # action for an unparseable reply. Record it as a model-side
            # failure and settle the case instead.
            print(f" [{agent_name}] unparseable model output, ending case.")
            _save_step_prompt(case_id, agent_name, step_idx, sent_payload,
                              None, raw_res, {"error": "model output failure"},
                              RUN_NAME, results_dir)
            final_status = "failed_model_output"
            _flush_case_log()
            break

        consecutive_api_failures = 0  # got action successfully, reset counter

        try:
            r2 = _post_with_retry(f"{server_url}/step",
                                  {"action": action, "agent_id": agent_name},
                                  timeout=30, retries=3)
            if r2 is None:
                print(f" [{agent_name}] step post failed after retries")
                _save_step_prompt(case_id, agent_name, step_idx, sent_payload,
                                  action, raw_res, {"error": "step post failed after retries"},
                                  RUN_NAME, results_dir)
                final_status = "server_error"
                _flush_case_log()
                break
            r2 = r2.json()
        except Exception as e:
            print(f"️ [{agent_name}] step post failed: {e}")
            _save_step_prompt(case_id, agent_name, step_idx, sent_payload,
                              action, raw_res, {"error": f"step post failed: {e}"},
                              RUN_NAME, results_dir)
            final_status = "server_error"
            _flush_case_log()
            break

        status = r2.get("status", "error")
        msg = r2.get("message", "")
        new_pose = r2.get("current_pose", pose)
        new_img = r2.get("visual_observations", {}).get("main_view", img_path)

        log_entry, display = _format_step_log(step_idx, action, status, pose, new_pose, msg)
        log_path.append(log_entry)

        img_history.append({
            "action": action,
            "status": display,
            "path": img_path,
            "pose": pose,
        })

        # Persist the full prompt + response for this step (base64 stripped,
        # image paths substituted). No-op if --save_prompts is off.
        _save_step_prompt(case_id, agent_name, step_idx, sent_payload, action,
                          raw_res, r2, RUN_NAME, results_dir)

        grid_state["visited"].add((new_pose["x"], new_pose["y"]))
        if status == "blocked":
            attempted = _attempted_position(pose, action)
            if attempted:
                grid_state["walls"].add(attempted)

        pose = new_pose
        img_path = new_img

        # Read stages progress from the server (incremented per stage hit).
        prev_stages = stages_completed
        stages_completed = r2.get("stages_completed", 0)
        stages_total = r2.get("stages_total", 0) or stages_total_for(data)
        # Detect a stage-completed event for the em tiebreaker.
        if stages_completed > prev_stages:
            last_stage_completion_step = step_idx

        if status == "task_completed":
            final_status = "task_completed"
            break
        if status == "failed_timeout":
            final_status = "failed_timeout"
            break
        if step_idx >= 200:
            final_status = "failed_timeout"
            break

        time.sleep(0.3)

    report = {
        "agent_name": agent_name,
        "case_id": case_id,
        "task_type": data.get("task_type"),
        "description": mission,
        "origin": {"coord": [origin_pose["x"], origin_pose["y"]], "heading": origin_pose["heading"]},
        "final_status": final_status,
        "success": final_status == "task_completed",
        "total_steps": step_idx,
        "steps_consumed": step_idx if final_status == "task_completed" else 200,
        "stages_completed": stages_completed,
        "stages_total": stages_total,
        "token_usage": token_usage,
        "step_tokens": step_tokens,
        "brief_execution_log": log_path,
    }
    fp = os.path.join(results_dir, f"client_case_{case_id}_{agent_name}.json")
    try:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ [{agent_name}] log write error: {e}")

    with lock:
        counters["done"] += 1
        if final_status == "task_completed":
            counters["success"] += 1
        print(f"✅ [{agent_name}] case {case_id} {final_status} ({counters['success']}/{counters['done']} done)")


def _preload_experiment_memory(cfg, group):
    """Load exactly the long-term memory this group is allowed to see.

    Loading is strict: a missing file, an empty file or a hash mismatch
    raises rather than silently producing a memory-free run. Called once
    before any case starts, so a data problem fails the group instead of
    corrupting 100 cases one at a time.
    """
    global EXPERIMENT_MEMORY_TEXT, EXPERIMENT_HISTORY, EXPERIMENT_MANIFEST
    import experiment as ex

    manifest = {"memory": group.memory, "versions": list(group.versions)}
    EXPERIMENT_MEMORY_TEXT = ""
    EXPERIMENT_HISTORY = {}

    if group.uses_text_memory:
        mems = prepare_experiment_text_memory(cfg, group)
        manifest["text_memories"] = [
            {"version": m.version, "kind": m.kind, "path": m.source_path,
             "sha256": m.source_sha256, "chars": len(m.text)} for m in mems]
        manifest["rendered_chars"] = sum(len(v) for v in EXPERIMENT_TEXT_BY_KIND.values())
        if group.memory == "hybrid":
            manifest["hybrid_components"] = ["summary", "semantic"]
            manifest["text_scope"] = "all 100 cases of each listed historical version"
        print("[memory] %s: %d version(s), %d chars rendered"
              % (group.memory, len(group.versions), manifest["rendered_chars"]))

    if group.uses_history_trajectories:
        names = ex.scene_base64_names(cfg)
        traj_manifest = []
        for key in cfg.case_map.all_keys():
            loaded = ex.load_history_set(cfg, key, group.versions, base64_names=names)
            EXPERIMENT_HISTORY[key.legacy_case_id] = [
                t.as_prompt_traj() for t in loaded]
            for t in loaded:
                traj_manifest.append({"legacy_case_id": key.legacy_case_id,
                                      "version": t.version, "length": t.length,
                                      "sha256": t.source_sha256,
                                      "notes": t.validation_notes})
        manifest["trajectories"] = traj_manifest
        print("[memory] loaded %d historical trajectories for %d cases"
              % (len(traj_manifest), len(EXPERIMENT_HISTORY)))

    EXPERIMENT_MANIFEST = manifest
    return manifest


def _experiment_handshake(server_url, cfg):
    """Refuse to run against a legacy v0 server or a mismatched config."""
    try:
        info = requests.get(f"{server_url}/get_server_time", timeout=10).json()
    except Exception as e:
        raise SystemExit("cannot reach server at %s: %s" % (server_url, e))
    exp = info.get("experiment") or {}
    if not exp:
        raise SystemExit(
            "server at %s is running the legacy v0 task set. Start it with "
            "--experiment_config to test v4." % server_url)
    if exp.get("test_version") != cfg.test_version:
        raise SystemExit("server test_version=%r, config expects %r"
                         % (exp.get("test_version"), cfg.test_version))
    if exp.get("config_fingerprint") != cfg.fingerprint():
        raise SystemExit(
            "experiment fingerprint mismatch.\n  server: %s\n  agent:  %s\n"
            "The server and the agent were started from different configs."
            % (exp.get("config_fingerprint"), cfg.fingerprint()))
    if exp.get("experiment_id") != cfg.experiment_id:
        raise SystemExit("server experiment_id=%r, config has %r"
                         % (exp.get("experiment_id"), cfg.experiment_id))
    return exp


def _assert_effective_flags(group):
    """Positively assert the runtime state, not just the absence of a flag.

    The legacy runner enables Summary and Semantic by default, so
    "we did not pass --sum" is not evidence that they are off.
    """
    problems = []
    if ENABLE_GRID_MEMORY:
        problems.append("grid memory is on")
    if ENABLE_PLACEHOLDERS:
        problems.append("placeholders are on")
    if PEER_REFERENCE:
        problems.append("peer reference is on")
    if REFERENCE_TRAJ_MODE != "none":
        problems.append("legacy reference_traj=%s is on" % REFERENCE_TRAJ_MODE)
    if ENABLE_EPISODIC_MEMORY:
        problems.append("legacy episodic_memory flag is on")
    if SUMMARY_PROMPT_TEXT or SEMANTIC_PROMPT_TEXT:
        problems.append("legacy summary/semantic prompt text is loaded")
    if not ENABLE_HISTORY:
        problems.append("working-memory history is off")
    if MAX_HISTORY_STEPS != EXPERIMENT.history_steps:
        problems.append("h=%d, config wants %d" % (MAX_HISTORY_STEPS, EXPERIMENT.history_steps))
    if LOOKBACK_IMAGE_COUNT != EXPERIMENT.history_images:
        problems.append("i=%d, config wants %d" % (LOOKBACK_IMAGE_COUNT, EXPERIMENT.history_images))
    if problems:
        raise SystemExit("effective config is wrong for %s: %s"
                         % (group.group_id, "; ".join(problems)))


def run_experiment_main(args):
    """Experiment-mode entry point: one group, a set of cases, one round."""
    global EXPERIMENT, EXPERIMENT_GROUP, MAX_HISTORY_STEPS, LOOKBACK_IMAGE_COUNT
    global TEMPERATURE, MODEL_NAME, API_URL, MAX_OUTPUT_TOKENS, MAX_PROMPT_IMAGES
    global MAX_API_ATTEMPTS, STEP_WALL_CLOCK, API_TIMEOUT, CONNECT_TIMEOUT
    global RETRY_INTERVAL, SERVER_RETRIES, API_KEYS, RUN_NAME, SAVE_PROMPTS
    global API_PROVIDER, MODEL_EXTRA_BODY, ACTION_EXTRACTION
    global MAX_TRUNCATION_RETRIES, MAX_OUTPUT_TOKENS_CEILING
    global MAX_CONSECUTIVE_TRUNCATED, MAX_CONSECUTIVE_MODEL_FAIL
    global MAX_CONSECUTIVE_TECHNICAL
    global ENABLE_GRID_MEMORY, ENABLE_PLACEHOLDERS, ENABLE_HISTORY
    global REFERENCE_TRAJ_SMART_LAYERS, REFERENCE_TRAJ_SMART_SAMPLE_COUNT
    global PEER_REFERENCE, REFERENCE_TRAJ_MODE, ENABLE_EPISODIC_MEMORY
    global SUMMARY_PROMPT_TEXT, SEMANTIC_PROMPT_TEXT
    import experiment as ex

    cfg = ex.load_config(args.experiment_config)
    if args.results_dir == DEFAULT_RESULTS_DIR:
        from result_paths import experiment_result_dir
        args.results_dir = experiment_result_dir(cfg, cfg.config_path)
    EXPERIMENT = cfg
    group = cfg.group
    EXPERIMENT_GROUP = group

    # Every knob comes from the config, so a stale CLI default cannot
    # quietly change what the model sees.
    MAX_HISTORY_STEPS = cfg.history_steps
    LOOKBACK_IMAGE_COUNT = cfg.history_images
    MAX_PROMPT_IMAGES = cfg.max_prompt_images
    TEMPERATURE = float(cfg.model.get("temperature", 0.0))
    MODEL_NAME = cfg.model.get("name", MODEL_NAME)
    API_URL = cfg.model.get("api_url", API_URL)
    MAX_OUTPUT_TOKENS = int(cfg.model.get("max_tokens", 30))
    API_PROVIDER = cfg.model.get("provider", "internai")
    MODEL_EXTRA_BODY = dict(cfg.model.get("extra_body") or {})
    ACTION_EXTRACTION = cfg.model.get("action_extraction", "strict")
    if API_PROVIDER not in ("internai", "openai"):
        raise SystemExit("model.provider must be 'internai' or 'openai', got %r"
                         % API_PROVIDER)
    if ACTION_EXTRACTION not in ("strict", "last"):
        raise SystemExit("model.action_extraction must be 'strict' or 'last'")
    print("[model] %s @ %s (provider=%s, max_tokens=%d, extraction=%s)"
          % (MODEL_NAME, API_URL, API_PROVIDER, MAX_OUTPUT_TOKENS, ACTION_EXTRACTION))
    ex_cfg = cfg.execution
    MAX_API_ATTEMPTS = int(ex_cfg.get("max_api_attempts_per_step", 10))
    STEP_WALL_CLOCK = float(ex_cfg.get("step_wall_clock_seconds", 900))
    API_TIMEOUT = int(ex_cfg.get("read_timeout", 300))
    CONNECT_TIMEOUT = int(ex_cfg.get("connect_timeout", 30))
    RETRY_INTERVAL = float(ex_cfg.get("retry_interval_seconds", 1.0))
    SERVER_RETRIES = int(ex_cfg.get("server_retry_attempts", 3))
    max_case_attempts = int(ex_cfg.get("max_case_attempts", 3))
    max_workers = args.max_workers or int(ex_cfg.get("max_workers", 8))
    # Output only: saving the first request never changes what is sent, so
    # it lives in ``execution`` and stays out of the fingerprint.
    # ``--save_prompts`` can still switch it on for a single run.
    SAVE_PROMPTS = bool(ex_cfg.get("save_prompts", False)) or bool(args.save_prompts)

    # Decision retries and give-up limits (config ``retry``). They decide
    # whether a cut-off or unparseable reply is re-asked and when an attempt
    # ends, so they belong to the experiment rather than to module constants.
    r = cfg.retry
    MAX_TRUNCATION_RETRIES = int(r["truncation_retries"])
    MAX_OUTPUT_TOKENS_CEILING = int(r["max_tokens_ceiling"])
    MAX_CONSECUTIVE_TRUNCATED = int(r["max_consecutive_truncated"])
    MAX_CONSECUTIVE_MODEL_FAIL = int(r["max_consecutive_model_failures"])
    MAX_CONSECUTIVE_TECHNICAL = int(r["max_consecutive_technical"])

    # Prompt-shaping parameters. These used to be module constants in three
    # different files, so a run was described by its config plus whatever
    # those files happened to say. Now the config is the whole description.
    p = cfg.prompt
    REFERENCE_TRAJ_SMART_LAYERS = set(p["reference_traj_smart_layers"])
    REFERENCE_TRAJ_SMART_SAMPLE_COUNT = p["reference_traj_smart_sample_count"]
    from prompt import _helpers as _ph
    from prompt import vit_matcher as _vm
    _ph.set_image_encoding(p["image_max_size"], p["image_quality"])
    try:
        _vm.set_thresholds(p["vit_exact_cosine"], p["vit_near_cosine"],
                           p["vit_cosine_tie_eps"])
    except ValueError as exc:
        raise SystemExit(str(exc))
    print("[prompt] images %dpx q%d | smart layers %s | vit bands %.2f/%.2f"
          % (p["image_max_size"], p["image_quality"],
             ",".join(sorted(REFERENCE_TRAJ_SMART_LAYERS)),
             p["vit_exact_cosine"], p["vit_near_cosine"]))

    # Legacy memory switches stay off; the group whitelist is the only
    # thing that turns memory on in this mode.
    ENABLE_GRID_MEMORY = False
    ENABLE_PLACEHOLDERS = False
    ENABLE_HISTORY = True
    PEER_REFERENCE = False
    REFERENCE_TRAJ_MODE = "none"
    ENABLE_EPISODIC_MEMORY = False
    SUMMARY_PROMPT_TEXT = None
    SEMANTIC_PROMPT_TEXT = None
    _assert_effective_flags(group)

    inline_keys = ex_cfg.get("api_keys")
    if inline_keys and not args.api_excel:
        api_excel = "<inline api_keys in %s>" % os.path.basename(cfg.config_path)
        API_KEYS = load_api_keys_from_list(
            inline_keys, default_url=API_URL, default_model=MODEL_NAME,
            default_provider=API_PROVIDER)
        key_source_sha = ex.sha256_text(
            "\n".join(sorted(e.key for e in API_KEYS)))
    else:
        api_excel = args.api_excel or resolve_repo_path(ex_cfg.get("api_excel", "api.xlsx"))
        API_KEYS = load_api_keys(api_excel)
        key_source_sha = ex.sha256_file(api_excel)
    if ex_cfg.get("validate_keys", True):
        # This endpoint reports dead keys as HTTP 200 with an error
        # envelope, and most keys in a big pool are dead. Filtering once
        # up front is what makes the per-step attempt cap meaningful.
        API_KEYS = validate_api_keys(
            API_KEYS, API_URL, MODEL_NAME,
            workers=int(ex_cfg.get("key_probe_workers", 16)),
            cache_path=resolve_repo_path(ex_cfg.get(
                "api_key_cache", "result/api_keys_valid_%s.json" % API_PROVIDER)),
            source_sha=key_source_sha,
            min_keys=int(ex_cfg.get("min_valid_keys", 1)),
            force=bool(args.revalidate_keys),
        )
    if len(API_KEYS) < MAX_API_ATTEMPTS:
        print("[keys] note: only %d usable keys, so a decision can make at "
              "most %d attempts (config asks for %d)"
              % (len(API_KEYS), len(API_KEYS), MAX_API_ATTEMPTS))
    RUN_NAME = args.run_name or ("%s_%s" % (cfg.experiment_id, group.group_id))

    server_info = _experiment_handshake(args.server_url, cfg)
    manifest = _preload_experiment_memory(cfg, group)

    if args.case_ids:
        wanted = [int(x) for x in args.case_ids.split(",") if x.strip() != ""]
    else:
        wanted = list(server_info.get("case_ids") or range(len(cfg.case_map)))
    case_keys = [cfg.case_map.by_legacy(c) for c in wanted]

    results_dir = os.path.join(args.results_dir, f"client_{RUN_NAME}")
    os.makedirs(results_dir, exist_ok=True)

    print("[run] %s %s | %d cases | workers=%d | h=%d i=%d | max_steps=%d"
          % (cfg.experiment_id, group.group_id, len(case_keys), max_workers,
             MAX_HISTORY_STEPS, LOOKBACK_IMAGE_COUNT, cfg.max_steps))

    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = Lock()
    counters = {"done": 0, "success": 0, "attempts": 0, "technical_steps": 0,
                "decisions": 0}
    settled = {}
    all_attempts = []
    start = time.time()

    breaker = dict(ex_cfg.get("circuit_breaker") or {})
    probe_cases = int(breaker.get("probe_cases", 0) or 0)
    max_ratio = float(breaker.get("max_technical_step_ratio", 1.0))
    tripped = {"reason": None}

    def _check_breaker():
        """Stop the group when the first N cases are mostly failing on
        technical errors — better than spending the rest of the quota
        producing a run that cannot be published anyway."""
        if not probe_cases or tripped["reason"]:
            return
        with lock:
            settled_n = counters["done"]
            unresolved = counters["attempts"] - counters["done"]
            tech = counters["technical_steps"]
            decisions = counters["decisions"]
            if settled_n + unresolved < probe_cases:
                return
            ratio = (tech / decisions) if decisions else 0.0
            if ratio > max_ratio:
                tripped["reason"] = (
                    "technical failures were %.1f%% of the first %d decisions "
                    "(limit %.0f%%)" % (ratio * 100, decisions, max_ratio * 100))

    def _settled_result_on_disk(case_key):
        """Return a previously settled result for this case, if any.

        A 100-case run takes many hours; without this, an interrupted run
        has to redo everything it already paid for. Only *settled* results
        are reused — an unresolved technical failure is retried.
        """
        if not args.resume:
            return None
        import glob as _glob
        pattern = os.path.join(results_dir, "case_%s_%03d_a*.json"
                               % (group.group_id, case_key.legacy_case_id))
        for path in sorted(_glob.glob(pattern)):
            try:
                with io.open(path, encoding="utf-8") as f:
                    prev = json.load(f)
            except Exception:
                continue
            if prev.get("settled") and                     prev.get("config_fingerprint") == cfg.fingerprint():
                prev["resumed_from_disk"] = os.path.basename(path)
                return prev
        return None

    def _run_case_with_retries(case_key):
        """Run one case, retrying only technical failures."""
        prev = _settled_result_on_disk(case_key)
        if prev is not None:
            with lock:
                counters["done"] += 1
                if prev.get("success"):
                    counters["success"] += 1
                print("[%s] SKIP case %d (already settled: %s)"
                      % (group.group_id, case_key.legacy_case_id,
                         prev.get("final_status")))
            return case_key.legacy_case_id, [prev]
        if tripped["reason"]:
            return case_key.legacy_case_id, [{
                "legacy_case_id": case_key.legacy_case_id,
                "group_id": group.group_id, "attempt_id": -1,
                "final_status": "skipped_circuit_breaker", "settled": False,
                "success": False, "error": tripped["reason"],
                "task_id": case_key.task_id, "case_no": case_key.case_no,
                "task_type": case_key.task_type}]
        attempts = []
        for attempt_id in range(max_case_attempts):
            res = run_experiment_case(case_key, attempt_id, args.server_url,
                                      results_dir, lock, counters,
                                      round_id=args.round_id,
                                      resume_steps=bool(getattr(args, "resume_steps", False)))
            attempts.append(res)
            with lock:
                counters["decisions"] += res.get("decisions", 0) or 0
            if res.get("settled"):
                break
        _check_breaker()
        if tripped["reason"]:
            print("[%s] CIRCUIT BREAKER: %s — remaining cases will be skipped."
                  % (group.group_id, tripped["reason"]))
        return case_key.legacy_case_id, attempts

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_case_with_retries, k): k for k in case_keys}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                legacy_id, attempts = fut.result()
            except Exception as e:
                # A thread that died still has to leave a record, or the
                # case silently disappears from the denominator.
                legacy_id = key.legacy_case_id
                attempts = [{"legacy_case_id": legacy_id,
                             "group_id": group.group_id,
                             "attempt_id": -1, "final_status": "thread_exception",
                             "settled": False, "success": False,
                             "error": "%s: %s" % (type(e).__name__, e)}]
                print("[thread] case %d raised: %s" % (legacy_id, e))
            all_attempts.extend(attempts)
            chosen = next((a for a in attempts if a.get("settled")), None)
            settled[legacy_id] = chosen or attempts[-1]

    elapsed = time.time() - start
    summary = _build_experiment_summary(cfg, group, case_keys, settled,
                                        all_attempts, elapsed, manifest,
                                        server_info, api_excel, max_workers,
                                        args.round_id)
    summary["circuit_breaker_tripped"] = tripped["reason"]
    with io.open(os.path.join(results_dir, "SUMMARY.json"), "w",
                 encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    print("[%s] resolved %d/%d | success %d | rate %s | unresolved %d | %.1fs"
          % (group.group_id, summary["cases_resolved"], summary["cases_expected"],
             summary["success"], summary["success_rate"],
             len(summary["cases_unresolved"]), elapsed))
    if summary["cases_unresolved"]:
        print("[%s] UNRESOLVED (technical): %s"
              % (group.group_id, summary["cases_unresolved"]))
        print("[%s] complete-set success rate is NOT published for this group."
              % group.group_id)
    print("[%s] logs: %s" % (group.group_id, results_dir))
    print("=" * 78)
    return summary


def _build_experiment_summary(cfg, group, case_keys, settled, all_attempts,
                              elapsed, manifest, server_info, api_excel,
                              max_workers, round_id):
    expected = [k.legacy_case_id for k in case_keys]
    resolved, unresolved = [], []
    per_case, by_task_type = [], {}
    tok_all = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tok_settled = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tok_success = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    unknown_usage_steps = 0
    technical, model_failures, retried, truncated = 0, 0, 0, 0

    for a in all_attempts:
        for k in tok_all:
            tok_all[k] += (a.get("token_usage") or {}).get(k, 0) or 0
        technical += a.get("technical_failures", 0) or 0
        model_failures += a.get("model_output_failures", 0) or 0
        truncated += a.get("truncated_decisions", 0) or 0
        unknown_usage_steps += a.get("steps_with_unknown_usage", 0) or 0
        if (a.get("attempt_id") or 0) > 0:
            retried += 1

    for cid in expected:
        res = settled.get(cid)
        if res is None:
            unresolved.append(cid)
            continue
        if not res.get("settled"):
            unresolved.append(cid)
        else:
            resolved.append(cid)
        for k in tok_settled:
            tok_settled[k] += (res.get("token_usage") or {}).get(k, 0) or 0
        if res.get("success"):
            for k in tok_success:
                tok_success[k] += (res.get("token_usage") or {}).get(k, 0) or 0
        tt = res.get("task_type")
        bucket = by_task_type.setdefault(tt, {"n": 0, "success": 0})
        bucket["n"] += 1
        bucket["success"] += 1 if res.get("success") else 0
        per_case.append({
            "legacy_case_id": cid, "task_id": res.get("task_id"),
            "case_no": res.get("case_no"), "task_type": tt,
            "final_status": res.get("final_status"), "settled": res.get("settled"),
            "success": res.get("success"),
            "actions_executed": res.get("actions_executed"),
            "decisions": res.get("decisions"),
            "stages_completed": res.get("stages_completed"),
            "stages_total": res.get("stages_total"),
            "attempt_id": res.get("attempt_id"),
            "token_usage": res.get("token_usage"),
            "initial_origin": res.get("initial_origin"),
        })

    success = sum(1 for c in per_case if c["success"])
    complete = not unresolved
    rate = ("%.2f%%" % (success / len(expected) * 100)) if complete and expected else "n/a (incomplete)"
    steps_all = [c["actions_executed"] for c in per_case
                 if isinstance(c["actions_executed"], int)]
    steps_ok = [c["actions_executed"] for c in per_case
                if c["success"] and isinstance(c["actions_executed"], int)]

    return {
        "experiment_id": cfg.experiment_id,
        "config_fingerprint": cfg.fingerprint(),
        "config_path": cfg.config_path,
        "group_id": group.group_id,
        "group": group.as_dict(),
        "round_id": round_id,
        "test_version": cfg.test_version,
        "run_name": RUN_NAME,
        "server_experiment": server_info,
        "cases_expected": len(expected),
        "cases_resolved": len(resolved),
        "cases_unresolved": unresolved,
        "complete": complete,
        "success": success,
        "failed": len(per_case) - success,
        "success_rate": rate,
        "success_rate_over_resolved": ("%.2f%%" % (success / len(per_case) * 100))
                                      if per_case else "n/a",
        "success_by_task_type": {str(k): v for k, v in sorted(by_task_type.items(), key=lambda item: str(item[0]))},
        "steps_mean_all": (sum(steps_all) / len(steps_all)) if steps_all else None,
        "steps_median_all": _median(steps_all),
        "steps_mean_success": (sum(steps_ok) / len(steps_ok)) if steps_ok else None,
        "steps_median_success": _median(steps_ok),
        "elapsed_seconds": round(elapsed, 2),
        "token_usage_settled": tok_settled,
        "token_usage_success_only": tok_success,
        "token_usage_all_attempts": tok_all,
        "steps_with_unknown_usage": unknown_usage_steps,
        "technical_failures": technical,
        "model_output_failures": model_failures,
        "truncated_decisions": truncated,
        "attempts_total": len(all_attempts),
        "cases_retried": retried,
        "hyperparams": {
            "model": MODEL_NAME, "api_url": API_URL, "temperature": TEMPERATURE,
            "provider": API_PROVIDER, "action_extraction": ACTION_EXTRACTION,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "retry": dict(EXPERIMENT.retry),
            "max_history_steps": MAX_HISTORY_STEPS,
            "lookback_image_count": LOOKBACK_IMAGE_COUNT,
            "max_prompt_images": MAX_PROMPT_IMAGES,
            "max_steps": cfg.max_steps,
            "max_api_attempts_per_step": MAX_API_ATTEMPTS,
            "step_wall_clock_seconds": STEP_WALL_CLOCK,
            "connect_timeout": CONNECT_TIMEOUT, "read_timeout": API_TIMEOUT,
            "retry_interval_seconds": RETRY_INTERVAL,
            "server_retry_attempts": SERVER_RETRIES,
            "max_workers": max_workers,
            "save_prompts": SAVE_PROMPTS,
            "api_excel": api_excel, "api_key_count": len(API_KEYS),
            "endpoints": [e.as_dict() for e in API_KEYS
                          if isinstance(e, Endpoint)],
            "enable_grid_memory": ENABLE_GRID_MEMORY,
            "enable_placeholders": ENABLE_PLACEHOLDERS,
            "peer_reference": PEER_REFERENCE,
            "legacy_reference_traj": REFERENCE_TRAJ_MODE,
            "legacy_episodic_memory": ENABLE_EPISODIC_MEMORY,
        },
        "memory_manifest": manifest,
        "task_results": sorted(per_case, key=lambda c: c["legacy_case_id"]),
        "attempts": all_attempts,
    }


def _median(values):
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def resolve_repo_path(path):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--api_excel", default=None,
                        help="path to the API-key spreadsheet (column C holds "
                             "the sk- keys). Default: <repo>/api.xlsx, or the "
                             "experiment config's execution.api_excel.")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--experiment_config", default=None,
                        help="run a config/**/*.yaml experiment (its one group). "
                             "Mutually exclusive with the legacy memory flags.")
    parser.add_argument("--case_ids", default=None,
                        help="comma-separated legacy case ids; default = all")
    parser.add_argument("--round_id", type=int, default=0,
                        help="repeat index, recorded in every result row")
    parser.add_argument("--resume", action="store_true",
                        help="reuse settled case results already present in "
                             "--results_dir instead of re-running them")
    parser.add_argument("--revalidate_keys", action="store_true",
                        help="re-probe every API key instead of reusing the "
                             "cached validated subset")
    parser.add_argument("--resume_steps", action="store_true",
                        help="restore interrupted decisions via verified local action replay")
    parser.add_argument("--case_count", type=int, default=None,
                        help="run only the first N cases")
    parser.add_argument("--max_history_steps", type=int, default=5,
                        help="number of recent step log lines included in the prompt (text count)")
    parser.add_argument("--lookback_image_count", type=int, default=5,
                        help="number of recent step images included in the prompt (image count, independent of text count)")
    parser.add_argument("--grid_memory", action="store_true",
                        help="enable the ASCII grid memory section (default off)")
    parser.add_argument("--no_precomputed_images", action="store_true",
                        help="force runtime image conversion instead of using pre-computed "
                             "base64 files from dataset_v1-v5_base64/scene_v0-v4/image_base64/ "
                             "(default: use pre-computed if available)")
    parser.add_argument("--no_history", action="store_true",
                        help="omit the entire history section (text + images)")
    parser.add_argument("--placeholders", action="store_true",
                        help="enable the future-memory placeholders block (default off)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_workers", type=int, default=None,
                        help="concurrent cases (default 8, or the experiment "
                             "config's execution.max_workers)")
    parser.add_argument("--max_api_retries", type=int, default=100,
                        help="max random API keys to try per call")
    parser.add_argument("--api_timeout", type=int, default=300,
                        help="per-request timeout in seconds (default 300 = 5 min)")
    parser.add_argument("--trajectory_json", default=DEFAULT_TRAJECTORY_JSON,
                        help="path to warehouse_01_01_trajectories.json")
    parser.add_argument("--reference_traj", default="none",
                        choices=["none", "forward_only", "all_actions"],
                        help="which reference trajectory to inject as text into the prompt")
    parser.add_argument("--reference_traj_images", default="none",
                        choices=["none", "all", "interval", "turn_only", "smart"],
                        help="interleave images into the reference trajectory block; "
                             "'smart' uses priority start→end→stages→turns→uniform-sample")
    parser.add_argument("--reference_traj_interval", type=int, default=3,
                        help="step interval for --reference_traj_images=interval")
    parser.add_argument("--reference_traj_max_smart_images", type=int, default=50,
                        help="upper bound on smart-mode images; effective max is also "
                             "capped by the remaining budget (MAX_PROMPT_IMAGES=50 minus "
                             "current image and history images)")
    parser.add_argument("--reference_traj_smart_layers", type=str, default=None,
                        help="comma-separated priority layers for smart mode; "
                             "valid: start,stage,end,turn. Default: all four. "
                             "Examples: 'start,end' for only start+end; "
                             "'start,stage,end' for no turns; 'start,end,turn' for no stages")
    parser.add_argument("--reference_traj_smart_sample_count", type=int, default=None,
                        help="max number of uniform-sampled images in smart mode; "
                             "default: fill remaining budget. Set to 0 to disable sampling")
    parser.add_argument("--peer_reference", action="store_true",
                        help="use trajectory from another case of the SAME task_type "
                             "as reference (mutually exclusive with own trajectory; "
                             "randomly picks one peer case per task)")
    parser.add_argument("--episodic_memory", action="store_true",
                        help="enable the episodic-memory prompt section: anchor the "
                             "agent at the closest matched view in the active "
                             "reference trajectory, then surface up to "
                             "--episodic_memory_horizon next expert actions with "
                             "their resulting views.")
    parser.add_argument("--episodic_memory_horizon", type=int, default=6,
                        help="max number of subsequent expert actions to render in "
                             "the episodic-memory section (default 6).")
    parser.add_argument("--episodic_memory_matcher", default="pose",
                        choices=["pose", "vit"],
                        help="matcher used to anchor the episodic-memory slice: "
                             "'pose' (default; filename/pose-based, no extra deps) "
                             "or 'vit' (ViT-B/16 cosine similarity over the "
                             "precomputed dataset_v1-v5_base64/scene_v0-v4/image_embeddings.npz; "
                             "auto-falls back to 'pose' if the .npz is missing)")
    parser.add_argument("--episodic_memory_content", default="all",
                        choices=["all", "actions", "images"],
                        help="what to include after the matched view in the "
                             "episodic-memory section: 'all' (default; actions + "
                             "resulting images), 'actions' (only action text, no "
                             "resulting images), 'images' (only resulting images, "
                             "no action text)")
    parser.add_argument("--save_prompts", action="store_true",
                        help="persist every step's full prompt + LLM response to "
                             "result/<run>/prompts/case_<id>.json (image base64 is "
                             "stripped; only the image path is recorded)")
    parser.add_argument("--summary_prompt_path",
                        default=DEFAULT_SUMMARY_PROMPT_PATH,
                        help="path to a .md file injected at the front of the "
                             "user message (after the semantic prompt, if any). "
                             "Default: empty string (feature OFF). Pass an "
                             "absolute path, or a path relative to PROJECT_ROOT "
                             "(the repo root), to opt in. Typical value: "
                             "dataset_v1-v3_memory/v0/summary_memory/summary.md "
                             "(running summary-memory state produced by "
                             "scripts/summary_memory_writer.py). "
                             "Independent of --semantic_prompt_path.")
    parser.add_argument("--semantic_prompt_path",
                        default=DEFAULT_SEMANTIC_PROMPT_PATH,
                        help="path to a .md file injected at the front of the "
                             "user message BEFORE any summary prompt. "
                             "Default: empty string (feature OFF). Pass an "
                             "absolute path, or a path relative to PROJECT_ROOT "
                             "(the repo root), to opt in. Typical value: "
                             "dataset_v1-v3_memory/v0/semantic_memory/memory.md "
                             "(running semantic-memory state produced by "
                             "scripts/semantic_memory_consolidation.py). "
                             "Independent of --summary_prompt_path.")
    args = parser.parse_args()

    # Refuse to write directly under the result/ root — only the
    # timestamped default (`result/agent_run/`) or a deeper path passed
    # by test_run.py is allowed. This stops a stray `python3 code/agent.py`
    # from polluting the test-run output folder.
    if args.results_dir == DEFAULT_RESULTS_DIR and not args.experiment_config:
        args.results_dir = os.path.join(args.results_dir, f"agent_{datetime_str()}")

    # ---- experiment mode --------------------------------------------
    # Two config sources for the same knob is how a run silently ends up
    # measuring something other than what the matrix says, so the legacy
    # memory flags are refused outright rather than merged.
    if args.experiment_config:
        conflicting = []
        if args.reference_traj != "none":
            conflicting.append("--reference_traj")
        if args.reference_traj_images != "none":
            conflicting.append("--reference_traj_images")
        if args.episodic_memory:
            conflicting.append("--episodic_memory")
        if args.peer_reference:
            conflicting.append("--peer_reference")
        if args.grid_memory:
            conflicting.append("--grid_memory")
        if args.placeholders:
            conflicting.append("--placeholders")
        if args.no_history:
            conflicting.append("--no_history")
        if args.summary_prompt_path:
            conflicting.append("--summary_prompt_path")
        if args.semantic_prompt_path:
            conflicting.append("--semantic_prompt_path")
        if conflicting:
            raise SystemExit(
                "these legacy flags conflict with --experiment_config: %s\n"
                "In experiment mode the group definition is the only source "
                "of memory configuration." % ", ".join(conflicting))
        set_use_precomputed(not args.no_precomputed_images)
        run_experiment_main(args)
        return

    global MAX_HISTORY_STEPS, LOOKBACK_IMAGE_COUNT, TEMPERATURE, CASE_START, RUN_NAME, API_KEYS
    global REFERENCE_TRAJ_MODE, REFERENCE_TRAJ_IMAGES_MODE, REFERENCE_TRAJ_INTERVAL, TRAJECTORY_JSON_PATH, TRAJECTORY_INDEX
    global ENABLE_GRID_MEMORY, ENABLE_HISTORY, ENABLE_PLACEHOLDERS
    global REFERENCE_TRAJ_MAX_SMART_IMAGES, REFERENCE_TRAJ_SMART_LAYERS, REFERENCE_TRAJ_SMART_SAMPLE_COUNT, SAVE_PROMPTS
    global MAX_API_RETRIES, API_TIMEOUT
    global PEER_REFERENCE
    global ENABLE_EPISODIC_MEMORY, EPISODIC_MEMORY_HORIZON, EPISODIC_MEMORY_MATCHER, EPISODIC_MEMORY_CONTENT
    global SUMMARY_PROMPT_TEXT, SEMANTIC_PROMPT_TEXT
    MAX_HISTORY_STEPS = args.max_history_steps
    LOOKBACK_IMAGE_COUNT = args.lookback_image_count
    ENABLE_GRID_MEMORY = args.grid_memory
    set_use_precomputed(not args.no_precomputed_images)
    if not args.no_precomputed_images:
        from prompt import IMAGE_BASE64_DIR
        print(f"📷 Using pre-computed base64 images from {IMAGE_BASE64_DIR}")
    else:
        print("📷 Runtime image conversion enabled (--no_precomputed_images)")
    ENABLE_HISTORY = not args.no_history
    ENABLE_PLACEHOLDERS = args.placeholders
    SAVE_PROMPTS = args.save_prompts
    TEMPERATURE = args.temperature
    MAX_API_RETRIES = args.max_api_retries
    API_TIMEOUT = args.api_timeout
    CASE_START = 0
    RUN_NAME = args.run_name or f"agent_{datetime_str()}"
    API_KEYS = load_api_keys(args.api_excel or DEFAULT_API_EXCEL)
    REFERENCE_TRAJ_MODE = args.reference_traj
    REFERENCE_TRAJ_IMAGES_MODE = args.reference_traj_images
    REFERENCE_TRAJ_INTERVAL = args.reference_traj_interval
    REFERENCE_TRAJ_MAX_SMART_IMAGES = args.reference_traj_max_smart_images
    if args.reference_traj_smart_layers is not None:
        REFERENCE_TRAJ_SMART_LAYERS = set(
            s.strip() for s in args.reference_traj_smart_layers.split(",") if s.strip()
        )
    if args.reference_traj_smart_sample_count is not None:
        REFERENCE_TRAJ_SMART_SAMPLE_COUNT = args.reference_traj_smart_sample_count
    TRAJECTORY_JSON_PATH = args.trajectory_json
    TRAJECTORY_INDEX = load_trajectories(TRAJECTORY_JSON_PATH)
    if TRAJECTORY_INDEX:
        print(f"📂 Loaded {len(TRAJECTORY_INDEX)} trajectories from {TRAJECTORY_JSON_PATH}")
    # Optional memory-recall prompts (semantic / summary) injected at the
    # very front of the user message. Each CLI flag accepts a path to a .md
    # file; both are independent and may be disabled (pass "" to skip).
    # Relative paths are resolved against PROJECT_ROOT (the repo root).
    def _load_memory_prompt(path):
        if not path:
            return None  # empty string / None -> feature disabled
        if not os.path.isabs(path):
            full = os.path.join(PROJECT_ROOT, path)
        else:
            full = path
        try:
            with open(full, encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            # The running-memory file may not exist yet (consolidation
            # script hasn't run). Skip silently — the agent just runs
            # without that memory recall.
            return None
        except Exception as e:
            print(f"⚠️ Failed to load memory prompt from {full}: {e}")
            return None

    SUMMARY_PROMPT_TEXT = _load_memory_prompt(args.summary_prompt_path)
    SEMANTIC_PROMPT_TEXT = _load_memory_prompt(args.semantic_prompt_path)
    if SUMMARY_PROMPT_TEXT is not None:
        print(f"📜 Loaded summary prompt from {args.summary_prompt_path}")
    if SEMANTIC_PROMPT_TEXT is not None:
        print(f"📜 Loaded semantic prompt from {args.semantic_prompt_path}")
    PEER_REFERENCE = args.peer_reference
    if PEER_REFERENCE:
        if not TRAJECTORY_INDEX:
            print(f"⚠️ --peer_reference requested but no trajectory JSON loaded; will be skipped")
            PEER_REFERENCE = False
        else:
            print(f"🔄 Peer reference enabled: will use trajectory from another case of same task_type")
    elif REFERENCE_TRAJ_MODE != "none" and not TRAJECTORY_INDEX:
        print(f"⚠️ --reference_traj={REFERENCE_TRAJ_MODE} requested but no trajectory JSON loaded; will be skipped")

    ENABLE_EPISODIC_MEMORY = args.episodic_memory
    EPISODIC_MEMORY_HORIZON = max(1, args.episodic_memory_horizon)
    EPISODIC_MEMORY_MATCHER = args.episodic_memory_matcher
    EPISODIC_MEMORY_CONTENT = args.episodic_memory_content
    if ENABLE_EPISODIC_MEMORY and EPISODIC_MEMORY_MATCHER == "vit":
        try:
            from prompt import vit_matcher
            ready = vit_matcher.scenes_ready()
            missing = sorted(s for s, ok in ready.items() if not ok)
            if not ready.get("base", False):
                print(f"⚠️ --episodic_memory_matcher=vit requested but the base "
                      f"embeddings table is missing; will auto-fall back to the "
                      f"'pose' matcher")
                EPISODIC_MEMORY_MATCHER = "pose"
            else:
                print(f"🧠 ViT matcher: {len(ready) - len(missing)}/{len(ready)} "
                      f"scene tables (lazy-loaded; 768-dim cosine)")
                if missing:
                    # A v5 case whose scene has no table silently degrades to
                    # the pose matcher for that case only, which would look
                    # like a matcher difference rather than a missing file.
                    print(f"⚠️ no ViT table for {len(missing)} scene(s): "
                          f"{', '.join(missing[:5])}"
                          f"{' ...' if len(missing) > 5 else ''} — cases on those "
                          f"scenes fall back to 'pose'. Run "
                          f"scripts/build_scene_embeddings.py")
        except Exception as e:
            print(f"⚠️ --episodic_memory_matcher=vit failed to initialise ({e}); "
                  f"falling back to 'pose'")
            EPISODIC_MEMORY_MATCHER = "pose"
    if ENABLE_EPISODIC_MEMORY:
        if not TRAJECTORY_INDEX:
            print(f"⚠️ --episodic_memory requested but no trajectory JSON loaded; will be skipped")
        else:
            print(f"🧠 Episodic-memory enabled: anchored on the case's own forward_only trajectory, "
                  f"rendering the closest matched view + the next {EPISODIC_MEMORY_HORIZON} "
                  f"actions (independent of §6's choice)")

    try:
        cfg = requests.get(f"{args.server_url}/get_server_time", timeout=5).json()
        total = cfg.get("total_tasks", 0)
        server_log_dir = cfg["server_paths"]["log_dir"]
        print(f"📡 Server ready. total_tasks={total}, log_dir={server_log_dir}")
    except Exception as e:
        print(f"❌ Cannot reach server at {args.server_url}: {e}")
        return

    end = total - 1
    if args.case_count is not None:
        end = min(CASE_START + args.case_count - 1, total - 1)
    case_ids = list(range(CASE_START, end + 1))
    legacy_workers = args.max_workers or MAX_WORKERS
    print(f"🎯 Running cases [{CASE_START}..{end}] = {len(case_ids)} cases | max_workers={legacy_workers}")

    results_dir = os.path.join(args.results_dir, f"client_{RUN_NAME}")
    os.makedirs(results_dir, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    lock = Lock()
    counters = {"done": 0, "success": 0}
    start = time.time()

    with ThreadPoolExecutor(max_workers=legacy_workers) as pool:
        futures = {pool.submit(run_case, cid, len(case_ids), args.server_url, results_dir, lock, counters): cid
                   for cid in case_ids}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"🚨 thread exception: {e}")

    elapsed = time.time() - start
    rate = (counters["success"] / counters["done"] * 100) if counters["done"] else 0.0

    # Collect per-case results from the per-case report files
    per_case_results = []
    total_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    success_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    run_tag = _safe_run_tag()
    for cid in range(counters["done"]):
        # Match the per-case file name built by run_case (which includes
        # the same run_tag) so we can read its report.
        agent_name = f"Agent_{run_tag}_{str(cid).zfill(3)}"
        report_path = os.path.join(results_dir, f"client_case_{cid}_{agent_name}.json")
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    r = json.load(f)
                case_tokens = r.get("token_usage", {})
                case_success = r.get("success", False)
                per_case_results.append({
                    "case_id": r.get("case_id", cid),
                    "task_type": r.get("task_type"),
                    "description": r.get("description"),
                    "origin": r.get("origin"),
                    "success": case_success,
                    "steps_consumed": r.get("steps_consumed", 200),
                    "stages_completed": r.get("stages_completed", 0),
                    "stages_total": r.get("stages_total", 0),
                    "final_status": r.get("final_status"),
                    "token_usage": case_tokens,
                })
                total_token_usage["prompt_tokens"] += case_tokens.get("prompt_tokens", 0)
                total_token_usage["completion_tokens"] += case_tokens.get("completion_tokens", 0)
                total_token_usage["total_tokens"] += case_tokens.get("total_tokens", 0)
                if case_success:
                    success_token_usage["prompt_tokens"] += case_tokens.get("prompt_tokens", 0)
                    success_token_usage["completion_tokens"] += case_tokens.get("completion_tokens", 0)
                    success_token_usage["total_tokens"] += case_tokens.get("total_tokens", 0)
            except Exception as e:
                per_case_results.append({"case_id": cid, "error": str(e)})

    # Embed scene metadata + tasks for offline analysis
    scene_metadata = None
    tasks = None
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            scene_metadata = json.load(f)
    except Exception:
        pass
    try:
        with open(TASKS_PATH, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except Exception:
        pass

    summary = {
        "run_name": RUN_NAME,
        "cases_run": counters["done"],
        "success": counters["success"],
        "failed": counters["done"] - counters["success"],
        "success_rate": f"{rate:.2f}%",
        "elapsed_seconds": round(elapsed, 2),
        "save_prompts": SAVE_PROMPTS,
        "token_usage": total_token_usage,
        "token_usage_success_only": success_token_usage,
        "hyperparams": {
            "model": MODEL_NAME,
            "max_history_steps": MAX_HISTORY_STEPS,
            "lookback_image_count": LOOKBACK_IMAGE_COUNT,
            "enable_grid_memory": ENABLE_GRID_MEMORY,
            "enable_history": ENABLE_HISTORY,
            "enable_placeholders": ENABLE_PLACEHOLDERS,
            "temperature": TEMPERATURE,
            "reference_traj": REFERENCE_TRAJ_MODE,
            "reference_traj_images": REFERENCE_TRAJ_IMAGES_MODE,
            "reference_traj_interval": REFERENCE_TRAJ_INTERVAL,
            "reference_traj_max_smart_images": REFERENCE_TRAJ_MAX_SMART_IMAGES,
            "reference_traj_smart_layers": sorted(REFERENCE_TRAJ_SMART_LAYERS),
            "reference_traj_smart_sample_count": REFERENCE_TRAJ_SMART_SAMPLE_COUNT,
            "episodic_memory": ENABLE_EPISODIC_MEMORY,
            "episodic_memory_horizon": EPISODIC_MEMORY_HORIZON,
            "episodic_memory_matcher": EPISODIC_MEMORY_MATCHER,
            "episodic_memory_content": EPISODIC_MEMORY_CONTENT,
        },
        "scene_metadata": scene_metadata,
        "tasks": tasks,
        "task_results": per_case_results,
    }
    with open(os.path.join(results_dir, "SUMMARY.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 80)
    print(f"🏁 {counters['done']} cases done, success {counters['success']}, rate {rate:.2f}%, {elapsed:.1f}s")
    print(f"📁 logs: {results_dir}")
    if SAVE_PROMPTS:
        print(f"📁 per-step prompts: {os.path.join(results_dir, 'prompts')}")
    print("=" * 80)


def datetime_str():
    return time.strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    main()
