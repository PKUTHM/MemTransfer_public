"""Restore an interrupted case by replaying its recorded actions locally.

No model calls are made during replay. Every environment transition is checked
against the saved log before new decisions are allowed.
"""
import datetime
import json
import os


def load_checkpoint(path, identity):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    headers = [r for r in records if r.get("record_type") == "run_header"]
    if len(headers) != 1:
        raise ValueError("checkpoint must contain exactly one original run header")
    header = headers[0]
    for key in ("config_fingerprint", "legacy_case_id", "group_id", "round_id", "attempt_id"):
        if header.get(key) != identity.get(key):
            raise ValueError("checkpoint identity mismatch: " + key)
    rows = [r for r in records if r.get("record_type") == "decision"]
    for index, row in enumerate(rows, 1):
        if row.get("decision_index") != index:
            raise ValueError("non-contiguous checkpoint decisions")
        for key in ("legacy_case_id", "group_id", "round_id", "attempt_id"):
            if row.get(key) != identity.get(key):
                raise ValueError("checkpoint decision identity mismatch: " + key)
        if row.get("server_error"):
            raise ValueError("checkpoint has an unconfirmed server transition")
    resumed = [r for r in records if r.get("record_type") == "step_resume"]
    if resumed:
        base = resumed[-1]["elapsed_before"]
        segment_start = resumed[-1]["resumed_at_epoch"]
    else:
        base = 0
        segment_start = datetime.datetime.fromisoformat(header["started_at"]).timestamp()
    elapsed = base + max(0, os.path.getmtime(path) - segment_start)
    return {"rows": rows, "elapsed_before": elapsed}


def replay_checkpoint(rows, initial, step, format_log):
    pose = initial["current_pose"]
    origin = dict(pose)
    image = initial["visual_observations"]["main_view"]
    state = dict(log_path=[], img_history=[], step_tokens=[], decision_idx=0,
                 action_seq=0, last_stage_completion_step=0, stages_completed=0,
                 stages_total=initial.get("stages_total", 0),
                 consecutive_technical=0, consecutive_truncated=0,
                 consecutive_model_fail=0, technical_failures=0,
                 model_output_failures=0, truncated_decisions=0,
                 steps_with_unknown_usage=0, terminal=None,
                 token_usage=dict(prompt_tokens=0, completion_tokens=0, total_tokens=0))
    for row in rows:
        if state["terminal"]:
            raise ValueError("checkpoint continues after terminal status")
        if row["pose_before"] != pose or row["initial_origin"] != origin:
            raise ValueError("checkpoint pose/origin mismatch before action")
        if row["action_seq_before"] != state["action_seq"]:
            raise ValueError("checkpoint action sequence mismatch")
        state["decision_idx"] = row["decision_index"]
        telemetry = row.get("request", {}).get("telemetry", {})
        state["truncated_decisions"] += len(telemetry.get("truncation_retries", []))
        usage = row.get("token_info")
        if usage:
            for key in state["token_usage"]:
                value = usage.get(key)
                if isinstance(value, int):
                    state["token_usage"][key] += value
            if not usage.get("usage_reported"):
                state["steps_with_unknown_usage"] += 1
            state["step_tokens"].append({"decision": row["decision_index"],
                **{key: usage.get(key) for key in state["token_usage"]}})
        elif row["outcome"] != "technical":
            # Older logs did not persist usage. Never invent recovered tokens.
            state["steps_with_unknown_usage"] += 1
        outcome = row["outcome"]
        if outcome == "technical":
            state["technical_failures"] += 1
            state["consecutive_technical"] += 1
            continue
        if outcome == "truncated":
            state["truncated_decisions"] += 1
            state["consecutive_truncated"] += 1
            continue
        if outcome == "model_output_failure":
            state["model_output_failures"] += 1
            state["consecutive_model_fail"] += 1
            continue
        if outcome != "ok" or not row.get("action"):
            raise ValueError("unsupported checkpoint outcome")
        seq = state["action_seq"] + 1
        if row.get("action_seq") != seq or not row.get("server_status"):
            raise ValueError("checkpoint lacks confirmed action result")
        response = step(row["action"], seq)
        if response.get("current_pose") != row.get("pose_after"):
            raise ValueError("environment replay pose mismatch")
        if response.get("status") != row["server_status"]:
            raise ValueError("environment replay status mismatch")
        if response.get("message", "") != row.get("server_message", ""):
            raise ValueError("environment replay feedback mismatch")
        new_pose = response["current_pose"]
        entry, display = format_log(seq, row["action"], response["status"],
                                    pose, new_pose, response.get("message", ""))
        state["log_path"].append(entry)
        state["img_history"].append({"action": row["action"], "status": display,
                                    "path": image, "pose": pose})
        pose = new_pose
        image = response.get("visual_observations", {}).get("main_view", image)
        state["action_seq"] = seq
        for key in ("consecutive_technical", "consecutive_truncated", "consecutive_model_fail"):
            state[key] = 0
        completed = response.get("stages_completed", 0)
        if completed > state["stages_completed"]:
            state["last_stage_completion_step"] = seq
        state["stages_completed"] = completed
        state["stages_total"] = response.get("stages_total", 0) or state["stages_total"]
        if response["status"] in ("task_completed", "failed_timeout"):
            state["terminal"] = response["status"]
    state.update(pose=pose, img_path=image)
    return state
