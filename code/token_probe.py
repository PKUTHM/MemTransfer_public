"""Real-request token probe for the v0-v3 -> v4 experiment.

Builds the config's largest realistic request and sends it once with
``max_tokens=1``, reading ``usage.prompt_tokens`` back. That number is
the only trustworthy input-length measurement for this endpoint: there
is no local tokenizer for intern-s2, and character counts do not predict
image token cost at all.

Run it before the smoke run. It answers three questions the offline
preflight cannot:

  * does the endpoint accept the four-full-text requests (G07/G09)?
  * does it accept the 47-image requests (G02/G03)?
  * which groups are actually expensive per decision?

Usage:
    python code/token_probe.py --experiment_config A06_episodic_v0__test_v0
    python code/token_probe.py ... --api_excel api.xlsx --out probe.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests                       # noqa: E402
import agent as ag                    # noqa: E402
import experiment as ex               # noqa: E402


def _worst_case_for(cfg, group, names):
    """Pick the case that makes this group's request largest."""
    if not group.uses_history_trajectories:
        return cfg.case_map.by_legacy(0)
    best, best_len = None, -1
    for key in cfg.case_map.all_keys():
        total = 0
        for v in group.versions:
            t = ex.load_history_trajectory(cfg, key, v, base64_names=names)
            total += t.length
        if total > best_len:
            best, best_len = key, total
    return best


def build_request(cfg, group, key, names, full_history=True):
    """Assemble one decision exactly as the agent would, at full history."""
    tasks, _ = ex.load_test_tasks(cfg)
    task = tasks[key.legacy_case_id]
    origin = {"x": int(task.origin["coord"][0]), "y": int(task.origin["coord"][1]),
              "heading": task.origin["heading"]}
    img = task.scene.image_path(origin["x"], origin["y"], origin["heading"])

    ag.EXPERIMENT, ag.EXPERIMENT_GROUP = cfg, group
    ag.MAX_HISTORY_STEPS = cfg.history_steps
    ag.LOOKBACK_IMAGE_COUNT = cfg.history_images
    ag.MAX_PROMPT_IMAGES = cfg.max_prompt_images
    ag.ENABLE_HISTORY = True
    ag.MODEL_NAME = cfg.model.get("name", ag.MODEL_NAME)
    ag.EXPERIMENT_MEMORY_TEXT = ""

    trajs = []
    if group.uses_text_memory:
        ag.prepare_experiment_text_memory(cfg, group)
    if group.uses_history_trajectories:
        loaded = ex.load_history_set(cfg, key, group.versions, base64_names=names)
        trajs = [t.as_prompt_traj() for t in loaded]

    # Saturated working memory: the steady-state request is bigger than
    # the first decision's, so probing step 1 would understate the cost.
    logs, imgs = [], []
    if full_history:
        for i in range(cfg.history_steps):
            logs.append("Step %d: Turn_Left -> success at (%d, %d) North (heading changed)"
                        % (i + 1, origin["x"], origin["y"]))
            imgs.append({"action": "Turn_Left", "status": "turned", "path": img,
                         "pose": dict(origin)})

    ctx = ag.DecisionContext(case_label=(key.task_id, key.case_no),
                             initial_origin=dict(origin), current_pose=dict(origin),
                             decision_index=cfg.history_steps + 1)
    telemetry = {}
    parts = ag.build_experiment_user_content(
        task.description, origin, origin, logs, img, imgs, trajs, ctx, telemetry)
    payload = {
        "model": cfg.model.get("name"),
        "messages": [{"role": "system", "content": ag.build_system_prompt()},
                     {"role": "user", "content": parts}],
        "temperature": cfg.model.get("temperature", 0.0),
        "max_tokens": 1,
    }
    if cfg.model.get("provider", "internai") == "internai":
        payload["thinking_mode"] = False
    payload.update(cfg.model.get("extra_body") or {})
    return ag._strip_private_blocks(payload), telemetry


def probe(cfg, api_excel, timeout=(30, 600)):
    names = ex.scene_base64_names(cfg)
    api_url = cfg.model.get("api_url", ag.API_URL)
    ag.API_PROVIDER = cfg.model.get("provider", "internai")
    inline = cfg.execution.get("api_keys")
    if inline and not api_excel:
        ag.API_KEYS = ag.load_api_keys_from_list(inline)
        source_sha = ex.sha256_text("\n".join(sorted(ag.API_KEYS)))
    else:
        api_excel = api_excel or ex.resolve_path(
            cfg.execution.get("api_excel", "api.xlsx"))
        ag.API_KEYS = ag.load_api_keys(api_excel)
        source_sha = ex.sha256_file(api_excel)
    if cfg.execution.get("validate_keys", True):
        ag.API_KEYS = ag.validate_api_keys(
            ag.API_KEYS, api_url, cfg.model.get("name"),
            workers=int(cfg.execution.get("key_probe_workers", 16)),
            cache_path=ex.resolve_path(cfg.execution.get(
                "api_key_cache",
                "result/api_keys_valid_%s.json" % ag.API_PROVIDER)),
            source_sha=source_sha,
            min_keys=int(cfg.execution.get("min_valid_keys", 1)))
    group, gid = cfg.group, cfg.group_id
    key = _worst_case_for(cfg, group, names)
    payload, telemetry = build_request(cfg, group, key, names)
    body = json.dumps(payload)
    row = {"group_id": gid, "slug": group.slug,
           "case": key.label(), "images": telemetry["images"]["total"],
           "request_bytes": len(body.encode("utf-8"))}
    t0 = time.time()
    last_err = None
    for key_idx in range(min(8, len(ag.API_KEYS))):
        headers = {"Content-Type": "application/json",
                   "Authorization": "Bearer %s" % ag.API_KEYS[key_idx]}
        try:
            r = requests.post(api_url, headers=headers, data=body, timeout=timeout)
        except Exception as e:
            last_err = "%s: %s" % (type(e).__name__, e)
            continue
        row["http_status"] = r.status_code
        if r.status_code == 200:
            try:
                body_json = r.json()
            except Exception as e:
                last_err = "unparseable body: %s" % e
                continue
            # This endpoint returns errors as HTTP 200 with a
            # code/msg envelope, so a 200 alone proves nothing.
            envelope_err = ag._response_error(body_json)
            if envelope_err is not None:
                last_err = envelope_err
                continue
            usage = (body_json.get("usage") or {})
            row["prompt_tokens"] = usage.get("prompt_tokens")
            row["total_tokens"] = usage.get("total_tokens")
            last_err = None if isinstance(row["prompt_tokens"], int) \
                else "HTTP 200 but no usage reported"
            break
        last_err = r.text[:300]
    row["latency_s"] = round(time.time() - t0, 2)
    if last_err:
        row["error"] = last_err
    status = ("prompt_tokens=%s" % row.get("prompt_tokens")) if isinstance(row.get("prompt_tokens"), int) \
        else "FAILED: %s" % str(row.get("error"))[:120]
    print("[%s] %-26s imgs=%-3d %8d bytes  %6.2fs  %s"
          % (gid, group.slug[:26], row["images"], row["request_bytes"],
             row["latency_s"], status))
    return [row]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment_config", required=True)
    p.add_argument("--api_excel", default=None)
    p.add_argument("--out", default=None, help="write the rows to this JSON file")
    args = p.parse_args()

    cfg = ex.load_config(args.experiment_config)
    api_excel = args.api_excel
    if not api_excel and not cfg.execution.get("api_keys"):
        api_excel = ex.resolve_path(cfg.execution.get("api_excel", "api.xlsx"))

    print("=" * 92)
    print("TOKEN PROBE  %s  (max_tokens=1, one request)" % cfg.experiment_id)
    print("=" * 92)
    rows = probe(cfg, api_excel)

    failed = [r for r in rows if not isinstance(r.get("prompt_tokens"), int)]
    print("-" * 92)
    ok = [r for r in rows if isinstance(r.get("prompt_tokens"), int)]
    if ok:
        widest = max(ok, key=lambda r: r["prompt_tokens"])
        print("largest accepted request: %s at %d prompt tokens"
              % (widest["group_id"], widest["prompt_tokens"]))
    if failed:
        print("REJECTED groups (treat as configuration errors, do not auto-shrink): %s"
              % ", ".join(r["group_id"] for r in failed))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print("wrote %s" % args.out)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
