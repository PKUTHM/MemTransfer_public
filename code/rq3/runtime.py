"""Independent RQ3 entry point. No edits to code/agent.py or code/server.py.

Preflight only loads/validates memory. The explicit ``serve``/``agent`` modes
run the existing RQ1 server/agent with a process-local RQ3 memory adapter.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from .build_bank import ROOT
from .cross_case_memory import config_from_rq3, imports, install_agent, install_server, preload


def default_results_dir(config_path: Path) -> Path:
    """Keep RQ3 runs beside the existing Intern-S2 experiment results."""
    return ROOT / 'result_Intern-S2-Preview-397B' / config_path.stem


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True, type=Path,
                        help='One of config/rq3/D01...D04 YAML files')
    parser.add_argument('--mode', choices=('preflight', 'serve', 'agent'), default='preflight')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--api-key-limit', type=int,
                        help='Agent mode: use at most this many validated API keys')
    args, extra = parser.parse_known_args()
    path = (ROOT / args.config).resolve()
    if args.mode == 'preflight':
        if extra:
            parser.error(f'Unexpected preflight arguments: {extra}')
        cfg = config_from_rq3(path)
        history, manifest = preload(cfg, cfg.group)
        print(json.dumps({'experiment_id': cfg.experiment_id,
                          'fingerprint': cfg.fingerprint(),
                          'cases': len(history),
                          'trajectories': len(manifest['trajectories']),
                          'replacement_bank_sha256': manifest['rq3_replacement_bank_sha256'],
                          'sample_case_0': [{'slot': t['_rq3_slot'], 'role': t['_rq3_role'],
                                             'episode_id': t['_rq3_episode_id']}
                                            for t in history[0]]}, ensure_ascii=False, indent=2))
        return
    if args.mode == 'serve':
        if extra:
            parser.error(f'Unexpected server arguments: {extra}')
        install_server()
        import server
        out = args.results_dir or default_results_dir(path)
        server.LOG_DIR = str((out / 'server_logs').resolve())
        server.load_experiment(str(path))
        server.load_environment()
        server.app.run(host=args.host, port=args.port, debug=False, threaded=True)
        return
    if extra and extra[0] == '--':
        extra = extra[1:]
    imports()
    import agent
    install_agent(agent)
    if args.api_key_limit is not None:
        if args.api_key_limit < 1:
            parser.error('--api-key-limit must be positive')
        original_validate = agent.validate_api_keys

        def limited_validate(*a, **kw):
            valid = original_validate(*a, **kw)
            if len(valid) < args.api_key_limit:
                raise SystemExit(f'Only {len(valid)} validated API keys; requested {args.api_key_limit}')
            # Stable selection despite the RQ1 loader shuffling its input.
            ordered = sorted(valid, key=lambda item: hashlib.sha256(
                (item.key if isinstance(item, agent.Endpoint) else item).encode('utf-8')).digest())
            print(f'[rq3] using exactly {args.api_key_limit} validated API keys')
            return ordered[:args.api_key_limit]

        agent.validate_api_keys = limited_validate
    out = args.results_dir or default_results_dir(path)
    agent_args = ['--experiment_config', str(path), '--results_dir', str(out.resolve()),
                  '--server_url', f'http://{args.host}:{args.port}'] + extra
    old_argv = sys.argv
    try:
        sys.argv = ['agent.py'] + agent_args
        agent.main()
    finally:
        sys.argv = old_argv


if __name__ == '__main__':
    main()
