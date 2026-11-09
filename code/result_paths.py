"""Default experiment output layout, shared by runner and standalone agent."""
import os
import re
from datetime import datetime


def experiment_result_dir(cfg, config_path, round_id=0, root=None, stamp=None):
    model = str(cfg.raw.get('ref_model') or cfg.model['name'])
    model = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', model).strip(' .')
    if not model:
        raise ValueError('Model name cannot form a result directory')
    project = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = root if root is not None else os.path.join(project, 'result')
    stem = os.path.splitext(os.path.basename(config_path))[0]
    stamp = stamp or datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    return os.path.join(root, model, stem, 'r%s_%s' % (round_id, stamp))
