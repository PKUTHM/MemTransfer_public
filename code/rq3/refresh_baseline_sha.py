"""Rewrite baseline_sha256 in every config/rq3/D*.yaml from the file on disk.

The check it refreshes is real: it catches an RQ1 baseline changing under an RQ3
spec.  Run this ONLY after confirming the baseline change was intentional; it
prints every old -> new value so the change is visible in the commit.

It is also needed after a checkout that rewrites line endings, because the
recorded hash is over raw bytes.
"""
import argparse
import re
from pathlib import Path

from .adapter import sha256_text
from .build_bank import ROOT


def refresh(root=ROOT, apply=False):
    changes = []
    for path in sorted((root / 'config/rq3').glob('D*.yaml')):
        text = path.read_text(encoding='utf-8')
        base = re.search(r'^baseline_config: (.+)$', text, re.M).group(1).strip()
        current = re.search(r'^baseline_sha256: (.+)$', text, re.M).group(1).strip()
        actual = sha256_text(root / base)
        if actual == current:
            continue
        changes.append((path.name, base, current, actual))
        if apply:
            path.write_text(re.sub(r'^baseline_sha256: .+$', 'baseline_sha256: ' + actual,
                                   text, count=1, flags=re.M), encoding='utf-8')
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='write the new hashes')
    args = parser.parse_args()
    changes = refresh(apply=args.apply)
    if not changes:
        print('All baseline_sha256 values already match.')
        return
    for name, base, old, new in changes:
        print(f'{name}\n  baseline {base}\n  {old} -> {new}')
    print(('applied' if args.apply else 'dry run; re-run with --apply') + f' ({len(changes)} files)')


if __name__ == '__main__':
    main()
