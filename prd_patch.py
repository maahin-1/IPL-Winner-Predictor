"""
Utility to safely patch ipl_prd.json without hand-editing.
Usage:
  python prd_patch.py --key "open_questions[0].resolution" --value "Resolved: ..."
  python prd_patch.py --key "meta.version" --value "1.1"
"""
import argparse
import json
import re
from pathlib import Path

PRD_PATH = Path(__file__).parent / "ipl_prd.json"


def _set_nested(obj, key_path: str, value):
    """Walk a dotted/indexed key path and set the terminal value."""
    parts = re.split(r"\.(?![^\[]*\])", key_path)
    cur = obj
    for i, part in enumerate(parts[:-1]):
        m = re.match(r"^(\w+)\[(\d+)\]$", part)
        if m:
            cur = cur[m.group(1)][int(m.group(2))]
        else:
            cur = cur[part]
    last = parts[-1]
    m = re.match(r"^(\w+)\[(\d+)\]$", last)
    if m:
        cur[m.group(1)][int(m.group(2))] = value
    else:
        cur[last] = value


def main():
    parser = argparse.ArgumentParser(description="Patch ipl_prd.json")
    parser.add_argument("--key", required=True, help="Dotted key path, e.g. meta.version")
    parser.add_argument("--value", required=True, help="New value (string)")
    args = parser.parse_args()

    prd = json.loads(PRD_PATH.read_text(encoding="utf-8"))
    _set_nested(prd, args.key, args.value)
    PRD_PATH.write_text(json.dumps(prd, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Patched: {args.key} = {args.value!r}")


if __name__ == "__main__":
    main()
