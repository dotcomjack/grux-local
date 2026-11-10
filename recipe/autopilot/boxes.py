#!/usr/bin/env python3
"""The lane's Daytona boxes: list them (the after-run check) or delete one model's box (the kill-resume proof).

  <venv>/bin/python recipe/autopilot/boxes.py list
  <venv>/bin/python recipe/autopilot/boxes.py kill <ollama tag>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from daytona import Daytona, ListSandboxesQuery  # noqa: E402


def main():
    d = Daytona()
    mine = d.list(ListSandboxesQuery(labels={"dcj.owner": common.OWNER}))
    items = list(getattr(mine, "items", mine))
    if sys.argv[1:2] == ["list"]:
        print(f"dcj.owner={common.OWNER} boxes: {len(items)}")
        for sb in items:
            m = sb.model_dump()
            print(f"  {m.get('id')} {m.get('state')} {m.get('labels', {}).get('grux.factory')} {m.get('gpu_type')}")
        return
    if sys.argv[1:2] == ["kill"] and len(sys.argv) == 3:
        want = "gl-" + common.stem(sys.argv[2])
        hit = [sb for sb in items if sb.model_dump().get("labels", {}).get("grux.factory") == want]
        for sb in hit:
            d.delete(sb, wait=False, timeout=60)
            print(f"deleted {sb.id} ({want})")
        if not hit:
            print(f"no box labelled {want}")
            sys.exit(1)
        return
    print(__doc__)
    sys.exit(2)


if __name__ == "__main__":
    main()
