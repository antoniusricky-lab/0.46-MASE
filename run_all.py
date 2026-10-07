"""Execute every code cell of cinema_v5.ipynb in order, like Jupyter's Run All.

Used to verify the notebook end-to-end before handing it over, so RUN_HEAVY = True
cannot fail on the user's machine for a reason we could have caught here.
"""
import json
import sys
import time
import traceback

nb = json.load(open("cinema_v5.ipynb", encoding="utf-8"))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
env = {"__name__": "__main__"}
failed = []

for i, c in enumerate(cells, 1):
    src = "".join(c["source"])
    head = src.strip().splitlines()[0][:70] if src.strip() else "(empty)"
    print(f"\n{'#' * 78}\n### CELL {i}: {head}\n{'#' * 78}", flush=True)
    t0 = time.time()
    try:
        exec(compile(src, f"CELL{i}", "exec"), env)
        print(f">>> CELL {i} PASS ({time.time() - t0:.0f}s)", flush=True)
    except Exception as exc:
        print(f">>> CELL {i} FAIL: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        failed.append(i)
        break

print(f"\n=== RUN COMPLETE | failed cells: {failed or 'none'} ===", flush=True)
sys.exit(1 if failed else 0)
