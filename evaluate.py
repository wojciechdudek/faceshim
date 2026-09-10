"""Score every image Double Take has stored, side by side with what Double Take recorded.

Reads Double Take's SQLite database and match images, sends each image to faceshim and prints,
per image, the stored detector result (name, %) next to faceshim's userid, calibrated confidence
and raw cosine similarity. Ends with per-person quantiles and a suggested SIM_LOW / SIM_HIGH.

Run on the Docker host (shares faceshim's network namespace, so no port or network name needed):

    docker run --rm --network container:faceshim -v /home/double-take:/dt:ro \\
        --entrypoint python ghcr.io/wojciechdudek/faceshim:0.1.4 evaluate.py --storage /dt

Ground truth is whatever Double Take stored from the *other* detectors (e.g. CompreFace): rows
with a match are "known:<name>", rows without one are "unknown". Rows from faceshim itself
(detector "deepstack") are ignored as ground truth.
"""

import argparse
import json
import os
import sqlite3
import statistics
import sys
from pathlib import Path

from selfcheck import post

SELF_DETECTOR = "deepstack"  # faceshim's name inside Double Take


def find_db(storage: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for name in ("database.db",):
        if (storage / name).exists():
            return storage / name
    dbs = sorted(storage.glob("*.db"))
    if not dbs:
        sys.exit(f"no SQLite database under {storage}")
    return dbs[0]


def load_rows(db: Path) -> list[tuple[str, str, str]]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})")]
        if "filename" in cols and "response" in cols:
            when = "createdAt" if "createdAt" in cols else "'?'"
            return list(cur.execute(f"SELECT filename, response, {when} FROM {table} ORDER BY 1"))
    sys.exit(f"no table with filename+response columns in {db} (tables: {tables})")


def walk_results(obj, detector: str | None = None):
    """Yield (detector, name, confidence, match) for every result dict, whatever the nesting."""
    if isinstance(obj, dict):
        detector = obj.get("detector", detector)
        if "name" in obj and "confidence" in obj:
            yield detector, obj["name"], float(obj["confidence"]), bool(obj.get("match", False))
        for v in obj.values():
            yield from walk_results(v, detector)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_results(v, detector)


def truth_from(response: str) -> tuple[str, str]:
    """Best non-faceshim result -> ("known:<name>" | "unknown", "compreface wojtek 99.4")."""
    best = None
    try:
        results = [r for r in walk_results(json.loads(response)) if r[0] != SELF_DETECTOR]
    except (json.JSONDecodeError, TypeError):
        results = []
    for det, name, conf, match in results:
        if best is None or conf > best[2]:
            best = (det, name, conf, match)
    if not best:
        return "unknown", "-"
    det, name, conf, match = best
    label = f"known:{name}" if match else "unknown"
    return label, f"{det} {name} {conf:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage", default="/dt", help="Double Take .storage directory")
    ap.add_argument("--db", default=None, help="SQLite file (default: <storage>/database.db)")
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--limit", type=int, default=300, help="newest N rows")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    import selfcheck
    selfcheck.TIMEOUT = args.timeout

    storage = Path(args.storage)
    rows = load_rows(find_db(storage, args.db))[-args.limit:]
    by_name = {p.name: p for p in storage.rglob("*.jpg") if "train" not in p.parts}
    print(f"{len(rows)} rows, {len(by_name)} images under {storage}\n")

    sims: dict[str, list[float]] = {}
    print(f"{'when':19} {'file':34} {'double take':28} {'faceshim':40}")
    for filename, response, when in rows:
        path = by_name.get(Path(filename).name)
        truth, dt_text = truth_from(response)
        if path is None:
            print(f"{str(when)[:19]:19} {filename[:34]:34} {dt_text:28} (image missing)")
            continue
        r = post(f"{args.url}/v1/vision/face/recognize", {}, (path.name, path.read_bytes()))
        preds = r.get("predictions") or []
        if not preds:
            fs_text = "no face"
        else:
            p = preds[0]  # largest face
            fs_text = f"{p['userid']:10} conf {p['confidence']:.2f} sim {p['similarity']:.3f} ({p['candidate']})"
            sims.setdefault(truth, []).append(p["similarity"])
        print(f"{str(when)[:19]:19} {path.name[:34]:34} {dt_text:28} {fs_text}")

    print("\nraw cosine similarity of faceshim's best candidate, grouped by what Double Take stored:")
    known_all, unknown_all = [], []
    for truth, values in sorted(sims.items()):
        values.sort()
        q = lambda f: values[min(len(values) - 1, int(f * len(values)))]
        print(f"  {truth:18} n={len(values):3}  min {values[0]:.3f}  p25 {q(0.25):.3f}  "
              f"median {statistics.median(values):.3f}  p75 {q(0.75):.3f}  max {values[-1]:.3f}")
        (known_all if truth.startswith("known:") else unknown_all).extend(values)
    if known_all:
        known_all.sort()
        p25 = known_all[min(len(known_all) - 1, len(known_all) // 4)]
        low = round(max(unknown_all) + 0.05, 2) if unknown_all else 0.30
        high = round(p25, 2)
        print(f"\nsuggestion: SIM_LOW={low:.2f} SIM_HIGH={high:.2f}  "
              f"(unknown max {max(unknown_all):.3f} + margin; known p25)" if unknown_all else
              f"\nsuggestion: SIM_HIGH={high:.2f} (known p25); no unknowns stored, keep SIM_LOW well below known min {known_all[0]:.3f}")
        if unknown_all and max(unknown_all) + 0.1 > p25:
            print("WARNING: unknown and known overlap - no threshold separates them cleanly; more/better training images first")
    return 0


if __name__ == "__main__":
    sys.exit(main())
