"""End-to-end verification harness for Phase 1 MVP.

Control assertion: for each seed clip, the nearest recommended neighbor
(excluding itself) should belong to the SAME synthetic family, because families
were generated with distinct tempo/pitch/energy/timbre.

Every arm self-reports its identity (per verification discipline): we print the
seed family and the top-1 family so a wrong result is visible, not silent.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from music_recommender.database import repository  # noqa: E402
from music_recommender.recommender import Recommender  # noqa: E402


def family_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def main():
    rec = Recommender()
    rows = repository.all_tracks(rec.conn)
    id2name = {r["track_id"]: Path(r["file_path"]).name for r in rows}
    assert len(rows) >= 8, f"expected >=8 indexed tracks, got {len(rows)}  (run pipeline first)"

    # This harness scores top-1 against the *synthetic family* encoded in the
    # filename (bright_pop_1 -> "bright_pop"), so it is meaningless on a real
    # library. Refuse rather than print a confident wrong number.
    if sum("_" in n for n in id2name.values()) < len(rows) * 0.8:
        print("SKIP: current DB is not the synthetic corpus (family_of() needs "
              "<family>_<n>.wav names). Re-index data/test_music to run this control.")
        sys.exit(0)

    checked = 0
    top1_same_family = 0
    failures = []
    for r in rows:
        tid = r["track_id"]
        seed_name = id2name[tid]
        seed_family = family_of(seed_name)
        recs = rec.similar(tid, limit=3)
        if not recs:
            failures.append(f"{seed_name}: empty recommendations")
            checked += 1
            continue
        top1 = recs[0]
        top1_family = family_of(id2name.get(top1.track_id, "?"))
        checked += 1
        ok = top1_family == seed_family
        top1_same_family += ok
        mark = "OK " if ok else "BAD"
        print(f"[{mark}] seed={seed_name:22s} -> top1={id2name.get(top1.track_id,'?'):22s} "
              f"score={top1.score:.3f}")
        if not ok:
            failures.append(f"{seed_name} matched a {top1_family} clip")

    print("\n==== SUMMARY ====")
    print(f"seeds checked        : {checked}")
    print(f"top1 same-family     : {top1_same_family}/{checked}")
    accuracy = top1_same_family / checked if checked else 0
    print(f"accuracy             : {accuracy:.0%}")
    # Pass bar: at least 80% of clips should rank a same-family clip first.
    passed = accuracy >= 0.8
    print(f"VERDICT              : {'PASS' if passed else 'FAIL'}")
    if failures:
        print("failures:")
        for f in failures:
            print("  -", f)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
