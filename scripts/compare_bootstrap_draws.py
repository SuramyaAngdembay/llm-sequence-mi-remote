#!/usr/bin/env python3
"""Compare every cluster-bootstrap interval in two analyzer JSON outputs that
differ only in the number of draws (package-4 declared 5,000; the analyzer
default is 10,000). Reports the largest bound difference, every interval whose
zero-exclusion status differs, and bounds that differ in the third significant
figure.

  python3 scripts/compare_bootstrap_draws.py A.json B.json [...pairs]
"""
import json
import math
import sys


def intervals(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("ci") and isinstance(v, list) and len(v) == 2:
                yield path + (k,), v
            else:
                yield from intervals(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from intervals(v, path + (str(i),))


def sig3(x):
    if x == 0 or not math.isfinite(x):
        return x
    return round(x, 2 - int(math.floor(math.log10(abs(x)))))


def main():
    files = sys.argv[1:]
    for a, b in zip(files[::2], files[1::2]):
        ia, ib = dict(intervals(json.load(open(a)))), dict(intervals(json.load(open(b))))
        assert ia.keys() == ib.keys(), "different interval sets"
        worst, flips, third = 0.0, [], 0
        for k in ia:
            (la, ha), (lb, hb) = ia[k], ib[k]
            if not all(map(math.isfinite, (la, ha, lb, hb))):
                continue
            worst = max(worst, abs(la - lb), abs(ha - hb))
            if (la > 0 or ha < 0) != (lb > 0 or hb < 0):
                flips.append(("/".join(k), (la, ha), (lb, hb)))
            third += (sig3(la) != sig3(lb)) + (sig3(ha) != sig3(hb))
        print(f"{a} vs {b}: {len(ia)} intervals, max bound difference {worst:.2e}, "
              f"bounds differing at 3 s.f.: {third}, zero-exclusion changes: {len(flips)}")
        for f in flips:
            print("   ", f)


if __name__ == "__main__":
    main()
