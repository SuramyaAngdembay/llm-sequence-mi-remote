#!/usr/bin/env python3
"""Checks for the H6 alignment helpers (standard library + numpy; toy digit-splitting tokenizer)."""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from h6_profile_patching import align, char_parts, swap_lines, token_keys  # noqa: E402

def toy_offsets(text):                      # words, single digits, '=', ' '+word, newline: like Qwen's digit splitting
    return [(m.start(), m.end()) for m in re.finditer(r"\n| ?[A-Za-z_]+|\d|=|[.:]", text)]

fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); fails.append(name) if not ok else None

o = "DAY week=3 role=39 dept=5\nPSY O=17 C=2\nSESSIONS total=1 kept=1\nSES idx=0 n=45"
r = swap_lines(o, "DAY week=9 role=7 dept=12", "PSY O=4 C=33")
check("swap keeps the receiver's week and session lines", r.split("\n")[0] == "DAY week=3 role=7 dept=12" and r.split("\n")[2:] == o.split("\n")[2:])
ko, kr = token_keys(o, toy_offsets(o)), token_keys(r, toy_offsets(r))
m = dict(align(ko, kr))
to, tr = [o[a:b] for a, b in toy_offsets(o)], [r[a:b] for a, b in toy_offsets(r)]
sess_o = [i for i, (a, b) in enumerate(toy_offsets(o)) if a >= o.index("SESSIONS")]
check("every session token aligns to an identical token", all(i in {s for s, _ in align(ko, kr)} for i in sess_o)
      and all(to[s] == tr[d] for s, d in align(ko, kr) if s in sess_o))
inv = {d: s for s, d in align(ko, kr)}
role_r = [i for i, t in enumerate(tr) if t == "7"]
check("value digits align from the end (role 39 -> 7: '9' <-> '7')", len(role_r) == 1 and to[inv[role_r[0]]] == "9")
dept_r = [i for i, t in enumerate(tr) if tr[i - 1:i] == ["="] and t == "1" and "dept" in tr[i - 2]]
check("a longer value leaves its extra leading digit unaligned (dept 5 -> 12)", dept_r and dept_r[0] not in inv)
check("keys and '=' align one to one", all(to[inv[i]] == tr[i] for i, t in enumerate(tr) if t in ("=", " role", " dept", " week", "PSY", " O", " C") and i in inv))
cp = char_parts("A b=1\nC")
check("char parts: newline is its own part", cp[5] == [0, -1, "nl"] and cp[4] == [0, 1, "v"] and cp[3] == [0, 1, "k"])
print(f"{len(fails)} failure(s)"); sys.exit(1 if fails else 0)
