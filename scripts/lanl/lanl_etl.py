#!/usr/bin/env python3
"""
LANL-2015 (cyber1) auth.txt -> per-user-day session serialization for the
identity-shortcut pilot. Streaming, memory-bounded, with a validate-first mode.

Modes:
  inspect : stream a sample, validate field parsing, report distributions.
            No output written; just sanity numbers + example serializations.
  build   : stream (optionally subsampled by user), group into (src_user, day)
            sessions, join red-team labels, write JSONL. Also emits identity
            ablation variants for a small check when --emit-ablations.

Design choices (documented so they are auditable):
  - session unit         = (src_user, day)  where day = time // 86400
  - modeled principals   = HUMAN users only (src_user matches ^U[0-9]+),
                           machine (C###$) / ANONYMOUS sessions are skipped as
                           MODELING UNITS but still appear as dst context inside
                           human sessions.
  - label                = a session is positive iff it contains >=1 auth event
                           whose (time, src_user, src_comp, dst_comp) is in
                           redteam.txt.
  - user subsample       = keep user iff (md5(user) % sample_mod == 0) OR user is
                           a red-team user (red-team users are ALWAYS kept).
"""
import gzip, sys, argparse, hashlib, json, os
from collections import Counter, defaultdict

SECONDS_PER_DAY = 86400

def uid(field):
    """'U620@DOM1' -> 'U620' ; 'C586$@DOM1' -> 'C586$' ; 'ANONYMOUS LOGON@C586' -> 'ANONYMOUS LOGON'"""
    return field.split('@', 1)[0]

def is_human(user_field):
    u = uid(user_field)
    return len(u) > 1 and u[0] == 'U' and u[1:].isdigit()

def load_redteam(red_path):
    red_events = set()   # (time, user_full, src_comp, dst_comp) — exact label-join key
    red_user_ids = set() # stripped uids (e.g. 'U620') — for force-keep in subsample
    n = 0
    with gzip.open(red_path, 'rt') as f:
        for line in f:
            p = line.rstrip('\n').split(',')
            if len(p) != 4:
                continue
            t, u, sc, dc = p
            red_events.add((t, u, sc, dc))
            red_user_ids.add(uid(u))
            n += 1
    return red_events, red_user_ids, n

def serialize_event(fields, mode='full', user_map=None):
    """fields = 9-tuple from auth.txt. mode in full|user_anon|host_anon|shuffle."""
    t, su, du, sc, dc, at, lt, orient, res = fields
    hh = (int(t) % SECONDS_PER_DAY) // 3600
    if mode == 'user_anon':
        su, du = 'USER', 'USER'
    elif mode == 'host_anon':
        sc, dc = 'HOST', 'HOST'
    elif mode == 'shuffle' and user_map is not None:
        su = user_map.get(uid(su), uid(su))
        du = user_map.get(uid(du), uid(du))
    return f"t{hh:02d} su={uid(su)} du={uid(du)} sc={sc} dc={dc} at={at} lt={lt} or={orient} res={res}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--auth', default='auth.txt.gz')
    ap.add_argument('--red', default='redteam.txt.gz')
    ap.add_argument('--mode', choices=['inspect', 'build', 'build-windows'], default='inspect')
    ap.add_argument('--window-events', type=int, default=32,
                    help='build-windows: events per window within a user-day')
    ap.add_argument('--limit', type=int, default=0, help='max auth lines to read (0=all)')
    ap.add_argument('--sample-mod', type=int, default=1, help='keep user if md5%%mod==0 (1=all users)')
    ap.add_argument('--out', default='sessions.jsonl')
    ap.add_argument('--emit-ablations', action='store_true')
    args = ap.parse_args()

    red_events, red_user_ids, n_red = load_redteam(args.red)
    sys.stderr.write(f"[redteam] events={n_red} unique_users={len(red_user_ids)}\n")

    # counters for validation
    nf_counter = Counter()
    bad_lines = 0
    prefix = Counter()
    at_c, lt_c, or_c, res_c = Counter(), Counter(), Counter(), Counter()
    day_min, day_max = 10**9, -1
    n_read = 0
    n_kept_events = 0

    # build-mode accumulation. auth.txt is time-ordered, so once the stream
    # advances past day d, every (user, d) session is complete -> flush to disk.
    # Memory is bounded by ONE day of kept events, not the whole corpus.
    sessions = defaultdict(list)   # (user, day) -> [event fields], current day only
    kept_users = set()
    cur_day = None
    out_f = None
    stats = {"n_sessions": 0, "n_pos": 0, "red_hit": 0}
    umap = {}  # filled lazily on first flush if --emit-ablations

    def flush_day_windows(day_sessions):
        """build-windows: chunk each user-day into W-event windows; label each
        window by whether IT contains a red-team event (so truncation can never
        drop a positive), emit all 4 serialization variants per window."""
        nonlocal umap
        W = args.window_events
        if args.emit_ablations and not umap:
            us = sorted(kept_users)
            rot = us[1:] + us[:1] if len(us) > 1 else us
            umap = dict(zip(us, rot))
        for (user, day), evs in day_sessions.items():
            evs.sort(key=lambda e: int(e[0]))
            for wi in range(0, len(evs), W):
                chunk = evs[wi:wi + W]
                label = 0
                for e in chunk:
                    if (e[0], e[1], e[3], e[4]) in red_events:
                        label = 1; stats["red_hit"] += 1
                rec = {
                    "example_id": f"{user}:{day}:{wi // W}",
                    "user_id": user, "day_index": day, "window": wi // W,
                    "split": "eval", "y": label, "n_events": len(chunk),
                    "text": " | ".join(serialize_event(e, 'full') for e in chunk),
                }
                if args.emit_ablations:
                    rec["text_user_anon"] = " | ".join(serialize_event(e, 'user_anon') for e in chunk)
                    rec["text_host_anon"] = " | ".join(serialize_event(e, 'host_anon') for e in chunk)
                    rec["text_shuffle"] = " | ".join(serialize_event(e, 'shuffle', umap) for e in chunk)
                out_f.write(json.dumps(rec) + "\n")
                stats["n_sessions"] += 1
                if label: stats["n_pos"] += 1

    def flush_day(day_sessions):
        nonlocal umap
        if args.emit_ablations and not umap:
            # deterministic rotation map over users seen so far (stable enough for
            # the ablation check; the GPU-stage shuffle uses its own seeded map)
            us = sorted(kept_users)
            rot = us[1:] + us[:1] if len(us) > 1 else us
            umap = dict(zip(us, rot))
        for (user, day), evs in day_sessions.items():
            evs.sort(key=lambda e: int(e[0]))
            label = 0
            for e in evs:
                if (e[0], e[1], e[3], e[4]) in red_events:
                    label = 1; stats["red_hit"] += 1
            text = " | ".join(serialize_event(e, 'full') for e in evs)
            rec = {"user": user, "day": day, "n_events": len(evs), "label": label, "text": text}
            if args.emit_ablations:
                rec["text_user_anon"] = " | ".join(serialize_event(e, 'user_anon') for e in evs)
                rec["text_host_anon"] = " | ".join(serialize_event(e, 'host_anon') for e in evs)
                rec["text_shuffle"] = " | ".join(serialize_event(e, 'shuffle', umap) for e in evs)
            out_f.write(json.dumps(rec) + "\n")
            stats["n_sessions"] += 1
            if label: stats["n_pos"] += 1

    def keep_user(su_field):
        if not is_human(su_field):
            return False
        u = uid(su_field)
        if u in red_user_ids:  # red-team users always kept
            return True
        if args.sample_mod <= 1:
            return True
        return (int(hashlib.md5(u.encode()).hexdigest(), 16) % args.sample_mod) == 0

    example_serials = []
    with gzip.open(args.auth, 'rt') as f:
        for line in f:
            n_read += 1
            if args.limit and n_read > args.limit:
                break
            p = line.rstrip('\n').split(',')
            nf_counter[len(p)] += 1
            if len(p) != 9:
                bad_lines += 1
                continue
            t, su, du, sc, dc, at, lt, orient, res = p
            # validate time is int
            if not t.isdigit():
                bad_lines += 1
                continue
            day = int(t) // SECONDS_PER_DAY
            day_min = min(day_min, day); day_max = max(day_max, day)
            # distributions (from the human-src rows to reflect modeled pop)
            if su.startswith('U') and uid(su)[1:].isdigit():
                prefix['human'] += 1
            elif su.startswith('ANONYMOUS'):
                prefix['anon'] += 1
            elif '$@' in su or su.split('@')[0].endswith('$'):
                prefix['machine'] += 1
            else:
                prefix['other'] += 1
            at_c[at] += 1; lt_c[lt] += 1; or_c[orient] += 1; res_c[res] += 1

            if args.mode in ('build', 'build-windows') and keep_user(su):
                if out_f is None:
                    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
                    out_f = open(args.out, 'w')
                if cur_day is not None and day != cur_day:
                    (flush_day if args.mode == 'build' else flush_day_windows)(sessions)
                    sessions = defaultdict(list)
                cur_day = day
                sessions[(uid(su), day)].append((t, su, du, sc, dc, at, lt, orient, res))
                kept_users.add(uid(su))
                n_kept_events += 1
            elif args.mode == 'inspect' and len(example_serials) < 5 and is_human(su):
                example_serials.append(serialize_event(p, 'full'))

    # ---- report ----
    print("=== VALIDATION REPORT ===")
    print("auth_lines_read:", n_read - (1 if args.limit and n_read > args.limit else 0))
    print("field_count_dist:", dict(nf_counter))
    print("bad_lines(!=9 or bad time):", bad_lines)
    print("src_user_prefix_dist:", dict(prefix))
    print("day_range:", day_min, "->", day_max)
    print("top_auth_type:", at_c.most_common(6))
    print("top_logon_type:", lt_c.most_common(6))
    print("orientation:", or_c.most_common(8))
    print("success/failure:", res_c.most_common())
    if example_serials:
        print("--- example serialized human events (full) ---")
        for s in example_serials:
            print("  ", s)

    if args.mode in ('build', 'build-windows'):
        if sessions:
            (flush_day if args.mode == 'build' else flush_day_windows)(sessions)
        if out_f is not None:
            out_f.close()
        print("=== BUILD REPORT ===")
        print("kept_users:", len(kept_users))
        print("kept_events:", n_kept_events)
        print("sessions(user-days):", stats["n_sessions"])
        print("positive_sessions:", stats["n_pos"])
        print("redteam_event_hits:", stats["red_hit"], "of", n_red, "(within read window)")
        print("out:", args.out)

if __name__ == '__main__':
    main()
