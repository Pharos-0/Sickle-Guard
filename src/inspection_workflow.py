"""Step 4, verification, as software: plan, inspect, record, audit.

    python src/inspection_workflow.py plan --budget 0.33 --reserve 0.10
    python src/inspection_workflow.py sheet --task 3
    python src/inspection_workflow.py record --task 3 --finding "no problem observed" --inspector "A. Officer"
    python src/inspection_workflow.py simulate --sensitivity 0.8 --false-alarm 0.1
    python src/inspection_workflow.py verify
    python src/inspection_workflow.py summary
    python src/inspection_workflow.py demo          # fresh database, simulated round

Experiments A-C ranked fields. This module is what an inspection office
would actually run, and each of its rules traces to a measurement:

  1  Fields WITH a harvest record are ranked by their own past harvests.
     (Experiment A: the prioritisation value belongs to the records.)
  2  A stated share of inspections is RESERVED for fields with no record,
     and chosen among them AT RANDOM with a logged seed, not by the
     satellite estimate. (Experiment B: satellite ordering did not beat a
     random order within cold starts; the gain came from looking at all.)
  3  Task sheets are BLIND: the inspector never sees the yield estimate.
     (Experiment C, observed: half the real problems the workflow sends an
     inspector to carry an estimate above the threshold, which an anchored
     inspector could wave through.)
  4  An inspection records an OBSERVATION, never a decision. The three
     findings are "below expectation observed", "no problem observed" and
     "could not assess". Language that decides a claim or certifies damage
     is refused.
  5  A finding of a problem needs evidence: a photograph or a written note.
  6  Every action is appended to a hash-chained audit log. Editing any past
     record, or the log itself, breaks the chain and `verify` says where.
  7  SIMULATED inspections are marked simulated in the record, in the audit
     log and in every summary, so a demonstration can never be mistaken for
     fieldwork.
  8  A field that was not inspected is NOT CLEARED. Summaries list it as not
     inspected, and the farmer's own route to report damage stays open.

Nothing here is connected to PMFBY or decides anything about a claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

DB = PROJ / "results" / "workflow" / "inspections.sqlite"
DEMO_OUT = PROJ / "results" / "workflow" / "demo_summary.json"
GENESIS = "0" * 64

FINDINGS = ("below expectation observed", "no problem observed",
            "could not assess")
POSITIVE = "below expectation observed"
MIN_NOTE = 20

# Decisions belong to the scheme, not to an inspection record.
DECISION_LANGUAGE = [
    r"\bclaim (is )?(approved|rejected|accepted|denied|valid|invalid)\b",
    r"\b(damage|loss) (is )?(confirmed|verified|certified)\b",
    r"\b(not )?eligible\b", r"\bentitled\b", r"\bcrop is (fine|ok|okay|healthy)\b",
    r"\bno (crop )?loss\b", r"\bpayout\b"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    plan_id INTEGER PRIMARY KEY, created_at TEXT, split TEXT, budget REAL,
    reserve REAL, seed INTEGER, n_pool INTEGER, n_tasks INTEGER,
    n_record INTEGER, n_reserve INTEGER, audit_seq INTEGER);
CREATE TABLE IF NOT EXISTS tasks (
    task_id INTEGER PRIMARY KEY, plan_id INTEGER, uid TEXT, plot REAL,
    season TEXT, year TEXT, reason TEXT, rank INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS inspections (
    inspection_id INTEGER PRIMARY KEY, task_id INTEGER UNIQUE,
    inspector TEXT, visited_on TEXT, finding TEXT, note TEXT,
    evidence TEXT, simulated INTEGER, audit_seq INTEGER);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY, ts TEXT, actor TEXT, action TEXT,
    payload TEXT, prev_hash TEXT, hash TEXT);
"""

REASON_RECORD = "has a harvest record: among the lowest past harvests"
REASON_RESERVE = ("no harvest record: part of the reserved share, "
                  "chosen at random")


class WorkflowError(ValueError):
    """A request the workflow refuses, with the rule that refuses it."""


# ------------------------------------------------------------ the database
def connect(path: Path | str = DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _hash(prev: str, ts: str, actor: str, action: str, payload: str) -> str:
    h = hashlib.sha256()
    h.update("|".join((prev, ts, actor, action, payload)).encode("utf8"))
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_audit(con, actor: str, action: str, payload: dict) -> int:
    row = con.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1"
                      ).fetchone()
    prev = row["hash"] if row else GENESIS
    ts = _now()
    body = _canon(payload)
    h = _hash(prev, ts, actor, action, body)
    cur = con.execute("INSERT INTO audit (ts, actor, action, payload, "
                      "prev_hash, hash) VALUES (?,?,?,?,?,?)",
                      (ts, actor, action, body, prev, h))
    return int(cur.lastrowid)


def _inspection_payload(r) -> dict:
    return {"task_id": r["task_id"], "inspector": r["inspector"],
            "visited_on": r["visited_on"], "finding": r["finding"],
            "note": r["note"], "evidence": json.loads(r["evidence"]),
            "simulated": bool(r["simulated"])}


def _task_payload(r) -> dict:
    return {"task_id": r["task_id"], "uid": r["uid"], "reason": r["reason"],
            "rank": r["rank"]}


def verify(con) -> dict:
    """Walk the chain, then check every record against its audit entry."""
    problems = []
    prev = GENESIS
    rows = con.execute("SELECT * FROM audit ORDER BY seq").fetchall()
    expected_seq = 1
    for r in rows:
        if r["seq"] != expected_seq:
            problems.append(f"audit entry {expected_seq} is missing")
            expected_seq = r["seq"]
        if r["prev_hash"] != prev:
            problems.append(f"audit entry {r['seq']} does not follow its "
                            f"predecessor")
        if _hash(r["prev_hash"], r["ts"], r["actor"], r["action"],
                 r["payload"]) != r["hash"]:
            problems.append(f"audit entry {r['seq']} was altered")
        prev = r["hash"]
        expected_seq += 1
    by_seq = {r["seq"]: r for r in rows}
    for r in con.execute("SELECT * FROM inspections").fetchall():
        a = by_seq.get(r["audit_seq"])
        if a is None or json.loads(a["payload"]) != _inspection_payload(r):
            problems.append(f"inspection for task {r['task_id']} differs "
                            f"from its audit entry")
    for p in con.execute("SELECT * FROM plans").fetchall():
        a = by_seq.get(p["audit_seq"])
        tasks = [_task_payload(t) for t in con.execute(
            "SELECT * FROM tasks WHERE plan_id=? ORDER BY rank",
            (p["plan_id"],)).fetchall()]
        if a is None or json.loads(a["payload"]).get("tasks") != tasks:
            problems.append(f"plan {p['plan_id']} task list differs from its "
                            f"audit entry")
    return {"ok": not problems, "n_entries": len(rows), "problems": problems,
            "head": prev}


# ------------------------------------------------------------- the inputs
def field_table():
    """Field-seasons, causal history, and the study's threshold flag."""
    import verification_policies as vp
    D = vp.build()
    import farmer_report as fr
    feats = fr.load()
    return feats, D


# ------------------------------------------------------------------- plan
def plan(con, feats, hist, split="test", budget=0.33, reserve=0.10, seed=0,
         actor="planner"):
    """Choose fields to inspect. Takes no satellite estimate, by design."""
    if not 0 < budget <= 1:
        raise WorkflowError("budget must be in (0, 1]")
    if not 0 <= reserve <= 1:
        raise WorkflowError("reserve must be in [0, 1]")
    pool = [i for i, r in enumerate(feats) if r["split"] == split]
    if not pool:
        raise WorkflowError(f"no field-seasons in split {split!r}")
    has = np.array([np.isfinite(hist[i]) for i in pool])
    n = len(pool)
    k = max(1, int(round(budget * n)))
    k_cold = min(int(round(reserve * k)), int((~has).sum()))
    k_rec = min(k - k_cold, int(has.sum()))
    k_cold = min(k - k_rec, int((~has).sum()))     # spill unused capacity
    rec = [pool[j] for j in np.where(has)[0]]
    rec = sorted(rec, key=lambda i: (hist[i], feats[i]["uid"]))[:k_rec]
    cold = [pool[j] for j in np.where(~has)[0]]
    rng = np.random.default_rng(seed)
    cold = [cold[j] for j in rng.permutation(len(cold))[:k_cold]]

    cur = con.execute(
        "INSERT INTO plans (created_at, split, budget, reserve, seed, n_pool, "
        "n_tasks, n_record, n_reserve) VALUES (?,?,?,?,?,?,?,?,?)",
        (_now(), split, budget, reserve, seed, n, k_rec + k_cold, k_rec,
         k_cold))
    pid = int(cur.lastrowid)
    for rank, (i, why) in enumerate(
            [(i, REASON_RECORD) for i in rec]
            + [(i, REASON_RESERVE) for i in cold], 1):
        r = feats[i]
        con.execute("INSERT INTO tasks (plan_id, uid, plot, season, year, "
                    "reason, rank, status) VALUES (?,?,?,?,?,?,?,?)",
                    (pid, r["uid"], r["plot"], r["season"], r["year"], why,
                     rank, "open"))
    tasks = [_task_payload(t) for t in con.execute(
        "SELECT * FROM tasks WHERE plan_id=? ORDER BY rank", (pid,))]
    seq = append_audit(con, actor, "plan", {
        "plan_id": pid, "split": split, "budget": budget, "reserve": reserve,
        "seed": seed, "n_pool": n, "rule": "records by own past harvests; "
        "reserve chosen at random among fields with no record; no satellite "
        "estimate used", "tasks": tasks})
    con.execute("UPDATE plans SET audit_seq=? WHERE plan_id=?", (seq, pid))
    con.commit()
    return pid


# ------------------------------------------------------------------ sheet
def task_sheet(con, task_id: int) -> str:
    """What the inspector carries. No yield estimate appears on it."""
    t = con.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)
                    ).fetchone()
    if t is None:
        raise WorkflowError(f"no task {task_id}")
    year = str(t["year"]).split(".")[0]
    lines = [
        "=" * 64,
        f"  INSPECTION TASK {t['task_id']}     field {t['plot']:.0f}     "
        f"season {t['season']} {year}",
        "=" * 64,
        f"  Why this field: {t['reason']}.",
        "",
        "  The yield estimate is withheld from this sheet on purpose. Record",
        "  what you see in the field, not what a model expected.",
        "",
        "  Record exactly one finding:",
        "    [ ] below expectation observed   (attach a photo or a note)",
        "    [ ] no problem observed",
        "    [ ] could not assess",
        "",
        "  Write what you saw. Do not write a decision: this record is an",
        "  observation, and decisions on any claim belong to the scheme.",
        "=" * 64]
    return "\n".join(lines)


# ----------------------------------------------------------------- record
def record(con, task_id: int, finding: str, inspector: str,
           visited_on: str | None = None, note: str = "",
           photos: list[str] | None = None, gps: str | None = None,
           simulated: bool = False, actor: str | None = None) -> int:
    t = con.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)
                    ).fetchone()
    if t is None:
        raise WorkflowError(f"no task {task_id}")
    if finding not in FINDINGS:
        raise WorkflowError(f"finding must be one of {FINDINGS}")
    note = note or ""
    photos = list(photos or [])
    if finding == POSITIVE and not photos and len(note.strip()) < MIN_NOTE:
        raise WorkflowError("a finding of a problem needs evidence: a photo "
                            f"or a note of at least {MIN_NOTE} characters")
    for pat in DECISION_LANGUAGE:
        if re.search(pat, note, re.I):
            raise WorkflowError("an inspection records what was observed; "
                                "decisions on a claim belong to the scheme "
                                f"(refused wording matching {pat!r})")
    if not inspector.strip():
        raise WorkflowError("an inspection must name its inspector")
    if con.execute("SELECT 1 FROM inspections WHERE task_id=?", (task_id,)
                   ).fetchone():
        raise WorkflowError(f"task {task_id} already has an inspection; the "
                            "record is append-only")
    evidence = {"photos": photos, "gps": gps}
    row = {"task_id": task_id, "inspector": inspector,
           "visited_on": visited_on or datetime.now().date().isoformat(),
           "finding": finding, "note": note,
           "evidence": json.dumps(evidence, sort_keys=True),
           "simulated": int(bool(simulated))}
    seq = append_audit(con, actor or inspector,
                       "record (SIMULATED)" if simulated else "record",
                       _inspection_payload(row))
    con.execute("INSERT INTO inspections (task_id, inspector, visited_on, "
                "finding, note, evidence, simulated, audit_seq) VALUES "
                "(?,?,?,?,?,?,?,?)",
                (task_id, row["inspector"], row["visited_on"], finding, note,
                 row["evidence"], row["simulated"], seq))
    con.execute("UPDATE tasks SET status='done' WHERE task_id=?", (task_id,))
    con.commit()
    return seq


# --------------------------------------------------------------- simulate
def simulate(con, below: dict, sensitivity: float, false_alarm: float,
             seed: int = 0) -> int:
    """Fill every open task with a SIMULATED inspection.

    `below` maps uid -> whether the field-season is truly below the study
    threshold. The inspector confirms a real problem with probability
    `sensitivity` and reports a false one with probability `false_alarm`.
    """
    rng = np.random.default_rng(seed)
    n = 0
    for t in con.execute("SELECT * FROM tasks WHERE status='open' ORDER BY "
                         "task_id").fetchall():
        truth = bool(below[t["uid"]])
        p = sensitivity if truth else false_alarm
        finding = POSITIVE if rng.random() < p else "no problem observed"
        record(con, t["task_id"], finding, inspector="simulated inspector",
               visited_on="simulated",
               note=(f"SIMULATED inspection: sensitivity {sensitivity:.2f}, "
                     f"false alarm {false_alarm:.2f}, seed {seed}. "
                     f"No field visit took place."),
               simulated=True, actor="simulator")
        n += 1
    return n


# ---------------------------------------------------------------- summary
def summary(con, below: dict | None = None) -> dict:
    p = con.execute("SELECT * FROM plans ORDER BY plan_id DESC LIMIT 1"
                    ).fetchone()
    if p is None:
        return {"plan": None}
    tasks = con.execute("SELECT * FROM tasks WHERE plan_id=?",
                        (p["plan_id"],)).fetchall()
    ins = {r["task_id"]: r for r in con.execute(
        "SELECT * FROM inspections").fetchall()}
    by = {f: 0 for f in FINDINGS}
    n_sim = 0
    for t in tasks:
        r = ins.get(t["task_id"])
        if r is not None:
            by[r["finding"]] += 1
            n_sim += int(r["simulated"])
    out = {"plan": dict(p), "n_tasks": len(tasks),
           "n_inspected": sum(1 for t in tasks if t["task_id"] in ins),
           "n_reserved_for_no_record": int(p["n_reserve"]),
           "findings": by, "n_simulated": n_sim,
           "label": ("SIMULATED inspections -- no field visit took place"
                     if n_sim else "recorded inspections"),
           "n_pool_not_inspected": int(p["n_pool"]) - len(tasks),
           "not_inspected_means": ("NOT INSPECTED. This is not evidence that "
                                   "the crop is fine; farmers can still "
                                   "report damage themselves."),
           "audit": verify(con)}
    if below is not None:
        # evaluation against the study threshold (farmer-recalled yield)
        found = confirmed = false_conf = 0
        for t in tasks:
            truth = bool(below[t["uid"]])
            found += truth
            r = ins.get(t["task_id"])
            if r is not None and r["finding"] == POSITIVE:
                confirmed += truth
                false_conf += (not truth)
        out["against_study_threshold"] = {
            "below_threshold_among_tasks": int(found),
            "confirmed_real": int(confirmed),
            "confirmed_not_below": int(false_conf),
            "note": "below threshold on farmer-recalled yield, not damage"}
    return out


def audit_tail(con, n=6) -> list[dict]:
    rows = con.execute("SELECT seq, ts, actor, action, prev_hash, hash FROM "
                       "audit ORDER BY seq DESC LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows][::-1]


# ------------------------------------------------------------------- demo
def demo(path: Path = DB, budget=0.33, reserve=0.10, sensitivity=0.8,
         false_alarm=0.10, seed=0) -> dict:
    """A fresh database, one plan on the held-out split, one SIMULATED round."""
    import verification_policies as vp
    path = Path(path)
    if path.exists():
        path.unlink()
    feats, D = field_table()
    con = connect(path)
    pid = plan(con, feats, D["hist"], "test", budget, reserve, seed)
    below = {feats[i]["uid"]: bool(D["bad"][i]) for i in range(len(feats))}
    simulate(con, below, sensitivity, false_alarm, seed)
    s = summary(con, below)
    s["audit_tail"] = audit_tail(con)
    first = con.execute("SELECT task_id FROM tasks WHERE plan_id=? ORDER BY "
                        "rank LIMIT 1", (pid,)).fetchone()["task_id"]
    s["example_task_sheet"] = task_sheet(con, first)
    s["tasks"] = [dict(t) | {"finding": (r["finding"] if r else None)}
                  for t in con.execute("SELECT * FROM tasks WHERE plan_id=? "
                                       "ORDER BY rank", (pid,)).fetchall()
                  for r in [con.execute("SELECT finding FROM inspections "
                                        "WHERE task_id=?",
                                        (t["task_id"],)).fetchone()]]
    s["inspector_model"] = {"sensitivity": sensitivity,
                            "false_alarm": false_alarm, "seed": seed,
                            "label": "SIMULATED"}
    s["threshold_kg_per_acre"] = vp.LEVEL * D["bench"]
    con.close()
    DEMO_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEMO_OUT.write_text(json.dumps(s, indent=2, default=str), encoding="utf8")
    return s


# -------------------------------------------------------------------- CLI
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=str(DB))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("plan")
    a.add_argument("--split", default="test")
    a.add_argument("--budget", type=float, default=0.33)
    a.add_argument("--reserve", type=float, default=0.10)
    a.add_argument("--seed", type=int, default=0)
    a = sub.add_parser("sheet")
    a.add_argument("--task", type=int, required=True)
    a = sub.add_parser("record")
    a.add_argument("--task", type=int, required=True)
    a.add_argument("--finding", required=True, choices=FINDINGS)
    a.add_argument("--inspector", required=True)
    a.add_argument("--visited-on")
    a.add_argument("--note", default="")
    a.add_argument("--photo", action="append", default=[])
    a.add_argument("--gps")
    a = sub.add_parser("simulate")
    a.add_argument("--sensitivity", type=float, default=0.8)
    a.add_argument("--false-alarm", type=float, default=0.1)
    a.add_argument("--seed", type=int, default=0)
    sub.add_parser("verify")
    sub.add_parser("summary")
    sub.add_parser("demo")
    args = ap.parse_args()

    if args.cmd == "demo":
        s = demo(Path(args.db))
        print(json.dumps({k: s[k] for k in ("label", "n_tasks", "findings",
                                            "against_study_threshold")},
                         indent=2))
        print(f"audit chain: {'INTACT' if s['audit']['ok'] else 'BROKEN'} "
              f"({s['audit']['n_entries']} entries)")
        print(f"wrote {DEMO_OUT.relative_to(PROJ)}")
        return 0

    con = connect(args.db)
    try:
        if args.cmd == "plan":
            feats, D = field_table()
            pid = plan(con, feats, D["hist"], args.split, args.budget,
                       args.reserve, args.seed)
            for t in con.execute("SELECT * FROM tasks WHERE plan_id=? ORDER "
                                 "BY rank", (pid,)):
                print(f"{t['rank']:>3d}  task {t['task_id']:>4d}  field "
                      f"{t['plot']:>5.0f}  {t['season']:<8s}  {t['reason']}")
            print("Fields not listed are NOT inspected, and NOT cleared.")
        elif args.cmd == "sheet":
            print(task_sheet(con, args.task))
        elif args.cmd == "record":
            record(con, args.task, args.finding, args.inspector,
                   args.visited_on, args.note, args.photo, args.gps)
            print("recorded; audit chain extended")
        elif args.cmd == "simulate":
            feats, D = field_table()
            below = {feats[i]["uid"]: bool(D["bad"][i])
                     for i in range(len(feats))}
            n = simulate(con, below, args.sensitivity, args.false_alarm,
                         args.seed)
            print(f"{n} SIMULATED inspections recorded (no field visits)")
        elif args.cmd == "verify":
            v = verify(con)
            print("audit chain INTACT" if v["ok"] else "audit chain BROKEN")
            for p in v["problems"]:
                print("  -", p)
            return 0 if v["ok"] else 1
        elif args.cmd == "summary":
            print(json.dumps(summary(con), indent=2, default=str))
    except WorkflowError as e:
        print(f"REFUSED: {e}")
        return 2
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
