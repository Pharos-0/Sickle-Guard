"""Build the verification console: one self-contained HTML page.

    python src/build_console.py

Reads only artefacts the pipeline already wrote, plus the workflow's demo
database, and injects them into src/console_template.html. The page
computes nothing that matters on its own: every figure it shows is copied
from a metrics file, and the inspection queue is the planner's own output
for each budget and reserve, so the console cannot drift from the code the
tests exercise. The one thing it does recompute in the browser is the
audit chain's SHA-256 links, so a viewer can watch tampering being caught.

Writes results/console/verification_console.html.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

import inspection_workflow as iw  # noqa: E402

M = PROJ / "results" / "metrics"
TEMPLATE = PROJ / "src" / "console_template.html"
OUT = PROJ / "results" / "console" / "verification_console.html"
BUDGETS = (0.10, 0.20, 0.33)
RESERVES = (0.0, 0.10, 0.25)


def j(name):
    p = M / name
    if not p.exists():
        raise SystemExit(f"missing {p}; run the script that writes it first")
    return json.loads(p.read_text(encoding="utf8"))


def main() -> int:
    vp = j("verification_policies.json")
    cs = j("inspection_sim.json")
    lr = j("official_records.json")
    yw = j("yestech_whatif.json")
    if not iw.DEMO_OUT.exists():
        iw.demo()
    demo = json.loads(iw.DEMO_OUT.read_text(encoding="utf8"))

    # ---- A and B, observed
    ea = vp["experiment_a"]["held_out"]
    A = {"n": ea["n"], "n_bad": ea["n_bad"], "verdict": ea["verdict"],
         "budgets": {b: {"random_mean": r["random"]["recall_mean"] * ea["n_bad"],
                         "random_p975": r["random"]["recall_p97_5"] * ea["n_bad"],
                         "register": r["register"]["n_found"],
                         "history_only": r["history_only"]["n_found"],
                         "workflow": r["workflow"]["n_found"],
                         "k": r["register"]["n_inspections"]}
                     for b, r in ea["budgets"].items()}}
    b33 = vp["experiment_b"]["all_oof"]["budgets"]["0.33"]
    B = {"none": b33["0.00|satellite"], "ten": b33["0.10|satellite"],
         "proportional": b33["proportional|satellite"],
         "within_cold": b33["_within_cold"]}

    # ---- C, simulated inspector on observed rankings
    def cset(name):
        s = cs["sets"][name]
        return {"n_bad": s["n_bad"],
                "budgets": {b: {"k": r["k"],
                                "random_mean": r["random"]["mean_found"],
                                "register": r["register"]["n_found"],
                                "history_only": r["history_only"]["n_found"],
                                "workflow": r["workflow"]["n_found"],
                                "break_even": r["register"]["break_even_sensitivity"]}
                            for b, r in s["budgets"].items()},
                "lowest_surviving": {b: v["lowest_surviving_sensitivity"]
                                     for b, v in s["monte_carlo_register_vs_random"].items()}}
    C = {"heldout": cset("held-out test split (VERDICT)"),
         "oof": cset("all field-seasons out-of-fold (secondary)"),
         "exposure": cs["observed_exposure"]["held-out test split"]["budgets"],
         "exposure_oof": cs["observed_exposure"]["all field-seasons out-of-fold"]["budgets"]}

    # ---- L, observed
    l1 = lr["l1_level"]["cells"]
    L = {"cells": [{"name": k.replace(" 2019-20", ""), "year": "2019-20",
                    "n": v["n_field_seasons"],
                    "mean": v["by_conversion"]["2/3 (official)"]["sickle_mean_rice_eq_kg_ha"],
                    "ci": v["by_conversion"]["2/3 (official)"]["ci95"],
                    "band": v["official_core_band"],
                    "districts": v["official_core_districts"],
                    "ratio": v["by_conversion"]["2/3 (official)"]["ratio_to_core_weighted"],
                    "verdict": v["verdict"]} for k, v in l1.items()],
         "event": {k: lr["l2_event"][k] for k in
                   ("samba_recalled", "kuruvai_control", "samba_model",
                    "contrast_samba_minus_kuruvai", "recalled_minus_model",
                    "verdict", "exposure")},
         "relief": lr["l2_event"]["relief"]}

    # ---- D, simulated
    cur = yw["conditional_curves"]
    D = {"curves": {}, "guard": {}, "verdict": yw["verdict"]}
    for sd in ("0.10", "0.20", "0.30"):
        D["curves"][sd] = {w: {s: cur[f"w={w}|sd_t={sd}|bias=+0.00|sd_c=0.10"][s]["p_paid_nothing"]
                               for s in ("0.05", "0.10", "0.20", "0.30")}
                           for w in ("0.30", "0.50", "1.00")}
        D["guard"][sd] = {}
        for w in ("0.30", "0.50", "1.00"):
            g = yw["guard"][f"loss=0.20|w={w}|sd_t={sd}|bias=+0.00"]
            D["guard"][sd][w] = {
                "n_wrongly_unpaid": g["n_wrongly_unpaid"],
                "n_wrongly_paid": g["n_wrongly_paid"],
                "budgets": {b: {p: {kk: row["policies"][p][kk] for kk in
                                    ("recovered", "recovered_share",
                                     "rechecks_per_recovery",
                                     "correctly_unpaid_flipped_to_paid")}
                                for p in ("near_threshold", "random",
                                          "lowest_estimate")}
                            for b, row in g["budgets"].items()}}

    # ---- the planner's own queue for every budget and reserve
    feats, Dw = iw.field_table()
    con = iw.connect(":memory:")
    plans = {}
    for b in BUDGETS:
        for r in RESERVES:
            pid = iw.plan(con, feats, Dw["hist"], "test", b, r, seed=0)
            plans[f"{b:.2f}|{r:.2f}"] = [
                {"rank": t["rank"], "uid": t["uid"], "plot": int(t["plot"]),
                 "season": t["season"], "year": str(t["year"]).split(".")[0],
                 "reserve": t["reason"] == iw.REASON_RESERVE}
                for t in con.execute("SELECT * FROM tasks WHERE plan_id=? "
                                     "ORDER BY rank", (pid,))]
    con.close()
    test = [i for i, r in enumerate(feats) if r["split"] == "test"]
    truth = {feats[i]["uid"]: bool(Dw["bad"][i]) for i in test}

    # ---- the demo round's full audit log, for in-browser verification
    dcon = sqlite3.connect(str(iw.DB))
    dcon.row_factory = sqlite3.Row
    audit = [dict(r) for r in dcon.execute("SELECT * FROM audit ORDER BY seq")]
    findings = {r["task_id"]: r["finding"] for r in dcon.execute(
        "SELECT task_id, finding FROM inspections")}
    dcon.close()

    data = {"generated": date.today().isoformat(),
            "split": {"n": A["n"], "n_bad": A["n_bad"], "n_plots": len(
                {feats[i]["plot"] for i in test}),
                "threshold": demo["threshold_kg_per_acre"]},
            "A": A, "B": B, "C": C, "L": L, "D": D,
            "plans": plans, "truth": truth,
            "demo": {"label": demo["label"],
                     "inspector": demo["inspector_model"],
                     "findings": demo["findings"],
                     "against": demo["against_study_threshold"],
                     "sheet": demo["example_task_sheet"],
                     "tasks": [{"rank": t["rank"], "uid": t["uid"],
                                "finding": t["finding"]} for t in demo["tasks"]],
                     "audit": audit}}
    html = TEMPLATE.read_text(encoding="utf8")
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    blob = blob.replace("</", "<\\/")
    if "/*__DATA__*/null" not in html:
        raise SystemExit("template has no data placeholder")
    html = html.replace("/*__DATA__*/null", blob)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf8")
    print(f"wrote {OUT.relative_to(PROJ)} ({len(html) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
