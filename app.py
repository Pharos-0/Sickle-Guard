"""SICKLE-Guard — farmer-facing field report.

    streamlit run app.py

This is a thin shell over src/farmer_report.py. It computes nothing of its
own: estimates, ranking and the safety wording all come from the same
functions the CLI and the test suite exercise, so the interface cannot
drift from what was validated.

The one design rule is carried through from the CLI: the app never tells a
farmer their crop is fine. There is no green "all clear" state. A field is
either flagged for attention or not flagged, and not-flagged is shown with
an explicit statement that it is not evidence of no loss. The costs are
asymmetric -- a false reassurance can cost a farmer a claim inside a
72-hour window; a false "go and check" costs a phone call.

Nothing here is a government service. No eligibility, no payout, no
Insurance Unit figure, because the data contains none of those.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJ / "src"))

import farmer_report as fr  # noqa: E402

st.set_page_config(page_title="SICKLE-Guard", page_icon="🌾",
                   layout="centered")


@st.cache_resource(show_spinner="Loading fields and fitting the model…")
def bootstrap():
    feats = fr.load()
    hist = fr.causal_history(feats)
    sat = fr.satellite(feats)
    est = np.where(np.isfinite(hist), hist, sat)
    basis = np.where(np.isfinite(hist), "history", "satellite")
    bench = float(np.mean([r["yield"] for r in feats
                           if r["split"] == "train"]))
    return feats, est, basis, bench, fr.contacts()


feats, est, basis, bench, cfg = bootstrap()

st.title("🌾 SICKLE-Guard")
st.caption("Research prototype · not a government service · no official "
           "standing")

with st.sidebar:
    st.header("Field")
    splits = sorted({r["split"] for r in feats})
    split = st.selectbox("Group", splits,
                         index=splits.index("test") if "test" in splits else 0)
    pool = [i for i, r in enumerate(feats) if r["split"] == split]
    order = sorted(pool, key=lambda i: est[i])
    labels = {i: (f"Field {feats[i]['plot']:.0f} · {feats[i]['season']} "
                  f"{feats[i]['year']}") for i in pool}
    pick = st.selectbox("Choose a field-season", order,
                        format_func=lambda i: labels[i])
    budget = st.slider("Verification budget", 0.05, 1.0, 0.33, 0.01,
                       help="Share of fields that can be visited.")
    st.divider()
    st.caption("Evidence behind this tool")
    st.caption("Ranking AUROC 0.904 [0.832, 0.964] on a forward-in-time "
               "split, catching 25 of 28 losses at a 33% budget. That "
               "number belongs to the **history table**: causal history "
               "covered 100% of that test set, so the satellite model "
               "contributed nothing to it.")
    st.caption("Tested separately where it could matter, the satellite "
               "component adds nothing measurable: AUROC 0.824 with it vs "
               "0.838 without, difference −0.014 [−0.058, +0.022].")
    st.caption("Season-specific deterioration: **not** detectable "
               "(AUROC 0.59, interval spans chance).")
    st.caption(":red[**What the ranking actually finds.** 60% of labelled "
               "losses sit at 12 plots that are below threshold in every "
               "season. On plots that are sometimes above and sometimes "
               "below — the only case where the answer is not already in "
               "the record — AUROC falls to 0.626 [0.386, 0.887], which "
               "includes chance. This ranks fields by their usual level; "
               "it does not detect crop damage.]")

k = max(1, int(round(budget * len(pool))))
flagged_set = set(order[:k])
i = pick
r = feats[i]
is_flagged = i in flagged_set
pct = 100 * est[i] / bench

st.subheader(f"Field {r['plot']:.0f} — {r['season']} {r['year']}")

c1, c2, c3 = st.columns(3)
c1.metric("Estimated yield", f"{est[i]:,.0f} kg/acre")
c2.metric("Compared with area", f"{pct:.0f}%")
c3.metric("Rank (lowest first)", f"{order.index(i) + 1} of {len(pool)}")

if basis[i] == "history":
    st.success("**Based on this field's own past harvests.** On test data "
               "that predicted yield far better than the satellite model "
               "(error 200 vs 346 kg/acre).", icon="📜")
else:
    st.warning("**No past harvest on record for this field, so there is "
               "no reliable estimate.** The figure above is a satellite "
               "guess. On test data it was *not shown* to beat simply using "
               "the area average (improvement 14 kg/acre, interval "
               "−30 to +4, which includes zero). Do not act on the number "
               "itself." + chr(10) + chr(10) +
               "This warning means there is **no record** for this field. It "
               "does **not** mean this estimate is worse than the ones "
               "without the warning — tested, the difference in error "
               "is +41 kg/acre, interval −58 to +130, which includes "
               "zero.", icon="🛰️")

if is_flagged:
    st.error("### Flagged for attention\n"
             "This field is among the lower estimates in its area. "
             "**That is a reason to look, not a finding of loss.**\n\n"
             "Most fields flagged this way are ones that yield below "
             "average *most years*. Being flagged is usually a statement "
             "about the land, not about this season.", icon="🔎")
else:
    st.info("### Not flagged by this system\n"
            "**This is not evidence that your crop is fine.** This system "
            "misses losses. If you think your crop was damaged, report it "
            "anyway.", icon="ℹ️")

st.warning("This system **cannot** tell whether this particular season was "
           "bad for this field. That was tested and it did not work.",
           icon="⚠️")

st.divider()
st.subheader("What to do now")
st.markdown("**1. If your crop was damaged, report it now.** Do not wait "
            "for any report, including this one.")

if cfg:
    for d in cfg.get("deadlines", []):
        tag = "" if d.get("verified") else " `UNVERIFIED`"
        st.markdown(f"- {d['applies_to']}: **{d['limit']}**{tag}")

    st.markdown("**2. Contact one of these:**")
    for c in cfg.get("contacts", []):
        if c.get("verified"):
            # A tick alone only says the config claims verification.
            # Print the source so a farmer or official can check it.
            st.markdown(f"- **{c['name']}: {c['value']}**")
            st.caption(f"   ↳ checked {str(c.get('source', ''))[:300]}")
            st.caption("   ↳ a tick in an app proves nothing on its own. "
                       "Confirm against the source above or with your bank "
                       "before relying on it.")
        else:
            st.markdown(f"- {c['name']}: {c['value']} `UNVERIFIED — confirm "
                        f"before relying on it`")
    st.markdown("**3.** Confirm anything marked `UNVERIFIED` with your bank, "
                "the Common Service Centre, or the District Agriculture "
                "Officer.")

    for key in ("never_discourage", "not_official"):
        txt = " ".join(cfg.get("mandatory_notices", {}).get(key, "").split())
        if txt:
            st.caption(txt)

with st.expander(f"Fields to visit first ({k} of {len(pool)})"):
    st.caption("Ranked lowest estimate first. Ranking, not diagnosis.")
    st.dataframe(
        [{"#": n, "Field": f"{feats[j]['plot']:.0f}",
          "Season": feats[j]["season"], "Estimate": f"{est[j]:,.0f}",
          "% of area": f"{100 * est[j] / bench:.0f}%",
          "Based on": ("past harvests" if basis[j] == "history"
                       else "satellite only")}
         for n, j in enumerate(order[:k], 1)],
        hide_index=True, use_container_width=True)
    nh = sum(1 for j in order[:k] if basis[j] == "history")
    st.caption(f"{nh} of {k} use the field's own past harvests; {k - nh} are "
               f"satellite-only and much less reliable. "
               f"**Fields not listed are not cleared.**")
