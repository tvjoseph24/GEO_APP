"""Matplotlib → PNG bytes. Deterministic, embeddable, no interactivity."""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import PercentFormatter   # noqa: E402
from theme import CHART

BRAND = CHART["brand"]
ALT    = CHART["alt"]
GOOD   = CHART["good"]
WARN   = CHART["warn"]
BAD    = CHART["bad"]
GREY   = CHART["grey"]

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.autolayout": True,
})

def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()

def mention_by_intent(rows, target="Brand", title="Unprompted visibility by query type"):
    """Mention rate per intent with Wilson CI error bars."""
    rows = [r for r in rows if r.get("intent")]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r.get("mention_rate", 0))
    labels = [r["intent"] for r in rows]
    vals = [r.get("mention_rate", 0) for r in rows]
    lo = [max(v - (r.get("ci_low") or v), 0) for v, r in zip(vals, rows)]
    hi = [max((r.get("ci_high") or v) - v, 0) for v, r in zip(vals, rows)]
    cols = [GOOD if v >= 0.5 else (WARN if v >= 0.2 else BAD) for v in vals]
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(rows) + 1.4))
    ax.barh(labels, vals, color=cols, height=0.6)
    ax.errorbar(vals, labels, xerr=[lo, hi], fmt="none", ecolor="#333", capsize=3, lw=1)
    for y, (v, r) in enumerate(zip(vals, rows)):
        ax.text(v + 0.02, y, f"{v:.1%}  (n={r.get('runs', 0):,})",
                va="center", fontsize=8)
    ax.set_xlim(0, 1.18)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel(f"Share of runs mentioning {target}")
    ax.set_title(title)
    return _png(fig)

def retrieval_gap(unprompted, in_context, title="The retrieval gap"):
    """The single most important comparison: found vs performs-when-found."""
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    bars = ["Unprompted\n(live retrieval)", "Own content in context\n(experiment control)"]
    vals = [unprompted, in_context]
    ax.bar(bars, vals, color=[BAD, GOOD], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.0%}", ha="center", fontweight="bold")
    gap = (in_context - unprompted) * 100
    ax.annotate("", xy=(1, in_context), xytext=(0, unprompted),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.2))
    ax.text(0.5, (unprompted + in_context) / 2 + 0.04, f"{gap:.0f}pp gap",
            ha="center", fontweight="bold", color="#333")
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("Brand mention rate")
    ax.set_title(title)
    return _png(fig)

def provenance_bar(prov, profile="internal", title="Who the engines cite"):
    keys = [("self_referential_pct", "Brand-owned", BRAND),
            ("corroborated_pct", "Editorial / authority", GOOD),
            ("competitor_owned_pct", "Competitor-owned", ALT),
            ("grey_market_pct", "Unauthorised resellers", BAD),
            ("affiliate_pct", "Affiliate / SEO", WARN),
            ("retailer_pct", "Authorised retail", "#5B8DEF"),
            ("community_pct", "Community", "#9B6BDF"),
            ("unclassified_pct", "Unclassified", GREY)]
    data = [(lbl, prov.get(k) or 0, c) for k, lbl, c in keys if (prov.get(k) or 0) > 0]
    if not data:
        return None
    data.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(data) + 1.3))
    ax.barh([d[0] for d in data], [d[1] for d in data],
            color=[d[2] for d in data], height=0.6)
    for y, d in enumerate(data):
        ax.text(d[1] + 0.005, y, f"{d[1]:.1%}", va="center", fontsize=8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Share of all citations in AI answers")
    ax.set_title(title)
    return _png(fig)

def richness_leaderboard(rows, target, title="Argument density: brand vs competitors"):
    """Reason density per 100 tokens — presence alone does not win recommendations."""
    rows = [r for r in rows if (r.get("spans") or 0) >= 5]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r.get("richness_per_100t") or 0)
    labels = [r["brand"] for r in rows]
    vals = [r.get("richness_per_100t") or 0 for r in rows]
    cols = [BRAND if str(l).lower() == str(target).lower() else GREY for l in labels]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.3))
    ax.barh(labels, vals, color=cols, height=0.62)
    for y, (v, r) in enumerate(zip(vals, rows)):
        ax.text(v + 0.15, y, f"{v:.1f}  ({r.get('spans')} spans)", va="center", fontsize=8)
    ax.set_xlabel("Reason density per 100 tokens")
    ax.set_title(title)
    return _png(fig)

def veracity_stack(per_claim, title="Claim accuracy in machine reality"):
    """Per-engine claim status. Visibility without accuracy is not a win."""
    from collections import Counter, defaultdict
    order = [("locked_in", "Consistently correct", GOOD),
             ("mostly_supported", "Mostly correct", "#7CC49A"),
             ("fragile", "Fragile", WARN),
             ("partial_only", "Present but diluted", "#E8C87D"),
             ("absent", "Absent", GREY),
             ("distorted", "Distorted", BAD)]
    per_eng = defaultdict(Counter)
    for r in per_claim:
        eng = r.get("engine")
        if eng and eng != "*":
            per_eng[eng][r.get("status")] += 1
    if not per_eng:
        return None
    engines = sorted(per_eng)
    fig, ax = plt.subplots(figsize=(6.4, 0.7 * len(engines) + 1.6))
    left = [0] * len(engines)
    for key, lbl, col in order:
        vals = [per_eng[e].get(key, 0) for e in engines]
        if not any(vals):
            continue
        ax.barh(engines, vals, left=left, color=col, label=lbl, height=0.5)
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlabel("Number of substantiated claims")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25),
              frameon=False)
    return _png(fig)

def experiment_forest(rows, title="Experimental results"):
    """Effect with CI per finding. Colour encodes evidence class."""
    if not rows:
        return None
    col = {"PROVEN": GOOD, "PROVEN_ENGINE_SPECIFIC": "#4FA97A",
           "PROVEN_WITH_CAVEAT": WARN, "DISPROVEN": BAD, "INCONCLUSIVE": GREY}
    rows = list(reversed(rows))
    labels = [f"{r['design_id']} · {r.get('label','')}"[:44] for r in rows]
    vals = [r.get("value") or 0 for r in rows]
    cols = [col.get(r.get("evidence_class"), GREY) for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 0.5 * len(rows) + 1.5))
    ax.axvline(0, color="#333", lw=1)
    ax.barh(labels, vals, color=cols, height=0.5)
    for y, r in enumerate(rows):
        v = r.get("value") or 0
        ax.text(v + (0.6 if v >= 0 else -0.6), y,
                f"{v:+.1f}{r.get('unit','pp')}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_xlabel("Measured effect (control → treatment)")
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (GOOD, "#4FA97A", WARN, BAD)]
    ax.legend(handles, ["Proven", "Proven (engine-specific)", "Proven (caveated)",
                        "Disproven — no effect"],
              fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22),
              frameon=False)
    return _png(fig)

def projection(traj, baseline, title="Projected trajectory (illustrative)"):
    if not traj:
        return None
    x = list(range(len(traj) + 1))
    mid = [baseline] + [t["projected_pct"] for t in traj]
    lo = [baseline] + [t["low_pct"] for t in traj]
    hi = [baseline] + [t["high_pct"] for t in traj]
    labels = ["Today"] + [str(t["wave"]).split(" (")[0] for t in traj]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.fill_between(x, lo, hi, color=BRAND, alpha=0.15, label="Range")
    ax.plot(x, mid, "-o", color=BRAND, lw=2, ms=5, label="Projection")
    for xi, v in zip(x, mid):
        ax.text(xi, v + 1.6, f"{v:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Need-query mention rate (%)")
    ax.set_ylim(0, max(hi) * 1.25 + 4)
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    return _png(fig)

def engine_compare(rows, metric="mention_rate", title="Engine comparison"):
    """Grouped bars: engines differ, and the differences are strategy inputs."""
    from collections import defaultdict
    import numpy as np
    by_eng = defaultdict(dict)
    for r in rows:
        if r.get("engine") and r.get("intent"):
            by_eng[r["engine"]][r["intent"]] = r.get(metric, 0)
    if len(by_eng) < 2:
        return None
    engines = sorted(by_eng)
    intents = sorted({i for d in by_eng.values() for i in d})
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    w = 0.8 / len(engines)
    xs = np.arange(len(intents))
    for i, e in enumerate(engines):
        ax.bar(xs + i * w, [by_eng[e].get(t, 0) for t in intents], width=w, label=e,
               color=[BRAND, WARN, GOOD, ALT][i % 4])
    ax.set_xticks(xs + w * (len(engines) - 1) / 2)
    ax.set_xticklabels(intents, fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    return _png(fig)