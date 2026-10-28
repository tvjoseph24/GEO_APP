"""visual.py — what the engines' own imagery shows about the brand. W-57.

Distinct from the creative bench, which tests assets YOU upload. This reads the images the
engines actually surface — often a retailer's photograph of the product rather than the
brand's own asset, shot at an angle where no claim is readable.

Three findings nothing else produces:
  * whether the brand is identifiable from the imagery engines return
  * whether its CLAIMS are legible there, against the substantiated register
  * whose product appears — visual share of voice, which no text metric captures
"""
from __future__ import annotations

from collections import Counter

from metrics_spec import build_registry, fold

def _as_list(v):
    """LLM json is not shape-guaranteed. A bare string iterated character-wise reports
    eighteen claims where there is one."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple, set)):
        return [x for x in v if x is not None and str(x).strip()]
    return [v]

def collect(df):
    """Every successful image read in the corpus, deduplicated by image url."""
    out = {}
    for _, r in df.iterrows():
        for i in (r.get("image_insights") or []):
            if not isinstance(i, dict) or not i.get("fetched") or i.get("error"):
                continue
            u = i.get("url")
            if not u or u in out:
                continue
            out[u] = {**i, "query": r.get("query"), "intent": r.get("intent"),
                      "engine": r.get("engine")}
    return list(out.values())

def visual_report(df, target_brand, competitors=(), claims=None,
                  alias_overrides=None):
    """The bundle. Consumed by metrics_bundle, gap_detect and the report."""
    reads = collect(df)
    pages = int(df["num_visual_pages"].fillna(0).sum()) \
        if "num_visual_pages" in df else 0

    if not reads:
        return {
            "images_read": 0, "visual_pages_returned": pages,
            "verdict": "unmeasured",
            "why": (f"{pages} visual page(s) were returned across the corpus and no "
                    f"readable image was recovered. Visual legibility is UNMEASURED — not "
                    f"confirmed good. Run the visual enrichment pass in the Audit tab."
                    if pages else
                    "No engine returned a visual page, so there is nothing to assess. "
                    "Visual legibility is unmeasured in this corpus."),
        }

    reg = build_registry(target_brand, list(competitors), alias_overrides)
    tpat = reg.target.pattern
    cpats = [(c.canonical, c.pattern) for c in reg.competitors]

    brand_imgs, comp_imgs, other, roles = [], [], [], Counter()
    legibility, unreadable = Counter(), []
    all_claims, all_numbers, all_certs = [], [], []
    comp_seen = Counter()

    for r in reads:
        roles[r.get("image_role") or "unknown"] += 1
        legibility[r.get("text_legibility") or "unknown"] += 1
        _bv = fold(r.get("brand_visible") or "")
        _txt = fold(" ".join(str(x) for x in _as_list(r.get("legible_text"))))
        is_brand = bool(tpat.search(_bv) or tpat.search(_txt))
        named = [n for n, p in cpats
                 if p.search(_bv) or p.search(_txt)
                 or any(p.search(fold(str(o)))
                        for o in _as_list(r.get("other_brands_visible")))]
        for n in named:
            comp_seen[n] += 1
        if is_brand:
            brand_imgs.append(r)
            all_claims += _as_list(r.get("visible_claims"))
            all_numbers += _as_list(r.get("numbers_read"))
            all_certs += _as_list(r.get("certifications_read"))
        elif named:
            comp_imgs.append(r)
        else:
            other.append(r)
        if r.get("what_is_unreadable"):
            unreadable.append({"url": r.get("url"),
                               "brand": r.get("brand_visible"),
                               "what": str(r["what_is_unreadable"])[:220]})

    n = len(reads)
    nb = len(brand_imgs)
    # ---- claim legibility against the register ----
    reg_claims = [c.get("claim") for c in (claims or []) if c.get("claim")]
    legible, absent = [], []
    for c in reg_claims:
        ct = {w for w in fold(c).split() if len(w) > 3}
        if not ct:
            continue
        hit = next((v for v in all_claims
                    if ct and len(ct & {w for w in fold(str(v)).split()
                                        if len(w) > 3}) / len(ct) >= 0.45), None)
        (legible if hit else absent).append(c)

    unread_share = (legibility.get("illegible", 0)
                    + legibility.get("partial", 0)) / max(n, 1)
    if nb == 0:
        verdict, why = "brand_absent_from_imagery", (
            f"{n} image(s) were read and the brand is identifiable in none of them. "
            f"{len(comp_imgs)} show a competitor's product. The engines are surfacing "
            f"imagery of this category that does not include the brand.")
    elif unread_share >= 0.5:
        verdict, why = "illegible", (
            f"The brand appears in {nb} of {n} image(s), but text is illegible or only "
            f"partially legible in {unread_share:.0%} of all images read. Whatever is "
            f"printed on the pack is not reaching a machine reader.")
    elif reg_claims and not legible:
        verdict, why = "no_claim_legible", (
            f"The brand is identifiable in {nb} of {n} image(s), but NONE of the "
            f"{len(reg_claims)} substantiated claims is legible in any of them. The "
            f"imagery establishes recognition and carries no substance.")
    elif legible:
        verdict, why = "claims_legible", (
            f"{len(legible)} of {len(reg_claims)} substantiated claim(s) are legible in "
            f"the imagery the engines return, across {nb} brand image(s).")
    else:
        verdict, why = "recognition_only", (
            f"The brand is identifiable in {nb} of {n} image(s). No claims register was "
            f"supplied, so claim legibility was not assessed.")

    return {
        "images_read": n, "visual_pages_returned": pages,
        "verdict": verdict, "why": why,
        # ---- visual share of voice: no text metric captures this ----
        "brand_images": nb,
        "brand_image_share": round(nb / n, 3),
        "competitor_images": len(comp_imgs),
        "competitor_image_share": round(len(comp_imgs) / n, 3),
        "unattributed_images": len(other),
        "competitor_appearances": dict(comp_seen.most_common()),
        "image_roles": dict(roles),
        "legibility": dict(legibility),
        "unreadable_share": round(unread_share, 3),
        # ---- claim legibility ----
        "claims_in_register": len(reg_claims),
        "claims_legible_on_pack": legible[:20],
        "n_claims_legible": len(legible),
        "claims_not_legible": absent[:20],
        "n_claims_not_legible": len(absent),
        "claims_read_from_imagery": list(dict.fromkeys(all_claims))[:20],
        "numbers_read": list(dict.fromkeys(str(x) for x in all_numbers))[:20],
        "certifications_read": list(dict.fromkeys(str(x) for x in all_certs))[:12],
        "what_is_unreadable": unreadable[:12],
        "sample": [{k: r.get(k) for k in
                    ("url", "page_url", "brand_visible", "product_type", "image_role",
                     "text_legibility", "what_is_unreadable", "engine", "intent")}
                   for r in reads[:15]],
        "note": (
            "Read from the imagery the ENGINES returned — frequently a retailer's "
            "photograph rather than the brand's own asset, so this measures what a "
            "machine can take from the product as it appears in the wild. It is not a "
            "test of the brand's own creative; the creative bench does that."),
    }

def visual_summary(rep):
    if not rep:
        return {}
    return {k: rep.get(k) for k in
            ("verdict", "images_read", "visual_pages_returned", "brand_image_share",
             "competitor_image_share", "unreadable_share", "n_claims_legible",
             "n_claims_not_legible")}