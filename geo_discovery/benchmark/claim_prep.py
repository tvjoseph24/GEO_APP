"""Normalise the Claims Register before veracity.

W-08 consolidates five defects:
  * near-duplicates survived — symmetric Jaccard at 0.70 missed paraphrase pairs where
    one entry carried analytical preamble (proven case: two register entries for the
    same dual-capability claim scored 0.39)
  * analyst commentary was treated as a claim ("X — publicly invoked via the Louvre
    exhibition, used as evidence for...") and generated high-severity gaps
  * advertising copy was treated as a claim ("protect against everything but admiring
    glances") and could reach a funded experiment
  * long multi-fact paragraphs were dropped as unjudgeable instead of being SPLIT
  * dropped claims carried no reason, so the register could not be improved

Claims are now classified three ways rather than kept/dropped:
  factual      -> veracity judging
  brand_voice  -> positioning evidence, not a testable fact
  analyst_note -> excluded entirely
"""
from __future__ import annotations

import re

from metrics_spec import brand_stop, fold

_META = re.compile(
    r"^\s*(?:the\s+brand\s+|it\s+)?(?:claims?|states?|says?|markets?|advertises?|"
    r"positions?|promotes?)\s+(?:that\s+|its\s+|it\s+as\s+|itself\s+as\s+)?", re.I)
_TRAIL = re.compile(r"\bas\s+(?:supporting|providing|offering|helping)\b", re.I)

# Meta-analytical framing — an observation ABOUT claims, not a claim.
_ANALYST = re.compile(
    r"\b(?:used as evidence|publicly invoked|cluster(?:s|ed)? around|recurring theme|"
    r"positioning pillar|claim to (?:luxury|premium|authority)|"
    r"invoked via|serves as (?:evidence|proof)|(?:this|the) (?:suggests|indicates) that|"
    r"public claims|brand narrative|messaging strateg)", re.I)

# Anchors that make a statement checkable against something.
_VERIFY_WORDS = re.compile(
    r"\b(?:certified|certification|approved|approval|tested|clinically|dermatologist|"
    r"registered|compliant|accredited|standard|regulation|patent|award|"
    r"contains|includes|free from|without|made with|derived from|"
    r"recommended by|endorsed by|rated)\b", re.I)

STOP = set("the a an and or of for with to in on is are be its their this that from as "
           "by it has have had was were will can may also more most than then them "
           "these such other others some".split())

def canonical_fact(claim: str, brand: str = "") -> str:
    """Strip brand name and meta-framing so the same fact compares equal regardless of
    how the source worded the act of claiming it."""
    t = str(claim or "").strip()
    if brand:
        t = re.sub(r"(?<![a-z0-9])" + re.escape(brand) + r"(?:'s)?(?![a-z0-9])", "",
                   t, flags=re.I)
    t = _META.sub("", t)
    t = _TRAIL.sub("", t)
    return re.sub(r"\s+", " ", t).strip(" .,-'")

def _tokens(s, stop):
    """Discriminative tokens for duplicate detection.

    Numerics are included deliberately. Without them "contains 1% oatmeal" and
    "contains 5% oatmeal" have identical token sets and MERGE — silently collapsing two
    different concentrations into one claim.
    """
    low = fold(s)
    words = {w for w in re.findall(r"[a-z][a-z\-]{2,}", low) if w not in stop}
    nums = {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)?", low)}
    return words | nums



def _proper_nouns(raw: str, brand: str = "") -> list:
    """Capitalised or all-caps tokens, excluding the brand and the first word."""
    bf = fold(brand)
    out = []
    for i, w in enumerate(str(raw or "").split()):
        t = w.strip(".,;:()\"'—–-")
        if not t or i == 0:
            continue
        if t.isupper() and len(t) >= 2:
            out.append(t)
        elif t[0].isupper() and len(t) > 2 and fold(t) not in bf:
            out.append(t)
    return out

def classify_claim(raw: str, brand: str = "", min_tokens=4) -> tuple[str, str]:
    """Returns (kind, reason). kind ∈ factual | brand_voice | analyst_note | too_short."""
    text = str(raw or "").strip()
    if _ANALYST.search(text):
        return "analyst_note", "meta-analytical commentary about claims, not a claim"
    fact = canonical_fact(text, brand)
    if len(fact.split()) < min_tokens:
        return "too_short", f"only {len(fact.split())} content word(s) after normalisation"

    has_number = bool(re.search(r"\d", fact))
    has_verify = bool(_VERIFY_WORDS.search(fact))
    has_entity = bool(_proper_nouns(text, brand))
    if has_number or has_verify or has_entity:
        return "factual", ""
    return "brand_voice", ("no verifiable anchor — no quantity, named component, "
                           "certification or comparison to check against")

def split_multiclaim(text: str, max_words=25, min_part_words=4) -> list:
    """A long register entry is usually MULTI-claim, not unjudgeable. Splitting recovers
    claims that were previously discarded wholesale."""
    t = str(text or "").strip()
    if len(t.split()) <= max_words:
        return [t]
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    parts = [p for p in parts if len(p.split()) >= min_part_words]
    if len(parts) > 1:
        return parts
    parts = [p.strip() for p in t.split(";") if len(p.split()) >= min_part_words]
    return parts if len(parts) > 1 else [t]

def _contained(small: set, large: set, thresh=0.80) -> bool:
    """Asymmetric containment. The symmetric Jaccard test missed pairs where one entry
    carried extra analytical wording, inflating its token count."""
    return bool(small) and bool(large) and len(small & large) / len(small) >= thresh

_NUMS = re.compile(r"\d+(?:[.,]\d+)?")

def _numeric_conflict(a: str, b: str) -> bool:
    """Two claims stating DIFFERENT numbers are never the same claim, however similar
    their wording. Blocks '1% oatmeal' merging with '5% oatmeal'."""
    na = {m.group(0).replace(",", "") for m in _NUMS.finditer(fold(a))}
    nb = {m.group(0).replace(",", "") for m in _NUMS.finditer(fold(b))}
    if not na or not nb:
        return False
    return bool(na ^ nb)          # any value present in one but not the other

def prepare_claims(claims, brand="", category="", competitors=(),
                   min_tokens=4, jaccard=0.7, contain=0.80, stop=None,
                   split_long=True, keep_brand_voice=False):
    """Returns (kept, report). Merged claims carry `_merged_from` for traceability;
    excluded claims are reported with a reason so the register can be improved."""
    stop = stop or brand_stop(brand, competitors, category)
    kept, excluded, merged, split_log = [], [], [], []

    # ---- 1. expand multi-claim paragraphs ----
    expanded = []
    for c in (claims or []):
        raw = (c.get("claim") or "").strip()
        if not raw:
            continue
        pieces = split_multiclaim(raw) if split_long else [raw]
        if len(pieces) > 1:
            split_log.append({"original": raw[:140], "into": len(pieces)})
            for i, p in enumerate(pieces):
                d = dict(c)
                d["claim"] = p
                d["_split_from"] = raw[:140]
                d["claim_id"] = f"{c.get('claim_id', 'CL')}-{i + 1}"
                expanded.append(d)
        else:
            expanded.append(dict(c))

    # ---- 2. classify ----
    candidates = []
    for c in expanded:
        raw = c["claim"]
        kind, reason = classify_claim(raw, brand, min_tokens)
        c["_claim_kind"] = kind
        if kind == "factual" or (kind == "brand_voice" and keep_brand_voice):
            c["_canonical_fact"] = canonical_fact(raw, brand)
            candidates.append(c)
        else:
            excluded.append({"claim": raw[:200], "kind": kind, "why": reason})

    # ---- 3. merge duplicates: symmetric Jaccard OR asymmetric containment ----
    for c in candidates:
        toks = _tokens(c["_canonical_fact"], stop)
        if len(toks) < 2:
            excluded.append({"claim": c["claim"][:200], "kind": "too_short",
                             "why": "fewer than 2 discriminative tokens"})
            continue
        hit = None
        for k in kept:
            a, b = k["_tokens"], toks
            if not a or not b:
                continue
            if _numeric_conflict(k["_canonical_fact"], c["_canonical_fact"]):
                continue  # different numbers — never the same claim
            jac = len(a & b) / len(a | b)
            if jac >= jaccard or _contained(b, a, contain) or _contained(a, b, contain):
                hit = k
                break
        if hit is not None:
            hit.setdefault("_merged_from", []).append(c["claim"])
            merged.append({"kept": hit["claim"][:120], "merged": c["claim"][:120],
                           "basis": "containment" if len(toks) != len(hit["_tokens"])
                                    else "jaccard"})
            # keep the richer wording, but prefer the one WITHOUT analytical preamble
            if len(c["_canonical_fact"]) > len(hit["_canonical_fact"]) \
                    and not _ANALYST.search(c["claim"]):
                hit["claim"] = c["claim"]
                hit["_canonical_fact"] = c["_canonical_fact"]
                hit["_tokens"] = toks
            continue
        c["_tokens"] = toks
        kept.append(c)

    for k in kept:
        k.pop("_tokens", None)

    from collections import Counter
    return kept, {
        "input": len(claims or []),
        "after_split": len(expanded),
        "kept": len(kept),
        "n_split": len(split_log), "split": split_log[:10],
        "n_merged": len(merged), "merged": merged[:20],
        "n_dropped": len(excluded), "dropped": excluded[:30],
        "dropped_by_kind": dict(Counter(e["kind"] for e in excluded)),
        "kept_by_kind": dict(Counter(k.get("_claim_kind") for k in kept)),
    }