"""Tier B — Diagnostic: veracity (hybrid), reason-richness, provenance, consistency."""
import re, json, unicodedata
from collections import Counter

def _norm(t):
    if not t: return ""
    return unicodedata.normalize("NFKC", str(t)).lower()

# ---------- VERACITY (hybrid) ----------
VERACITY_PROMPT = """Compare what an AI said about "{brand}" against its VERIFIED claims.
For each verified claim, judge if the AI response is: correct / missing / contradicted.
Return STRICT JSON list mirroring the claims:
[{{"claim":"", "verdict":"correct|missing|contradicted", "evidence":""}}]

VERIFIED CLAIMS:
{claims}

AI RESPONSES (concatenated):
{responses}"""

def veracity_keyword(df, claims, target_brand):
    """Free: does the corpus mention each registered claim?"""
    corpus = _norm(" ".join(df["response_text"].tolist()))
    out = []
    for c in claims:
        claim_txt = c.get("claim", "")
        # crude keyword: check key nouns of the claim appear
        keywords = [w for w in _norm(claim_txt).split() if len(w) > 4][:3]
        present = all(k in corpus for k in keywords) if keywords else False
        out.append({"claim": claim_txt, "verdict": "correct" if present else "missing",
                    "method": "keyword"})
    return out

def veracity_llm(gateway, df, claims, target_brand, judge_model, module="benchmark"):
    """Opt-in: LLM judges correct/missing/contradicted."""
    sample = "\n---\n".join(df["response_text"].head(15).tolist())[:8000]
    claims_txt = json.dumps([c.get("claim") for c in claims])[:4000]
    r = gateway.generate(judge_model,
        VERACITY_PROMPT.format(brand=target_brand, claims=claims_txt, responses=sample),
        json_mode=True, module=module, task="veracity")
    res = r["parsed"] if isinstance(r["parsed"], list) else []
    for x in res: x["method"] = "llm"
    return res or veracity_keyword(df, claims, target_brand)

def run_veracity(df, claims, target_brand, *, use_llm=False, gateway=None,
                 judge_model="gemini-2.5-flash"):
    if use_llm and gateway:
        try: return veracity_llm(gateway, df, claims, target_brand, judge_model)
        except Exception: pass
    return veracity_keyword(df, claims, target_brand)

# ---------- REASON-RICHNESS (reuse) ----------
QUANT = [r"\b\d+%\b", r"\b\d+[- ]?hour", r"\btriple\b", r"\bceramide", r"\burea\b",
         r"\bclinically\b", r"\bproven\b", r"\bdermatologist"]

def reason_richness(df, brands):
    rows = []
    for _, r in df.iterrows():
        low = _norm(r["response_text"])
        for b in brands:
            if _norm(b) not in low: continue
            m = re.search(re.escape(_norm(b)), low)
            ctx = low[max(0,m.start()-20):m.start()+240] if m else ""
            nq = sum(len(re.findall(p, ctx)) for p in QUANT)
            rows.append({"brand": b, "quantified_signals": nq})
    import pandas as pd
    if not rows: return pd.DataFrame()
    return (pd.DataFrame(rows).groupby("brand")["quantified_signals"]
              .agg(avg_richness="mean", mentions="count").round(2)
              .sort_values("avg_richness", ascending=False).reset_index())

# ---------- CLAIM PROVENANCE (reuse) ----------
def claim_provenance(df):
    """Which source types back the answers (self-referential vs corroborated)."""
    t = Counter()
    for tally in df["source_type_tally"]:
        t.update(tally or {})
    total = sum(t.values()) or 1
    brand_owned = t.get("brand_owned", 0)
    corroborated = t.get("editorial_authority", 0) + t.get("community_reviews", 0)
    return {"by_type": {k: round(v/total,3) for k,v in t.items()},
            "self_referential_pct": round(brand_owned/total, 3),
            "corroborated_pct": round(corroborated/total, 3)}

# ---------- ANSWER CONSISTENCY ----------
def answer_consistency(df, target_brand):
    """Per query: fraction of runs mentioning brand (fragile vs locked-in)."""
    g = df.groupby("query")["mentioned"].mean()
    return {"locked_in": g[g >= 0.9].index.tolist(),
            "fragile": g[(g > 0) & (g < 0.5)].index.tolist(),
            "absent": g[g == 0].index.tolist()}