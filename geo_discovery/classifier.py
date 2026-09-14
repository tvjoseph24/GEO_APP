"""Hybrid outcome classifier: free keyword rules OR opt-in LLM-combined."""
import re, unicodedata, json

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
def _norm(t):
    if not t: return ""
    for d in _DASHES: t = t.replace(d, "-")
    return unicodedata.normalize("NFKC", t).lower()

REFUSAL_CUES = ["consult a", "see a doctor", "speak with your", "cannot recommend",
                "i can't provide", "i cannot provide", "talk to your", "seek medical",
                "healthcare professional", "i'm not able to recommend", "consult a professional",
                "please consult", "recommend seeing"]

# ---------- FREE: keyword classifier ----------
'''

def classify_keyword(text, target_brand, competitors):
    low = _norm(text)
    named_target = target_brand.lower() in low
    named_any = named_target or any(c.lower() in low for c in competitors)
    is_refusal = any(cue in low for cue in REFUSAL_CUES) and not named_any
    outcome = ("refusal" if is_refusal else
               "brand_recommended" if named_target else "outranked")
    return {"outcome": outcome, "method": "keyword",
            "target_mentioned": named_target, "sentiment": None}
'''

def classify_keyword(text, target_brand, competitors, precomputed=None):
    """Now derives sentiment from metrics_spec polarity — previously always None,
    which silently disabled Tier-B sentiment analysis unless LLM mode was paid for."""
    if precomputed is None:
        from metrics_spec import build_registry, measure
        precomputed = measure(text, build_registry(target_brand, competitors))
    pol = precomputed.get("polarity", "absent")
    return {"outcome": precomputed.get("outcome"),
            "method": "keyword+metrics_spec",
            "target_mentioned": precomputed.get("mentioned"),
            "sentiment_toward_target": {"pos": "positive", "neg": "negative",
                                        "neu": "neutral", "absent": "absent"}.get(pol),
            "refusal_reason": precomputed.get("refusal_kind", "none"),
            "advisory_referral": precomputed.get("advisory_referral"),
            "recommended_as_top": precomputed.get("is_first"),
            "hedged": precomputed.get("hedged")}

# ---------- OPT-IN: LLM-combined (richer) ----------
CLASSIFY_PROMPT = """Classify this AI response about product recommendations. Return STRICT JSON:
{{
 "outcome": "brand_recommended|outranked|refusal|partial",
 "refusal_reason": "safety|policy|insufficient_info|none",
 "sentiment_toward_target": "positive|neutral|negative|absent",
 "target_mentioned": true,
 "recommended_as_top": false
}}
- brand_recommended = target is recommended
- outranked = others recommended, target not / weakly
- refusal = engine declines to recommend ANY product
- partial = hedged / conditional

TARGET BRAND: {brand}
RESPONSE:
{text}"""

def classify_llm(gateway, text, target_brand, judge_model, module="audit"):
    r = gateway.generate(judge_model,
                         CLASSIFY_PROMPT.format(brand=target_brand, text=text[:4000]),
                         json_mode=True, module=module, task="classify", temperature=0.1)
    d = r["parsed"] if isinstance(r["parsed"], dict) else {}
    d["method"] = "llm"
    d.setdefault("outcome", "outranked")
    d.setdefault("sentiment_toward_target", None)
    return d

'''

def classify(gateway, text, target_brand, competitors, *, use_llm=False,
             judge_model="gemini-2.5-flash"):
    if use_llm:
        try:
            return classify_llm(gateway, text, target_brand, judge_model)
        except Exception:
            pass  # fall back to keyword on any failure
    return classify_keyword(text, target_brand, competitors)
'''
def classify(gateway, text, target_brand, competitors, *, use_llm=False,
             judge_model="gemini-2.5-flash", precomputed=None):
    if use_llm:
        try:
            d = classify_llm(gateway, text, target_brand, judge_model)
            if precomputed:      # keep deterministic facts authoritative
                d["target_mentioned"] = precomputed.get("mentioned")
                d["advisory_referral"] = precomputed.get("advisory_referral")
                d["_keyword_outcome"] = precomputed.get("outcome")
            return d
        except Exception:
            pass
    return classify_keyword(text, target_brand, competitors, precomputed)