"""metrics_spec.py — SINGLE SOURCE OF TRUTH for brand + outcome measurement.

Every phase (audit, benchmark, experiment) MUST measure mention / first-mention /
rank / refusal / polarity through this module. Divergent implementations were causing
baseline and lift to be measured on different rulers.

Design notes
------------
* `_fold()` is LENGTH-PRESERVING (accents mapped 1:1, dashes unified, lowercased) so
  character spans found in folded text map back to the ORIGINAL text exactly.
* Aliases are DERIVED from the brand name — nothing is hardcoded per brand.
* Overlapping matches are resolved longest-alias-first, so tracking a parent brand does
  not steal a mention belonging to one of its sub-brands (and vice versa).
* A mention is still a mention when negated; polarity is recorded separately so
  share-of-voice numerators can exclude anti-recommendations.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------- folding
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"

def _build_accent_map() -> dict:
    pairs = {"àáâãäåāă": "a", "çćč": "c", "èéêëēĕė": "e", "ìíîïīĭ": "i",
             "ñńň": "n", "òóôõöøōŏ": "o", "ùúûüūŭ": "u", "ýÿ": "y",
             "šś": "s", "žźż": "z", "ďđ": "d", "ģğ": "g", "ĺľł": "l",
             "ŕř": "r", "ťţ": "t"}
    m = {}
    for src, dst in pairs.items():
        for ch in src:
            m[ord(ch)] = dst
            m[ord(ch.upper())] = dst
    for d in _DASHES:
        m[ord(d)] = "-"
    m[ord("\u2019")] = "'"   # curly apostrophe
    m[ord("\u2018")] = "'"
    m[ord("\u201c")] = '"'
    m[ord("\u201d")] = '"'
    m[ord("\u00a0")] = " "   # nbsp
    return m

_ACCENT_MAP = _build_accent_map()

def fold(text) -> str:
    """Length-preserving normalisation. len(fold(t)) == len(t) always."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", str(text))   # NFC keeps length for Latin-1
    return t.translate(_ACCENT_MAP).lower()

# kept as `_norm` alias so existing call-sites can migrate gradually
_norm = fold

# ---------------------------------------------------------------- aliases
_MIN_ALIAS_LEN = 3

# Tokens that must never survive as a standalone alias — too generic to discriminate.
_ALIAS_BLOCKLIST = {"the", "and", "for", "with", "baby", "kids", "junior", "plus",
                    "pro", "max", "new", "original", "classic", "brand", "company",
                    "inc", "ltd", "llc", "gmbh", "group", "holdings"}

def _apostrophe_variants(s: str) -> set:
    """'childs farm' <-> "child's farm". Engines write the possessive; registers often
    don't. Targeted: only inserts before a trailing s on a word of 4+ chars."""
    out = set()
    if "'" in s:
        out.add(s.replace("'", ""))
        out.add(s.replace("'s", "s"))
    else:
        parts = s.split()
        for i, w in enumerate(parts):
            if len(w) >= 4 and w.endswith("s") and not w.endswith("ss"):
                alt = parts[:]
                alt[i] = w[:-1] + "'s"
                out.add(" ".join(alt))
    return out

def _parenthetical_variants(s: str) -> set:
    """'first aid beauty (fab)' -> 'first aid beauty' + 'fab'."""
    m = re.search(r"^(.*?)\s*\(([^)]{2,})\)\s*(.*)$", s)
    if not m:
        return set()
    head, inner, tail = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    out = set()
    if head:
        out.add((head + " " + tail).strip())
    if inner and inner not in _ALIAS_BLOCKLIST:
        out.add(inner)
    return out

def _slash_variants(s: str) -> set:
    """'audi q7/q8' -> 'audi q7', 'audi q8'. Distributes a shared prefix."""
    if "/" not in s:
        return set()
    out = {s.replace("/", " "), s.replace("/", "")}
    parts = s.split()
    for i, w in enumerate(parts):
        if "/" in w:
            prefix = " ".join(parts[:i])
            suffix = " ".join(parts[i + 1:])
            for seg in w.split("/"):
                seg = seg.strip()
                if seg:
                    out.add(" ".join(x for x in (prefix, seg, suffix) if x))
    return out

def _shorten_variants(s: str) -> set:
    """Drop a trailing sub-brand token, but ONLY when >=2 tokens remain.
    'burt's bees baby' -> "burt's bees"   (kept)
    'childs farm'      -> 'childs'        (blocked — 1 token remains)
    'range rover'      -> 'range'         (blocked — 1 token remains)"""
    parts = s.split()
    if len(parts) < 3:
        return set()
    head = " ".join(parts[:-1])
    if len(head.split()) >= 2 and parts[-1] not in {"&", "and"}:
        return {head}
    return set()

def derive_aliases(name: str, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Automatic alias derivation. Brand-agnostic.

    W-12: previously only varied the FULL string, so any brand with an apostrophe,
    parenthetical, slash or trailing sub-brand was systematically untracked.
    """
    if not name:
        return ()
    f = fold(name).strip()
    cands = {f}

    # ---- punctuation normalisation (original behaviour) ----
    cands.add(f.replace("'", ""))
    cands.add(re.sub(r"[^a-z0-9]+", " ", f).strip())
    cands.add(re.sub(r"[^a-z0-9]+", "", f))
    cands.add(f.replace("&", "and"))
    cands.add(f.replace(" and ", " & "))
    cands.add(f.replace("-", " "))
    cands.add(f.replace(" ", "-"))

    # ---- structural variants (new) ----
    for v in (_apostrophe_variants(f) | _parenthetical_variants(f)
              | _slash_variants(f) | _shorten_variants(f)):
        cands.add(v)
        cands.add(re.sub(r"\s+", " ", v).strip())

    for e in extra:
        cands.add(fold(e).strip())

    cands = {re.sub(r"\s+", " ", c).strip() for c in cands}
    cands = {c for c in cands
             if len(c) >= _MIN_ALIAS_LEN and c not in _ALIAS_BLOCKLIST}
    return tuple(sorted(cands, key=lambda x: (-len(x), x)))

@dataclass(frozen=True)
class Brand:
    canonical: str
    aliases: tuple[str, ...]

    @property
    def pattern(self) -> re.Pattern:
        body = "|".join(re.escape(a) for a in self.aliases)
        return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")

def make_brand(name: str, extra_aliases=()) -> Brand:
    return Brand(canonical=name, aliases=derive_aliases(name, tuple(extra_aliases)))

@dataclass
class BrandRegistry:
    target: Brand
    competitors: tuple[Brand, ...] = ()

    @property
    def all(self) -> tuple[Brand, ...]:
        return (self.target,) + tuple(self.competitors)

    def names(self) -> list[str]:
        return [b.canonical for b in self.all]

def build_registry(target_brand: str, competitors, alias_overrides: dict | None = None
                   ) -> BrandRegistry:
    """competitors: list[str] | list[{'brand':..,'aliases':[..]}]
    alias_overrides: {canonical_name: [extra aliases]} — persisted in competitive_set."""
    ov = {fold(k): v for k, v in (alias_overrides or {}).items()}

    def _mk(n, inline=()):
        return make_brand(n, tuple(inline) + tuple(ov.get(fold(n), ())))

    comps = []
    for c in (competitors or []):
        if isinstance(c, dict):
            comps.append(_mk(c.get("brand", ""), tuple(c.get("aliases") or ())))
        elif c:
            comps.append(_mk(c))
    comps = [c for c in comps if c.canonical and fold(c.canonical) != fold(target_brand)]
    return BrandRegistry(target=_mk(target_brand), competitors=tuple(comps))

# ---------------------------------------------------------------- sentences
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in _SENT_SPLIT.finditer(text or ""):
        spans.append((pos, m.start()))
        pos = m.end()
    if text and pos < len(text):
        spans.append((pos, len(text)))
    return [(a, b) for a, b in spans if b > a]

def _sentence_for(text: str, idx: int) -> tuple[int, int]:
    for a, b in sentence_spans(text):
        if a <= idx < b:
            return a, b
    return max(0, idx - 120), min(len(text), idx + 200)

# ---------------------------------------------------------------- polarity
# STRONG_NEG = criticism directed AT a product/brand. Only these can produce 'neg'.
STRONG_NEG_CUES = [
    r"\bavoid\s+(?:this|that|it\b|the\s+\w+|using\b)",   # imperative → directed at product
    r"\bbest\s+avoided\b", r"\bwould\s+avoid\b", r"\bshould\s+be\s+avoided\b",
    r"\b(?:would|do|does|did|could|should)\s*n[o']?t\s+recommend\b",
    r"\bnot\s+recommend(?:ed)?\b", r"\bcan(?:not|'t)\s+recommend\b",
    r"\bsteer clear\b",
    r"\bnot\s+(?:a\s+|the\s+)?(?:good|great|ideal|suitable|safe|best)\b",
    r"\bless\s+effective\b", r"\bineffective\b", r"\bunderperform",
    r"\bworse\b", r"\binferior\b", r"\bsubpar\b",
    r"\bdrawback", r"\bdownside", r"\bshortcoming",
    r"\bcriticis(?:e|ed|es|m)\b", r"\bcriticiz(?:e|ed|es|m)\b",
    r"\bcomplain(?:t|ts|ed|s)?\b", r"\bbacklash\b",
    r"\brecall(?:ed|s)?\b", r"\bcontrovers(?:y|ial)\b", r"\bscandal\b",
    r"\blawsuit\b", r"\bbanned\b", r"\bwithdrawn\b",
    r"\bexpensive\b", r"\boverpriced\b", r"\bpricey\b",
    r"\bcause[sd]?\s+(?:constipation|reflux|gas|bloating|discomfort|irritation|colic)\b",
    r"\bdisappoint(?:ed|ing|ment)?\b", r"\bpoor(?:ly|er)?\b",
    r"\bharsh\b", r"\bunsuitable\b", r"\bnot\s+worth\b",
]
# WEAK_CONTRAST = discourse markers. NEVER produce 'neg'. Also act as clause boundaries.
WEAK_CONTRAST_CUES = [
    r"\bhowever\b", r"\bbut\b", r"\bwhereas\b", r"\bwhilst\b", r"\bwhile\b",
    r"\binstead of\b", r"\brather than\b", r"\b(?:al)?though\b",
    r"\bon the other hand\b", r"\bin contrast\b",
    r"\bdiffer(?:s|ent|ently|ence[s]?)?\b", r"\bvar(?:y|ies|ying)\b",
]
POS_CUES = [
    r"\brecommend(?:ed|s|ation)?\b", r"\bbest\b", r"\btop (?:pick|choice|rated)\b",
    r"\bideal\b", r"\bexcellent\b", r"\bgreat (?:choice|option)\b",
    r"\b(?:well|highly|widely)[- ]regarded\b", r"\bhighly rated\b",
    r"\btrusted\b", r"\bpreferred\b", r"\bgold standard\b", r"\bwidely used\b",
    r"\bleading\b", r"\breputable\b", r"\bpopular\b", r"\bclinically proven\b",
    r"\bwell[- ]established\b", r"\bnutritionally complete\b",
    r"\b(?:avoids|free\s+from|without)\s+(?:palm\s+oil|added\s+sugar|maltodextrin|"
    r"corn\s+syrup|artificial)\b",
]
HEDGE_CUES = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\bmay not be\b",
    r"\bsome (?:parents|users|people|reviewers|studies)\b",
    r"\bgenerally\b", r"\breportedly\b", r"\ballegedly\b", r"\bit'?s said\b",
    r"\bmixed reviews\b", r"\bdepend(?:s|ing)?\b",
    r"\bthey claim\b", r"\bclaims? (?:that|to)\b", r"\bmarketed as\b",
    r"\bnot (?:clinically )?(?:proven|confirmed)\b",
]

def _compile(pats):
    return [re.compile(p) for p in pats]

_STRONG_NEG = _compile(STRONG_NEG_CUES)
_WEAK = _compile(WEAK_CONTRAST_CUES)
_POS = _compile(POS_CUES)
_HEDGE = _compile(HEDGE_CUES)

# clause boundaries: contrast markers + strong punctuation. Scopes cue attribution so
# "Aptamil is great but Enfamil is not suitable" does not make Aptamil negative.
_CLAUSE_SPLIT = re.compile(
    r"\b(?:but|however|whereas|whilst|while|although|though|instead of|rather than"
    r"|in contrast|on the other hand)\b|[;:]|\s[—–-]\s")

def _clause_spans(folded_sent):
    bounds = [0] + [m.start() for m in _CLAUSE_SPLIT.finditer(folded_sent)] \
             + [len(folded_sent)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]

def _find_cues(folded, compiled):
    return [(m.start(), m.group(0)) for rx in compiled for m in rx.finditer(folded)]

def _attribute_cues(folded_sent, brand_rel_positions):
    """Attribute each cue to the brand(s) in ITS OWN CLAUSE.
    Falls back to the nearest preceding brand in the sentence (marked low-confidence).
    brand_rel_positions: sorted [(rel_start, brand)]."""
    clauses = _clause_spans(folded_sent)

    def _clause_of(pos):
        for a, b in clauses:
            if a <= pos < b:
                return a, b
        return 0, len(folded_sent)

    scores = {}

    def _slot(b):
        return scores.setdefault(b, {"pos": 0, "neg": 0, "hedge": 0, "contrast": 0,
                                     "cues": [], "low_confidence": False})

    for kind, compiled in (("neg", _STRONG_NEG), ("pos", _POS),
                           ("hedge", _HEDGE), ("contrast", _WEAK)):
        for pos, txt in _find_cues(folded_sent, compiled):
            ca, cb = _clause_of(pos)
            owners = [b for p, b in brand_rel_positions if ca <= p < cb]
            confident = bool(owners)
            if not owners:                       # cue in a brand-free clause
                prev = [b for p, b in brand_rel_positions if p <= pos]
                owners = [prev[-1]] if prev else (
                    [brand_rel_positions[0][1]] if brand_rel_positions else [])
            for ow in set(owners):
                d = _slot(ow)
                d[kind] += 1
                if kind in ("neg", "pos"):
                    d["cues"].append(f"{kind}:{txt}")
                if kind == "neg" and not confident:
                    d["low_confidence"] = True

    for _, d in scores.items():
        # a low-confidence negative never wins outright
        if d["neg"] and d["neg"] > d["pos"] and not d["low_confidence"]:
            d["polarity"] = "neg"
        elif d["pos"] > d["neg"]:
            d["polarity"] = "pos"
        else:
            d["polarity"] = "neu"
    return scores

# ---------------------------------------------------------------- mentions
@dataclass
class MentionHit:
    brand: str
    alias: str
    start: int
    end: int
    sentence: str
    polarity: str
    hedged: bool
    pos_cues: int = 0
    neg_cues: int = 0
    contrast_cues: int = 0
    cue_detail: list = field(default_factory=list)
    n_brands_in_sentence: int = 1
    low_confidence: bool = False

def detect_mentions(text: str, registry: BrandRegistry) -> list[MentionHit]:
    if not text:
        return []
    folded = fold(text)
    raw = []
    for b in registry.all:
        for m in b.pattern.finditer(folded):
            raw.append((m.start(), m.end(), b.canonical, m.group(0)))
    if not raw:
        return []
    raw.sort(key=lambda r: (r[0], -(r[1] - r[0])))
    kept, last_end = [], -1
    for s, e, brand, alias in raw:
        if s < last_end:
            continue
        kept.append((s, e, brand, alias))
        last_end = e

    sents = sentence_spans(text)

    def _sent_idx(i):
        for k, (a, b) in enumerate(sents):
            if a <= i < b:
                return k
        return -1

    by_sent = {}
    for h in kept:
        by_sent.setdefault(_sent_idx(h[0]), []).append(h)

    hits = []
    for si, group in by_sent.items():
        if si == -1:
            a = max(0, group[0][0] - 120)
            b_ = min(len(text), group[0][0] + 200)
        else:
            a, b_ = sents[si]
        fs = folded[a:b_]
        rel = sorted((g[0] - a, g[2]) for g in group)
        n_distinct = len({g[2] for g in group})
        attr = _attribute_cues(fs, rel)
        for s, e, brand, alias in group:
            d = attr.get(brand, {})
            hits.append(MentionHit(
                brand=brand, alias=alias, start=s, end=e,
                sentence=text[a:b_].strip(),
                polarity=d.get("polarity", "neu"), hedged=d.get("hedge", 0) > 0,
                pos_cues=d.get("pos", 0), neg_cues=d.get("neg", 0),
                contrast_cues=d.get("contrast", 0),
                cue_detail=d.get("cues", [])[:4],
                n_brands_in_sentence=n_distinct,
                low_confidence=d.get("low_confidence", False)))
    hits.sort(key=lambda h: h.start)
    return hits

# a response is 'neg' only when negatives are a MATERIAL share of its mentions
NEG_SHARE_THRESHOLD = 0.34

def mention_summary(text: str, registry: BrandRegistry) -> dict:
    """Per-brand summary. BACKWARD COMPATIBLE: keeps `mentioned` + `first_index`."""
    hits = detect_mentions(text, registry)
    by_brand: dict[str, list[MentionHit]] = {}
    for h in hits:
        by_brand.setdefault(h.brand, []).append(h)

    ordered = sorted(by_brand.items(), key=lambda kv: kv[1][0].start)
    rank_of = {name: i + 1 for i, (name, _) in enumerate(ordered)}

    out = {}
    for b in registry.all:
        hs = by_brand.get(b.canonical, [])
        if not hs:
            out[b.canonical] = {"mentioned": False, "first_index": -1, "rank": None,
                                "n_mentions": 0, "alias_matched": None,
                                "polarity": "absent", "hedged": False,
                                "polarity_mix": {}, "neg_evidence": [],
                                "cue_detail": [], "first_sentence": ""}
            continue
        pols = [h.polarity for h in hs]
        mix = {k: pols.count(k) for k in ("pos", "neu", "neg") if pols.count(k)}
        neg_n, pos_n, tot = pols.count("neg"), pols.count("pos"), len(pols)
        # share-based, thresholded — a single stray cue in a long answer no longer
        # flips the whole response to negative
        if neg_n and (neg_n / tot) >= NEG_SHARE_THRESHOLD and neg_n >= pos_n:
            pol = "neg"
        elif pos_n:
            pol = "pos"
        else:
            pol = "neu"
        neg_ev = [{"sentence": h.sentence[:300], "cues": h.cue_detail,
                   "n_brands_in_sentence": h.n_brands_in_sentence,
                   "low_confidence": h.low_confidence}
                  for h in hs if h.polarity == "neg"][:2]
        out[b.canonical] = {
            "mentioned": True, "first_index": hs[0].start,
            "rank": rank_of.get(b.canonical), "n_mentions": len(hs),
            "alias_matched": hs[0].alias, "polarity": pol,
            "polarity_mix": mix,
            "hedged": any(h.hedged for h in hs),
            "neg_evidence": neg_ev,                      # ← the audit trail
            "cue_detail": [c for h in hs for c in h.cue_detail][:6],
            "first_sentence": hs[0].sentence[:400],
        }
    return out

def first_brand(mention_summary_dict: dict) -> str | None:
    present = [(b, v.get("first_index", -1)) for b, v in (mention_summary_dict or {}).items()
               if v.get("first_index", -1) >= 0]
    return min(present, key=lambda x: x[1])[0] if present else None

def positive_first_brand(mention_summary_dict: dict) -> str | None:
    """First brand that is NOT negatively framed — the honest share-of-voice winner."""
    present = [(b, v.get("first_index", -1))
               for b, v in (mention_summary_dict or {}).items()
               if v.get("first_index", -1) >= 0 and v.get("polarity") != "neg"]
    return min(present, key=lambda x: x[1])[0] if present else None

def brand_key(mention_summary_dict: dict, brand_name: str) -> str | None:
    """Case/accent-insensitive key lookup into a brand_mentions dict.

    W-73: brand_mentions keys carry the CANONICAL brand name (typically title-cased)
    while the UI supplies the storage slug (lower-cased). An exact .get() therefore
    returns nothing on every row, silently zeroing baseline mention rate, leaderboard
    first-share and winnability attribution. Every consumer of brand_mentions must
    route through this.
    """
    if not mention_summary_dict or not brand_name:
        return None
    if brand_name in mention_summary_dict:
        return brand_name
    target = fold(brand_name)
    for k in mention_summary_dict:
        if fold(k) == target:
            return k
    return None

def brand_entry(mention_summary_dict: dict, brand_name: str) -> dict:
    """The per-brand record, resolved case-insensitively. Empty dict if absent."""
    k = brand_key(mention_summary_dict, brand_name)
    return (mention_summary_dict or {}).get(k, {}) if k else {}

def same_brand(a, b) -> bool:
    """Case/accent-insensitive brand comparison."""
    return bool(a) and bool(b) and fold(a) == fold(b)

# ---------------------------------------------------------------- refusal
REFUSAL_PATTERNS = {
    "policy": [r"\bi (?:can'?t|cannot|am unable to) (?:provide|recommend|help with)\b",
               r"\bi'?m not able to recommend\b", r"\bcannot recommend\b",
               r"\bi (?:don'?t|do not) (?:provide|give) (?:medical|health) advice\b",
               r"\bagainst my guidelines\b"],
    "safety":  [r"\bseek (?:immediate )?medical\b", r"\bemergency\b.*\bdoctor\b",
                r"\bcould be (?:dangerous|harmful)\b", r"\bnot safe to\b"],
    # NOTE: advisory referral CO-EXISTS with brand recommendations (regulated categories:
    # infant formula, OTC pharma, supplements). Recorded independently — NOT a refusal.
    "advisory_referral": [r"\bconsult (?:a|your|with) (?:doctor|physician|pediatrician|paediatrician|healthcare|professional|gp)\b",
                          r"\btalk to your (?:doctor|pediatrician|paediatrician|gp|midwife|health visitor)\b",
                          r"\bspeak (?:with|to) (?:a|your) (?:doctor|pediatrician|paediatrician|healthcare)\b",
                          r"\bhealthcare (?:professional|provider)\b",
                          r"\bunder medical supervision\b",
                          r"\bplease consult\b", r"\brecommend seeing\b"],
    "insufficient_info": [r"\bi (?:don'?t|do not) have (?:enough|specific) information\b",
                          r"\bmy (?:knowledge|training data) (?:cutoff|ends)\b",
                          r"\bi'?m not aware of\b"],
}
_REFUSAL_RE = {k: re.compile("|".join(v)) for k, v in REFUSAL_PATTERNS.items()}

@dataclass
class RefusalVerdict:
    is_refusal: bool
    kind: str                      # policy|safety|insufficient_info|none
    advisory_referral: bool        # independent of is_refusal
    cues: list = field(default_factory=list)

def refusal_verdict(text: str, any_brand_named: bool) -> RefusalVerdict:
    folded = fold(text)
    cues, kind = [], "none"
    for k in ("policy", "safety", "insufficient_info"):
        found = _REFUSAL_RE[k].findall(folded)
        if found:
            cues += [k] * len(found)
            if kind == "none":
                kind = k
    advisory = bool(_REFUSAL_RE["advisory_referral"].search(folded))
    # a hard refusal requires a refusal cue AND no product actually recommended
    is_ref = kind != "none" and not any_brand_named
    return RefusalVerdict(is_refusal=is_ref, kind=kind if is_ref else "none",
                          advisory_referral=advisory, cues=cues)

# ---------------------------------------------------------------- outcome
def outcome_label(msum: dict, target_brand: str, refusal: RefusalVerdict) -> str:
    """Canonical outcome vocabulary used by audit AND experiment."""
    if refusal.is_refusal:
        return "refusal"
    t = msum.get(target_brand, {})
    if not t.get("mentioned"):
        return "outranked" if any(v.get("mentioned") for v in msum.values()) else "no_brands"
    if t.get("polarity") == "neg":
        return "negatively_framed"
    if first_brand(msum) == target_brand:
        return "brand_first"
    if t.get("hedged") or refusal.advisory_referral:
        return "brand_hedged"
    return "brand_recommended"

def measure(text: str, registry: BrandRegistry) -> dict:
    """THE canonical measurement of one response. Use this everywhere."""
    msum = mention_summary(text, registry)
    any_named = any(v.get("mentioned") for v in msum.values())
    rv = refusal_verdict(text, any_named)
    t = msum.get(registry.target.canonical, {})
    fb = first_brand(msum)
    return {
        "brand_mentions": msum,
        "mentioned": bool(t.get("mentioned")),
        "first_index": t.get("first_index", -1),
        "rank": t.get("rank"),
        "is_first": fb == registry.target.canonical,
        "first_brand": fb,
        "positive_first_brand": positive_first_brand(msum),
        "polarity": t.get("polarity", "absent"),
        "hedged": bool(t.get("hedged")),
        "refusal": rv.is_refusal,
        "refusal_kind": rv.kind,
        "advisory_referral": rv.advisory_referral,
        "outcome": outcome_label(msum, registry.target.canonical, rv),
        "n_brands_present": sum(1 for v in msum.values() if v.get("mentioned")),
        "polarity_mix": t.get("polarity_mix", {}),
        "neg_evidence": t.get("neg_evidence", []),
        "cue_detail": t.get("cue_detail", []),
    }

# ---------------------------------------------------------------- token noise
GENERIC_STOP = set(
    "the a an and or of for with to in on at is are was were be been being this that "
    "these those it its their there here as by from into over under about than then "
    "has have had will would can could may might should must not no yes more most less "
    "least very much many some any all both each every other others such same "
    "brand brands product products range ranges line lines variant variants "
    "claim claims claimed states stated says said markets marketed advertises "
    "supports support supporting helps help helping provides provide providing "
    "designed aimed intended made makes contains contain including included".split())

# Measurement nouns that are never brand-discriminative but frequently appear AS brand
# tokens. Protected from brand-derived stopping so the category lexicon can still see
# them. Category-specific additions come from the Category Profile (W-07).
_PROTECTED_SPEC_NOUNS = {
    "range", "power", "capacity", "speed", "size", "weight", "level", "grade",
    "rate", "volume", "length", "width", "height", "depth", "strength", "load",
}

def brand_stop(target_brand="", competitors=(), category="", protect=()):
    """Stop-token set for clustering / lexicon work.

    W-09: brand tokens were stopped unconditionally, so a measurement noun that is also
    a brand token was removed from the vocabulary — killing every downstream term
    containing it. Measurement nouns are now protected.

    protect: extra tokens to keep, supplied by the Category Profile.
    """
    keep = _PROTECTED_SPEC_NOUNS | {fold(p) for p in (protect or ()) if p}
    s = set(GENERIC_STOP)
    names = [target_brand or "", category or ""] + [
        (c.get("brand") if isinstance(c, dict) else c) or "" for c in (competitors or [])]
    for n in names:
        s |= {w for w in re.findall(r"[a-z]{3,}", fold(n)) if w not in keep}
    return s - keep