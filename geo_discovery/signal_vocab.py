"""signal_vocab.py — canonical weakness-flag vocabulary.

B3 root cause: metrics_bundle emitted 'visibility_gap' / 'need_visibility_gap' while
signal_designer matched 'intent_gap' / 'need_first_low'. SIGNAL 1 therefore never fired.
Both sides now import from here so the mismatch cannot recur.
"""

VISIBILITY_GAP      = "visibility_gap"
NEED_VISIBILITY_GAP = "need_visibility_gap"
RANK_GAP            = "rank_gap"
RICHNESS_DEFICIT    = "richness_deficit"
REFUSAL_HEAVY       = "refusal_heavy"
NEGATIVE_FRAMING    = "negative_framing"
ADVISORY_HEAVY      = "advisory_heavy"      # regulated categories: hedged, not refused
VERACITY_GAP        = "veracity_gap"

ALL_FLAGS = {VISIBILITY_GAP, NEED_VISIBILITY_GAP, RANK_GAP, RICHNESS_DEFICIT,
             REFUSAL_HEAVY, NEGATIVE_FRAMING, ADVISORY_HEAVY, VERACITY_GAP}

# legacy names seen in older artifacts → canonical
LEGACY_ALIASES = {"intent_gap": VISIBILITY_GAP, "need_first_low": NEED_VISIBILITY_GAP}

# flags that should trigger a VISIBILITY experiment (E4 need-query design)
VISIBILITY_FLAGS = {VISIBILITY_GAP, NEED_VISIBILITY_GAP} | set(LEGACY_ALIASES)

# flags → experiment template hint
FLAG_TO_EXPERIMENT = {
    VISIBILITY_GAP: "E4", NEED_VISIBILITY_GAP: "E4", RANK_GAP: "E2",
    RICHNESS_DEFICIT: "E2", NEGATIVE_FRAMING: "E9", VERACITY_GAP: "E3",
    REFUSAL_HEAVY: "E9", ADVISORY_HEAVY: "E9",
}

def canonical(flag_type: str) -> str:
    return LEGACY_ALIASES.get(flag_type, flag_type)

def matches(flag_type: str, group: set) -> bool:
    return canonical(flag_type) in {canonical(g) for g in group}