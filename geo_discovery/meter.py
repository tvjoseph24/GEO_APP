"""Provider-agnostic cost/time meter with budget guard."""
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from config import PRICING

@dataclass
class Meter:
    budget_usd: float = 50.0                # hard cap per engagement
    events: list = field(default_factory=list)
    calls: int = 0
    tok_in: int = 0
    tok_out: int = 0
    elapsed: float = 0.0
    last_image_tokens: int = 0  # images, video and audio — see note_image_tokens
    last_media_kinds: list = field(default_factory=list)
    # ---- autosave. A manual save button lost a six-hour audit's cost record on any
    # ---- restart, and the engagement cap reverted to its default.
    autosave_cb: object = None  # (meter, cursor) -> new cursor
    _save_cursor: int = 0
    autosave_every: int = 25  # calls between writes
    _since_save: int = 0

    def extract_usage(self, resp, provider):
        if provider == "vertex":
            um = getattr(resp, "usage_metadata", None)
            return (getattr(um, "prompt_token_count", 0) or 0,
                    getattr(um, "candidates_token_count", 0) or 0)
        else:  # litellm
            u = getattr(resp, "usage", None) or {}
            get = (u.get if isinstance(u, dict) else lambda k, d=0: getattr(u, k, d))
            return (get("prompt_tokens", 0) or 0, get("completion_tokens", 0) or 0)

    def record(self, model, provider, tin, tout, seconds, module, task):
        pin, pout = PRICING.get(model, (0, 0))
        cost = tin * pin + tout * pout
        self.calls += 1
        self.tok_in += int(tin); self.tok_out += int(tout)
        self.elapsed += seconds
        self.events.append({"module": module, "task": task, "model": model,
                            "provider": provider, "tok_in": int(tin),
                            "tok_out": int(tout), "seconds": round(seconds, 2),
                            "cost_usd": round(cost, 6),
                            "ts": datetime.now(timezone.utc).isoformat()})
        self._maybe_autosave()
        return cost

    def _maybe_autosave(self, force=False):
        """Persist if the threshold is reached. Never raises — a failed cost write must
        not abort a paid run."""
        if self.autosave_cb is None:
            return
        self._since_save += 1
        if not force and self._since_save < self.autosave_every:
            return
        try:
            self._save_cursor = self.autosave_cb(self, self._save_cursor) or 0
            self._since_save = 0
        except Exception:
            pass

    def flush(self):
        """Force a write. Call at the end of every run block, and on tab render."""
        self._maybe_autosave(force=True)

    def total_cost(self):
        return sum(e["cost_usd"] for e in self.events)

    def check_budget(self):
        if self.total_cost() >= self.budget_usd:
            raise RuntimeError(f"Budget cap ${self.budget_usd} reached "
                              f"(spent ${self.total_cost():.2f}).")

    def summary_by_task(self):
        import pandas as pd
        if not self.events:
            return pd.DataFrame()
        df = pd.DataFrame(self.events)
        return (df.groupby(["module", "task"])
                  .agg(calls=("cost_usd", "count"), tok_in=("tok_in", "sum"),
                       tok_out=("tok_out", "sum"), seconds=("seconds", "sum"),
                       cost_usd=("cost_usd", "sum")).round(4).reset_index())

    def restore(self, events):
        """Restore prior events so cost carries over across sessions.

        The cursor is set to the restored length, so the first autosave appends only NEW
        events — otherwise every restored event is written again and the ledger doubles
        on each restart.
        """
        for e in events:
            self.events.append(e)
            self.calls += 1
            self.tok_in += e.get("tok_in", 0)
            self.tok_out += e.get("tok_out", 0)
            self.elapsed += e.get("seconds", 0)
        self._save_cursor = len(self.events)
        self._since_save = 0

    def note_image_tokens(self, resp, provider):
        """Records MEDIA tokens for the LAST call. 0 with media attached ⇒ the payload
        never reached the model.

        Previously matched only "IMAGE", so every video run reported zero tokens and
        results._visual_validity marked the finding INVALID — a correct guard firing on
        a working experiment. Name and attribute retained for call-site compatibility.
        """
        n, kinds = 0, []
        try:
            if provider == "vertex":
                um = getattr(resp, "usage_metadata", None)
                for d in (getattr(um, "prompt_tokens_details", None) or []):
                    mod = str(getattr(d, "modality", "")).upper()
                    if "IMAGE" in mod or "VIDEO" in mod or "AUDIO" in mod:
                        n += int(getattr(d, "token_count", 0) or 0)
                        kinds.append(mod)
            else:
                u = getattr(resp, "usage", None) or {}
                det = (u.get("prompt_tokens_details") if isinstance(u, dict)
                       else getattr(u, "prompt_tokens_details", None)) or {}
                for key in ("image_tokens", "video_tokens", "media_tokens"):
                    v = (det.get(key) if isinstance(det, dict)
                         else getattr(det, key, 0)) or 0
                    if v:
                        n += int(v)
                        kinds.append(key.replace("_tokens", "").upper())
        except Exception:
            n, kinds = 0, []
        self.last_image_tokens = n
        self.last_media_kinds = sorted(set(kinds))
        return n

    def unpriced_models(self):
        """Models with no PRICING entry — their cost silently reads as $0."""
        from config import PRICING
        return sorted({e["model"] for e in self.events if e["model"] not in PRICING})