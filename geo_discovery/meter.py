"""Provider-agnostic cost/time meter with budget guard."""
import time
from dataclasses import dataclass, field
from config import PRICING

@dataclass
class Meter:
    budget_usd: float = 50.0                # hard cap per engagement
    events: list = field(default_factory=list)
    calls: int = 0
    tok_in: int = 0
    tok_out: int = 0
    elapsed: float = 0.0
    last_image_tokens: int = 0

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
                            "cost_usd": round(cost, 6)})
        return cost

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
        """Restore prior events (from a saved cost_log) so cost carries over."""
        for e in events:
            self.events.append(e)
            self.calls += 1
            self.tok_in += e.get("tok_in", 0)
            self.tok_out += e.get("tok_out", 0)
            self.elapsed += e.get("seconds", 0)

    def note_image_tokens(self, resp, provider):
        """Records image tokens for the LAST call. 0 with images attached ⇒ vision
        payload never reached the model."""
        n = 0
        try:
            if provider == "vertex":
                um = getattr(resp, "usage_metadata", None)
                for d in (getattr(um, "prompt_tokens_details", None) or []):
                    mod = str(getattr(d, "modality", "")).upper()
                    if "IMAGE" in mod:
                        n += int(getattr(d, "token_count", 0) or 0)
            else:
                u = getattr(resp, "usage", None) or {}
                det = (u.get("prompt_tokens_details") if isinstance(u, dict)
                       else getattr(u, "prompt_tokens_details", None)) or {}
                n = int((det.get("image_tokens") if isinstance(det, dict)
                         else getattr(det, "image_tokens", 0)) or 0)
        except Exception:
            n = 0
        self.last_image_tokens = n
        return n

    def unpriced_models(self):
        """Models with no PRICING entry — their cost silently reads as $0."""
        from config import PRICING
        return sorted({e["model"] for e in self.events if e["model"] not in PRICING})