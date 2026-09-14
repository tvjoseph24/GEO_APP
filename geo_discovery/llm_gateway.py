"""Unified LLM interface. Gemini→Vertex (always); others→LiteLLM.
Supports text and (image) multimodal; returns unified dict + meters cost."""
import time, json, re
from config import MODEL_REGISTRY, GCP_PROJECT, GCP_REGION
import litellm
litellm.drop_params = True   # ← safety: let litellm drop unsupported params
litellm.num_retries = 2
from url_resolver import resolve_sources
import random

# HTTP statuses worth retrying — the server is unwell, the request is fine.
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_TRANSIENT_MARKERS = ("502", "503", "504", "bad gateway", "service unavailable",
                      "deadline exceeded", "timed out", "timeout", "temporarily",
                      "overloaded", "resource exhausted", "rate limit",
                      "connection reset", "connection aborted", "remote end closed",
                      "internal error", "try again")
# Never retry these — retrying wastes time and money on a request that cannot succeed.
_PERMANENT_MARKERS = ("invalid_argument", "permission denied", "unauthenticated",
                      "not found", "unsupported", "api key", "quota exceeded for quota"
                      , "safety", "blocked", "400 bad request", "401", "403", "404")

def _is_transient(e) -> bool:
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if isinstance(code, int):
        if code in _TRANSIENT_STATUS:
            return True
        if 400 <= code < 500:
            return False                      # client error — permanent
    s = f"{type(e).__name__} {e}".lower()
    if any(m in s for m in _PERMANENT_MARKERS):
        return False
    return any(m in s for m in _TRANSIENT_MARKERS)

class LLMGateway:
    def __init__(self, meter):
        self.meter = meter
        self._vertex = None
        self.transient_events = []

    def _vertex_client(self):
        if self._vertex is None:
            from google import genai
            self._vertex = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_REGION)
        return self._vertex

    def _retry_call(self, fn, *args, what="", attempts=4, base=2.0, **kwargs):
        """Retry ONLY transient failures, with exponential backoff + jitter.
        A failed API call returns no response, so no tokens are billed for it."""
        last = None
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last = e
                if not _is_transient(e) or i == attempts - 1:
                    raise
                delay = base * (2 ** i) + random.uniform(0, 1.5)
                self.transient_events.append({
                    "what": what, "attempt": i + 1, "attempts": attempts,
                    "error": f"{type(e).__name__}: {str(e)[:120]}",
                    "sleep_s": round(delay, 1), "ts": time.time()})
                time.sleep(delay)
        raise last

    def generate(self, model, prompt, *, system=None, images=None,
                 temperature=0.2, module="", task="", json_mode=False,
                 max_tokens=None, thinking_budget=None):
        provider = MODEL_REGISTRY.get(model, {}).get("provider", "litellm")
        if images and "vision" not in MODEL_REGISTRY.get(model, {}).get("supports", []):
            raise ValueError(f"{model} has no declared vision support")
        self.meter.check_budget()
        t0 = time.time()
        if provider == "vertex":
            text, tin, tout, finish = self._retry_call(
                self._vertex_call, model, prompt, system, images, temperature,
                json_mode, max_tokens, thinking_budget, what=f"{module}/{task}")
        else:
            text, tin, tout, finish = self._retry_call(
                self._litellm_call, model, prompt, system, images, temperature,
                json_mode, max_tokens, what=f"{module}/{task}")
        self.meter.record(model, provider, tin, tout, time.time() - t0, module, task)
        img_tok = self.meter.last_image_tokens
        truncated = str(finish).upper() in ("LENGTH", "MAX_TOKENS")
        out = {"text": text, "model": model, "provider": provider,
               "finish_reason": finish, "truncated": truncated,
               "empty": not (text or "").strip(),
               "image_tokens": img_tok,
               "parsed": self._try_json(text) if json_mode else None}
        if images and not img_tok:
            out["vision_warning"] = f"{len(images)} image(s) attached, 0 image tokens"
        return out

    def _vertex_call(self, model, prompt, system, images, temperature, json_mode,
                     max_tokens=None, thinking_budget=None):
        from google.genai import types
        client = self._vertex_client()
        parts = [(f"{system}\n\n{prompt}" if system else prompt)]
        for data, mime in (images or []):
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))

        spec = MODEL_REGISTRY.get(model, {})
        kw = dict(temperature=temperature,
                  response_mime_type="application/json" if json_mode else None)
        if max_tokens:
            budget = 0
            if spec.get("thinking"):
                # thinking tokens are charged against max_output_tokens — reserve headroom
                budget = (spec.get("min_think", 0) if thinking_budget is None
                          else max(thinking_budget, spec.get("min_think", 0)))
                try:
                    kw["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
                except Exception:
                    budget = max(budget, 1024)      # SDK too old — just add headroom
            kw["max_output_tokens"] = int(max_tokens) + int(budget) + 256
        cfg = types.GenerateContentConfig(**{k: v for k, v in kw.items() if v is not None})

        resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        tin, tout = self.meter.extract_usage(resp, "vertex")
        self.meter.note_image_tokens(resp, "vertex")
        finish = ""
        try:
            finish = str(getattr(resp.candidates[0], "finish_reason", "") or "")
        except Exception:
            pass
        try:
            text = resp.text or ""
        except Exception:
            text = ""                                # blocked / no candidate
        return text, tin, tout, finish

    def generate_json(self, model, prompt, *, expect="any", module="", task="",
                      temperature=0.0, max_tokens=3000, thinking_budget=None,
                      retries=2, coerce_keys=("items", "results", "data", "gaps",
                                              "classifications", "output", "response")):
        """JSON call with escalation on empty/truncated/unparseable output.
        Returns {'data', 'ok', 'attempts', 'failure', 'finish_reason'}.
        Fixes the silent-empty failure that zeroed veracity and gap classification."""
        last, budget, mt = None, thinking_budget, max_tokens
        for attempt in range(retries + 1):
            r = self.generate(model, prompt, json_mode=True, module=module,
                              task=task if attempt == 0 else f"{task}_retry{attempt}",
                              temperature=temperature, max_tokens=mt,
                              thinking_budget=budget)
            last = r
            data = r.get("parsed")
            if data is None and r.get("text"):
                data = self._try_json(r["text"])
            data = self._coerce(data, expect, coerce_keys)
            if data is not None:
                return {"data": data, "ok": True, "attempts": attempt + 1,
                        "failure": None, "finish_reason": r.get("finish_reason")}
            mt = int(mt * 2)  # empty ⇒ thinking ate the budget
            budget = (budget or 512) * 2 if MODEL_REGISTRY.get(
                model, {}).get("thinking") else budget
        why = ("empty_response" if last.get("empty") else
               "truncated" if last.get("truncated") else "unparseable")
        return {"data": ([] if expect == "list" else {} if expect == "dict" else None),
                "ok": False, "attempts": retries + 1, "failure": why,
                "finish_reason": last.get("finish_reason"),
                "text_preview": (last.get("text") or "")[:400]}

    @staticmethod
    def _coerce(data, expect, keys):
        if data is None:
            return None
        if expect == "list":
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for k in keys:
                    if isinstance(data.get(k), list):
                        return data[k]
                if data and all(isinstance(v, dict) for v in data.values()):
                    out = []
                    for k, v in data.items():
                        d = dict(v);
                        d.setdefault("gap_id", k);
                        d.setdefault("span_id", k)
                        out.append(d)
                    return out
            return None
        if expect == "dict":
            return data if isinstance(data, dict) else None
        return data
    def _resolve_temperature(self, model, requested):
        """Respect per-model temperature constraints."""
        policy = MODEL_REGISTRY.get(model, {}).get("temp_policy", "free")
        if policy == "fixed_1":
            return 1.0  # GPT-5 family: only 1 allowed
        return requested

    def _min_tokens(self, model, requested=None):
        if requested:
            return int(requested)
        return 4000 if MODEL_REGISTRY.get(model, {}).get("reasoning") else 1500

    def _litellm_call(self, model, prompt, system, images, temperature, json_mode,
                      max_tokens=None):
        content = [{"type": "text", "text": prompt}]
        for data, mime in (images or []):
            import base64
            b64 = base64.b64encode(data).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"}})
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": content if images else prompt}]

        temp = self._resolve_temperature(model, temperature)  # ← model-aware temp
        kwargs = dict(model=model, messages=msgs, temperature=temp,
                      max_tokens=self._min_tokens(model, max_tokens))  # ← reasoning-safe tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = litellm.completion(**kwargs)
        except Exception as e:
            # retry once: force temp=1 then drop temp entirely
            if "temperature" in str(e).lower():
                kwargs["temperature"] = 1.0
                try:
                    resp = litellm.completion(**kwargs)
                except Exception:
                    kwargs.pop("temperature", None)
                    resp = litellm.completion(**kwargs)
            else:
                raise
        text = resp.choices[0].message.content or ""
        tin, tout = self.meter.extract_usage(resp, "litellm")
        self.meter.note_image_tokens(resp, "litellm")
        finish = str(getattr(resp.choices[0], "finish_reason", "") or "")
        return text, tin, tout, finish

    def generate_grounded(self, prompt, *, model="gemini-2.5-flash",
                          temperature=0.3, module="", task="", json_mode=False):
        """Gemini + Google Search grounding. Returns text + sources (real web)
        + the actual search queries Gemini ran. Vertex-only."""
        from google.genai import types
        self.meter.check_budget()
        import time
        from config import MODEL_REGISTRY
        if MODEL_REGISTRY.get(model, {}).get("provider") != "vertex":
            raise ValueError(f"Grounding requires a Vertex/Gemini model, got '{model}'. "
                             f"(Grounding is Gemini-only.)")
        t0 = time.time()
        client = self._vertex_client()
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        resp = self._retry_call(client.models.generate_content,
                                model=model, contents=prompt, config=cfg,
                                what=f"{module}/{task}(grounded)")
        tin, tout = self.meter.extract_usage(resp, "vertex")
        self.meter.record(model, "vertex", tin, tout, time.time() - t0, module, task)

        text = resp.text or ""
        sources = []
        web_queries = []                                            # ← ADD
        try:
            meta = resp.candidates[0].grounding_metadata
            web_queries = list(getattr(meta, "web_search_queries", []) or [])   # ← ADD
            for ch in (getattr(meta, "grounding_chunks", None) or []):
                if getattr(ch, "web", None):
                    sources.append({"title": ch.web.title, "url": ch.web.uri})
        except Exception:
            pass
        parsed = self._try_json(text) if json_mode else None
        sources = resolve_sources(sources)  # concurrent resolve + fallback
        finish = ""
        try:
            finish = str(getattr(resp.candidates[0], "finish_reason", "") or "")
        except Exception:
            pass
        return {"text": text, "sources": sources, "parsed": parsed,
                "finish_reason": finish,
                "truncated": str(finish).upper() in ("LENGTH", "MAX_TOKENS"),
                "web_search_queries": web_queries}                  # ← ADD to return

    def generate_web_litellm(self, model, prompt, *, temperature=0.3,
                             module="", task=""):
        """ChatGPT-style web search via litellm.responses + url_citation parsing.
        Returns text + REAL source URLs (no redirect problem). Meters cost."""
        import litellm, time
        litellm.drop_params = True
        self.meter.check_budget()
        t0 = time.time()

        # GPT-5 temperature constraint (reuse gateway policy)
        temp = self._resolve_temperature(model, temperature)
        kwargs = dict(model=model, input=prompt,
                      tools=[{"type": "web_search_preview"}])
        if temp is not None:
            kwargs["temperature"] = temp

        try:
            resp = self._retry_call(litellm.responses, what=f"{module}/{task}(web)",
                                    **kwargs)
        except Exception as e:
            # retry: drop temp if rejected (reasoning models)
            if "temperature" in str(e).lower():
                kwargs.pop("temperature", None)
                resp = self._retry_call(litellm.responses, what=f"{module}/{task}(web)",
                                        **kwargs)
            else:
                raise

        text, citations, web_queries = self._parse_responses_output(resp)

        # usage extraction from responses object
        tin, tout = self._responses_usage(resp)
        self.meter.record(model, "litellm", tin, tout, time.time() - t0, module, task)

        # unique cited URLs → sources (REAL urls; classify from URL host)
        from urllib.parse import urlparse
        seen, sources = set(), []
        for c in citations:
            url = c.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({"title": c.get("title"), "url": url,
                            "resolved_url": url, "resolved_ok": True,  # already real
                            "domain": urlparse(url).netloc.replace("www.", "")})
        return {"text": text, "sources": sources, "citations": citations,
                "web_search_queries": web_queries}

    # ---- helpers (your proven parsing, adapted) ----
    @staticmethod
    def _g(obj, key, default=None):
        if obj is None: return default
        if isinstance(obj, dict): return obj.get(key, default)
        return getattr(obj, key, default)

    def _parse_responses_output(self, resp):
        """Walk responses output: text spans + url_citation annotations + web queries."""
        g = self._g
        text_parts, citations, web_queries = [], [], []
        for item in (g(resp, "output", []) or []):
            itype = g(item, "type")
            if itype == "web_search_call":
                q = g(g(item, "action"), "query")
                if q: web_queries.append(q)
            elif itype == "message":
                for c in (g(item, "content", []) or []):
                    if g(c, "type") == "output_text":
                        text_parts.append(g(c, "text", "") or "")
                        for ann in (g(c, "annotations", []) or []):
                            if g(ann, "type") == "url_citation":
                                citations.append({
                                    "url": g(ann, "url"),
                                    "title": g(ann, "title"),
                                    "start_index": g(ann, "start_index"),
                                    "end_index": g(ann, "end_index")})
        text = "".join(text_parts) or (g(resp, "output_text", "") or "")
        return text, citations, web_queries

    def _responses_usage(self, resp):
        """Extract token usage from a responses object (shape varies)."""
        g = self._g
        u = g(resp, "usage", None)
        if u is None:
            return 0, 0
        # responses API uses input_tokens/output_tokens
        tin = g(u, "input_tokens", None)
        tout = g(u, "output_tokens", None)
        if tin is None:  # fall back to chat-style names
            tin = g(u, "prompt_tokens", 0)
            tout = g(u, "completion_tokens", 0)
        return int(tin or 0), int(tout or 0)
    @staticmethod
    def _try_json(text):
        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}|\[.*\]", text, re.S)
            if m:
                try: return json.loads(m.group())
                except Exception: return None
            return None