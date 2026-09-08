"""Unified LLM interface. Gemini→Vertex (always); others→LiteLLM.
Supports text and (image) multimodal; returns unified dict + meters cost."""
import time, json, re
from config import MODEL_REGISTRY, GCP_PROJECT, GCP_REGION
import litellm
litellm.drop_params = True   # ← safety: let litellm drop unsupported params
from url_resolver import resolve_sources

class LLMGateway:
    def __init__(self, meter):
        self.meter = meter
        self._vertex = None

    def _vertex_client(self):
        if self._vertex is None:
            from google import genai
            self._vertex = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_REGION)
        return self._vertex

    def generate(self, model, prompt, *, system=None, images=None,
                 temperature=0.2, module="", task="", json_mode=False):
        """images: list of (bytes, mime_type). Returns {'text','model','provider'}."""
        provider = MODEL_REGISTRY.get(model, {}).get("provider", "litellm")
        self.meter.check_budget()
        t0 = time.time()

        if provider == "vertex":
            text, tin, tout = self._vertex_call(model, prompt, system, images,
                                                temperature, json_mode)
        else:
            text, tin, tout = self._litellm_call(model, prompt, system, images,
                                                 temperature, json_mode)

        self.meter.record(model, provider, tin, tout, time.time() - t0, module, task)
        return {"text": text, "model": model, "provider": provider,
                "parsed": self._try_json(text) if json_mode else None}

    def _vertex_call(self, model, prompt, system, images, temperature, json_mode):
        from google.genai import types
        client = self._vertex_client()
        parts = []
        full = (f"{system}\n\n{prompt}" if system else prompt)
        parts.append(full)
        for data, mime in (images or []):
            parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        resp = client.models.generate_content(model=model, contents=parts, config=cfg)
        tin, tout = self.meter.extract_usage(resp, "vertex")
        return (resp.text or ""), tin, tout

    def _resolve_temperature(self, model, requested):
        """Respect per-model temperature constraints."""
        policy = MODEL_REGISTRY.get(model, {}).get("temp_policy", "free")
        if policy == "fixed_1":
            return 1.0  # GPT-5 family: only 1 allowed
        return requested

    def _min_tokens(self, model):
        """Reasoning models must have generous max_tokens or they output nothing."""
        return 4000 if MODEL_REGISTRY.get(model, {}).get("reasoning") else 1500

    def _litellm_call(self, model, prompt, system, images, temperature, json_mode):
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
                      max_tokens=self._min_tokens(model))  # ← reasoning-safe tokens
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
        return text, tin, tout

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
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
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
        return {"text": text, "sources": sources, "parsed": parsed,
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
            resp = litellm.responses(**kwargs)
        except Exception as e:
            # retry: drop temp if rejected (reasoning models)
            if "temperature" in str(e).lower():
                kwargs.pop("temperature", None)
                resp = litellm.responses(**kwargs)
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