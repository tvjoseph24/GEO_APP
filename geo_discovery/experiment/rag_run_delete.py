"""RAG-proxy runner (reuses Phase-4 logic): inject docs, web-off, N-runs,
position control, high-temp. Text + visual paths."""
import re, json, random, unicodedata

def _norm(t):
    if not t: return ""
    for d in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212": t = t.replace(d, "-")
    return unicodedata.normalize("NFKC", t).lower()

SYSTEM = ("You are a shopping assistant. Answer the question using ONLY the source "
          "documents provided. Recommend specific products where appropriate. "
          "Do not mention products not listed.")

def _doc_order(target_pos, run_index, n_comp=3):
    slots = ["TARGET"] + [f"COMP{i}" for i in range(n_comp)]
    if target_pos == "top": return slots
    if target_pos == "bottom": return slots[1:] + ["TARGET"]
    random.Random(run_index).shuffle(slots); return slots

def _build_prompt(target_text, comp_context, question, order):
    parts = [SYSTEM, ""]
    comp_blocks = comp_context.split("\n\n")
    ci = 0
    for slot in order:
        if slot == "TARGET":
            parts.append(f"[TARGET]\n{target_text}\n")
        else:
            if ci < len(comp_blocks):
                parts.append(comp_blocks[ci] + "\n"); ci += 1
    parts.append(f"QUESTION: {question}")
    return "\n".join(parts)

def _outcome(text, target_brand, competitors):
    low = _norm(text)
    positions = {}
    for b in [target_brand] + list(competitors):
        idx = low.find(_norm(b))
        if idx >= 0: positions[b] = idx
    mentioned = target_brand in positions
    first_brand = min(positions, key=positions.get) if positions else None
    return {"mentioned": mentioned,
            "first_index": positions.get(target_brand, -1),
            "is_first": first_brand == target_brand,
            "first_brand": first_brand,
            "response_text": text}          # ← KEEP raw response (Fix 3)

def run_variant(gateway, variant_text, comp_context, question, target_brand,
                competitors, *, n_runs, temperature, target_pos, model,
                visual=None, module="experiment", progress_cb=None,
                raw_writer=None, design_id="", arm=""):     # ← new params
    hits_mentioned = hits_first = 0; results = []
    for r in range(n_runs):
        order = _doc_order(target_pos, r)
        prompt = _build_prompt(variant_text, comp_context, question, order)
        images = [visual] if visual else None
        try:
            resp = gateway.generate(model, prompt, images=images,
                                    temperature=temperature, module=module, task="rag_run")
            o = _outcome(resp["text"], target_brand, competitors)
        except Exception as e:
            o = {"mentioned": None, "error": str(e)[:100], "response_text": ""}
        # ── FIX 3: stream raw output ──
        if raw_writer:
            raw_writer({"design_id": design_id, "arm": arm, "model": model,
                        "run": r, "question": question,
                        "mentioned": o.get("mentioned"), "is_first": o.get("is_first"),
                        "first_brand": o.get("first_brand"),
                        "response_text": o.get("response_text","")})
        # strip response_text from the compact result (keep results light)
        compact = {k: v for k, v in o.items() if k != "response_text"}
        results.append(compact)
        if o.get("mentioned"): hits_mentioned += 1
        if o.get("is_first"): hits_first += 1
        if progress_cb: progress_cb()
    return {"n": n_runs, "mentioned": hits_mentioned, "is_first": hits_first,
            "results": results}

def run_experiment(gateway, design, variants, comp_context, question, target_brand,
                   competitors, *, n_runs=30, temperature=0.8, engines, target_pos="top",
                   progress_cb=None, raw_writer=None):     # ← add raw_writer
    _id = design.get("design_id") or design.get("gap_id") or "unknown"
    out = {"gap_id": _id, "experiment_id": design.get("experiment_id"), "engines": {}}
    visual = None
    if design.get("visual_asset"):
        va = design["visual_asset"]; visual = (va["bytes"], va["mime"])
    for eng in engines:
        model = eng["model"]
        # visual variants only on vision-capable models
        vis_ok = "vision" in eng.get("supports", [])
        eng_out = {}
        for vk in ("variant_A", "variant_B"):
            use_visual = visual if (vk == "variant_B" and vis_ok) else None
            eng_out[vk] = run_variant(
                gateway, variants[vk]["text"], comp_context, question, target_brand,competitors,
                n_runs=n_runs, temperature=temperature, target_pos=target_pos,
                model=model, visual=use_visual, progress_cb=progress_cb,raw_writer=raw_writer,
                                      design_id=_id, arm=vk)
        out["engines"][eng["name"]] = eng_out
    return out

def run_progressive_experiment(gateway, design, arms, comp_context, question,
                               target_brand, competitors, *, n_runs=20,
                               temperature=0.8, engines, target_pos="top",
                               progress_cb=None, raw_writer=None):
    """Run all cumulative arms (A-E) across engines. Returns per-engine, per-arm."""
    _id = design.get("design_id") or design.get("gap_id") or "unknown"
    out = {"gap_id": _id, "experiment_id": "E1", "arms": list(arms.keys()), "engines": {}}

    visual = None
    if design.get("visual_asset"):
        va = design["visual_asset"]; visual = (va["bytes"], va["mime"])
    for eng in engines:
        vis_ok = "vision" in eng.get("supports", [])
        eng_out = {}
        for arm_key, arm in arms.items():
            use_visual = visual if (arm_key == "E" and vis_ok) else None   # visual on top arm
            eng_out[arm_key] = run_variant(
                gateway, arm["text"], comp_context, question, target_brand,
                competitors, n_runs=n_runs, temperature=temperature,
                target_pos=target_pos, model=eng["model"],
                visual=use_visual, progress_cb=progress_cb,raw_writer=raw_writer,
                                      design_id=_id, arm=arm_key)
        out["engines"][eng["name"]] = eng_out
    return out

##############

d["variants"] = generate_text_variants(
                    gateway, d, claims, t, model=variant_model,
                    hypothesis=d.get("_hypothesis"), target_words=target_words,
                    hitlist=hitlist, audit_raw=audit_raw,
                    parent_brand=p, target_brand=t,
                    comp_docs=ss.get("exp_comp_docs"),
                    include_negative_control=...)