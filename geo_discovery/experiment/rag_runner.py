#################### New code #############
"""RAG-proxy runner: inject docs, web-off, N-runs, position control, high-temp.
All measurement via metrics_spec so experiment lift is on the SAME ruler as the audit."""
import random
import re

from metrics_spec import build_registry, measure
import json as _json
from pathlib import Path as _Path

class BudgetStop(Exception):
    """Raised inside a run loop so spend stops at the cap, not at the next design."""

def completed_runs(raw_path, design_ids=None, arms=None, models=None, modes=None):
    """(design_id, mode, arm, model, run) already in the log.

    W-28: this counted EVERY row in the file, so with 8,800 rows against a 5,600-call
    plan `remaining` clamped to 0 and a partially-complete run reported "0 remaining,
    $0.00" while arms were still missing. Filters narrow the set to the space actually
    being run.
    """
    done = set()
    p = _Path(raw_path)
    if not p.exists():
        return done
    want_d = set(design_ids) if design_ids else None
    want_a = {str(a).replace("variant_", "") for a in arms} if arms else None
    want_m = set(models) if models else None
    # "creative" is the bench mode. It must never merge with "ab" or "progressive" runs
    # of a same-named design — aggregate_from_raw warns on contamination, and the right
    # answer is to give it nothing to warn about.
    want_mode = set(modes) if modes else None
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                r = _json.loads(line)
            except Exception:
                continue
            d = r.get("design_id")
            a = str(r.get("arm", "")).replace("variant_", "")
            mdl, mode = r.get("model"), r.get("mode", "")
            if want_d and d not in want_d:
                continue
            if want_a and a not in want_a:
                continue
            if want_m and mdl not in want_m:
                continue
            if want_mode and mode not in want_mode:
                continue
            done.add((d, mode, a, mdl, r.get("run")))
    return done

def aggregate_from_raw(raw_path, design_id, mode=None):
    """Rebuild per-engine/arm counts from the raw log — the only correct source when
    resuming. If the log contains MORE THAN ONE mode for this design, aggregate only the
    requested one; merging an A/B run with a progressive ladder produces meaningless
    arm comparisons."""
    import pandas as _pd
    rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = _json.loads(line)
            except Exception:
                continue
            if r.get("design_id") != design_id:
                continue
            r["arm"] = str(r.get("arm", "")).replace("variant_", "")
            rows.append(r)
    if not rows:
        return {}, {}
    df = _pd.DataFrame(rows)
    modes = sorted(df.get("mode", _pd.Series(dtype=str)).dropna().unique().tolist())
    meta = {"modes_in_log": modes, "mode_used": mode}
    if mode and "mode" in df and len(modes) > 1:
        df = df[df["mode"] == mode]
        meta["contamination_warning"] = (
            f"{design_id} has runs from {len(modes)} modes {modes} in the raw log — "
            f"only '{mode}' was aggregated. Purge the others for a clean result.")
    out = {}
    for (model, arm), g in df.groupby(["model", "arm"]):
        valid = g[g["error"].isna()] if "error" in g else g
        out.setdefault(model, {})[arm] = {
            "n": int(len(valid)), "n_attempted": int(len(g)),
            "n_errors": int(len(g) - len(valid)),
            "mentioned": int(valid["mentioned"].fillna(False).sum()),
            "is_first": int(valid["is_first"].fillna(False).sum()),
            "refusals": int(valid.get("refusal", _pd.Series(dtype=bool))
                            .fillna(False).sum())}
    return out, meta

SYSTEM = ("You are a shopping assistant. Answer the question using ONLY the source "
          "documents provided. Recommend specific products where appropriate. "
          "Do not mention products not listed.")

_BLOCK_HEAD = re.compile(r"^\s*\[([^\]\n]{1,80})\]\s*$", re.M)

def split_competitor_blocks(comp_context):
    """Accept list[str] OR the analyst-edited markdown string.
    B6 root cause: `.split("\\n\\n")` fragmented any block containing a blank line,
    so a competitor's table became a separate pseudo-document."""
    if isinstance(comp_context, (list, tuple)):
        return [b for b in comp_context if str(b).strip()]
    text = comp_context or ""
    heads = list(_BLOCK_HEAD.finditer(text))
    if not heads:
        return [b.strip() for b in text.split("\n\n") if b.strip()]
    blocks = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        blk = text[h.start():end].strip()
        if blk:
            blocks.append(blk)
    return blocks

def _doc_order(target_pos, run_index, n_comp):
    """B6 root cause: n_comp defaulted to 3, so competitors 4+ were never rendered.
    'balanced' counterbalances position WITHIN each arm (recommended default)."""
    slots = ["TARGET"] + [f"COMP{i}" for i in range(n_comp)]
    mode = target_pos
    if target_pos == "balanced":
        mode = ("top", "bottom", "shuffle")[run_index % 3]
    if mode == "top":
        return slots, mode
    if mode == "bottom":
        return slots[1:] + ["TARGET"], mode
    sh = list(slots)
    random.Random(run_index * 7919).shuffle(sh)
    return sh, mode

def _build_prompt(target_text, comp_blocks, question, order):
    parts = [SYSTEM, ""]
    ci = 0
    for slot in order:
        if slot == "TARGET":
            parts.append(f"[TARGET]\n{target_text}\n")
        elif ci < len(comp_blocks):
            parts.append(str(comp_blocks[ci]).strip() + "\n")
            ci += 1
    parts.append(f"QUESTION: {question}")
    return "\n".join(parts)

def run_variant(gateway, variant_text, comp_context, question, target_brand,
                competitors, *, n_runs, temperature, target_pos, model,
                visual=None, module="experiment", progress_cb=None,
                raw_writer=None, design_id="", arm="", alias_overrides=None,
                done=None, budget_check=None, mode="", comp_mode="full",
                video_supported=True):
    registry = build_registry(target_brand, competitors, alias_overrides)
    comp_blocks = split_competitor_blocks(comp_context)
    if comp_mode == "thin":
        # A visual arm may be 30 words against 900 words of competitor field. Keeping
        # only each competitor's opening line preserves the field without burying the
        # brand by construction.
        comp_blocks = [b.split("\n\n")[0] if "\n\n" in b else
                       "\n".join(b.splitlines()[:3]) for b in comp_blocks]
    elif comp_mode == "none":
        comp_blocks = []
    n_comp = len(comp_blocks)

    hits_mentioned = hits_first = hits_pos_first = refusals = errors = 0
    skipped = img_runs = img_token_runs = skipped_video = 0
    img_tokens_total = 0
    video_ok = bool(video_supported)
    results, positions = [], {}

    for r in range(n_runs):
        if done and (design_id, mode, str(arm).replace("variant_", ""),
                     model, r) in done:
            skipped += 1
            continue
        if budget_check is not None:
            stop = budget_check()
            if stop:
                raise BudgetStop(stop)
        order, pos_mode = _doc_order(target_pos, r, n_comp)
        prompt = _build_prompt(variant_text, comp_blocks, question, order)
        images = [visual] if visual else None
        if images and str(visual[1]).startswith("video/") and not video_ok:
            images = None
            skipped_video += 1
        try:
            resp = gateway.generate(model, prompt, images=images,
                                    temperature=temperature, module=module,
                                    task="rag_run")
            m = measure(resp["text"], registry)
            m["error"] = None
            text = resp["text"]
            itok = int(resp.get("image_tokens") or 0)
            vwarn = resp.get("vision_warning")
        except Exception as e:
            m = {"mentioned": None, "is_first": None, "refusal": None,
                 "error": str(e)[:200], "polarity": None, "rank": None,
                 "first_brand": None, "positive_first_brand": None,
                 "outcome": "error", "advisory_referral": None, "hedged": None}
            text, itok, vwarn = "", 0, None

        if images:
            img_runs += 1
            img_tokens_total += itok
            if itok > 0:
                img_token_runs += 1

        positions[pos_mode] = positions.get(pos_mode, 0) + 1

        if raw_writer:
            raw_writer({"design_id": design_id, "arm": arm, "model": model,
                        "mode": mode,  # ← the only addition
                        "run": r, "question": question, "position": pos_mode,
                        "target_slot": order.index("TARGET"), "n_comp": n_comp,
                        "mentioned": m.get("mentioned"), "is_first": m.get("is_first"),
                        "rank": m.get("rank"), "polarity": m.get("polarity"),
                        "refusal": m.get("refusal"),
                        "advisory_referral": m.get("advisory_referral"),
                        "first_brand": m.get("first_brand"),
                        "outcome": m.get("outcome"), "error": m.get("error"),
                        "image_attached": bool(images),
                        "image_tokens": itok,
                        "vision_warning": vwarn,
                        "comp_mode": comp_mode,
                        "response_text": text})

        results.append({k: v for k, v in m.items() if k != "brand_mentions"})

        if m.get("error"):
            errors += 1
            continue
        if m.get("mentioned"):
            hits_mentioned += 1
        if m.get("is_first"):
            hits_first += 1
        if m.get("positive_first_brand") == target_brand:
            hits_pos_first += 1
        if m.get("refusal"):
            refusals += 1
        if progress_cb:
            progress_cb()

    n_valid = (n_runs - skipped) - errors
    out = {"n": n_valid, "n_attempted": n_runs - skipped, "n_skipped": skipped,
           "n_errors": errors,
           "mentioned": hits_mentioned, "is_first": hits_first,
           "positive_first": hits_pos_first, "refusals": refusals,
           "n_comp_blocks": n_comp, "comp_mode": comp_mode,
           "position_mix": positions, "results": results}
    if img_runs or skipped_video:
        from config import MODEL_REGISTRY
        _prov = MODEL_REGISTRY.get(model, {}).get("provider", "litellm")
        out["visual"] = {
            "provider": _prov,
            # `provider` decides whether tokens are reported; `model` is what an operator
            # can act on. "litellm publishes no breakdown" names a gateway, not an engine.
            "model": model,
            # OpenAI folds image tokens into prompt_tokens with no breakdown, so a
            # LiteLLM engine reports zero media tokens on EVERY vision call — whether the
            # payload arrived or not. Treating that as a delivery failure produced a false
            # caveat on a clean finding.
            "reports_media_tokens": _prov == "vertex",
            "runs_with_image": img_runs,
            "runs_with_image_tokens": img_token_runs,
            "image_tokens_total": img_tokens_total,
            "mean_image_tokens": round(img_tokens_total / max(img_runs, 1), 1),
            # 0 media tokens with media attached means the payload never reached the
            # model — the arm is text-only and any null is an artifact
            "payload_delivered": img_token_runs > 0,
            "delivery_rate": round(img_token_runs / max(img_runs, 1), 3),
            "video_runs_skipped": skipped_video,
            "video_note": (f"{skipped_video} run(s) dropped a VIDEO payload because "
                           f"{model} declares no video support — those runs were "
                           f"text-only and are not comparable to a video-capable engine."
                           if skipped_video else None)}
    return out

def _arms_from_variants(variants):
    """Support A/B AND arbitrary multi-arm (E5 3-arm no longer degrades to A/B)."""
    if all(k in variants for k in ("variant_A", "variant_B")):
        return {k: variants[k] for k in sorted(variants) if k.startswith("variant_")}
    return {k: v for k, v in variants.items() if isinstance(v, dict) and "text" in v}

def run_experiment(gateway, design, variants, comp_context, question, target_brand,
                   competitors, *, n_runs=30, temperature=0.8, engines,
                   target_pos="balanced", progress_cb=None, raw_writer=None,
                   alias_overrides=None, done=None, budget_check=None,
                   comp_mode="full", asset_store=None, mode="ab"):
    from experiment.visual import asset_for_arm, assets, has_video
    store = asset_store if asset_store is not None else {}
    _id = design.get("design_id") or design.get("gap_id") or "unknown"
    arms = _arms_from_variants(variants)
    _has_assets = bool(assets(store, design))
    out = {"gap_id": _id, "experiment_id": design.get("experiment_id"),
           "arms": list(arms.keys()), "position_mode": target_pos,
           "resumed": bool(done), "comp_mode": comp_mode, "mode": mode,
           "visual_arms": sorted(assets(store, design)), "engines": {}}
    for eng in engines:
        vis_ok = "vision" in eng.get("supports", [])
        vid_ok = "video" in eng.get("supports", [])
        eng_out = {}
        for arm_key, arm in arms.items():
            bare = str(arm_key).replace("variant_", "")
            # PER-ARM asset. Previously a single image was hardcoded to arm B, so
            # creative-vs-creative and image-presence tests were both impossible.
            use_visual = asset_for_arm(store, design, bare) if vis_ok else None
            eng_out[bare] = run_variant(
                gateway, arm.get("text", ""), comp_context, question, target_brand,
                competitors, n_runs=n_runs, temperature=temperature,
                target_pos=target_pos, model=eng["model"], visual=use_visual,
                progress_cb=progress_cb, raw_writer=raw_writer,
                design_id=_id, arm=bare, alias_overrides=alias_overrides,
                done=done, budget_check=budget_check, mode=mode,
                comp_mode=comp_mode, video_supported=vid_ok)
        if _has_assets and not vis_ok:
            eng_out["_skipped_visual"] = (
                f"{eng['name']} declares no vision support — its arms ran TEXT-ONLY and "
                f"are not comparable to a vision engine's arms")
        elif has_video(store, design) and not vid_ok:
            eng_out["_skipped_video"] = (
                f"{eng['name']} declares no video support — video arms ran TEXT-ONLY "
                f"here. Video is Vertex-only in the current registry, so this design "
                f"has no cross-engine transfer verdict.")
        out["engines"][eng["name"]] = eng_out
    return out

def run_progressive_experiment(gateway, design, arms, comp_context, question,
                               target_brand, competitors, *, n_runs=20,
                               temperature=0.8, engines, target_pos="balanced",
                               progress_cb=None, raw_writer=None,
                               alias_overrides=None, done=None, budget_check=None,
                               comp_mode="full", asset_store=None):
    from experiment.visual import asset_for_arm, assets
    store = asset_store if asset_store is not None else {}
    _id = design.get("design_id") or design.get("gap_id") or "unknown"
    out = {"gap_id": _id, "experiment_id": "E1", "arms": list(arms.keys()),
           "position_mode": target_pos, "resumed": bool(done),
           "comp_mode": comp_mode, "visual_arms": sorted(assets(store, design)),
           "engines": {}}
    if assets(store, design):
        out["attribution_warning"] = (
            "This progressive ladder carries a media asset. Element attribution "
            "accumulates STRUCTURE across arms, so a constant image confounds every "
            "marginal effect and a varying image is not a structure test at all. The "
            "reported best element is not interpretable.")
    for eng in engines:
        vis_ok = "vision" in eng.get("supports", [])
        vid_ok = "video" in eng.get("supports", [])
        eng_out = {}
        for arm_key, arm in arms.items():
            # PER-ARM asset. Previously a single image was hardcoded to the final arm,
            # which conflated the image with the accumulated structure of the ladder.
            use_visual = asset_for_arm(store, design, arm_key) if vis_ok else None
            eng_out[arm_key] = run_variant(
                gateway, arm.get("text", ""), comp_context, question, target_brand,
                competitors, n_runs=n_runs, temperature=temperature,
                target_pos=target_pos, model=eng["model"], visual=use_visual,
                progress_cb=progress_cb, raw_writer=raw_writer,
                design_id=_id, arm=arm_key, alias_overrides=alias_overrides,
                done=done, budget_check=budget_check, mode="progressive",
                comp_mode=comp_mode, video_supported=vid_ok)
        if assets(store, design) and not vis_ok:
            eng_out["_skipped_visual"] = (
                f"{eng['name']} declares no vision support — its arms ran TEXT-ONLY and "
                f"are not comparable to a vision engine's arms")
        out["engines"][eng["name"]] = eng_out
    return out