"""Command line.

  python -m lifemate.cli check-labels       replay the train answers and compare with the released results
  python -m lifemate.cli asr                speech -> text for a split (cached); on train also the word error rate
  python -m lifemate.cli predict            run the agent and write the submission file
  python -m lifemate.cli eval               score a prediction file against the train answers
  python -m lifemate.cli export-sft         write fine-tuning data for the language model

Sample selection, shared by asr / predict / export-sft:  --every K (every K-th sample), --limit N (first N of
those), --ids a,b,c.  `--every 10` on train is the dev part that no training ever sees (lifemate.data.dev_split).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

from .data import dev_split, load_catalog, load_samples, open_env, use_split
from .score import call_f1, fuzzy_params


def _pick(samples: list[dict], a) -> list[dict]:
    if a.ids:
        want = set(a.ids.split(","))
        return [s for s in samples if s["id"] in want]
    if a.every > 1:
        samples = samples[:: a.every]
    return samples[: a.limit] if a.limit else samples


def _add_pick(p) -> None:
    p.add_argument("--every", type=int, default=1, help="take every K-th sample")
    p.add_argument("--limit", type=int, default=0, help="take only the first N of the selected samples")
    p.add_argument("--ids", default="", help="comma-separated sample ids")


def gold_hash(split: str, sample: dict) -> str:
    env = open_env(split, sample["id"])
    for c in sample["label"]["tool_calls"]:
        env.call(c["name"], c["arguments"])
    return env.get_state_hash()


def cmd_check_labels(a) -> None:
    zt = use_split("train")
    samples = load_samples("train")
    bad_result = bad_state = rejected = 0
    for s in samples:
        env = open_env("train", s["id"])
        lab = s["label"]
        for c, want in zip(lab["tool_calls"], lab["tool_results"]):
            got = env.call(c["name"], c["arguments"])
            rejected += got.get("ok") is False
            if got != want:
                bad_result += 1
                if a.show:
                    print("result differs", s["id"], c["name"], json.dumps(got, ensure_ascii=False)[:200])
        state = env.get_state()
        for app, want in lab["final_state"].items():
            if zt.canon({app: state.get(app)}) != zt.canon({app: want}):
                bad_state += 1
                if a.show:
                    print("state differs", s["id"], app)
    print(f"{len(samples)} samples replayed: {bad_result} tool results differ, {bad_state} final app states differ, "
          f"{rejected} label calls rejected by the schema check")


def cmd_eval(a) -> None:
    samples = {s["id"]: s for s in load_samples("train")}
    fuzzy = fuzzy_params(load_catalog("train"))
    preds = [json.loads(line) for line in open(a.pred, encoding="utf-8") if line.strip()]
    n = state_ok = exact = 0
    f1_sum = 0.0
    by_len = collections.defaultdict(lambda: [0, 0, 0.0])
    shown = 0
    for p in preds:
        s = samples[p["id"]]
        gold = s["label"]["tool_calls"]
        ok = p["state_hash"] == gold_hash("train", s)
        f1, _hit, _n_pred, _n_gold = call_f1(p["tool_calls"], gold, fuzzy)
        n += 1
        state_ok += ok
        f1_sum += f1
        exact += ok and f1 == 1.0
        b = by_len[min(len(gold), 5)]
        b[0] += 1
        b[1] += ok
        b[2] += f1
        if a.show and shown < a.show and not (ok and f1 == 1.0):
            shown += 1
            print(json.dumps({"id": p["id"], "request": s["audio_text"], "state_ok": ok, "f1": round(f1, 2),
                              "pred": p["tool_calls"], "gold": gold}, ensure_ascii=False))
    if not n:
        print("no predictions")
        return
    print(f"n={n} state_match={state_ok / n:.4f} call_f1={f1_sum / n:.4f} both_perfect={exact / n:.4f}")
    for k in sorted(by_len):
        c, so, fs = by_len[k]
        print(f"  gold calls {'5+' if k == 5 else k}: n={c} state_match={so / c:.3f} call_f1={fs / c:.3f}")


def cmd_asr(a) -> None:
    from .asr import audio_paths, make_transcriber, transcribe_split, wer
    samples = _pick(load_samples(a.split), a)
    cache_path = a.cache or f"result/asr_{a.split}_{a.model.strip('/').replace('/', '_')}.jsonl"
    tr = make_transcriber(a.model, device=a.device, compute_type=a.compute_type, beam_size=a.beam, prompt=a.prompt, backend=a.backend)
    t0 = time.time()
    cache = transcribe_split(a.split, samples, cache_path, tr, final_only=a.final_only)
    n_clips = len(samples) if a.final_only else sum(len(audio_paths(s)) for s in samples)
    print(f"{n_clips} clips of {len(samples)} samples -> {cache_path} ({time.time() - t0:.0f}s for the clips not cached before)")
    if a.split == "train":
        final = [(s["audio_text"], cache[s["audio"]]) for s in samples]
        hist = [(h["audio_text"], cache[h["audio"]]) for s in samples for h in s["history"] if h["role"] == "user" and h["audio"] in cache]
        print(f"word error rate: final requests {wer(final):.4f} ({len(final)} clips), history turns {wer(hist):.4f} ({len(hist)} clips)")
        exact = sum(wer([p]) == 0 for p in final)
        print(f"final requests transcribed with no word error: {exact}/{len(final)}")
        for ref, hyp in final[: a.show]:
            if wer([(ref, hyp)]) > 0:
                print("  ref:", ref)
                print("  hyp:", hyp)


def _oracle_chat(sample: dict):
    """Stand-in for the model that replays the train answer; used to test the loop itself."""
    from .prompt import call_json
    gold = list(sample["label"]["tool_calls"])
    state = {"i": 0}

    def chat(_messages):
        i = state["i"]
        state["i"] += 1
        return call_json(gold[i]["name"], gold[i]["arguments"]) if i < len(gold) else '{"reply": "xong"}'
    return chat


def _pool(name: str) -> list[dict]:
    """Train samples the demonstrations are taken from: all of them, or only the fit part (honest scores on dev)."""
    train = load_samples("train")
    return dev_split(train)[0] if name == "fit" else train


def cmd_predict(a) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from .agent import run_sample
    from .asr import load_cache
    from .llm import LLM
    from .retrieve import Retriever

    samples = _pick(load_samples(a.split), a)
    catalog = load_catalog(a.split)
    texts = load_cache(a.asr) if a.asr else {}
    retriever = Retriever(_pool(a.pool)) if a.shots > 0 else None
    llm = None if a.oracle else LLM(a.base_url, a.model, a.api_key, max_tokens=a.max_tokens, extra=json.loads(a.extra) if a.extra else None)
    if a.asr:
        missing = [s["id"] for s in samples if s["audio"] not in texts]
        if missing and not a.allow_missing:
            raise SystemExit(f"no transcript for {len(missing)} samples (first: {missing[0]}) in {a.asr}; run the asr command first")
        for s in samples:
            texts.setdefault(s["audio"], "")  # --allow-missing: an empty request rather than no submission line
    done = set()
    if os.path.exists(a.out) and not a.fresh:
        done = {json.loads(line)["id"] for line in open(a.out, encoding="utf-8") if line.strip()}
    todo = [s for s in samples if s["id"] not in done]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    use_split(a.split)  # import the split's zaic_tools once, before any worker thread needs it

    def work(s: dict) -> dict:
        try:
            request = texts[s["audio"]] if a.asr else s["audio_text"]  # without --asr: the released train transcript
            shots = retriever.top(request, a.shots, exclude_id=s["id"], installed=set(s["context"]["installed_apps"])) if retriever else None
            chat = _oracle_chat(s) if a.oracle else llm.chat
            return run_sample(a.split, s, request, texts, catalog, chat, shots)
        except Exception as e:  # never lose a line of the submission: answer "nothing done" for this sample
            try:
                h = open_env(a.split, s["id"]).get_state_hash()
            except Exception:
                h = ""
            return {"id": s["id"], "tool_calls": [], "state_hash": h, "trace": {"id": s["id"], "steps": [], "error": "sample failed: " + repr(e)[:300]}}

    t0 = time.time()
    n = errors = 0
    trace_f = open(a.trace, "w" if a.fresh else "a", encoding="utf-8") if a.trace else None
    with open(a.out, "w" if a.fresh else "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        for fut in as_completed([pool.submit(work, s) for s in todo]):
            res = fut.result()
            out.write(json.dumps({"id": res["id"], "tool_calls": res["tool_calls"], "state_hash": res["state_hash"]}, ensure_ascii=False) + "\n")
            out.flush()
            if trace_f:
                trace_f.write(json.dumps(res["trace"], ensure_ascii=False) + "\n")
                trace_f.flush()
            n += 1
            if res["trace"].get("error"):
                errors += 1
                print(f"[predict] {res['id']} stopped early: {res['trace']['error']}", flush=True)
            if n % 10 == 0:
                print(f"[predict] {n}/{len(todo)} samples, {time.time() - t0:.0f}s", flush=True)
    if trace_f:
        trace_f.close()
    # put the lines back in the order of the split file (they were written as the workers finished)
    rows = {json.loads(line)["id"]: line for line in open(a.out, encoding="utf-8") if line.strip()}
    order = [s["id"] for s in load_samples(a.split) if s["id"] in rows]
    with open(a.out, "w", encoding="utf-8") as out:
        out.writelines(rows[i] if rows[i].endswith("\n") else rows[i] + "\n" for i in order)
    msg = f"{n} samples run ({errors} stopped early by an error), {len(order)} lines in {a.out}, {time.time() - t0:.0f}s"
    if llm and llm.calls:
        msg += f"; {llm.calls} model calls, {llm.prompt_tokens} prompt + {llm.completion_tokens} completion tokens, {llm.seconds:.0f}s in the model"
    print(msg)


def cmd_export_sft(a) -> None:
    from .asr import load_cache
    from .retrieve import Retriever
    from .sft import export
    train = load_samples("train")
    fit, dev = dev_split(train)
    part = {"fit": fit, "dev": dev, "all": train}[a.part]
    samples = _pick(part, a)
    retriever = Retriever(_pool(a.pool)) if a.shots > 0 else None
    stats = export(samples, load_catalog("train"), a.out, retriever, a.shots, load_cache(a.asr) if a.asr else {})
    print(f"{stats['records']} records -> {a.out}; about {stats['mean_chars']} characters each, {stats['target_messages']} target messages")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="lifemate")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check-labels")
    c.add_argument("--show", action="store_true")
    c.set_defaults(fn=cmd_check_labels)

    e = sub.add_parser("eval")
    e.add_argument("--pred", required=True)
    e.add_argument("--show", type=int, default=0)
    e.set_defaults(fn=cmd_eval)

    r = sub.add_parser("asr")
    r.add_argument("--split", default="train")
    _add_pick(r)
    r.add_argument("--model", default="large-v3-turbo", help="faster-whisper model name, or a model folder")
    r.add_argument("--backend", default="auto", choices=["auto", "ct2", "hf"], help="ct2 = faster-whisper, hf = transformers")
    r.add_argument("--device", default="auto")
    r.add_argument("--compute-type", default="default")
    r.add_argument("--beam", type=int, default=5)
    r.add_argument("--cache", default="")
    r.add_argument("--show", type=int, default=0)
    r.add_argument("--final-only", action="store_true", help="only the final request of each sample, not the history turns")
    r.add_argument("--prompt", default="", help="context text given to the speech model to bias it towards domain words (ct2 backend)")
    r.set_defaults(fn=cmd_asr)

    p = sub.add_parser("predict")
    p.add_argument("--split", default="public_test")
    p.add_argument("--out", required=True)
    p.add_argument("--trace", default="")
    p.add_argument("--asr", default="", help="transcript cache written by the asr command; without it the train transcripts are used")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="default")
    p.add_argument("--api-key", default="")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--extra", default="", help="JSON object merged into every request body (server-specific options)")
    p.add_argument("--shots", type=int, default=4, help="solved train samples shown to the model")
    p.add_argument("--pool", default="all", choices=["all", "fit"], help="where the solved samples come from; use fit when scoring the dev part")
    p.add_argument("--workers", type=int, default=1, help="samples run in parallel (the model server batches the requests)")
    _add_pick(p)
    p.add_argument("--fresh", action="store_true", help="overwrite --out instead of resuming")
    p.add_argument("--allow-missing", action="store_true", help="run samples that have no transcript with an empty request (final Docker run)")
    p.add_argument("--oracle", action="store_true", help="replay the train answers instead of asking a model (tests the loop)")
    p.set_defaults(fn=cmd_predict)

    x = sub.add_parser("export-sft")
    x.add_argument("--out", required=True)
    x.add_argument("--part", default="fit", choices=["fit", "dev", "all"], help="which train samples to export")
    x.add_argument("--shots", type=int, default=4)
    x.add_argument("--pool", default="fit", choices=["all", "fit"])
    x.add_argument("--asr", default="", help="use these speech transcripts instead of the released ones where available")
    _add_pick(x)
    x.set_defaults(fn=cmd_export_sft)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
