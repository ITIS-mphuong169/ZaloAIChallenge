"""Training data for fine-tuning the language model, built from the train answers.

One record per train sample: the exact chat the agent sees at inference time (system prompt with context, tool specs
and retrieved examples; the earlier turns; the request), followed by the answer written in the agent's protocol.
`n_context` is the index of the first message of that answer: only the assistant messages from there on are targets.
"""
from __future__ import annotations

import json

from .prompt import build_messages, call_json, result_text

FINAL_REPLY = '{"reply": "…"}'


def record(sample: dict, catalog: dict, examples: list[dict] | None, texts: dict) -> dict:
    request = texts.get(sample["audio"]) or sample["audio_text"]
    msgs = build_messages(sample, request, texts, catalog, examples)
    n_context = len(msgs)
    lab = sample["label"]
    calls = lab["tool_calls"]
    for i, (c, r) in enumerate(zip(calls, lab["tool_results"])):
        msgs.append({"role": "assistant", "content": call_json(c["name"], c.get("arguments"))})
        last = i == len(calls) - 1
        if last and c["name"] == "sys.ask_missing_info":
            break  # the agent stops here and is never asked for a reply
        msgs.append({"role": "user", "content": result_text(r)})
    else:
        msgs.append({"role": "assistant", "content": FINAL_REPLY})
    return {"id": sample["id"], "n_context": n_context, "messages": msgs}


def export(samples: list[dict], catalog: dict, out_path: str, retriever=None, shots: int = 4, texts: dict | None = None) -> dict:
    texts = texts or {}
    n = chars = targets = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            request = texts.get(s["audio"]) or s["audio_text"]
            ex = retriever.top(request, shots, exclude_id=s["id"], installed=set(s["context"]["installed_apps"])) if retriever and shots else None
            rec = record(s, catalog, ex, texts)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            chars += sum(len(m["content"]) for m in rec["messages"])
            targets += sum(m["role"] == "assistant" for m in rec["messages"][rec["n_context"]:])
    return {"records": n, "mean_chars": chars // max(1, n), "target_messages": targets}
