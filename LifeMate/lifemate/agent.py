"""The tool-calling loop for one sample.

One ToolEnv is opened per sample and never reopened (competition rule: every call made counts). To keep mistakes out
of the submitted call list, a call is checked against the schema first and only executed when it is valid; an invalid
call is reported back to the model instead. A read that was already made is answered from memory, not called again.
"""
from __future__ import annotations

import json

from .data import open_env, use_split
from .llm import parse_step
from .prompt import build_messages, result_text

MAX_PROMPT_CHARS = 40000  # about 13k tokens; the demonstrations are dropped one by one above this


def fit_examples(sample: dict, request_text: str, texts: dict, catalog: dict, examples, limit: int = MAX_PROMPT_CHARS):
    """Drop demonstrations from the end (the least similar first) until the prompt fits the model's context."""
    examples = list(examples or [])
    while examples and sum(len(m["content"]) for m in build_messages(sample, request_text, texts, catalog, examples)) > limit:
        examples.pop()
    return examples


MAX_STEPS = 12          # the longest train answer has 6 calls
MAX_REJECTED = 3        # schema-invalid proposals tolerated before giving up
MAX_UNPARSED = 2        # answers that are not a protocol step


def run_sample(split: str, sample: dict, request_text: str, texts: dict, catalog: dict, chat, examples=None) -> dict:
    """Run the agent on one sample. `chat(messages) -> str` is the model. Returns the submission fields and a trace."""
    zt = use_split(split)
    env = open_env(split, sample["id"])
    examples = fit_examples(sample, request_text, texts, catalog, examples)
    messages = build_messages(sample, request_text, texts, catalog, examples)
    trace = {"id": sample["id"], "request": request_text, "shots": len(examples), "steps": []}
    reads: dict = {}
    rejected = unparsed = 0
    for _ in range(MAX_STEPS):
        try:
            raw = chat(messages)
        except Exception as e:  # the model server failed: stop here and keep what was done, so the sample still gets its line
            trace["error"] = repr(e)[:300]
            break
        step = parse_step(raw)
        if step is None:
            unparsed += 1
            trace["steps"].append({"raw": raw[:500], "note": "not a protocol step"})
            if unparsed >= MAX_UNPARSED:
                break
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user", "content": 'Answer with one JSON object only: {"call": "<app.tool>", "args": {...}} or {"reply": "..."}'}]
            continue
        if "call" not in step:
            trace["reply"] = step.get("reply")
            break
        name = step.get("call")
        args = step.get("args")
        if not isinstance(args, dict):
            args = {}
        name = name if isinstance(name, str) else str(name)
        check = env.validate(name, args)
        key = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        if not check["ok"]:
            rejected += 1
            result = {"ok": False, "error": check["error"], "note": "not executed"}
            trace["steps"].append({"call": name, "args": args, "rejected": check["error"]})
            if rejected >= MAX_REJECTED:
                break
        elif key in reads:
            result = reads[key]
            trace["steps"].append({"call": name, "args": args, "note": "repeated read answered from memory"})
        else:
            result = env.call(name, args)
            if zt.TOOLS.get(name) == "read":
                reads[key] = result
            trace["steps"].append({"call": name, "args": args, "result": result})
        messages += [{"role": "assistant", "content": json.dumps({"call": name, "args": args}, ensure_ascii=False)},
                     {"role": "user", "content": result_text(result)}]
        if name == "sys.ask_missing_info" and check["ok"]:
            break  # the assistant now waits for the user; nothing follows it in any train answer
    return {"id": sample["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(), "trace": trace}
