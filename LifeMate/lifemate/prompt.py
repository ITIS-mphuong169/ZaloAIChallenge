"""Turns one sample into the chat messages the model sees.

The model talks in a tiny text protocol that every chat model can follow, so nothing here depends on a particular
server or tool-calling template:

    {"call": "<app.tool>", "args": {...}}     one tool call; the next user message is "RESULT {...}"
    {"reply": "<short Vietnamese answer>"}     finished
"""
from __future__ import annotations

import datetime as dt
import json

SYSTEM = """You are LifeMate, a Vietnamese voice assistant on the user's phone. The user speaks Vietnamese; the text you receive is an automatic transcript, so it may contain recognition errors and regional dialect words. You help only by calling the tools of the apps installed on this phone.

Answer with exactly one JSON object and nothing else:
  {"call": "<app.tool>", "args": {...}}    to call one tool; you then receive its result as "RESULT {...}"
  {"reply": "<short answer in Vietnamese>"}  when you are finished

Rules
1. Only the tools listed under "Installed apps" exist. If the request needs an app or a function that is not installed, call nothing and reply that you cannot do it. Never use another app as a substitute.
2. Small talk, thanks, or a request the user takes back in the same sentence ("à thôi khỏi", "đừng ...") needs no tool call.
3. When the user corrects themself ("à không", "mà thôi", "nhầm"), follow only the final version.
4. Get IDs from read tools first (search / list / get). Never invent an ID; copy values exactly as the tools return them. Facts already shown earlier in the conversation can be reused without reading them again.
5. If a required argument was not given and cannot be found with a read tool or in the conversation, or a lookup returns several equally good matches (for example two contacts with the same name), do not guess. Finish the other parts of the request first, then call sys.ask_missing_info with the tool name and the names of the missing parameters, and stop.
6. Conditional requests ("nếu ... thì ..."): check the condition with a read tool, then do only the branch that applies. If no branch applies, stop.
7. Do each requested action exactly once, in the order asked. Make no call that was not asked for and never repeat a read.
8. Dates and times are relative to the current time below; use the calendar to resolve weekdays. A date-time argument is ISO-8601 with seconds and offset, for example 2026-11-11T05:30:00+07:00.
9. Free-text arguments (content, title, note ...) keep the user's own words, short, without the command itself."""

_WEEKDAY_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]


def calendar_lines(now_iso: str) -> str:
    """Weekday of today and the dates of last / this / next week, so the model never has to compute a weekday."""
    now = dt.datetime.fromisoformat(now_iso)
    today = now.date()
    monday = today - dt.timedelta(days=today.weekday())
    rows = [f"Now: {now_iso} — {_WEEKDAY_VI[today.weekday()]}"]
    for label, start in (("tuần trước", monday - dt.timedelta(days=7)), ("tuần này", monday),
                         ("tuần sau", monday + dt.timedelta(days=7)), ("tuần sau nữa", monday + dt.timedelta(days=14))):
        days = [start + dt.timedelta(days=i) for i in range(7)]
        rows.append(f"{label}: " + ", ".join(f"{_WEEKDAY_VI[d.weekday()]} {d.isoformat()}" + (" (hôm nay)" if d == today else "") for d in days))
    return "\n".join(rows)


def context_block(sample: dict) -> str:
    c = sample["context"]
    p = c.get("user_profile") or {}
    loc = c.get("location") or {}
    lines = [calendar_lines(c["current_time"]), f"Timezone: {c.get('timezone', '')}",
             f"User: {p.get('name', '')} | home: {p.get('home_address', '')} | work: {p.get('work_address', '')} | city: {p.get('city', '')}"]
    if loc:
        lines.append(f"Current location: lat {loc.get('lat')}, lng {loc.get('lng')}")
    return "\n".join(lines)


def _param_line(name: str, p: dict, required: bool) -> str:
    t = p.get("type", "any")
    if t == "array":
        it = p.get("items") or {}
        if it.get("type") == "object":
            inner = ", ".join(f"{k}{'*' if k in it.get('required', []) else ''}: {v.get('type', 'any')}" for k, v in (it.get("properties") or {}).items())
            t = "array of {" + inner + "}"
        else:
            t = f"array of {it.get('type', 'any')}"
            if it.get("enum"):
                t += " in " + "|".join(map(str, it["enum"]))
    bits = [t]
    if p.get("enum"):
        bits.append("one of " + "|".join(json.dumps(x, ensure_ascii=False) if not isinstance(x, str) else x for x in p["enum"]))
    if p.get("format"):
        bits.append(p["format"])
    for k, sym in (("minimum", ">="), ("exclusiveMinimum", ">"), ("maximum", "<="), ("exclusiveMaximum", "<")):
        if k in p:
            bits.append(f"{sym} {p[k]}")
    if p.get("x-match") == "fuzzy":
        bits.append("free text")
    if p.get("description"):
        bits.append(p["description"])
    return f"    {name}{'*' if required else ''}: " + "; ".join(bits)


def tool_text(tool: dict) -> str:
    spec = tool.get("parameters") or {}
    req = set(spec.get("required") or [])
    head = f"- {tool['name']} [{tool.get('kind', '')}] {tool.get('description', '')}"
    if tool.get("returns"):
        head += f" -> {tool['returns']}"
    lines = [head] + [_param_line(k, p, k in req) for k, p in (spec.get("properties") or {}).items()]
    if spec.get("x-one-of"):
        lines.append("    needs one of: " + " | ".join(" + ".join(g) for g in spec["x-one-of"]))
    return "\n".join(lines)


def tools_block(sample: dict, catalog: dict) -> str:
    """Specs of the installed apps' tools plus sys.ask_missing_info; * marks a required parameter."""
    apps = {a["id"]: a for a in catalog["apps"]}
    by_name = {t["name"]: t for a in catalog["apps"] for t in a["tools"]}
    out = []
    for app_id, inst in sample["context"]["installed_apps"].items():
        a = apps.get(app_id, {})
        out.append(f"## {app_id} — {a.get('name_vi', '')}: {a.get('description', '')}")
        out += [tool_text(by_name[t]) for t in inst.get("tools", []) if t in by_name]
    if "sys.ask_missing_info" in by_name:
        out.append("## sys — always available")
        out.append(tool_text(by_name["sys.ask_missing_info"]))
    return "\n".join(out)


def call_json(name: str, args: dict) -> str:
    return json.dumps({"call": name, "args": args or {}}, ensure_ascii=False)


def result_text(result, limit: int = 3000) -> str:
    s = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if len(s) > limit:
        s = s[:limit] + " …(truncated)"
    return "RESULT " + s


def reply_json(text: str) -> str:
    return json.dumps({"reply": text or ""}, ensure_ascii=False)


def history_messages(sample: dict, texts: dict, result_limit: int = 1500) -> list[dict]:
    """Earlier turns in the protocol format. `texts` maps an audio path to its transcript."""
    msgs = []
    for h in sample.get("history", []):
        if h["role"] == "user":
            # a transcript from the speech model, when one was given, is what the agent really hears; the released
            # transcript (train only) is the fallback
            msgs.append({"role": "user", "content": texts.get(h.get("audio")) or h.get("audio_text") or ""})
            continue
        calls = h.get("tool_calls") or []
        results = h.get("tool_results") or []
        for i, c in enumerate(calls):
            msgs.append({"role": "assistant", "content": call_json(c["name"], c.get("arguments"))})
            msgs.append({"role": "user", "content": result_text(results[i], result_limit) if i < len(results) else "RESULT (not recorded)"})
        msgs.append({"role": "assistant", "content": reply_json(h.get("text", ""))})
    return msgs


def example_text(ex: dict, result_limit: int = 350) -> str:
    """One solved train sample as a compact demonstration."""
    lines = [f"### Example — now {ex['context']['current_time']}; installed: {', '.join(ex['context']['installed_apps'])}"]
    for h in ex.get("history", [])[-4:]:
        if h["role"] == "user":
            lines.append("User: " + (h.get("audio_text") or ""))
        else:
            for c in h.get("tool_calls") or []:
                lines.append(call_json(c["name"], c.get("arguments")))
            lines.append(reply_json(h.get("text", "")))
    lines.append("User: " + ex["audio_text"])
    lab = ex["label"]
    for c, r in zip(lab["tool_calls"], lab["tool_results"]):
        lines.append(call_json(c["name"], c.get("arguments")))
        lines.append(result_text(r, result_limit))
    lines.append('{"reply": "…"}')
    return "\n".join(lines)


def build_messages(sample: dict, request_text: str, texts: dict, catalog: dict, examples: list[dict] | None = None) -> list[dict]:
    system = [SYSTEM, "# Context\n" + context_block(sample), "# Installed apps (* = required parameter)\n" + tools_block(sample, catalog)]
    if examples:
        system.append("# Solved examples from other users (their phones have other apps and data; learn the conventions, do not copy IDs)\n"
                      + "\n\n".join(example_text(e) for e in examples))
    msgs = [{"role": "system", "content": "\n\n".join(system)}]
    msgs += history_messages(sample, texts)
    msgs.append({"role": "user", "content": request_text})
    return msgs
