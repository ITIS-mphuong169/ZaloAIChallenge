"""Penalized Accuracy scorer following the official matching rules."""
from __future__ import annotations

import json

from .textnorm import norm_answer


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num_eq(a, b) -> bool:
    return _is_num(a) and _is_num(b) and abs(round(float(a), 2) - round(float(b), 2)) < 1e-6


def _str_eq(a, ref, aliases) -> bool:
    if not isinstance(a, str):
        return False
    forms = {norm_answer(ref)} | {norm_answer(x) for x in (aliases or []) if isinstance(x, str)}
    return norm_answer(a) in forms


def _cell_eq(a, ref, aliases) -> bool:
    if _is_num(ref):
        return _num_eq(a, ref)
    return _str_eq(a, ref, aliases)


def _table_eq(pred, ref, alias) -> bool:
    if not isinstance(pred, dict) or "columns" not in pred or "rows" not in pred:
        return False
    pc, rc = pred["columns"], ref["columns"]
    if len(pc) != len(rc) or not isinstance(pred["rows"], list):
        return False
    npc, nrc = [norm_answer(c) for c in pc], [norm_answer(c) for c in rc]
    order = [npc.index(c) for c in nrc] if sorted(npc) == sorted(nrc) and len(set(npc)) == len(npc) else list(range(len(rc)))
    prow = [[r[i] for i in order] for r in pred["rows"] if isinstance(r, list) and len(r) == len(rc)]
    if len(prow) != len(ref["rows"]) or len(prow) != len(pred["rows"]):
        return False
    used = set()
    for ri, rr in enumerate(ref["rows"]):
        al = alias[ri] if isinstance(alias, list) and ri < len(alias) and isinstance(alias[ri], list) else []
        hit = None
        for pi, pr in enumerate(prow):
            if pi in used:
                continue
            if all(_cell_eq(pr[k], rr[k], al if k == 0 else None) for k in range(len(rr))):
                hit = pi
                break
        if hit is None:
            return False
        used.add(hit)
    return True


def score_one(pred: dict | None, ref: dict) -> int:
    """+1 correct, 0 abstain, -1 wrong."""
    if pred is None or pred.get("type") in (None, "abstain"):
        return 0
    if pred.get("type") != ref.get("type"):
        return -1
    t, a, r, alias = ref["type"], pred.get("answer"), ref.get("answer"), ref.get("alias")
    if t == "unanswerable":
        return 1
    if t in ("number", "percent"):
        ok = _num_eq(a, r)
    elif t == "string":
        ok = _str_eq(a, r, alias)
    elif t == "date":
        ok = isinstance(a, str) and a == r
    elif t == "boolean":
        ok = isinstance(a, bool) and a == r
    elif t == "list":
        ok = isinstance(a, list) and len(a) == len(r) and all(
            _cell_eq(x, y, alias[i] if isinstance(alias, list) and i < len(alias) and isinstance(alias[i], list) else None)
            for i, (x, y) in enumerate(zip(a, r)))
    elif t == "table":
        ok = _table_eq(a, r, alias)
    else:
        ok = False
    return 1 if ok else -1


def load_jsonl(path: str) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def penalized_accuracy(preds: dict, refs: dict) -> dict:
    c = w = ab = 0
    for qid, ref in refs.items():
        s = score_one(preds.get(qid), ref)
        c += s == 1
        w += s == -1
        ab += s == 0
    n = max(1, len(refs))
    return {"n": len(refs), "correct": c, "wrong": w, "abstain": ab, "pa": (c - w) / n}
