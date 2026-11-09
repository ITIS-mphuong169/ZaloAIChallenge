"""Local scoring of predictions against the train labels.

The organisers replay the submitted tool_calls on the initial state, compare the resulting state with the answer and
compute an F1 over the calls. Their exact formula (how the two parts are combined, how close a free-text argument has
to be) is not in the released package, so both parts are reported separately here and the free-text rule is our own.
"""
from __future__ import annotations

import json
import re
import unicodedata


def _fold(s) -> str:
    t = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).split())


def _tokens(s) -> set:
    return set(re.findall(r"[^\W_]+", _fold(s)))


def _canon(v):
    if isinstance(v, str):
        return re.sub(r"(\d)\s+(?=[^\W\d_])", r"\1", " ".join(v.lower().split()))
    if isinstance(v, list):
        out = [_canon(x) for x in v]
        if all(not isinstance(x, (dict, list)) for x in out):
            out = sorted(out, key=lambda x: (type(x).__name__, str(x)))
        return out
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in v.items()}
    return v


def fuzzy_params(catalog: dict) -> dict:
    """{tool name: set of argument names the catalog marks `x-match: fuzzy`}"""
    out = {}
    for app in catalog["apps"]:
        for t in app["tools"]:
            props = (t.get("parameters") or {}).get("properties") or {}
            out[t["name"]] = {k for k, p in props.items() if p.get("x-match") == "fuzzy"}
    return out


def text_close(a, b, threshold: float = 0.5) -> bool:
    """Free text counts as the same when the word overlap (F1 of the two word sets, accents ignored) reaches the threshold."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return ta == tb
    inter = len(ta & tb)
    if not inter:
        return False
    p, r = inter / len(ta), inter / len(tb)
    return 2 * p * r / (p + r) >= threshold


def call_match(pred: dict, gold: dict, fuzzy: dict) -> bool:
    if pred.get("name") != gold.get("name"):
        return False
    pa = {k: v for k, v in (pred.get("arguments") or {}).items() if v not in (None, "", [], {})}
    ga = {k: v for k, v in (gold.get("arguments") or {}).items() if v not in (None, "", [], {})}
    if set(pa) != set(ga):
        return False
    fz = fuzzy.get(gold["name"], set())
    for k, gv in ga.items():
        if k in fz and isinstance(gv, str) and isinstance(pa[k], str):
            if not text_close(pa[k], gv):
                return False
        elif json.dumps(_canon(pa[k]), sort_keys=True) != json.dumps(_canon(gv), sort_keys=True):
            return False
    return True


def call_f1(pred: list, gold: list, fuzzy: dict) -> tuple[float, int, int, int]:
    """(F1, matched, n_pred, n_gold); each gold call is matched at most once, order is ignored."""
    used = [False] * len(gold)
    hit = 0
    for p in pred:
        for i, g in enumerate(gold):
            if not used[i] and call_match(p, g, fuzzy):
                used[i] = True
                hit += 1
                break
    if not pred and not gold:
        return 1.0, 0, 0, 0
    if not hit:
        return 0.0, 0, len(pred), len(gold)
    pr, rc = hit / len(pred), hit / len(gold)
    return 2 * pr * rc / (pr + rc), hit, len(pred), len(gold)
