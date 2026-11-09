"""Command line entry points.

  python -m datamate.cli predict --data <split_dir> --out submission.jsonl [--log run_log.jsonl]
  python -m datamate.cli eval    --data <split_dir with answers.jsonl> [--show wrong]

<split_dir> holds questions.jsonl, calendar.csv, rulebook.md and data/<business_id>/...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from .calendar import load_calendar
from .engine import answer_question
from .model import build
from .score import load_jsonl, score_one


def _find(split: str, name: str) -> str:
    for cand in (os.path.join(split, name), os.path.join(split, "data", name)):
        if os.path.exists(cand):
            return cand
    return os.path.join(split, name)


def _biz_dir(split: str, biz_id: str) -> str:
    for cand in (os.path.join(split, "data", biz_id), os.path.join(split, biz_id)):
        if os.path.isdir(cand):
            return cand
    return os.path.join(split, "data", biz_id)


class Runner:
    def __init__(self, split: str):
        self.split = split
        self.cal = load_calendar(_find(split, "calendar.csv"))
        self.bizs: dict = {}

    def biz(self, biz_id: str):
        if biz_id not in self.bizs:
            t = time.time()
            try:
                self.bizs[biz_id] = build(_biz_dir(self.split, biz_id))
                print(f"[datamate] loaded {biz_id} in {time.time() - t:.1f}s", file=sys.stderr, flush=True)
            except Exception:  # a broken package must not stop the run
                traceback.print_exc()
                self.bizs[biz_id] = None
        return self.bizs[biz_id]

    def answer(self, q: dict) -> dict:
        b = self.biz(q["business_id"])
        if b is None:
            return {"type": "abstain", "why": "business package failed to load"}
        try:
            return answer_question(b, q["question"], q["as_of_date"], self.cal)
        except Exception as e:
            return {"type": "abstain", "why": "error: " + repr(e)[:200]}


def to_submission(qid: str, res: dict) -> dict:
    t = res.get("type", "abstain")
    if t == "abstain":
        return {"id": qid, "type": "abstain"}
    if t == "unanswerable":
        return {"id": qid, "type": "unanswerable", "reason": str(res.get("reason", ""))}
    return {"id": qid, "type": t, "answer": res.get("answer")}


def cmd_predict(a) -> None:
    qs = load_jsonl(_find(a.data, "questions.jsonl"))
    run = Runner(a.data)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    logf = open(a.log, "w", encoding="utf-8") if a.log else None
    n_ans, t0 = 0, time.time()
    with open(a.out, "w", encoding="utf-8") as out:
        for i, q in enumerate(qs):
            res = run.answer(q)
            sub = to_submission(q["id"], res)
            n_ans += sub["type"] != "abstain"
            out.write(json.dumps(sub, ensure_ascii=False) + "\n")
            out.flush()
            if logf:
                logf.write(json.dumps({"id": q["id"], "business_id": q["business_id"], "question": q["question"],
                                       "as_of_date": q["as_of_date"], "result": res}, ensure_ascii=False, default=str) + "\n")
                logf.flush()
    if logf:
        logf.close()
    print(f"[datamate] {len(qs)} questions, {n_ans} answered, {len(qs) - n_ans} abstained, {time.time() - t0:.0f}s -> {a.out}",
          file=sys.stderr)


def cmd_eval(a) -> None:
    qs = load_jsonl(_find(a.data, "questions.jsonl"))
    refs = {r["id"]: r for r in load_jsonl(_find(a.data, "answers.jsonl"))}
    run = Runner(a.data)
    c = w = ab = 0
    for q in qs:
        res = run.answer(q)
        s = score_one(res, refs[q["id"]])
        c += s == 1
        w += s == -1
        ab += s == 0
        if (a.show == "wrong" and s == -1) or (a.show == "abstain" and s == 0):
            print(json.dumps({"q": q["question"], "biz": q["business_id"], "as_of": q["as_of_date"], "got": res,
                              "ref": refs[q["id"]].get("answer")}, ensure_ascii=False, default=str))
    n = max(1, len(qs))
    print(f"n={len(qs)} correct={c} wrong={w} abstain={ab} PA={(c - w) / n:.4f} "
          f"coverage={(c + w) / n:.3f} precision={(c / (c + w) if c + w else 0):.4f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="datamate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", default="")
    p.set_defaults(fn=cmd_predict)
    e = sub.add_parser("eval")
    e.add_argument("--data", required=True)
    e.add_argument("--show", default="")
    e.set_defaults(fn=cmd_eval)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
