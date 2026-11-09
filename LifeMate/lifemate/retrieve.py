"""Picks solved train samples that resemble a new request, to show the model as demonstrations.

Similarity is TF-IDF cosine over the words of the request (accents removed), so it needs no model and no GPU.
"""
from __future__ import annotations

import collections
import math
import re
import unicodedata


def _words(s: str) -> list[str]:
    t = unicodedata.normalize("NFD", (s or "").lower()).replace("đ", "d")
    t = "".join(c for c in t if not unicodedata.combining(c))
    w = re.findall(r"[^\W_]+", t)
    return w + [a + "_" + b for a, b in zip(w, w[1:])]


class Retriever:
    def __init__(self, samples: list[dict]):
        self.samples = samples
        docs = [collections.Counter(_words(s["audio_text"])) for s in samples]
        df = collections.Counter(w for d in docs for w in d)
        n = len(docs)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items()}
        self.vecs = []
        for d in docs:
            v = {w: (1 + math.log(c)) * self.idf[w] for w, c in d.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            self.vecs.append({w: x / norm for w, x in v.items()})
        self.index = collections.defaultdict(list)
        for i, v in enumerate(self.vecs):
            for w, x in v.items():
                self.index[w].append((i, x))

    def top(self, text: str, k: int = 4, exclude_id: str | None = None, installed: set | None = None) -> list[dict]:
        """The k most similar samples. With `installed`, samples whose answer uses an app the phone does not have rank lower."""
        q = collections.Counter(_words(text))
        qv = {w: (1 + math.log(c)) * self.idf[w] for w, c in q.items() if w in self.idf}
        norm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
        score = collections.defaultdict(float)
        for w, x in qv.items():
            for i, y in self.index[w]:
                score[i] += x / norm * y
        ranked = []
        for i, sc in score.items():
            s = self.samples[i]
            if s["id"] == exclude_id:
                continue
            if installed is not None:
                used = {c["name"].split(".")[0] for c in s["label"]["tool_calls"]} - {"sys"}
                if used - installed:
                    sc *= 0.5
            ranked.append((sc, i))
        ranked.sort(key=lambda t: (-t[0], t[1]))
        return [self.samples[i] for _sc, i in ranked[:k]]
