"""Load every file of a business package into raw tables (messy values kept as-is)."""
from __future__ import annotations

import csv
import os
import re
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from .readme import Readme, parse_readme
from .textnorm import is_blank, nfc, parse_datetime, parse_num


@dataclass
class RawTable:
    name: str
    file: str
    sheet: str | None
    columns: list
    df: pd.DataFrame  # raw object values + "_sheet" / "_file" / "_row"
    meanings: dict = field(default_factory=dict)  # column -> README meaning
    parts: list = field(default_factory=list)  # (file, sheet) merged in


def _label_like(v) -> bool:
    if is_blank(v) or not isinstance(v, str):
        return False
    s = v.strip()
    if len(s) > 60:
        return False
    if parse_num(s) is not None and re.fullmatch(r"[-\d.,\s]+", s):
        return False
    if parse_datetime(s) is not None:
        return False
    return True


def detect_header(rows: list, known: set) -> int | None:
    """Index of the header row among the first rows of a sheet."""
    lim = min(len(rows), 25)
    width = max((sum(1 for v in r if not is_blank(v)) for r in rows[:lim]), default=0)
    if width == 0:
        return None
    if known:
        best, best_i = 0, None
        for i in range(lim):
            hit = sum(1 for v in rows[i] if isinstance(v, str) and nfc(v.strip()) in known)
            if hit > best:
                best, best_i = hit, i
        if best_i is not None and best >= max(1, min(2, len(known))):
            return best_i
    for i in range(lim):
        r = rows[i]
        non_empty = [v for v in r if not is_blank(v)]
        if len(non_empty) < max(2, 0.6 * width) and not (width == 1 and len(non_empty) == 1):
            continue
        if not all(_label_like(v) for v in non_empty):
            continue
        if len(set(str(v).strip() for v in non_empty)) < len(non_empty):
            continue
        return i
    return None


def _frame(header: list, body: list) -> tuple[list, pd.DataFrame]:
    idx = [j for j, h in enumerate(header) if not is_blank(h)]
    cols, seen = [], {}
    for j in idx:
        c = nfc(str(header[j]).strip())
        if c in seen:
            seen[c] += 1
            c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    hdr_vals = [str(header[j]).strip() for j in idx]
    data = []
    for rn, r in enumerate(body):
        vals = [r[j] if j < len(r) else None for j in idx]
        if all(is_blank(v) for v in vals):
            continue
        if [str(v).strip() if v is not None else "" for v in vals] == hdr_vals:
            continue  # repeated header line
        data.append(vals + [rn])
    df = pd.DataFrame(data, columns=cols + ["_row"], dtype=object)
    return cols, df


def _read_csv(path: str) -> list:
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-16", "cp1258", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeError:
            continue
    first = text.split("\n", 1)[0]
    delim = max([",", ";", "\t", "|"], key=first.count)
    return [row for row in csv.reader(text.splitlines(), delimiter=delim)]


def load_raw(biz_dir: str) -> tuple[Readme, list]:
    """Return (readme, [RawTable]) for one business folder."""
    rd = parse_readme(os.path.join(biz_dir, "README.md"))
    pieces = []  # (file, sheet, cols, df, meanings)
    for f in sorted(os.listdir(biz_dir)):
        p = os.path.join(biz_dir, f)
        low = f.lower()
        if low.endswith(".csv") or low.endswith(".tsv") or low.endswith(".txt"):
            rows = _read_csv(p)
            _add_sheet(pieces, rd, f, None, rows)
        elif low.endswith((".xlsx", ".xlsm", ".xls")):
            from python_calamine import CalamineWorkbook

            wb = CalamineWorkbook.from_path(p)
            for sn in wb.sheet_names:
                rows = wb.get_sheet_by_name(sn).to_python(skip_empty_area=False)
                _add_sheet(pieces, rd, f, sn, rows)
        elif low.endswith((".sqlite", ".db", ".sqlite3")):
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            try:
                names = [r[0] for r in con.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'")]
                for t in names:
                    cur = con.execute(f'select * from "{t}"')
                    cols = [nfc(c[0]) for c in cur.description]
                    df = pd.DataFrame(cur.fetchall(), columns=cols, dtype=object)
                    df["_row"] = range(len(df))
                    pieces.append((f, t, cols, df, rd.meanings_for(f, t), True))
            finally:
                con.close()
    return rd, _merge(pieces)


def _add_sheet(pieces, rd, f, sn, rows):
    if not rows:
        return
    meanings = rd.meanings_for(f, sn)
    hi = detect_header(rows, set(meanings))
    if hi is None:
        return
    cols, df = _frame(rows[hi], rows[hi + 1:])
    if not cols:
        return
    pieces.append((f, sn, cols, df, meanings, False))


def _stem(f: str) -> str:
    return os.path.splitext(f)[0]


def _merge(pieces) -> list:
    groups: dict = {}
    for f, sn, cols, df, meanings, is_sql in pieces:
        key = (("sql", sn) if is_sql else ("x", tuple(sorted(cols))))
        groups.setdefault(key, []).append((f, sn, cols, df, meanings))
    tables, used = [], set()
    for key, items in groups.items():
        f0, sn0, cols0, _, _ = items[0]
        dfs, meanings = [], {}
        for f, sn, cols, df, mn in items:
            df = df.copy()
            df["_sheet"] = sn
            df["_file"] = f
            dfs.append(df)
            meanings.update(mn)
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        files = sorted({f for f, *_ in items})
        if key[0] == "sql":
            name = sn0
        elif len(items) == 1:
            n_sheets = sum(1 for (ff, *_r) in pieces if ff == f0)
            name = _stem(f0) if (sn0 is None or n_sheets == 1) else sn0
        elif len(files) == 1:
            name = _stem(f0)
        else:
            name = os.path.commonprefix([_stem(x) for x in files]).rstrip("_- .0123456789") or _stem(f0)
        base, k = name, 2
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        tables.append(RawTable(name=name, file=f0, sheet=sn0, columns=list(cols0), df=df, meanings=meanings,
                               parts=[(f, sn) for f, sn, *_ in items]))
    return tables
