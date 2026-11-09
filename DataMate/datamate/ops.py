"""Side books: deliveries, price history and operating expenses."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .model import Biz, _date_col, _num_col, split_name
from .textnorm import fold, is_blank, parse_datetime, parse_num, squash


def _s(v):
    return None if is_blank(v) else str(v).strip()


def _col(tb, meaning_rx: str, names: tuple):
    for c in tb.columns:
        if c in tb.meanings and re.search(meaning_rx, fold(tb.meanings[c])):
            return c
    for c in tb.columns:
        if c not in tb.meanings and split_name(c) in names:
            return c
    return None


def _pkey(biz: Biz, tb, row_code, row_name):
    if row_code is not None and getattr(biz, "prod_has_code", False):
        return row_code
    if row_name is not None:
        return biz.name2key.get(squash(row_name))
    return None


def build_ops(biz: Biz) -> None:
    biz.deliveries, biz.price_hist, biz.expenses = None, None, None
    biz.due_dates = None
    for tb in biz.tables:
        r = biz.roles.get(tb.name, {})
        hc = _col(tb, r"^han thanh toan", ("han thanh toan", "han tra", "ngay phai tra"))
        if hc and "order_id" in r and tb.name not in (biz.lines_table, biz.orders_table):
            biz.due_dates = pd.DataFrame({"order_id": tb.df[r["order_id"]].map(_s), "due": _date_col(tb.df, hc).dt.normalize().values}).dropna()
            break
    for tb in biz.tables:
        if tb.name in (biz.lines_table, biz.orders_table):
            continue
        r = biz.roles.get(tb.name, {})
        # ---- deliveries: one row per item per delivery run
        dcol = _col(tb, r"^ngay giao", ("ngay giao hang", "ngay giao", "ngay cho", "thoi gian giao"))
        bcol = _col(tb, r"^dot giao", ("lan giao", "luot giao", "dot giao", "dot"))
        if dcol and bcol and "order_id" in r and biz.deliveries is None:
            D = pd.DataFrame({"order_id": tb.df[r["order_id"]].map(_s)})
            D["date"] = _date_col(tb.df, dcol).dt.normalize().values
            D["batch"] = _num_col(tb.df[bcol]).values
            qc = r.get("qty_base") or r.get("qty")
            D["qty"] = _num_col(tb.df[qc]).values if qc else np.nan
            code = tb.df[r["prod_code"]].map(_s) if "prod_code" in r else None
            name = tb.df[r["prod_name"]].map(_s) if "prod_name" in r else None
            D["pkey"] = [_pkey(biz, tb, code.iloc[i] if code is not None else None, name.iloc[i] if name is not None else None)
                         for i in range(len(tb.df))]
            biz.deliveries = D.dropna(subset=["order_id", "date"])
            continue
        # ---- price history: price per base unit effective from a date
        acol = _col(tb, r"^ngay bat dau ap dung", ("ngay ap dung", "ap dung tu", "ngay doi gia", "hieu luc tu"))
        if acol and biz.price_hist is None:
            pcol = _col(tb, r"^gia ban ap dung", ("gia moi", "gia ban le", "gia ban", "gia", "gia ap dung"))
            if pcol:
                code = tb.df[r["prod_code"]].map(_s) if "prod_code" in r else None
                ncol = r.get("prod_name") or r.get("name") or next((c for c in tb.columns if split_name(c) in ("item", "san pham", "ten hang", "hang")), None)
                name = tb.df[ncol].map(_s) if ncol else None
                H = pd.DataFrame({"pkey": [_pkey(biz, tb, code.iloc[i] if code is not None else None, name.iloc[i] if name is not None else None)
                                           for i in range(len(tb.df))]})
                H["date"] = _date_col(tb.df, acol).dt.normalize().values
                H["price"] = _num_col(tb.df[pcol]).values
                biz.price_hist = H.dropna(subset=["pkey", "date", "price"]).sort_values(["pkey", "date"])
            continue
        # ---- operating expenses
        if re.search(r"chi phi|chiphi|chi tieu", split_name(tb.name)) and biz.expenses is None:
            amt = next((c for c in tb.columns if split_name(c) in ("so tien", "tien chi", "tien", "thanh tien", "gia tri", "chi")), None)
            dc = None
            for c in tb.columns:
                vals = [v for v in tb.df[c].head(100) if not is_blank(v)]
                if vals and sum(parse_datetime(v) is not None for v in vals) >= 0.9 * len(vals) and c != amt:
                    dc = c
                    break
            if amt and dc:
                X = pd.DataFrame({"date": _date_col(tb.df, dc).dt.normalize().values, "amount": _num_col(tb.df[amt]).values})
                biz.expenses = X.dropna()


def price_change_date(biz: Biz, pkey, price: float):
    """First day the product's list price rose to `price` (rulebook §5 'Hiệu ứng tăng giá')."""
    H = getattr(biz, "price_hist", None)
    if H is None:
        return None
    h = H[H["pkey"] == pkey].sort_values("date").reset_index(drop=True)
    for i in range(len(h)):
        if abs(h.loc[i, "price"] - price) < 0.5 and (i == 0 or h.loc[i - 1, "price"] < price):
            if i == 0:
                return None  # opening price, not an increase
            return h.loc[i, "date"].date()
    return None
