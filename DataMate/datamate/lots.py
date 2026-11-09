"""Lots and expiry dates (rulebook §6 'Nhà thuốc'): remaining quantity per lot at the end of a day."""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

from .model import Biz, _date_col, _num_col, split_name
from .textnorm import fold, is_blank, squash


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


LOT_NAMES = ("lo", "ma lo", "so lo", "lo hang")


def build_lots(biz: Biz) -> None:
    biz.lots, biz.lot_moves = None, None
    master = sales = None
    for tb in biz.tables:
        lc = _col(tb, r"^so lo", LOT_NAMES)
        if not lc:
            continue
        r = biz.roles.get(tb.name, {})
        ec = _col(tb, r"^han su dung", ("han su dung", "hsd", "han dung"))
        sold = _col(tb, r"^so luong ban lay tu lo", ("sl quy doi", "sl quy ra le", "so luong", "so luong (dvt co ban)", "sl ban"))
        ret = _col(tb, r"^so luong khach tra lai", ("sl tra quy doi", "sl tra (dvt co ban)", "so luong tra", "so luong tra (dvt co ban)", "sl tra"))
        oc = r.get("order_id")
        if oc and ret and sold and not ec:
            sales = (tb, lc, oc, sold, ret)
        elif ec and tb.df[lc].nunique() >= 0.98 * len(tb.df):
            qin = _col(tb, r"^so luong nhap \(don vi co ban\)|^so luong nhap", ("sl nhap quy doi", "sl nhap (dvt co ban)", "so luong (dvt co ban)", "so luong nhap"))
            based = qin is not None and (re.search(r"co ban|quy doi", split_name(qin)) is not None or "co ban" in fold(tb.meanings.get(qin, "")))
            score = (2 if based else 0) + (1 if re.search(r"\blo\b", split_name(tb.name)) else 0)
            if master is None or score > master[-1]:
                master = (tb, lc, ec, qin, based, score)
    if master is None or sales is None:
        return
    tb, lc, ec, qin, based, _ = master
    r = biz.roles.get(tb.name, {})
    code = tb.df[r["prod_code"]].map(_s) if "prod_code" in r else None
    name = tb.df[r.get("prod_name") or r.get("name")].map(_s) if (r.get("prod_name") or r.get("name")) else None
    if code is not None and getattr(biz, "prod_has_code", False):
        pkey = code
    elif name is not None:
        pkey = name.map(lambda v: biz.name2key.get(squash(v)) if v else None)
    else:
        return
    Lt = pd.DataFrame({"lot": tb.df[lc].map(_s), "pkey": pkey.values})
    Lt["expiry"] = _date_col(tb.df, ec).dt.normalize().values
    Lt["qty_in"] = _num_col(tb.df[qin]).values if qin else np.nan
    # purchase date and unit cost of each lot
    imp = getattr(biz, "imports", None)
    slips = getattr(biz, "import_slips", None)
    Lt["in_date"] = pd.NaT
    Lt["unit_cost"] = np.nan
    ic = _col(tb, r"^ma phieu nhap", ("ma phieu nhap", "ma nhap hang", "ma don nhap", "so phieu"))
    if ic and slips is not None:
        d = dict(zip(slips["imp_id"], slips["date"]))
        Lt["in_date"] = tb.df[ic].map(lambda v: d.get(_s(v))).values
    pc = next((c for c in tb.columns if split_name(c) in ("don gia nhap", "gia nhap", "gia von lo")), None)
    if pc and based:
        Lt["unit_cost"] = _num_col(tb.df[pc]).values
    if imp is not None and "lot" in imp:
        im = imp.dropna(subset=["lot"]).drop_duplicates("lot").set_index("lot")
        if Lt["in_date"].isna().all():
            Lt["in_date"] = Lt["lot"].map(im["date"])
        if Lt["unit_cost"].isna().all() and im["price"].notna().any():
            fac = pd.Series([biz.conv.get((k, u), 1.0) for k, u in zip(im["pkey"], im["unit_f"])], index=im.index)
            Lt["unit_cost"] = Lt["lot"].map(im["price"] / fac)
        if not based or Lt["qty_in"].isna().all():
            fac = pd.Series([biz.conv.get((k, u), 1.0) for k, u in zip(im["pkey"], im["unit_f"])], index=im.index)
            Lt["qty_in"] = Lt["lot"].map(im["qty"] * fac)
    biz.lots = Lt.dropna(subset=["lot", "pkey", "qty_in"])
    stb, slc, soc, ssold, sret = sales
    M = pd.DataFrame({"lot": stb.df[slc].map(_s), "order_id": stb.df[soc].map(_s)})
    M["net"] = _num_col(stb.df[ssold]).fillna(0.0).values - _num_col(stb.df[sret]).fillna(0.0).values
    O = biz.orders
    M = M.merge(O[["order_id", "date", "st"]], on="order_id", how="inner")
    biz.lot_moves = M


def remaining(biz: Biz, x) -> pd.DataFrame:
    """Lots with the quantity left (base units) at the end of day x."""
    x = pd.Timestamp(x)
    Lt = biz.lots
    Lt = Lt[Lt["in_date"].isna() | (Lt["in_date"] <= x)]
    M = biz.lot_moves
    M = M[(M["date"] <= x) & (M["st"] != "cancel")]
    used = M.groupby("lot")["net"].sum()
    out = Lt.copy()
    out["left"] = out["qty_in"] - out["lot"].map(used).fillna(0.0)
    return out


def stock_from_lots(biz: Biz, x) -> dict:
    R = remaining(biz, x)
    return R.groupby("pkey")["left"].sum().to_dict()


def lots_intent(ctx, E):
    """Expiry questions. Returns None when the question is not about lots / expiry dates."""
    biz, text, P = ctx.biz, ctx.text, ctx.periods
    # 'lô' (lot) and 'lọ' (bottle) look the same once accents are stripped
    lot_word = re.search(r"\blô\b", ctx.low) is not None or (re.search(r"\blo (?:hang|nao|con|gan|sap)\b", text) is not None
                                                                and not re.search(r"[àáảãạăâđèéêìíòóôơùúưỳý]", ctx.low))
    accented = re.search(r"[àáảãạăâđèéêìíòóôơùúưỳý]", ctx.low) is not None
    if accented:
        exp_word = re.search(r"hạn dùng|hết hạn|hạn sử dụng|cận date|\bdate\b|\bhsd\b|hạn sd", ctx.low) is not None
    else:
        exp_word = re.search(r"han dung|het han\b|han su dung|can date|\bdate\b", text) is not None
    if not (lot_word or exp_word):
        return None
    if biz.lots is None:
        from .extra import _has_column
        if _has_column(biz, r"han su dung|\bhsd\b|han dung|expiry"):
            raise E.Abstain("expiry data present but not modelled")
        raise E.Unanswerable("dữ liệu không ghi lô và hạn dùng")
    asof = [p for p in P if p.kind in ("day", "upto")]
    ranges = [p for p in P if p.kind == "range"]
    if len(asof) == 2 and not ranges and re.search(r"tu (?:hom )?nay (?:den|toi)|tu nay", text):
        # 'theo tồn kho ngày X ... hết hạn từ nay đến hết ngày Y'
        first, second = sorted(asof, key=lambda p: p.end)
        ranges = [type(first)(min(first.end, ctx.today), second.end, "range")]
        asof = [first]
        P = asof + ranges
    if len(asof) > 1 or len(ranges) > 1 or len(asof) + len(ranges) != len(P):
        raise E.Abstain("expiry question with unexpected periods")
    x = min(asof[0].end, ctx.today) if asof else ctx.today
    R = remaining(biz, x)
    R = R[R["left"] > 1e-9]
    prods = ctx.ent("prod")
    if len(prods) != len(ctx.ents) or len(prods) > 1:
        raise E.Abstain("expiry question with extra conditions")
    xs = pd.Timestamp(x)
    if prods:
        if re.search(r"gan het han nhat|can date nhat|sap het han nhat|han (?:dung |su dung )?(?:gan|som) nhat|het han som nhat", text):
            r = R[(R["pkey"] == prods[0].value) & (R["expiry"] >= xs)]
            if not len(r):
                raise E.Abstain("no unexpired lot in stock")
            return {"type": "date", "answer": r["expiry"].min().strftime("%Y-%m-%d")}
        raise E.Abstain("lot question about one product not modelled")
    # expiry window
    if ranges:
        a, e = pd.Timestamp(ranges[0].start), pd.Timestamp(ranges[0].end)
    else:
        m = re.search(r"(zq[nk])?(\d+) ngay (?:toi|nua|sap toi)", text)
        if not m:
            raise E.Abstain("expiry window not found")
        n_days = int(ctx.nums[int(m.group(2))][1]) if m.group(1) else int(m.group(2))
        a, e = xs, xs + pd.Timedelta(days=n_days)
    W = R[(R["expiry"] >= a) & (R["expiry"] <= e)]
    if re.search(r"tien von|gia von|\bvon\b", text):
        if W["unit_cost"].isna().any():
            raise E.missing(biz, "dữ liệu không ghi giá nhập của lô")
        return {"type": "number", "answer": E.money((W["left"] * W["unit_cost"]).sum())}
    if re.search(r"(?:may|bao nhieu|so) lo\b", text):
        return {"type": "number", "answer": int(len(W))}
    if re.search(r"(?:may|bao nhieu|so) (?:mat hang|san pham|mon|loai)", text):
        return {"type": "number", "answer": int(W["pkey"].nunique())}
    if re.search(r"hang nao|san pham nao|mat hang nao|nhung (?:san pham|mat hang|hang)|liet ke|\bke\b|nao", text):
        first = W.groupby("pkey")["expiry"].min()
        code = getattr(biz, "prod_has_code", False)
        order = sorted(first.items(), key=lambda kv: (kv[1], str(kv[0]) if code else fold(E._prod_label(biz, kv[0]))))
        return {"type": "list", "answer": [E._prod_label(biz, k) for k, _ in order]}
    raise E.Abstain("expiry question not modelled")
