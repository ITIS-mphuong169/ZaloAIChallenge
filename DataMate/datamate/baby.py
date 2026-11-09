"""Catalogue attributes (brand, size / formula number) and non-item rows (voucher, points, shipping fee).

Rulebook §6 'Mẹ và bé': amounts are reported as positive numbers by invoice date for invoices that are neither
cancelled nor returned; a size or a brand share is taken over all nappies sold in the period.
"""
from __future__ import annotations

import re

import pandas as pd

from .textnorm import fold, is_blank


def classify_extra(name) -> str | None:
    f = fold(name)
    if re.match(r"^voucher", f):
        return "voucher"
    if re.match(r"^dung( \d+)? diem", f):
        return "points"
    if re.match(r"^phi (ship|giao|van chuyen)", f):
        return "ship"
    return None


def attr_intent(ctx, E):
    biz, text, P = ctx.biz, ctx.text, ctx.periods
    X = getattr(biz, "extras", None)
    kind = None
    if re.search(r"voucher|vau cho|ma giam gia|phieu giam", text):
        kind = "voucher"
    elif re.search(r"dung diem|doi diem|tru diem|diem (?:tru|doi|tich)|\bdiem\b", text) and not re.search(r"diem (?:hien|con)|cao diem", text):
        kind = "points"
    elif re.search(r"phi ship|tien ship|phi giao|phi van chuyen|\bship\b", text):
        kind = "ship"
    if kind is not None:
        if X is None or not (X["kind"] == kind).any():
            raise E.missing(biz, "dữ liệu không ghi khoản này")
        if len(P) != 1 or ctx.ents or ctx.nums or "%" in text:
            raise E.Abstain("voucher / points / shipping question with extra conditions")
        a, e = ctx.clip(P[0])
        O = biz.orders
        ok = set(O[(O["date"] >= a) & (O["date"] <= e) & O["st"].isin(["done", "ship"])]["order_id"])
        sel = X[(X["kind"] == kind) & X["order_id"].isin(ok)]
        if re.search(r"bao nhieu don|may don|so don", text):
            return {"type": "number", "answer": int(sel["order_id"].nunique())}
        return {"type": "number", "answer": E.money(sel["amount"].abs().sum())}
    # ---- new members
    if re.search(r"thanh vien", text) and re.search(r"dang ky|\bmoi\b", text):
        reg = getattr(biz, "member_dates", None)
        if reg is None:
            raise E.missing(biz, "dữ liệu không ghi ngày đăng ký thành viên")
        if len(P) != 1 or ctx.ents or ctx.nums:
            raise E.Abstain("member question with extra conditions")
        a, e = ctx.clip(P[0]) if P[0].end >= biz.data_start else (pd.Timestamp(P[0].start), pd.Timestamp(P[0].end))
        return {"type": "number", "answer": int(((reg >= a) & (reg <= e)).sum())}
    # ---- brand / size questions inside a product group
    prod = biz.prod
    brands = {fold(v["brand"]): v["brand"] for v in prod.values() if v.get("brand")}
    sizes = {fold(v["size"]) for v in prod.values() if v.get("size")}
    if not brands and not sizes:
        return None
    groups = {fold(v["group"]): v["group"] for v in prod.values() if v.get("group")}
    flat = " " + " ".join(re.findall(r"[a-z0-9+]+", ctx.norm)) + " "
    grp = None
    if re.search(r"\bbim\b|\bta\b", flat) and not ctx.ent("prod"):
        grp = next((g for g in groups if re.search(r"\bbim\b|\bta\b", g)), None)
    elif re.search(r"sua bot", flat) and not ctx.ent("prod"):
        grp = next((g for g in groups if g == "sua bot" or g.startswith("sua bot")), None)
    if grp is None:
        return None
    size = None
    m = re.search(r"\bsize (xxl|xl|nb|newborn|l|m|s)\b", flat)
    if m:
        size = "size " + ("nb" if m.group(1) == "newborn" else m.group(1))
    m2 = re.search(r"\bso (\d{1,2})\b", flat)
    if m2 and "sua" in grp:
        size = "so " + m2.group(1)
    if size is not None and size not in sizes:
        raise E.Abstain("size / number not found in the catalogue")
    brand = [b for b in brands if b and f" {b} " in flat]
    if len(brand) > 1:
        raise E.Abstain("several brands mentioned")
    ask_brand = re.search(r"hang nao", text) is not None or "hãng nào" in ctx.low
    if size is None and not brand and not ask_brand:
        return None
    other = [e for e in ctx.ents if e.kind not in ("size", "group")]
    if len(P) != 1 or other:
        raise E.Abstain("attribute question with extra conditions")
    a, e = ctx.clip(P[0])
    L = biz.lines
    Lv = L[(L["date"] >= a) & (L["date"] <= e) & L["st"].isin(["done", "ship"]) & L["pkey"].notna()]
    gkey = Lv["pkey"].map(lambda k: fold(prod.get(k, {}).get("group") or ""))
    G = Lv[gkey == grp]
    sel = G
    if size is not None:
        sel = sel[sel["pkey"].map(lambda k: fold(prod.get(k, {}).get("size") or "") == size)]
    if brand:
        sel = sel[sel["pkey"].map(lambda k: fold(prod.get(k, {}).get("brand") or "") == brand[0])]
    if "%" in text:
        if size is None and not brand:
            raise E.Abstain("share without a size or brand")
        tot = float(G["qty_base"].sum())
        if not tot:
            raise E.Abstain("zero base")
        return {"type": "percent", "answer": E.pct(100.0 * float(sel["qty_base"].sum()) / tot)}
    if re.search(r"hang nao", text) or "hãng nào" in ctx.low:
        if brand:
            raise E.Abstain("brand ranking with a brand filter")
        g = sel.groupby(sel["pkey"].map(lambda k: prod.get(k, {}).get("brand")))["qty_base"].sum()
        return {"type": "string", "answer": E._rank(g, 1, lambda b: fold(b))[0]}
    if size is None and not brand:
        return None
    if re.search(r"\btien\b|doanh thu", text):
        return {"type": "number", "answer": E.money(sel["amount"].sum())}
    if re.search(r"bao nhieu|\bmay\b|nhieu", text):
        return {"type": "number", "answer": E.num(sel["qty_base"].sum())}
    return None
