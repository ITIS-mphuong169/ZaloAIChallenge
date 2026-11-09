"""Customer debt (rulebook §4): payments table, outstanding debt, collections, ageing."""
from __future__ import annotations

import datetime as dt
import re

import numpy as np
import pandas as pd

from .model import Biz, _date_col, _num_col, split_name
from .textnorm import fold, is_blank, parse_datetime, parse_num

AMOUNT_NAMES = {"gia tri", "tien", "so tien", "so tien thu", "tien thu", "so tien tra", "tien tra", "thanh toan", "da thu"}


def build_payments(biz: Biz) -> None:
    """biz.payments: order_id, date, amount (one row per payment)."""
    biz.payments = None
    O = biz.orders
    if O is None or not biz.has_order_id:
        return
    ids = set(O["order_id"].dropna().astype(str))
    best = None
    for tb in biz.tables:
        if tb.name in (biz.lines_table, biz.orders_table) or len(tb.df) < 10:
            continue
        names = " | ".join(split_name(c) for c in tb.columns) + " | " + split_name(tb.name)
        if not re.search(r"quy|thanh toan|phieu thu|thu tien|tien thu|so tien thu|ngay thu|thoi gian thu|thoi diem thu", names):
            continue
        if re.search(r"nhap|ncc|giao hang|lo hang", names):
            continue
        oc = dc = ac = None
        for c in tb.columns:
            vals = [str(v).strip() for v in tb.df[c].head(300) if not is_blank(v)]
            if vals and sum(v in ids for v in vals) >= 0.8 * len(vals):
                oc = c
                break
        if oc is None:
            continue
        for c in tb.columns:
            if c == oc:
                continue
            vals = [v for v in tb.df[c].head(200) if not is_blank(v)]
            if not vals:
                continue
            if dc is None and sum(parse_datetime(v) is not None for v in vals) >= 0.9 * len(vals) \
                    and not all(isinstance(v, (int, float)) for v in vals):
                dc = c
            m = fold(tb.meanings.get(c, ""))
            if ac is None and (m.startswith("so tien") or split_name(c) in AMOUNT_NAMES) \
                    and sum(parse_num(v) is not None for v in vals) >= 0.9 * len(vals):
                ac = c
        if dc and ac:
            best = (tb, oc, dc, ac)
            break
    if best is None:
        return
    tb, oc, dc, ac = best
    P = pd.DataFrame({"order_id": tb.df[oc].map(lambda v: None if is_blank(v) else str(v).strip())})
    P["date"] = _date_col(tb.df, dc).dt.normalize().values
    P["amount"] = _num_col(tb.df[ac]).values
    biz.payments = P.dropna(subset=["order_id", "date", "amount"])
    biz.payments_table = tb.name


_AMT = r"(\d+(?:[.,]\d+)*\s*(?:tr\s*\d*|trieu|k\b|nghin|ngan|d\b|dong)?)"


def _note_amount(t: str):
    t = t.strip()
    m = re.fullmatch(r"(\d+)\s*tr\s*(\d*)", t)
    if m:
        return int(m.group(1)) * 1e6 + (int(m.group(2).ljust(3, "0")) * 1e3 if m.group(2) else 0.0)
    m = re.fullmatch(r"(\d+(?:[.,]\d+)?)\s*(?:k|nghin|ngan)", t)
    if m:
        return float(m.group(1).replace(",", ".")) * 1e3
    return parse_num(re.sub(r"\s*(?:d|dong)$", "", t))


def note_owed(note, total):
    """Amount still owed according to a free-text remark ('trả 259k, nợ 449.500đ'); None when the remark says nothing about debt."""
    if is_blank(note):
        return None
    f = fold(note)
    if not re.search(r"\bno\b|thieu|ghi so|chua tra|tra sau|\bchiu\b", f):
        return None
    if re.search(r"\bdu\b|tra het|het roi|tra xong|thanh toan xong|da xong", f):
        return 0.0
    m = re.search(r"(?:con thieu|thieu lai|thieu|con no|ghi no|no lai|\bno\b|ghi so)\s*(?:lai\s*)?" + _AMT, f)
    if m:
        return _note_amount(m.group(1))
    m = re.search(r"(?:dua truoc|tra truoc|da tra|da dua|tra|ck|chuyen khoan|dua)\s*" + _AMT + r".*(?:con lai|phan con lai).*(?:ghi so|ghi no|\bno\b)", f)
    if m and total is not None and not pd.isna(total):
        v = _note_amount(m.group(1))
        return None if v is None else float(total) - v
    if not re.search(r"\d", f) and re.search(r"no ca don|no het|no toan bo|chua tra|ghi so ca don|ghi no ca don", f) and total is not None and not pd.isna(total):
        return float(total)
    return float("nan")  # a debt remark that could not be read


def _static_owed(biz: Biz, E):
    """Unpaid amount per order as recorded at the end of the data (a 'còn nợ' column, or remarks in the note column)."""
    O = biz.orders
    if O is None:
        return None
    if "debt" in O and O["debt"].notna().any():
        return O["debt"].fillna(0.0)
    if "note" in O and "order_total" in O:
        owed = pd.Series([note_owed(n, t) for n, t in zip(O["note"], O["order_total"])], index=O.index, dtype=object)
        seen = owed.map(lambda v: v is not None)
        if not seen.any():
            return None
        if "pay_method" in O:
            cls = O["pay_method"].map(lambda v: E.classify(v, E.PAY_CLASSES))
            if ((cls == "debt") & ~seen & (O["st"] != "cancel")).any():
                return None  # credit orders without a readable remark
        vals = owed.map(lambda v: 0.0 if v is None else v).astype(float)
        if vals.isna().any():
            return None
        return vals
    return None


class Debt:
    """Debt view of one business. All figures follow rulebook §4."""

    def __init__(self, ctx, E):
        self.ctx, self.E, self.biz = ctx, E, ctx.biz
        biz = ctx.biz
        O = biz.orders
        self.static = None
        if getattr(biz, "payments", None) is None or O is None or "order_total" not in O or O["order_total"].isna().all():
            self.static = _static_owed(biz, E) if getattr(biz, "payments", None) is None else None
            if self.static is None:
                has_any = O is not None and ("debt" in O or "paid" in O)
                if has_any or getattr(biz, "payments", None) is not None:
                    raise E.Abstain("debt data present but not modelled")
                raise E.missing(biz, "dữ liệu không ghi công nợ")
        o = O[O["st"] != "cancel"].copy()
        # COD and marketplace orders are not customer debt
        if "pay_method" in o:
            cls = o["pay_method"].map(lambda v: E.classify(v, E.PAY_CLASSES))
            o = o[cls != "cod"]
            self.debt_flag = (cls.reindex(o.index) == "debt")
        else:
            self.debt_flag = None
        if "channel" in o:
            ch = o["channel"].map(lambda v: E.classify(v, E.CHANNEL_CLASSES))
            o = o[ch != "market"]
        self.o = o
        if self.static is not None:
            self.pay = None
            return
        pay = biz.payments.merge(o[["order_id", "date"]].rename(columns={"date": "odate"}), on="order_id", how="inner")
        self.pay = pay

    def outstanding(self, x: pd.Timestamp) -> pd.DataFrame:
        """Orders placed up to X with their unpaid amount at the end of day X (only > 0)."""
        o = self.o[self.o["date"] <= x]
        if self.static is not None:
            # the book only records what is still owed on its last day
            if self.biz.data_end is None or x.date() < self.biz.data_end:
                raise self.E.Abstain("debt is only recorded as of the last day of the data")
            d = o.assign(owed=self.static.reindex(o.index))
            return d[d["owed"] > 0.5]
        paid = self.pay[self.pay["date"] <= x].groupby("order_id")["amount"].sum()
        d = o.assign(owed=o["order_total"] - o["order_id"].map(paid).fillna(0.0))
        return d[d["owed"] > 0.5]

    def is_debt_order(self) -> pd.Series:
        """Orders sold on credit: not fully paid on the purchase day."""
        if self.static is not None:
            raise self.E.Abstain("payments are not recorded one by one")
        same = self.pay[self.pay["date"] <= self.pay["odate"]].groupby("order_id")["amount"].sum()
        first_owed = self.o["order_total"] - self.o["order_id"].map(same).fillna(0.0)
        return first_owed


def _labels(E, d: pd.DataFrame) -> dict:
    return E._cust_labels(d)


def debt_intent(ctx, E):
    text, biz = ctx.text, ctx.biz
    if not re.search(r"\bno\b|mua chiu|con thieu|chua tra het|ban chiu|chua thu|thieu tien", text):
        return None
    if re.search(r"gop|tra gop|han thanh toan|han tra|qua han|ngay phai tra|dung han|tre han", text):
        return None
    P = ctx.periods
    if len(P) > 1:
        return None
    ents = [e for e in ctx.ents if not (e.kind == "pay" and e.value == "debt")]
    custs = [e for e in ents if e.kind == "cust"]
    if len(ents) != len(custs) or len(custs) > 1:
        return None
    D = Debt(ctx, E)
    ks = [v for (op, v) in ctx.nums if op == "k"]
    has = lambda rx: re.search(rx, text) is not None
    add_pay = has(r"no them|chiu them") and has(r"tra")
    collect = has(r"tra bot no|tra no|thu no|thu duoc[a-z ]* cong no|tien cong no|tra bot|doi no|doi duoc|thu hoi") and not add_pay

    # ----- period based: new credit and repayments
    if add_pay or collect:
        if len(P) != 1:
            return None
        if D.static is not None:
            raise E.Abstain("payments are not recorded one by one")
        a, e = ctx.clip(P[0])
        o = D.o
        first_owed = D.is_debt_order()
        credit = o[first_owed > 0.5]
        if custs:
            credit = credit[credit["cust"].isin(custs[0].value)]
        credit_ids = set(credit["order_id"])
        pay = D.pay[D.pay["order_id"].isin(credit_ids)]
        repaid = pay[(pay["date"] > pay["odate"]) & (pay["date"] >= a) & (pay["date"] <= e)]["amount"].sum()
        if add_pay:
            if not custs:
                return None
            newc = credit[(credit["date"] >= a) & (credit["date"] <= e)]
            added = first_owed.reindex(newc.index).sum()
            return {"type": "list", "answer": [E.money(added), E.money(repaid)]}
        if custs or ks or ctx.nums:
            return None
        return {"type": "number", "answer": E.money(repaid)}

    # ----- point in time
    if custs:
        return None
    if len(P) == 1 and P[0].kind not in ("upto", "day"):
        return None
    x = pd.Timestamp(min(P[0].end, ctx.today)) if P else pd.Timestamp(ctx.today)
    if x.date() < biz.data_start:
        raise E.Unanswerable("ngày hỏi trước khoảng ngày có dữ liệu")
    d = D.outstanding(x)
    # ageing
    if has(r"tuoi no|don (?:da )?dat|ngay ma|cach |co tuoi|tuoi (?:qua|tren|hon)"):
        age = (x - d["date"]).dt.days
        nums = [ctx.nums[int(i)] for i in re.findall(r"zq[nk](\d+)", text)]  # in text order
        vals = [v for (_op, v) in nums]
        if len(vals) == 6 and vals[0] == 3:
            vals = vals[1:]  # '3 nhóm 0-30, 31-60, trên 60'
        if len(vals) == 5:
            a1, b1, a2, b2, c3 = vals
            if not (a1 <= b1 < a2 <= b2 <= c3):
                return None
            out = [d[(age >= a1) & (age <= b1)]["owed"].sum(), d[(age >= a2) & (age <= b2)]["owed"].sum(), d[age > c3]["owed"].sum()]
            return {"type": "list", "answer": [E.money(v) for v in out]}
        if len(nums) == 1:
            op, v = nums[0]
            sel = d[age >= v] if op == "ge" else d[age > v]
            return {"type": "number", "answer": E.money(sel["owed"].sum())}
        return None
    if ctx.nums and not ks:
        return None
    by = d[d["cust"].notna()].groupby("cust")["owed"].sum()
    by = by[by > 0.5]
    if has(r"top|liet ke|bang|zqk\d+ khach|khach nao") and len(ks) == 1:
        labels = _labels(E, d)
        top = E._rank(by, int(ks[0]), lambda key: str(key) if biz.has_cust_code else fold(labels.get(key, str(key))))
        rows = [[labels.get(k, str(k)), E.money(by[k])] for k in top]
        return {"type": "table", "answer": {"columns": ["khach_hang", "so_tien_no"], "rows": rows}}
    if ks:
        return None
    if has(r"bao nhieu khach|so khach|may khach|bao nhieu nguoi|may nguoi|dem khach|nhieu khach"):
        return {"type": "number", "answer": int(len(by))}
    if has(r"bao nhieu|tong|nhieu"):
        return {"type": "number", "answer": E.money(d["owed"].sum())}
    return None
