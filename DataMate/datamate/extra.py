"""Additional intents: purchases (imports) and unit price on a given day."""
from __future__ import annotations

import pandas as pd

from .textnorm import fold


def imports_intent(ctx, f, ks, thr, E):
    """E is the engine module (passed in to avoid a circular import)."""
    T, P, biz = ctx.toks, ctx.periods, ctx.biz
    if thr or ks or len(P) != 1:
        return None
    slips, lines = getattr(biz, "import_slips", None), getattr(biz, "imports", None)
    if slips is None:
        raise E.missing(biz, "dữ liệu không ghi nhập hàng")
    p = P[0]
    a, e = ctx.clip(p)
    S = slips[(slips["date"] >= a) & (slips["date"] <= e) & (~slips["cancelled"].astype(bool))]
    # ----- average purchase price of one product
    if "AVG" in T and f.prod is not None and len(ctx.ents) == 1 and len(ctx.units) <= 1 \
            and E.only(T, "IMPORT AVG PRICE Q MONEY SOLD"):
        if lines is None or lines["amount"].isna().all():
            raise E.missing(biz, "dữ liệu không ghi giá nhập")
        L = lines[(lines["date"] >= a) & (lines["date"] <= e) & (~lines["cancelled"]) & (lines["pkey"] == f.prod)]
        if not len(L):
            raise E.Abstain("product not purchased in the period")
        if ctx.units:
            unit = ctx.units[0]
        else:
            # unit named with several words ('xe 4m3') or hidden inside the product name ('Bộ van ...')
            import re as _re
            rest = ctx.norm.replace(ctx.ents[0].text, " ")
            cands = [u for (kk, u) in biz.conv if kk == f.prod and _re.search(r"(?<![a-z0-9])" + _re.escape(u) + r"(?![a-z0-9])", rest)]
            used = set(L["unit_f"].unique())
            if len(cands) == 1:
                unit = cands[0]
            elif not cands and len(used) == 1:
                unit = used.pop()
            else:
                raise E.Abstain("unit unclear")
        den = biz.conv.get((f.prod, unit), 1.0)
        qty = (L["qty"] * L["unit_f"].map(lambda u: biz.conv.get((f.prod, u), 1.0)) / den).sum()
        if not qty or L["amount"].isna().any():
            raise E.Abstain("no quantity")
        return {"type": "number", "answer": E.money(L["amount"].sum() / qty)}
    if f.prod is not None and len(ctx.ents) == 1 and len(ctx.units) <= 1 and "AVG" not in T and not (T & {"MONEY", "REV", "PRICE"}) \
            and E.only(T, "IMPORT Q SOLD TOTAL COUNT"):
        if lines is None:
            raise E.Abstain("no purchase lines")
        L = lines[(lines["date"] >= a) & (lines["date"] <= e) & (~lines["cancelled"]) & (lines["pkey"] == f.prod)]
        base = fold(biz.prod.get(f.prod, {}).get("base_unit") or "")
        den = 1.0
        if ctx.units:
            if ctx.units[0] != base:
                den = biz.conv.get((f.prod, ctx.units[0]))
                if not den:
                    raise E.Abstain("unit unknown")
        qty = (L["qty"] * L["unit_f"].map(lambda u: biz.conv.get((f.prod, u), 1.0))).sum() / den
        return {"type": "number", "answer": E.num(qty)}
    if f.any():
        return None
    # ----- supplier with the largest purchase value
    if "SUPPLIER" in T and "MAX" in T and E.only(T, "IMPORT SUPPLIER MAX MONEY Q TOTAL VALUE SOLD REV"):
        if S["supplier"].isna().all() or S["total"].isna().all():
            raise E.Abstain("no supplier / totals")
        g = S.groupby("supplier")["total"].sum()
        top = E._rank(g, 1, str)[0]
        return {"type": "string", "answer": getattr(biz, "supplier_names", {}).get(top) or top}
    # ----- total purchase value
    if "SUPPLIER" not in T and E.only(T, "IMPORT MONEY VALUE Q TOTAL"):
        if not (T & {"MONEY", "VALUE"}):
            return None
        if len(S) and S["total"].isna().any():
            raise E.Abstain("slip totals unknown")
        return {"type": "number", "answer": E.money(S["total"].sum())}
    return None


def price_intent(ctx, f, ks, thr, E):
    """Unit selling price of one product on one day, read from that day's sales lines."""
    T, P, biz = ctx.toks, ctx.periods, ctx.biz
    if thr or ks or len(P) != 1 or P[0].kind != "day" or f.prod is None or len(ctx.ents) != 1 or len(ctx.units) > 1:
        return None
    if not E.only(T, "PRICE Q SOLD MONEY"):
        return None
    if getattr(biz, "combo_names", False):
        raise E.Abstain("items and toppings share one cell")
    L = ctx.lines(P[0], f)
    if ctx.units and "unit_f" in L:
        L = L[L["unit_f"] == ctx.units[0]]
    elif ctx.units:
        bu = fold(biz.prod.get(f.prod, {}).get("base_unit") or "")
        if bu != ctx.units[0]:
            raise E.Abstain("unit unknown")
    if not len(L):
        raise E.Abstain("no sale of that product/unit on the day")
    if "price" in L and L["price"].notna().all():
        prices = L["price"]
    elif "qty" in L:
        prices = L["amount"] / L["qty"]
    else:
        raise E.Abstain("no price")
    vals = sorted(set(round(float(v), 2) for v in prices if pd.notna(v)))
    if len(vals) != 1:
        raise E.Abstain("several prices on that day")
    return {"type": "number", "answer": E.num(vals[0])}


def profit_intent(ctx, f, ks, thr, E):
    """Gross profit = revenue - quantity x cost recorded on the sale line (rulebook §5)."""
    T, P, biz = ctx.toks, ctx.periods, ctx.biz
    if thr or len(P) != 1 or len(ks) > 1:
        return None
    L0 = biz.lines
    if L0 is None or "cost" not in L0 or L0["cost"].isna().all():
        if not _has_column(biz, r"gia von|von luc ban|cost"):
            raise E.Unanswerable("dữ liệu không ghi giá vốn")
        raise E.missing(biz, "dữ liệu không ghi giá vốn trên dòng bán")
    if getattr(biz, "combo_names", False):
        raise E.Abstain("items and toppings share one cell")
    L = ctx.lines(P[0], f)
    L = L[L["pkey"].notna()] if "pkey" in L else L
    if L["cost"].isna().any() or L["qty"].isna().any():
        raise E.Abstain("missing cost on some lines")
    prof = L["amount"] - L["qty"] * L["cost"]
    prody = bool(T & {"PRODN", "WHICHPROD"})
    if prody and "MAX" in T and E.only(T, "PROFIT MAX PRODN WHICHPROD Q TOTAL LIST SOLD MONEY") and f.prod is None:
        k = int(ks[0]) if ks else None
        if k is None and "LIST" in T:
            return None
        g = prof.groupby(L["pkey"]).sum()
        top = E._rank(g, k or 1, lambda key: str(key) if biz.prod_has_code else fold(E._prod_label(biz, key)))
        names = [E._prod_label(biz, key) for key in top]
        return {"type": "list", "answer": names} if k is not None else {"type": "string", "answer": names[0]}
    if not ks and "MAX" not in T and E.only(T, "PROFIT Q MONEY TOTAL SOLD PRODN"):
        if "PRODN" in T and not f.line_level():
            return None
        return {"type": "number", "answer": E.money(prof.sum())}
    return None


_SUBJECTIVE = (r"than thien|le phep|chu dao|\bkhen\b|lanh nghe|nhiet tinh|hai long|thai do|de thuong|vui tinh|kheo tay|tay nghe|"
               r"tan tam|niem no|de men|quy men|tot bung|xinh|dep trai|uy tin|de chiu|chat luong (?:tot|cao)|\bngon\b|"
               r"trung thuc|dang tin cay|tin cay|ba bau|cho con bu|nen uong|nen dung|lieu dung|tac dung|cong dung|"
               r"(?:dung|uong) [a-z0-9 %().,+-]* duoc khong")
_REASON = (r"\bvi sao\b|\btai sao\b|nguyen nhan|\bvi (?:giao tre|goi nham|mua nham|mon bi loi|hang loi|bi loi|loi|nham|het hang|doi y)|"
           r"goi nham|mua nham|giao tre|\bly do (?:gi|nao|huy|tra|chinh|chu yeu)")
GROUP_VALUES = {"le", "si", "than thiet", "vip", "thuong", "moi", "khach le", "khach si", "dai ly", "ctv", "thanh vien", "bac", "vang",
                "kim cuong", "dong", "quen", "vang lai", "ban buon", "ban le", "tiem nang", "khach quen", "khach moi"}


def _has_column(biz, rx: str) -> bool:
    import re

    from .model import split_name
    from .textnorm import is_blank
    for tb in biz.tables:
        for c in tb.columns:
            if re.search(rx, split_name(c)) or re.search(rx, fold(tb.meanings.get(c, ""))):
                if "cot trong" in fold(tb.meanings.get(c, "")):
                    continue
                if tb.df[c].map(lambda v: not is_blank(v)).mean() > 0.02:
                    return True
    return False


def unanswerable_attr(ctx, E):
    """Questions about things the released data does not record (rulebook §7)."""
    import re
    n = re.sub(r"zq[enk]\d+", " ", ctx.text)  # entity names (products, customers) are masked out
    if re.search(_SUBJECTIVE, n) or re.search(r"khó tính|dễ tính|lịch sự|có tâm", ctx.low):
        raise E.Unanswerable("dữ liệu không ghi thông tin này")
    if re.search(_REASON, n) or "lý do" in ctx.low:
        if _has_column(ctx.biz, r"ly do|nguyen nhan|reason"):
            raise E.Abstain("reason column present, not modelled")
        raise E.Unanswerable("dữ liệu không ghi lý do")
    if re.search(r"khach nu\b|khach nam\b|gioi tinh|nu gioi|nam gioi|phai nu|khach la nu|khach la nam", n):
        if _has_column(ctx.biz, r"gioi tinh|nam nu|gender"):
            raise E.Abstain("gender column present, not modelled")
        raise E.Unanswerable("dữ liệu không ghi giới tính khách")
    if re.search(r"\bo duong\b|nha o duong|khach duong\b|dia chi|khach o [a-z]", n):
        if _has_column(ctx.biz, r"dia chi|addr"):
            raise E.Abstain("address column present, not modelled")
        raise E.Unanswerable("dữ liệu không ghi địa chỉ khách")
    return None


def group_missing(ctx, E):
    """'nhóm ...' asked of a shop that keeps no such grouping (rulebook §7)."""
    import re
    biz, text = ctx.biz, ctx.text
    if not re.search(r"\bnhom\b", text):
        return
    if any(fold(tb.name) == "ghi tat" for tb in biz.tables):
        return
    lex = biz._lex
    cust_ctx = re.search(r"khach (?:hang )?(?:thuoc )?nhom|nhom khach", text)
    if cust_ctx and not lex["cgroup"]:
        if not getattr(biz, "group_col_by_values", False):
            raise E.Unanswerable("dữ liệu không ghi nhóm khách")
        return
    if not cust_ctx and not lex["group"]:
        raise E.missing(biz, "dữ liệu không ghi nhóm hàng")


def named_metrics(ctx, E):
    """Metrics the rulebook defines by name (§3): 'khách sộp nhất', 'giờ cao điểm', ..."""
    import re
    raw, P, biz = ctx.raw, ctx.periods, ctx.biz
    rules = [
        ("sop", r"khach (?:hang )?(?:nao )?(?:la )?sop nhat|sop nhat"),
        ("loyal", r"trung thanh nhat"),
        ("fickle", r"hay doi y nhat|doi y nhieu nhat"),
        ("popular", r"ua chuong nhat"),
        ("busy_day", r"dat khach nhat|dong khach nhat(?! .*gio)"),
        ("peak_hour", r"gio cao diem|dong khach nhat .*may gio|gio nao dong khach nhat"),
        ("main_supplier", r"nha cung cap chinh|ncc chinh"),
        ("hardworking", r"cham chi nhat|sieng nhat|sieng nang nhat"),
    ]
    kind = next((k for k, rx in rules if re.search(rx, raw)), None)
    if kind is None:
        return None
    if kind == "peak_hour" and biz.orders is not None and ("dt" not in biz.orders or biz.orders["dt"].isna().all()):
        raise E.missing(biz, "dữ liệu không ghi giờ đặt đơn")
    if len(P) != 1 or ctx.ents or ctx.nums:
        raise E.Abstain("named metric with extra conditions")
    p = P[0]

    def one(series, tiekey=str):
        return E._rank(series, 1, tiekey)[0]

    if kind == "main_supplier":
        slips = getattr(biz, "import_slips", None)
        if slips is None:
            raise E.missing(biz, "dữ liệu không ghi nhập hàng")
        a, e = ctx.clip(p)
        S = slips[(slips["date"] >= a) & (slips["date"] <= e) & (~slips["cancelled"].astype(bool)) & slips["supplier"].notna()]
        top = one(S.groupby("supplier").size().astype(float))
        return {"type": "string", "answer": getattr(biz, "supplier_names", {}).get(top) or top}
    if kind == "hardworking":
        O = ctx.orders(p)
        if "staff" not in O or O["staff"].isna().all():
            raise E.missing(biz, "dữ liệu không ghi nhân viên / thợ")
        O = O[O["staff"].notna()]
        return {"type": "string", "answer": one(O.groupby(O["staff"].map(str))["date"].nunique().astype(float))}
    E.need_orders(ctx)
    if kind == "popular":
        if getattr(biz, "combo_names", False):
            raise E.Abstain("items and toppings share one cell")
        L = ctx.lines(p)
        L = L[L["pkey"].notna()]
        L = L[~L["pkey"].map(lambda k: fold(E._prod_label(biz, k)).startswith("phi phuc vu"))]
        top = one(L.groupby("pkey")["order_id"].nunique().astype(float),
                  lambda k: str(k) if biz.prod_has_code else fold(E._prod_label(biz, k)))
        return {"type": "string", "answer": E._prod_label(biz, top)}
    if kind == "busy_day":
        O = ctx.orders(p)
        g = O.groupby("date").size().astype(float)
        if (g == g.max()).sum() != 1:
            raise E.Abstain("tie")
        return {"type": "date", "answer": g.idxmax().strftime("%Y-%m-%d")}
    if kind == "peak_hour":
        O = ctx.orders(p)
        if "dt" not in O or O["dt"].isna().any():
            raise E.missing(biz, "dữ liệu không ghi giờ đặt đơn")
        g = O.groupby(O["dt"].dt.hour).size().astype(float)
        if (g == g.max()).sum() != 1:
            raise E.Abstain("tie")
        return {"type": "number", "answer": int(g.idxmax())}
    # customer metrics
    if biz.orders["cust"].isna().all():
        raise E.missing(biz, "dữ liệu không ghi khách hàng")
    tie = (lambda k: str(k)) if biz.has_cust_code else None
    if kind == "fickle":
        O = ctx.orders(p, valid=None)
        O = O[(O["st"] == "cancel") & O["cust"].notna()]
        g = O.groupby("cust").size().astype(float)
    else:
        O = ctx.orders(p)
        O = O[O["cust"].notna()]
        if kind == "loyal":
            g = O.groupby("cust")["date"].nunique().astype(float)
        else:  # sộp: highest average order value among customers with >= 3 orders
            agg = O.groupby("cust")["revenue"].agg(["sum", "count"])
            agg = agg[agg["count"] >= 3]
            g = (agg["sum"] / agg["count"]).apply(lambda v: float(E.money(v)))
    labels = E._cust_labels(O)
    top = one(g, tie or (lambda k: fold(labels.get(k, str(k)))))
    return {"type": "string", "answer": labels.get(top, str(top))}


def stock_intent(ctx, E):
    """Stock on hand at the end of a day (base units), optionally split into whole big units + remainder."""
    import re
    text, biz, P = ctx.text, ctx.biz, ctx.periods
    if not re.search(r"ton kho|trong kho|\bkho\b|\bton\b|con lai|con (?:bao nhieu|may|duoc|nhieu)|con hang", text):
        return None
    if re.search(r"han dung|het han\b|han su dung|nhap|\bno\b|het hang|dut hang|chay hang|gop|von", text) or re.search(r"\blô\b", ctx.low):
        return None
    prods = ctx.ent("prod")
    if len(prods) != 1 or len(ctx.ents) != 1 or len(P) > 1 or ctx.nums:
        return None
    if P and P[0].kind not in ("day", "upto"):
        return None
    stock, ledger = getattr(biz, "stock", None), getattr(biz, "stock_ledger", None)
    lots = getattr(biz, "lots", None)
    if stock is None and ledger is None and lots is None:
        if getattr(biz, "imports", None) is not None or getattr(biz, "import_slips", None) is not None:
            raise E.Abstain("stock must be derived from purchases and sales (not modelled)")
        raise E.missing(biz, "dữ liệu không ghi tồn kho")
    x = min(P[0].end, ctx.today) if P else ctx.today
    k = prods[0].value
    if ledger is not None:
        g = ledger[(ledger["pkey"] == k) & (ledger["time"] < pd.Timestamp(x) + pd.Timedelta(days=1))]
        if not len(g):
            raise E.Abstain("no stock movement before that day")
        qty = float(g.sort_values(["time", "seq"])["bal"].iloc[-1])
    elif stock is not None and x == biz.data_end and k in stock:
        qty = stock[k]
    elif lots is not None:
        from .lots import stock_from_lots
        qty = stock_from_lots(biz, x).get(k)
        if qty is None:
            raise E.Abstain("product has no lots")
    else:
        raise E.Abstain("stock only known at the end of the data")
    units = ctx.units
    base = fold(biz.prod.get(k, {}).get("base_unit") or "")
    if re.search(r"\bchan\b|\ble\b|nguyen", text):
        # '<n> big units and <m> loose base units'; big unit names may have several words ('xe 4m3').
        # ctx.text has the product name masked, so units inside the name ('hộp 3 vỉ x 10 viên') do not count
        flat_t = text.replace("_", " ")
        cands = [u for (kk, u) in biz.conv if kk == k and re.search(r"(?<![a-z0-9])" + re.escape(u) + r"(?![a-z0-9])", flat_t)]
        if len(cands) != 1 or not base:
            raise E.Abstain("unit split unclear")
        fac = biz.conv[(k, cands[0])]
        if not fac or fac <= 1:
            raise E.Abstain("unit split unclear")
        return {"type": "list", "answer": [int(qty // fac), E.num(qty - (qty // fac) * fac)]}
    if len(units) == 1 and base and units[0] != base:
        raise E.Abstain("unit differs from base unit")
    if len(units) > 1:
        return None
    return {"type": "number", "answer": E.num(qty)}


def stock_value_intent(ctx, E):
    """Value of the stock on hand at cost: quantity on hand x catalogue cost per base unit, items in stock only (§5)."""
    import re
    text, biz, P = ctx.text, ctx.biz, ctx.periods
    if not re.search(r"gia von|tien von|\bvon\b", text) or not re.search(r"\bkho\b|\bton\b|con hang|hang con", text):
        return None
    if re.search(r"han dung|het han\b|han su dung|nhap|loinhuan|\bloi\b", text) or re.search(r"\blô\b", ctx.low) or ctx.ents or ctx.nums or len(P) > 1:
        return None
    if P and P[0].kind not in ("day", "upto"):
        return None
    stock, ledger = getattr(biz, "stock", None), getattr(biz, "stock_ledger", None)
    lots = getattr(biz, "lots", None)
    if stock is None and ledger is None and lots is None:
        if getattr(biz, "imports", None) is not None or getattr(biz, "import_slips", None) is not None:
            raise E.Abstain("stock must be derived from purchases and sales (not modelled)")
        raise E.missing(biz, "dữ liệu không ghi tồn kho")
    cost = {k: v.get("cost") for k, v in biz.prod.items() if v.get("cost") is not None}
    if not cost:
        raise E.missing(biz, "danh mục hàng không ghi giá vốn")
    x = min(P[0].end, ctx.today) if P else ctx.today
    if ledger is not None:
        g = ledger[ledger["time"] < pd.Timestamp(x) + pd.Timedelta(days=1)].sort_values(["time", "seq"])
        qty = g.groupby("pkey")["bal"].last().to_dict()
    elif stock is not None and x == biz.data_end:
        qty = stock
    elif lots is not None:
        from .lots import stock_from_lots
        qty = stock_from_lots(biz, x)
    else:
        raise E.Abstain("stock only known at the end of the data")
    total = 0.0
    for k, q in qty.items():
        if q and q > 0:
            if k not in cost:
                raise E.Abstain("an item in stock has no catalogue cost")
            total += q * cost[k]
    return {"type": "number", "answer": E.money(total)}


def import_minus_sales(ctx, E):
    """'Nhập trừ bán': quantity purchased minus quantity sold in the period, in base units (rulebook §5)."""
    import re
    text, biz, P = ctx.text, ctx.biz, ctx.periods
    if not re.search(r"nhap\b.{0,30}\btru\b.{0,30}\bban\b|nhap tru ban", text):
        return None
    prods = ctx.ent("prod")
    if len(prods) != 1 or len(ctx.ents) != 1 or len(P) != 1 or ctx.nums:
        return None
    lines = getattr(biz, "imports", None)
    if lines is None:
        raise E.missing(biz, "dữ liệu không ghi nhập hàng")
    if getattr(biz, "combo_names", False):
        raise E.Abstain("items and toppings share one cell")
    k = prods[0].value
    a, e = ctx.clip(P[0])
    L = lines[(lines["date"] >= a) & (lines["date"] <= e) & (~lines["cancelled"]) & (lines["pkey"] == k)]
    imp = float((L["qty"] * L["unit_f"].map(lambda u: biz.conv.get((k, u), 1.0))).sum())
    S = biz.lines
    S = S[(S["pkey"] == k) & S["st"].isin(["done", "ship"]) & (S["date"] >= a) & (S["date"] <= e)]
    base = fold(biz.prod.get(k, {}).get("base_unit") or "")
    if len(ctx.units) == 1 and base and ctx.units[0] != base:
        raise E.Abstain("unit differs from base unit")
    return {"type": "number", "answer": E.num(imp - float(S["qty_base"].sum()))}


def delivery_intent(ctx, E):
    """Delivery runs (rulebook §6 'Vật liệu xây dựng'): each run of each order counts once."""
    import re
    text, biz, P = ctx.text, ctx.biz, ctx.periods
    if not re.search(r"\bdot\b|\bgiao\b|\bcho di\b|da cho\b|xe (?:da )?(?:giao|cho)|duoc cho\b", ctx.norm):
        return None
    if re.search(r"giao tre|ngay giao|phi giao|dang giao|chua giao|khong giao", text):
        return None
    D = getattr(biz, "deliveries", None)
    if D is None:
        if re.search(r"\bdot\b", text):
            raise E.missing(biz, "dữ liệu không ghi giao hàng")
        return None
    P = [p for p in P if not (p.kind == "day" and p.start == ctx.today)] or P
    if len(P) != 1:
        return None
    a, e = ctx.clip(P[0])
    today = pd.Timestamp(ctx.today)
    D = D[D["date"] <= today]
    prods = ctx.ent("prod")
    placed = re.search(r"don (?:hang )?(?:da )?dat|khach dat|dat (?:trong )?<P>|don <P>|cho khach dat", text) is not None
    O = biz.orders
    ids = set(O[(O["date"] >= a) & (O["date"] <= e)]["order_id"])
    nums = [(op, v) for (op, v) in ctx.nums]
    if prods:
        if len(prods) != 1 or len(ctx.ents) != 1 or nums or not placed:
            return None
        k = prods[0].value
        base = fold(biz.prod.get(k, {}).get("base_unit") or "")
        if len(ctx.units) == 1 and base and ctx.units[0] != base:
            raise E.Abstain("unit differs from base unit")
        sel = D[D["order_id"].isin(ids) & (D["pkey"] == k)]
        return {"type": "number", "answer": E.num(sel["qty"].sum())}
    if ctx.ents:
        return None
    if placed and len(nums) == 1 and re.search(r"\bdot\b", ctx.norm):
        op, v = nums[0]
        runs = D[D["order_id"].isin(ids)].groupby("order_id")["batch"].nunique()
        n = int((runs >= v).sum()) if op in ("ge", "k") else int((runs > v).sum()) if op == "gt" else None
        if n is None:
            return None
        return {"type": "number", "answer": n}
    if nums or placed or not re.search(r"\bdot\b", ctx.norm):
        return None
    sel = D[(D["date"] >= a) & (D["date"] <= e)]
    return {"type": "number", "answer": int(len(sel.drop_duplicates(["order_id", "batch"])))}


def net_profit_intent(ctx, E):
    """Net profit of a month = gross profit minus everything in the expense book for that month (rulebook §5)."""
    import re
    text, biz, P = ctx.text, ctx.biz, ctx.periods
    if not re.search(r"lai rong|loi nhuan rong|lai thuan|loinhuan rong|loinhuan thuan|loi that|tru (?:het |di |xong )?(?:moi |cac |toan bo )?chi phi", text):
        return None
    if ctx.ents or ctx.nums or len(P) != 1:
        return None
    p = P[0]
    if p.grain != "month" and p.kind != "mtd":
        raise E.Abstain("net profit is defined per month")
    L0 = biz.lines
    if L0 is None or "cost" not in L0 or L0["cost"].isna().all():
        raise E.missing(biz, "dữ liệu không ghi giá vốn trên dòng bán")
    X = getattr(biz, "expenses", None)
    if X is None:
        raise E.missing(biz, "dữ liệu không ghi sổ chi phí")
    if getattr(biz, "combo_names", False):
        raise E.Abstain("items and toppings share one cell")
    L = ctx.lines(p, None)
    if L["cost"].isna().any() or L["qty"].isna().any():
        raise E.Abstain("missing cost on some lines")
    gross = float((L["amount"] - L["qty"] * L["cost"]).sum())
    a, e = ctx.clip(p)
    exp = float(X[(X["date"] >= a) & (X["date"] <= e)]["amount"].sum())
    return {"type": "number", "answer": E.money(gross - exp)}


def out_of_stock_intent(ctx, E):
    """Items bought for resale whose stock is zero at the end of the day (rulebook §5 'Mặt hàng hết hàng cuối ngày')."""
    import re
    text, biz, P = ctx.text, ctx.biz, ctx.periods
    if not re.search(r"het hang|dut hang|chay hang|het sach hang|khong con hang", text):
        return None
    if re.search(r"han dung|het han\b", text) or re.search(r"\blô\b", ctx.low) or ctx.ents or ctx.nums or len(P) > 1:
        return None
    if P and P[0].kind not in ("day", "upto"):
        return None
    if not re.search(r"may|bao nhieu|dem|so", text):
        return None
    stock, ledger, lots = getattr(biz, "stock", None), getattr(biz, "stock_ledger", None), getattr(biz, "lots", None)
    x = min(P[0].end, ctx.today) if P else ctx.today
    if ledger is not None:
        g = ledger[ledger["time"] < pd.Timestamp(x) + pd.Timedelta(days=1)].sort_values(["time", "seq"])
        qty = g.groupby("pkey")["bal"].last().to_dict()
    elif lots is not None:
        from .lots import stock_from_lots
        qty = stock_from_lots(biz, x)
    elif stock is not None and x == biz.data_end:
        imp = getattr(biz, "imports", None)
        bought = set(imp["pkey"].dropna().unique()) if imp is not None else set(stock)
        qty = {k: v for k, v in stock.items() if k in bought}
    else:
        if stock is None and getattr(biz, "imports", None) is None:
            raise E.missing(biz, "dữ liệu không ghi tồn kho")
        raise E.Abstain("stock on that day unknown")
    return {"type": "number", "answer": int(sum(1 for v in qty.values() if abs(v) < 1e-9))}


def due_date_intent(ctx, E):
    """Credit orders by the payment due date written in the book ('hạn thanh toán')."""
    import re
    text, biz, P = ctx.text, ctx.biz, ctx.periods
    if not re.search(r"ngay phai tra|han thanh toan|han tra\b|han phai tra", text):
        return None
    if not re.search(r"\bdon\b", ctx.norm) or not re.search(r"may|bao nhieu|dem|so don", text):
        return None
    due = getattr(biz, "due_dates", None)
    if due is None:
        raise E.missing(biz, "dữ liệu không ghi hạn thanh toán")
    ents = [e for e in ctx.ents if not (e.kind == "pay" and e.value == "debt")]
    if ents or ctx.nums:
        raise E.Abstain("due date question with extra conditions")
    today = pd.Timestamp(ctx.today)
    O = biz.orders
    placed = set(O[(O["date"] <= today) & (O["st"] != "cancel")]["order_id"])
    D = due[due["order_id"].isin(placed)]
    Pm = [p for p in P if not (p.kind == "day" and p.start == ctx.today)]
    if re.search(r"muon hon hom nay|sau hom nay|sau ngay hom nay|chua (?:toi|den) han|con sau", ctx.norm) and not Pm:
        return {"type": "number", "answer": int((D["due"] > today).sum())}
    if len(Pm) == 1 and Pm[0].kind in ("month", "range", "quarter", "week", "year") and re.search(r"trong <P>|vao <P>|roi vao|thuoc <P>|nam trong", text):
        a, e = pd.Timestamp(Pm[0].start), pd.Timestamp(Pm[0].end)
        return {"type": "number", "answer": int(((D["due"] >= a) & (D["due"] <= e)).sum())}
    raise E.Abstain("due date question not modelled")


_WEEKDAYS = [(r"thứ hai|thứ 2\b", 0), (r"thứ ba|thứ 3\b", 1), (r"thứ tư|thứ 4\b", 2), (r"thứ năm|thứ 5\b", 3), (r"thứ sáu|thứ 6\b", 4),
             (r"thứ bảy|thứ 7\b", 5), (r"chủ nhật|chúa nhật", 6)]
_WD = r"(?<!doanh )(?<!da )(?<!tien )\bthu "
_WEEKDAYS_PLAIN = [(_WD + r"(?:hai|2)\b", 0), (_WD + r"(?:ba|3)\b", 1), (_WD + r"(?:tu|4)\b(?! (?:cao|thap|nhieu|it|lon|nho|tren|duoi))", 2), (_WD + r"(?:nam|5)\b(?! (?:nay|truoc|ngoai|roi|\d))", 3),
                   (_WD + r"(?:sau|6)\b", 4), (_WD + r"(?:bay|7)\b", 5), (r"\bchu nhat\b|\bcn\b", 6)]


def weekday_avg_intent(ctx, E):
    """Average per given weekday: divide by the number of such weekdays in the period, including days without orders (§2)."""
    import re
    low, text, biz, P = ctx.low, ctx.text, ctx.biz, ctx.periods
    if not re.search(r"trung binh|binh quan", text):
        return None
    accented = re.search(r"[àáảãạăâđèéêìíòóôơùúưỳý]", low) is not None
    hits = [w for rx, w in (_WEEKDAYS if accented else _WEEKDAYS_PLAIN) if re.search(rx, low if accented else ctx.raw)]
    if len(hits) != 1:
        return None
    if len(P) != 1 or ctx.ents:
        raise E.Abstain("weekday average with extra conditions")
    a, e = ctx.clip(P[0])
    days = pd.date_range(a, e, freq="D")
    n_days = int((days.weekday == hits[0]).sum())
    if not n_days:
        raise E.Abstain("no such weekday in the period")
    O = biz.orders
    O = O[(O["date"] >= a) & (O["date"] <= e) & O["st"].isin(["done", "ship"]) & (O["date"].dt.weekday == hits[0])]
    if re.search(r"doanh thu|\btien\b", text) and not re.search(r"\bdon\b|luot|phieu", ctx.norm):
        return {"type": "number", "answer": E.money(O["revenue"].sum() / n_days)}
    if re.search(r"\bdon\b|luot|phieu", ctx.norm):
        E.need_orders(ctx)
        return {"type": "number", "answer": E.num(len(O) / n_days)}
    raise E.Abstain("weekday average of an unknown measure")
