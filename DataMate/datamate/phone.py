"""Phone shops (rulebook §6 'Cửa hàng điện thoại'): abbreviated item names, trade-ins, model lines."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .textnorm import fold, is_blank, parse_num, squash


def _hdr(tb, names: tuple, default=None):
    from .model import split_name
    for c in tb.columns:
        if split_name(c) in names:
            return c
    return default


def prepare_abbreviations(tables) -> dict | None:
    """Expand abbreviated item names using the 'Ghi tắt' sheet and attach the item type as a product group."""
    gt = next((t for t in tables if fold(t.name) == "ghi tat"), None)
    if gt is None or len(gt.columns) < 2 or not len(gt.df):
        return None
    ac = _hdr(gt, ("ghi tat", "viet tat", "ten tat", "cach ghi"), gt.columns[0])
    nc = _hdr(gt, ("ten hang", "ten day du", "ten", "hang", "ten san pham"), gt.columns[1])
    tc = _hdr(gt, ("loai", "loai hang", "nhom", "phan loai", "nhom hang"), gt.columns[2] if len(gt.columns) > 2 else None)
    amap, kind = {}, {}
    for _, row in gt.df.iterrows():
        if is_blank(row[nc]):
            continue
        full = str(row[nc]).strip()
        amap[squash(full)] = full
        if not is_blank(row[ac]):
            amap[squash(row[ac])] = full
        if tc and not is_blank(row[tc]):
            kind[full] = str(row[tc]).strip()
    done = False
    for tb in tables:
        if tb is gt or len(tb.df) < 50:
            continue
        for c in list(tb.columns):
            vals = [v for v in tb.df[c].head(400) if not is_blank(v)]
            if len(vals) < 20 or sum(squash(v) in amap for v in vals) < 0.9 * len(vals):
                continue
            tb.df[c] = tb.df[c].map(lambda v: amap.get(squash(v), v) if not is_blank(v) else v)
            if kind and "Loại hàng" not in tb.columns:
                tb.df["Loại hàng"] = tb.df[c].map(lambda v: kind.get(v) if not is_blank(v) else None)
                tb.columns.append("Loại hàng")
                tb.meanings["Loại hàng"] = "nhóm hàng"
            done = True
            break
    return {"kind": kind} if done else None


def model_line(name: str) -> str:
    """'dòng máy' = the part of the name before the storage size '…GB'; the whole name when there is none."""
    m = re.search(r"\s\d+\s?(?:GB|TB)\b", name, flags=re.I)
    return (name[:m.start()] if m else name).strip()


def brand_of(name: str) -> str:
    w = name.split()[0] if name.split() else ""
    return "Apple" if fold(w) in ("iphone", "ipad") else w


def attach_tradeins(biz) -> None:
    """orders['tradein'] = money for the customer's old phone that was deducted from the order."""
    from .model import split_name
    O = biz.orders
    if O is None:
        return
    O["tradein"] = 0.0
    lt = biz.table(biz.lines_table) if biz.lines_table else None
    if lt is not None:
        col = next((c for c in lt.columns if fold(lt.meanings.get(c, "")).startswith("tien thu may cu") or split_name(c) == "thu may cu"), None)
        oc = biz.roles[lt.name].get("order_id")
        if col and oc and biz.lines is not None and "order_no" in O:
            t = pd.DataFrame({"no": lt.df[oc].map(lambda v: None if is_blank(v) else str(v).strip()).ffill(),
                              "v": lt.df[col].map(parse_num)})
            per = t.dropna().groupby("no")["v"].first()
            O["tradein"] = O["order_no"].map(per).fillna(0.0)
            return
    for tb in biz.tables:
        ref = next((c for c in tb.columns if split_name(c).startswith("tru vao")), None)
        val = next((c for c in tb.columns if split_name(c) in ("gia thu", "tien thu", "gia thu may")), None)
        if ref and val and "order_no" in O:
            t = pd.DataFrame({"no": tb.df[ref].map(parse_num), "v": tb.df[val].map(parse_num)}).dropna()
            per = t.groupby("no")["v"].sum()
            O["tradein"] = O["order_no"].map(lambda s: per.get(parse_num(s), 0.0) if s is not None else 0.0)
            return


def phone_pretext(text: str) -> str:
    text = re.sub(r"\b(?:1|mot) may\b", "moi may", text)
    text = re.sub(r"\bdt\b", "dien thoai", text)
    return re.sub(r"\s+", " ", re.sub(r"\b(?:chiec|cai)\b", " ", text)).strip()


def phone_intent(ctx, E):
    """Shop-type specific questions; returns None when the generic rules should handle the question."""
    biz, text, P, low = ctx.biz, ctx.text, ctx.periods, ctx.low
    L, O = biz.lines, biz.orders
    bk = books_intent(ctx, E)
    if bk is not None:
        return bk
    tradein = re.search(r"thu (?:mua )?(?:lai )?may cu|may cu thu|tien may cu", text) is not None
    payable = re.search(r"khach phai tra|phai tra", text) is not None and not re.search(r"ngay phai tra", text)
    grp = L["group"].map(lambda v: fold(v) if not is_blank(v) else "") if (L is not None and "group" in L) else None
    names = {k: v["name"] for k, v in biz.prod.items()}
    has_model = False
    if grp is not None:
        phone_keys = set(L[grp.isin(["may moi", "may cu"])]["pkey"].dropna().unique())
        ctx._models = {}
        for k in phone_keys:
            ctx._models.setdefault(model_line(names.get(k, "")), set()).add(k)
        from .question import normalize
        ctx._model_norm = {" ".join(re.findall(r"[a-z0-9]+", normalize(m))): m for m in ctx._models if m}
        flat = " " + " ".join(re.findall(r"[a-z0-9]+", ctx.norm)) + " "
        ctx._model_hit = next((m for nm, m in sorted(ctx._model_norm.items(), key=lambda kv: -len(kv[0])) if f" {nm} " in flat), None)
        has_model = ctx._model_hit is not None
    asks_phone = has_model or re.search(r"dien thoai|\bmay (?:moi|cu)\b|khong (?:tinh|ke) phu kien|tru phu kien|dong may|hang may|"
                           r"bao nhieu may\b|may may\b|tien ban may|moi may\b|mot may\b|1 may\b|so may\b|may ban ra|may [a-z]+ chiem", text) is not None
    if not (tradein or payable or asks_phone):
        return None
    if len(P) != 1:
        raise E.Abstain("phone question needs exactly one period")
    p = P[0]
    a, e = ctx.clip(p)
    Ov = O[(O["date"] >= a) & (O["date"] <= e) & O["st"].isin(["done", "ship"])]
    pctq = "%" in text
    avg = re.search(r"trung binh|binh quan", text) is not None
    if payable:
        if ctx.ents or ctx.nums or "order_total" not in Ov or Ov["order_total"].isna().any():
            raise E.Abstain("payable question not computable")
        if avg:
            if not len(Ov):
                raise E.Abstain("no orders")
            return {"type": "number", "answer": E.money(Ov["order_total"].sum() / len(Ov))}
        return {"type": "number", "answer": E.money(Ov["order_total"].sum())}
    if tradein and not asks_phone or (tradein and re.search(r"thu (?:mua )?(?:lai )?may cu", text) and not re.search(r"ban", text)):
        if ctx.ents or ctx.nums:
            raise E.Abstain("trade-in question with extra conditions")
        if pctq:
            if not len(Ov):
                raise E.Abstain("no orders")
            return {"type": "percent", "answer": E.pct(100.0 * (Ov["tradein"] > 0).sum() / len(Ov))}
        if re.search(r"bao nhieu don|may don|so don", text):
            return {"type": "number", "answer": int((Ov["tradein"] > 0).sum())}
        return {"type": "number", "answer": E.money(Ov["tradein"].sum())}
    if payable:
        if ctx.ents or ctx.nums or "order_total" not in Ov or Ov["order_total"].isna().any():
            raise E.Abstain("payable question not computable")
        if avg:
            if not len(Ov):
                raise E.Abstain("no orders")
            return {"type": "number", "answer": E.money(Ov["order_total"].sum() / len(Ov))}
        return {"type": "number", "answer": E.money(Ov["order_total"].sum())}
    # ---- phones (new + used) vs accessories
    if grp is None:
        raise E.Abstain("item types unknown")
    mdl = _model_question(ctx, E, L, grp, a, e)
    if mdl is not None:
        return mdl
    if ctx.ents or [v for (op, v) in ctx.nums if not (op == "k" and v == 1)]:
        raise E.Abstain("phone category question with extra conditions")
    Lv = L[(L["date"] >= a) & (L["date"] <= e) & L["st"].isin(["done", "ship"])]
    g = grp.reindex(Lv.index)
    new, used = g == "may moi", g == "may cu"
    has_new = bool(re.search(r"máy mới|điện thoại mới", low) or re.search(r"\bmay moi\b|dien thoai moi", text) and "mới" not in low and "mọi" not in low)
    has_used = bool(re.search(r"máy cũ|điện thoại cũ", low) or re.search(r"\bmay cu\b|dien thoai cu", text))
    both = (has_new and has_used) or not (has_new or has_used)
    if re.search(r"dong may|hang may", text) or "hãng" in low:
        raise E.Abstain("model line / brand questions not modelled")
    phones = Lv[new | used]
    if pctq:
        if has_new != has_used and re.search(r"tien ban may|doanh thu may|tien may", text):
            tot = float(phones["amount"].sum())
            if not tot:
                raise E.Abstain("zero base")
            part = float(Lv[new if has_new else used]["amount"].sum())
            return {"type": "percent", "answer": E.pct(100.0 * part / tot)}
        raise E.Abstain("phone share question not modelled")
    sel = phones if both else Lv[new if has_new else used]
    money_q = re.search(r"\btien\b|doanh thu|\bgia\b", text) is not None
    if avg and re.search(r"moi may|mot may|1 may|moi dien thoai|mot dien thoai", text):
        q = float(sel["qty"].sum())
        if not q:
            raise E.Abstain("no phones sold")
        return {"type": "number", "answer": E.money(float(sel["amount"].sum()) / q)}
    if avg:
        raise E.Abstain("phone average not modelled")
    if money_q:
        return {"type": "number", "answer": E.money(sel["amount"].sum())}
    return {"type": "number", "answer": E.num(sel["qty"].sum())}


def _model_question(ctx, E, L, grp, a, e):
    """Questions about one model line, the best-selling model lines, or a brand's share of phones sold."""
    biz, text, low = ctx.biz, ctx.text, ctx.low
    Lv = L[(L["date"] >= a) & (L["date"] <= e) & L["st"].isin(["done", "ship"])]
    g = grp.reindex(Lv.index)
    phones = Lv[g.isin(["may moi", "may cu"])]
    names = {k: v["name"] for k, v in biz.prod.items()}
    hit = getattr(ctx, "_model_hit", None)
    ks = [int(v) for (op, v) in ctx.nums if op == "k"]
    if re.search(r"dong may", text) and re.search(r"\btop\b|nhieu (?:may )?nhat|ban chay", text) and not hit:
        q = phones.groupby(phones["pkey"].map(lambda k: model_line(names.get(k, ""))))["qty"].sum()
        k = ks[0] if len(ks) == 1 else 1
        top = E._rank(q, k, lambda m: fold(m))
        return {"type": "list", "answer": top} if len(ks) == 1 else {"type": "string", "answer": top[0]}
    if hit:
        if "%" in text or re.search(r"trung binh|binh quan|nhat\b|tro len|it nhat", text):
            raise E.Abstain("model line question with extra conditions")
        sel = phones[phones["pkey"].isin(ctx._models[hit])]
        if re.search(r"\btien\b|doanh thu", text):
            return {"type": "number", "answer": E.money(sel["amount"].sum())}
        return {"type": "number", "answer": E.num(sel["qty"].sum())}
    if "%" in text and re.search(r"so may|tong so may|luong may|may ban ra", text):
        brands = {}
        for k in phones["pkey"].dropna().unique():
            brands.setdefault(fold(brand_of(names.get(k, ""))), set()).add(k)
        flat = " " + " ".join(re.findall(r"[a-z0-9]+", ctx.norm)) + " "
        found = [b for b in brands if b and f" {b} " in flat]
        if "apple" in brands and " iphone " in flat and "apple" not in found:
            found.append("apple")
        if len(found) != 1 or ctx.nums:
            raise E.Abstain("brand not identified")
        tot = float(phones["qty"].sum())
        if not tot:
            raise E.Abstain("zero base")
        return {"type": "percent", "answer": E.pct(100.0 * float(phones[phones["pkey"].isin(brands[found[0]])]["qty"].sum()) / tot)}
    return None


# ------------------------------------------------------------------ instalment, debt and warranty books
def _find(biz, need: tuple):
    from .model import split_name
    for tb in biz.tables:
        names = {split_name(c): c for c in tb.columns}
        if all(any(n.startswith(w) for n in names) for w in need):
            return tb, names
    return None, {}


def _c(names: dict, *prefixes):
    for p in prefixes:
        for n, c in names.items():
            if n.startswith(p):
                return c
    return None


def build_phone_books(biz) -> None:
    """Parse the hand-kept side books: instalments ('Trả góp'), debts ('Sổ nợ') and warranty visits ('Bảo hành')."""
    from .model import _strip_phone
    from .textnorm import parse_datetime
    biz.instalments, biz.debt_book, biz.warranty = None, None, None

    def d(v):
        x = parse_datetime(v, None)
        return pd.Timestamp(x).normalize() if x is not None else None

    tb, nm = _find(biz, ("so ky", "da gop"))
    if tb is not None:
        cg, cdown, cn, cper, cdue = _c(nm, "gia"), _c(nm, "tra truoc"), _c(nm, "so ky"), _c(nm, "gop/thang", "gop thang", "gop moi"), _c(nm, "han gop")
        cdate, cstt, ckh, csdt, cpd, cpa = _c(nm, "ngay"), _c(nm, "stt"), _c(nm, "khach"), _c(nm, "sdt", "so dien thoai", "dien thoai"), _c(nm, "ngay gop"), _c(nm, "da gop")
        cmay = _c(nm, "may")
        rows, cur = [], None
        for _, r in tb.df.iterrows():
            if cg and not is_blank(r[cg]):
                cur = {"no": parse_num(r[cstt]) if cstt else None, "date": d(r[cdate]), "cust": _strip_phone(str(r[ckh])).strip() if ckh and not is_blank(r[ckh]) else None,
                       "phone": re.sub(r"\D", "", str(r[csdt])) if csdt and not is_blank(r[csdt]) else "", "price": parse_num(r[cg]),
                       "down": (parse_num(r[cdown]) or 0.0) if cdown else 0.0, "n": int(parse_num(r[cn]) or 0), "per": parse_num(r[cper]) or 0.0,
                       "due_day": int(parse_num(r[cdue]) or 1), "model": str(r[cmay]).strip() if cmay and not is_blank(r[cmay]) else None, "pays": []}
                rows.append(cur)
            if cur is not None and cpa and not is_blank(r[cpa]):
                cur["pays"].append((d(r[cpd]) if cpd else None, parse_num(r[cpa]) or 0.0))
        biz.instalments = [r for r in rows if r["date"] is not None and r["price"] is not None]
    tb, nm = _find(biz, ("no", "da tra", "ngay tra"))
    if tb is not None:
        cdate, ckh, csdt, cno, cpa, cpd, cnote = _c(nm, "ngay"), _c(nm, "khach"), _c(nm, "sdt", "so dien thoai"), nm.get("no"), _c(nm, "da tra"), _c(nm, "ngay tra"), _c(nm, "ghi chu")
        rows, cur = [], None
        for _, r in tb.df.iterrows():
            if cno and not is_blank(r[cno]):
                cur = {"date": d(r[cdate]), "cust": _strip_phone(str(r[ckh])).strip() if ckh and not is_blank(r[ckh]) else None,
                       "phone": re.sub(r"\D", "", str(r[csdt])) if csdt and not is_blank(r[csdt]) else "", "owed": parse_num(r[cno]) or 0.0, "pays": []}
                rows.append(cur)
            if cur is not None and cpa and not is_blank(r[cpa]):
                cur["pays"].append((d(r[cpd]) if cpd else None, parse_num(r[cpa]) or 0.0))
        biz.debt_book = [r for r in rows if r["date"] is not None]
    tb, nm = _find(biz, ("ngay nhan", "imei"))
    if tb is not None:
        W = pd.DataFrame({"date": tb.df[_c(nm, "ngay nhan")].map(d), "imei": tb.df[_c(nm, "imei")].map(lambda v: re.sub(r"\D", "", str(v)) if not is_blank(v) else None)})
        cm = _c(nm, "may")
        W["model"] = tb.df[cm].map(lambda v: str(v).strip() if not is_blank(v) else None) if cm else None
        biz.warranty = W.dropna(subset=["date"])
    # IMEIs of phones sold, with the status of their order
    biz.sold_imei = None
    lt = biz.table(biz.lines_table) if biz.lines_table else None
    if lt is not None and biz.orders is not None and "order_no" in biz.orders:
        ic = next((c for c in lt.columns if fold(c) == "imei"), None)
        r = biz.roles[lt.name]
        if ic and "order_id" in r and "date" in r:
            st = dict(zip(biz.orders["order_no"], biz.orders["st"]))
            od = dict(zip(biz.orders["order_no"], biz.orders["date"]))
            no = lt.df[r["order_id"]].map(lambda v: None if is_blank(v) else str(v).strip()).ffill()
            out = []
            for n_, v in zip(no, lt.df[ic]):
                if is_blank(v) or n_ not in st:
                    continue
                for im in re.split(r"\s*/\s*", str(v)):
                    im = re.sub(r"\D", "", im)
                    if im:
                        out.append((im, od[n_], st[n_]))
            biz.sold_imei = pd.DataFrame(out, columns=["imei", "date", "st"])


def _due_dates(c: dict):
    import calendar as _cal
    out, y, m = [], c["date"].year, c["date"].month
    for _ in range(c["n"]):
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append(pd.Timestamp(y, m, min(c["due_day"], _cal.monthrange(y, m)[1])))
    return out


def _inst_state(c: dict, x: pd.Timestamp):
    """(remaining incl. not yet due, overdue amount, instalments due up to and including x) at the end of day x."""
    financed = c["price"] - c["down"]
    paid = sum(a for (dt_, a) in c["pays"] if dt_ is not None and dt_ <= x)
    dues = _due_dates(c)
    n_incl = sum(1 for dd in dues if dd <= x)
    n_before = sum(1 for dd in dues if dd < x)
    overdue = max(0.0, min(n_before * c["per"], financed) - paid)
    return max(0.0, financed - paid), overdue, n_incl


def books_intent(ctx, E):
    """Instalment / debt / warranty questions of a phone shop. Returns None when not applicable."""
    biz, text, low, P = ctx.biz, ctx.text, ctx.low, ctx.periods
    inst = re.search(r"tra gop|han gop|ky gop|tien gop|khoan gop|hop dong gop", text) is not None or "góp" in low
    warr = re.search(r"bao hanh", text) is not None
    debt = re.search(r"\bno\b|con thieu|mua chiu|cong no", text) is not None
    if not (inst or warr or debt):
        return None
    x = pd.Timestamp(min(P[0].end, ctx.today)) if (len(P) == 1 and P[0].kind in ("day", "upto")) else pd.Timestamp(ctx.today)
    flat = " " + " ".join(re.findall(r"[a-z0-9]+", ctx.norm)) + " "
    if warr:
        W = getattr(biz, "warranty", None)
        if W is None:
            raise E.Abstain("no warranty book")
        Pw = [p for p in P if not (p.kind == "day" and p.start == ctx.today)] or P
        if len(Pw) != 1:
            raise E.Abstain("warranty question needs one period")
        a, e = ctx.clip(Pw[0])
        if re.search(r"may ban|ban ra|ban (?:trong|thang)|da ban", text) and re.search(r"toi hom nay|den hom nay|toi nay|da (?:duoc )?mang|it nhat", ctx.norm) and "%" not in text:
            S = getattr(biz, "sold_imei", None)
            if S is None:
                raise E.Abstain("no IMEI of sold phones")
            sold = set(S[(S["date"] >= a) & (S["date"] <= e) & S["st"].isin(["done", "ship"])]["imei"])
            came = set(W[W["date"] <= pd.Timestamp(ctx.today)]["imei"].dropna())
            return {"type": "number", "answer": int(len(sold & came))}
        Wp = W[(W["date"] >= a) & (W["date"] <= e)]
        if re.search(r"dong may", text) and re.search(r"nhat", text):
            g = Wp.dropna(subset=["model"]).groupby("model").size().astype(float)
            return {"type": "string", "answer": E._rank(g, 1, lambda m: fold(m))[0]}
        if "%" in text or re.search(r"ti le|ty le", text):
            L = biz.lines
            grp = L["group"].map(lambda v: fold(v) if not is_blank(v) else "")
            Lv = L[(L["date"] >= a) & (L["date"] <= e) & L["st"].isin(["done", "ship"]) & grp.isin(["may moi", "may cu"])]
            sold = float(Lv["qty"].sum())
            if not sold:
                raise E.Abstain("no phones sold")
            return {"type": "percent", "answer": E.pct(100.0 * len(Wp) / sold)}
        if ctx.ents or re.search(r"khach nao|\bai\b", text):
            raise E.Abstain("warranty question not modelled")
        return {"type": "number", "answer": int(len(Wp))}
    C = getattr(biz, "instalments", None) or []
    Dk = getattr(biz, "debt_book", None) or []
    if len(P) > 1 or (len(P) == 1 and P[0].kind not in ("day", "upto")):
        raise E.Abstain("debt / instalment question over a period is not modelled")

    def named(rows):
        hits = [r for r in rows if r["cust"] and f" {' '.join(re.findall(r'[a-z0-9]+', fold(r['cust'])))} " in flat]
        return hits

    if inst:
        if not C:
            raise E.Abstain("no instalment book")
        act = [c for c in C if c["date"] <= x]
        mine = named(act)
        who = mine if mine else (act if not re.search(r"\bkhach [a-z]+ [a-z]+ [a-z]+\b", text) or re.search(r"khach (?:tra gop|con|qua|dang|da|chua)", text) else None)
        if who is None:
            raise E.Abstain("customer not found in the instalment book")
        st = [_inst_state(c, x) for c in who]
        if re.search(r"may ky|bao nhieu ky|so ky", text) and re.search(r"toi han|den han", text) and not re.search(r"qua han", text):
            if not mine:
                raise E.Abstain("instalment count needs a customer")
            return {"type": "number", "answer": int(sum(s[2] for s in st))}
        overdue_q = re.search(r"qua han|cham gop|tre han|cham dong", text) is not None
        if re.search(r"may khach|bao nhieu khach|may nguoi|bao nhieu nguoi|so khach", text):
            if not overdue_q:
                raise E.Abstain("customer count without overdue condition")
            keys = {(fold(c["cust"] or ""), c["phone"]) for c, s in zip(who, st) if s[1] > 0.5}
            return {"type": "number", "answer": int(len(keys))}
        if overdue_q:
            return {"type": "number", "answer": E.money(sum(s[1] for s in st))}
        if re.search(r"chua gop|con lai|con thieu|chua tra|chua dong", text):
            return {"type": "number", "answer": E.money(sum(s[0] for s in st))}
        raise E.Abstain("instalment question not modelled")
    # ---- plain debt questions include instalment contracts (rulebook: trả góp là đơn ghi nợ)
    if ctx.ents or re.search(r"tra no|thu no|no them|tra bot", text):
        raise E.Abstain("debt flow questions of a phone shop are not modelled")
    items = []  # (purchase date, customer key, owed at x)
    for r in Dk:
        if r["date"] <= x:
            owed = r["owed"] - sum(a for (dt_, a) in r["pays"] if dt_ is not None and dt_ <= x)
            if owed > 0.5:
                items.append((r["date"], (fold(r["cust"] or ""), r["phone"]), owed))
    for c in C:
        if c["date"] <= x:
            rem = _inst_state(c, x)[0]
            if rem > 0.5:
                items.append((c["date"], (fold(c["cust"] or ""), c["phone"]), rem))
    nums = [ctx.nums[int(i)] for i in re.findall(r"zq[nk](\d+)", text)]
    if re.search(r"tuoi no", text):
        vals = [v for (_op, v) in nums]
        if len(vals) == 6 and vals[0] == 3:
            vals = vals[1:]
        if len(vals) != 5:
            raise E.Abstain("ageing buckets not understood")
        a1, b1, a2, b2, c3 = vals
        age = [(x - d0).days for d0, _k, _o in items]
        out = [sum(o for (ag, (_d, _k, o)) in zip(age, items) if a1 <= ag <= b1), sum(o for (ag, (_d, _k, o)) in zip(age, items) if a2 <= ag <= b2),
               sum(o for (ag, (_d, _k, o)) in zip(age, items) if ag > c3)]
        return {"type": "list", "answer": [E.money(v) for v in out]}
    if nums:
        raise E.Abstain("debt question with numbers not modelled")
    if re.search(r"may khach|bao nhieu khach|may nguoi|bao nhieu nguoi|so khach", text):
        return {"type": "number", "answer": int(len({k for _d, k, _o in items}))}
    if re.search(r"bao nhieu|tong|nhieu", text) and not re.search(r"\btop\b|liet ke|khach nao|\bai\b", text):
        return {"type": "number", "answer": E.money(sum(o for _d, _k, o in items))}
    raise E.Abstain("debt question not modelled")
