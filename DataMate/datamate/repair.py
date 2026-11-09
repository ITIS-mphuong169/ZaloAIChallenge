"""Motorbike repair shops (rulebook §6 'Sửa xe'): one row = one visit (phiếu / lượt xe)."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .model import Biz, _date_col, _num_col
from .textnorm import fold, is_blank

_PLATE = re.compile(r"\b\d{2}[a-z]{1,2}\d?\s*-\s*\d{3,5}(?:\.\d{2})?\b")


def _col_by_meaning(tb, rx: str):
    for c in tb.columns:
        if re.search(rx, fold(tb.meanings.get(c, ""))):
            return c
    return None


def build_repair(biz: Biz) -> None:
    biz.repair = False
    book = None
    for tb in biz.tables:
        if _col_by_meaning(tb, r"dong xe va bien so|bien so") and _col_by_meaning(tb, r"^ngay sua|^ngay"):
            book = tb
            break
    if book is None or biz.orders is not None:
        return
    dc = _col_by_meaning(book, r"^ngay sua|^ngay")
    bc = _col_by_meaning(book, r"dong xe va bien so|bien so")
    mc = _col_by_meaning(book, r"^tho ")
    ac = _col_by_meaning(book, r"^tong tien ca phieu|^tong tien|^tien")
    nc = _col_by_meaning(book, r"^ghi chu")
    jc = _col_by_meaning(book, r"cac viec da lam|viec")
    if not (dc and ac):
        return
    df = book.df
    O = pd.DataFrame({"order_id": [f"r{i}" for i in range(len(df))]})
    O["date"] = _date_col(df, dc).dt.normalize().values
    O["dt"] = pd.NaT
    O["revenue"] = _num_col(df[ac]).fillna(0.0).values
    note = df[nc].map(lambda v: fold(v)) if nc else pd.Series([""] * len(df))
    O["warranty"] = note.map(lambda s: bool(re.search(r"bao hanh|\bbh\b", s))).values
    O["st"] = "done"
    O["cust"] = None
    O["cust_label"] = None
    bike = df[bc].map(lambda v: fold(v)) if bc else pd.Series([""] * len(df))
    O["plate"] = bike.map(lambda s: (_PLATE.search(s).group(0).replace(" ", "") if _PLATE.search(s) else None)).values
    O["jobs"] = df[jc].map(lambda v: fold(v)).values if jc else ""
    raw_model = [_model_text(v) for v in df[bc]] if bc else [None] * len(df)
    biz.bike_forms, biz.bike_label = _cluster_models([m for m in raw_model if m], list(zip(raw_model, O["plate"])))
    O["model"] = [biz.bike_forms.get(_form(m)) if m else None for m in raw_model]
    O["jobset"] = [frozenset() if w else _job_concepts(j) for j, w in zip(O["jobs"], O["warranty"])]
    # mechanics: nickname in the book ('A.Đức', 'a Đức', 'Đức') -> full name in the staff list
    full = {}
    biz.commission = {}
    for tb in biz.tables:
        nm = _col_by_meaning(tb, r"^ho ten tho|^ho ten")
        pc = _col_by_meaning(tb, r"hoa hong")
        if nm and tb.name != book.name:
            for _, row in tb.df.iterrows():
                if is_blank(row[nm]):
                    continue
                name = str(row[nm]).strip()
                full.setdefault(fold(name).split()[-1], []).append(name)
                if pc and not is_blank(row[pc]):
                    v = _num_col(pd.Series([row[pc]])).iloc[0]
                    if not pd.isna(v):
                        biz.commission[name] = float(v)

    def mech(v):
        if is_blank(v):
            return None
        s = re.sub(r"^(a\.|a |anh |c\.|c |chi |e\.|em |tho )\s*", "", fold(v)).strip()
        nick = s.split()[-1] if s else ""
        names = full.get(nick, [])
        return names[0] if len(names) == 1 else None

    O["staff"] = df[mc].map(mech).values if mc else None
    O = O.dropna(subset=["date"])
    biz.orders = O
    biz.lines = None
    biz.has_order_id = True
    biz.repair = True
    biz.schema_confident = True
    biz.unreliable = bool(mc) and O["staff"].isna().mean() > 0.05
    biz.data_start = O["date"].min().date()
    biz.data_end = biz.rd.data_end or O["date"].max().date()
    biz.staff_names = sorted({n for ns in full.values() for n in ns})


_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma"}


def _model_text(v):
    """The model part of 'Wave Alpha 59N9-950.06' (as written)."""
    if is_blank(v):
        return None
    t = str(v).strip()
    m = _PLATE.search(fold(t))
    if m:
        t = t[:m.start()] if len(fold(t)) == len(t) else re.sub(r"\s*\S+$", "", t)
    return t.strip() or None


def _form(m: str) -> str:
    t = fold(m)
    for g, w in _GREEK.items():
        t = t.replace(g, w)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t)).strip()


def _cluster_models(names: list, plates: list = ()):
    """Group spellings of one model line: case, spacing, initials ('AB'), a dropped first word ('Alpha'),
    a version suffix ('Vision 110', 'Sirius FI') or a cut-off word ('Ex'); two spellings seen on the same
    plates are the same line too."""
    from collections import Counter
    forms = Counter(_form(n) for n in names)
    spell = {}
    for n in names:
        spell.setdefault(_form(n), Counter())[n] += 1
    keys = {}
    for f in forms:
        keys.setdefault(f.replace(" ", ""), []).append(f)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(x, y):
        x, y = find(x), find(y)
        if x != y:
            parent[y] = x

    # the same bike (plate) written two ways
    by_plate = {}
    for m, pl in plates:
        if m and pl:
            by_plate.setdefault(pl, set()).add(_form(m).replace(" ", ""))
    shared = Counter()
    for ks in by_plate.values():
        ks = sorted(ks)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                shared[(ks[i], ks[j])] += 1
    for (x, y), n in shared.items():
        if n >= 2 and x in parent and y in parent:
            union(x, y)
    multi = [f for f in forms if " " in f]
    for k in list(keys):
        hits = set()
        for f in multi:
            w = f.split()
            fk = f.replace(" ", "")
            if fk == k:
                continue
            if "".join(x[0] for x in w) == k and len(k) >= 2:
                hits.add(fk)                      # initials
            elif w[-1] == k and len(k) >= 3:
                hits.add(fk)                      # first word dropped
            elif w[0] == k and all(x.isdigit() or len(x) <= 2 for x in w[1:]):
                hits.add(fk)                      # version suffix
        if not hits and len(k) >= 2:
            hits = {o for o in keys if o != k and o.startswith(k) and o.isalpha() and k.isalpha() and len(o) >= len(k) + 3}
            hits = hits if len({find(h) for h in hits}) == 1 else set()
        if len({find(h) for h in hits}) == 1:
            union(next(iter(hits)), k)
    form2c, label = {}, {}
    for k, fs in keys.items():
        for f in fs:
            form2c[f] = find(k)
    for c in set(form2c.values()):
        cnt = Counter()
        for f, cc in form2c.items():
            if cc == c:
                cnt.update(spell[f])
        best = sorted(cnt.items(), key=lambda kv: (-(" " in kv[0] or len(kv[0]) > 3), -kv[1], kv[0]))
        label[c] = best[0][0]
    return form2c, label


# (concept, pattern on a job as written in the book, pattern on an accented question, pattern on an unaccented question)
JOBS = [
    ("va", r"^va\b", r"\bvá\b", r"\bva (?:ruot|xe|vo|lop|banh|sam)\b"),
    ("thay_ruot", r"ruot|\bsam\b", r"thay ruột|ruột (?:xe )?mới|thay săm", r"thay ruot|ruot (?:xe )?moi|thay sam"),
    ("nhot_lap", r"\blap\b|hop so", r"\bláp\b|hộp số", r"(?:nhot|dau|thay) lap\b|hop so"),
    ("nhot_may", r"nhot|castrol|motul|shell|\bdau\b", r"nhớt|dầu máy|thay dầu", r"nhot|dau may|thay dau"),
    ("sua_dien", r"^(?:sua|lam) dien$|^dien$", r"(?:sửa|làm) điện", r"(?:sua|lam) dien\b"),
    ("bao_duong", r"^bao duong|^bd$|^bdtq$", r"bảo dưỡng", r"bao duong"),
    ("noi", r"\bnoi$", r"(?:vệ sinh|làm|vs) nồi|\bnồi\b", r"(?:ve sinh|lam|vs) noi\b"),
    ("bugi", r"bu ?gi", r"bu ?gi", r"bu ?gi"),
    ("binh", r"ac quy|^binh\b|thay binh", r"ắc quy|thay bình|bình (?:điện|gs)", r"ac quy|thay binh"),
    ("nsd", r"nhong|^nsd$", r"nhông|\bnsd\b", r"nhong|\bnsd\b"),
    ("bo_sau", r"(?:bo|thang|phanh) sau$", r"(?:bố|thắng|phanh) sau", r"\bbo (?:thang )?sau\b|phanh sau\b"),
    ("bo_truoc", r"(?:bo|thang|phanh) truoc$", r"(?:bố|thắng|phanh) trước", r"\bbo (?:thang )?truoc\b|phanh truoc\b"),
    ("den", r"bong den|den pha", r"bóng đèn|đèn pha|thay đèn", r"bong den|den pha|thay den"),
    ("sen", r"(?:can|chinh|tang) sen", r"(?:cân|chỉnh|tăng) sên", r"(?:can|chinh|tang) sen\b"),
    ("curoa", r"cu ?roa", r"cu ?roa", r"cu ?roa"),
    ("loc_gio", r"loc gio", r"lọc gió", r"loc gio"),
    ("nuoc_mat", r"nuoc (?:lam )?mat", r"nước (?:làm )?mát", r"nuoc (?:lam )?mat"),
    ("kim_phun", r"kim phun", r"kim phun", r"kim phun"),
    ("rua_xe", r"^rua(?: xe)?$", r"rửa xe", r"rua xe"),
    ("vo", r"\bvo\b", r"\bvỏ\b|thay lốp", r"thay vo\b|vo xe|thay lop"),
]
_ACCENT = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]")


def _job_concepts(jobs: str) -> frozenset:
    out = set()
    for tok in re.split(r"[,+;/]", jobs or ""):
        tok = tok.strip()
        if not tok or re.match(r"^(?:bao hanh|bh)\b", tok):
            continue
        for name, rx, _qa, _qp in JOBS:
            if re.search(rx, tok):
                out.add(name)
                break
    return frozenset(out)


def _asked_jobs(ctx) -> set:
    accented = _ACCENT.search(ctx.low) is not None
    src = re.sub(r"bình quân|trung bình", " ", ctx.low) if accented else re.sub(r"binh quan|trung binh", " ", ctx.raw)
    hit = {name for name, _rx, qa, qp in JOBS if re.search(qa if accented else qp, src)}
    if "va" in hit:
        hit.discard("thay_ruot")
    if "nhot_lap" in hit:
        hit.discard("nhot_may")
    return hit


def _asked_model(ctx):
    """Model line named in the question, or None; 'ambiguous' when two lines match."""
    biz = ctx.biz
    q = " " + _form(ctx.raw) + " "
    found = set()
    names = sorted(set(biz.bike_forms) | {f.replace(" ", "") for f in biz.bike_forms}, key=len, reverse=True)
    sq = {f.replace(" ", ""): c for f, c in biz.bike_forms.items()}
    for f in names:
        if len(f) < 2 or f.isdigit():
            continue
        if (" " + f + " ") in q:
            found.add(biz.bike_forms.get(f) or sq[f])
            q = q.replace(" " + f + " ", "  ")
    if not found:
        return None
    return next(iter(found)) if len(found) == 1 else "ambiguous"


def jobs_models(ctx, E):
    """Questions about the job done or the bike model / plate. None when the question is about neither."""
    biz, P, text, raw = ctx.biz, ctx.periods, ctx.text, ctx.raw
    O = biz.orders
    if "jobset" not in O:
        return None
    jobs = _asked_jobs(ctx)
    model = _asked_model(ctx)
    which_model = re.search(r"(?:dong|loai|hieu|mau) xe (?:nao|gi|mo)\b|xe (?:dong|loai|hieu) (?:nao|gi|mo)\b", raw) is not None \
        or re.search(r"(?:dong|loai|hieu|mau) xe (?:nao|gi)\b", ctx.norm) is not None
    which_bike = not which_model and re.search(r"\bxe (?:nao|mo)\b|bien so (?:nao|mo)\b", raw + " " + ctx.norm) is not None
    distinct = re.search(r"\bchiec\b|khac nhau|moi xe (?:chi )?(?:tinh|dem) (?:mot|1) lan", ctx.norm) is not None \
        and not re.search(r"\bluot\b|phieu", ctx.norm)
    if not (jobs or model or which_model or which_bike or distinct):
        return None
    if re.search(r"khach quen|xe quen|tro len|bao hanh|hoa hong|hue hong", text) and not (jobs or model or which_model or which_bike):
        return None
    if model == "ambiguous" or len(jobs) > 1:
        raise E.Abstain("more than one job / model in the question")
    staff = [e for e in ctx.ents if e.kind == "staff"]
    if len(P) != 1 or len(staff) != len(ctx.ents) or len(staff) > 1:
        raise E.Abstain("repair question with extra conditions")
    if re.search(r"khach quen|xe quen|tro len|bao hanh|hoa hong|hue hong|trung binh|binh quan|ti le|ty le|%|\bso voi\b|tang(?! sen)|giam", text):
        raise E.Abstain("job / model question with extra conditions")
    a, e = ctx.clip(P[0])
    O = O[(O["date"] >= a) & (O["date"] <= e)]
    if staff:
        O = O[O["staff"].map(lambda v: str(v).upper() if v else None) == staff[0].value]
    money_q = re.search(r"tien|doanh thu|thu ve|thu duoc", ctx.norm) is not None
    if which_model:
        if jobs or model or "nhat" not in ctx.norm.split():
            raise E.Abstain("model ranking with extra conditions")
        M = O[O["model"].notna()]
        if money_q:
            g = M.groupby("model")["revenue"].sum()
        elif re.search(r"\bluot\b|phieu", ctx.norm):
            g = M.groupby("model").size().astype(float)
        elif distinct or re.search(r"\bchiec\b|nhieu xe", ctx.norm):
            g = M.groupby("model")["plate"].nunique().astype(float)
        else:
            raise E.Abstain("model ranking by an unknown measure")
        if not len(g):
            raise E.Abstain("no tickets")
        lab = {c: biz.bike_label[c] for c in g.index}
        top = sorted(g.items(), key=lambda kv: (-kv[1], fold(lab[kv[0]])))[0][0]
        return {"type": "string", "answer": lab[top]}
    if model:
        O = O[O["model"] == model]
    if jobs:
        job = next(iter(jobs))
        O = O[O["jobset"].map(lambda js: job in js)]
    if which_bike:
        if "nhat" not in ctx.norm.split():
            raise E.Abstain("bike question not modelled")
        paid = O[(O["revenue"] > 0) & (~O["warranty"]) & O["plate"].notna()]
        g = paid.groupby("plate")["revenue"].sum() if money_q else paid.groupby("plate").size().astype(float)
        if not len(g):
            raise E.Abstain("no tickets")
        top = sorted(g.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return {"type": "string", "answer": str(top).upper()}
    if not re.search(r"bao nhieu|\bmay\b|\bdem\b|\bso\b|tong", ctx.norm):
        raise E.Abstain("job / model question not modelled")
    if distinct:
        return {"type": "number", "answer": int(O["plate"].nunique())}
    if money_q:
        if jobs:
            raise E.Abstain("money of one job is not recorded")
        return {"type": "number", "answer": E.money(O["revenue"].sum())}
    if re.search(r"\bluot\b|phieu|\blan\b", ctx.norm):
        return {"type": "number", "answer": int(len(O))}
    raise E.Abstain("job / model question not modelled")


def repair_pretext(text: str) -> str:
    """Map the trade's words onto the generic vocabulary before tagging."""
    text = re.sub(r"\b(?:cac )?phieu sua xe\b|\b(?:cac )?phieu sua\b|\bluot xe\b|\b(?:cac )?phieu\b|\bluot\b", " don ", text)
    text = re.sub(r"\btho nao\b", " nhan vien nao ", text)
    text = re.sub(r"\btho\b", " nhan vien ", text)
    text = re.sub(r"\b(sua xe|sua|man|lam|nhan(?! vien))\b", " ban ", text)
    return re.sub(r"\s+", " ", text).strip()


def repair_intent(ctx, f, E):
    """Trade-specific questions. Returns None to fall through to the generic rules."""
    biz, T, P, text = ctx.biz, ctx.toks, ctx.periods, ctx.text
    O_all = biz.orders
    if re.search(r"dong xe|chiec xe|\bxe (?:wave|vision|sh|air|lead|future|vario|ex|sirius|winner|dream|ab)\b|bien so|nhot|\blap\b|\bva\b|ruot|"
                 r"bugi|bu gi|sua dien|lam dien|\bnoi\b|bao duong|rua xe|loc gio|\bsen\b|bong den|ac quy|phanh|thay dau|thay nhot", ctx.raw):
        raise E.Abstain("repair job / bike model questions are not modelled")
    if len(P) != 1:
        return None if not re.search(r"bao hanh|hoa hong|hue hong|khach quen|xe quen", text) else E_abstain(E)
    p = P[0]
    a, e = ctx.clip(p)
    O = O_all[(O_all["date"] >= a) & (O_all["date"] <= e)]
    if f.staff:
        O = O[O["staff"].map(lambda v: str(v).upper() if v else None) == f.staff]
    paid = O[(O["revenue"] > 0) & (~O["warranty"])]
    if re.search(r"hoa hong|hue hong", text):
        if not f.staff or ctx.nums or p.grain != "month" and p.kind != "mtd":
            raise E.Abstain("commission needs one mechanic and one month")
        name = next((n for n in biz.commission if n.upper() == f.staff), None)
        if name is None:
            raise E.Abstain("no commission rate")
        return {"type": "number", "answer": E.money(O["revenue"].sum() * biz.commission[name] / 100.0)}
    if re.search(r"khach quen|xe quen", text):
        if f.any() or ctx.nums and not re.search(r"2 lan|hai lan", ctx.norm):
            raise E.Abstain("regular bikes with extra conditions")
        n = paid[paid["plate"].notna()].groupby("plate").size()
        return {"type": "number", "answer": int((n >= 2).sum())}
    if re.search(r"bao hanh", text):
        w = int(O["warranty"].sum())
        if re.search(r"ti le|ty le|%", text):
            if not len(paid) or f.any():
                raise E.Abstain("warranty rate undefined")
            return {"type": "percent", "answer": E.pct(100.0 * w / len(paid))}
        if re.search(r"khong tinh|khong ke|tru|ngoai", text):
            if "AVG" in T:
                return {"type": "number", "answer": E.money(paid["revenue"].sum() / len(paid))} if len(paid) else None
            raise E.Abstain("warranty exclusion not modelled")
        if "ORDER" in T and E.only(T, "ORDER Q COUNT TOTAL SOLD"):
            return {"type": "number", "answer": w}
        raise E.Abstain("warranty question not modelled")
    if "AVG" in T and ("PERORD" in T or "ORDER" in T):
        # average money per ticket that was charged (warranty tickets excluded)
        if not E.only(T, "AVG PERORD VALUE REV Q SOLD MONEY TOTAL ORDER") or ctx.nums:
            raise E.Abstain("average with extra conditions")
        if not len(paid):
            raise E.Abstain("no paid tickets")
        return {"type": "number", "answer": E.money(paid["revenue"].sum() / len(paid))}
    if "STAFFN" in T and "MAX" in T and "ORDER" in T and not (T & {"REV", "MONEY"}) and E.only(T, "STAFFN WHO MAX ORDER SOLD Q"):
        g = O[O["staff"].notna()].groupby("staff").size().astype(float)
        return {"type": "string", "answer": E._rank(g, 1, str)[0]}
    if "CANCEL" in T or "CUSTN" in T or "WHO" in T and "STAFFN" not in T:
        raise E.Abstain("not applicable to a repair book")
    return None


def E_abstain(E):
    raise E.Abstain("needs exactly one period")


def repair_special(ctx, E):
    """Warranty, commission, regular bikes and 'average per charged ticket' (checked before the generic rules)."""
    biz, P, text = ctx.biz, ctx.periods, ctx.text
    jm = jobs_models(ctx, E)
    if jm is not None:
        return jm
    kind = None
    if re.search(r"hoa hong|hue hong", text):
        kind = "commission"
    elif re.search(r"khach quen|xe quen|(?:vo|vao|toi|den|ghe)(?: tiem)? (?:sua |ban )?(?:tu )?(?:zq[nk]\d+ )?(?:lan )?tro len", text) and re.search(r"\bxe\b", text):
        kind = "regular"
    elif re.search(r"bao hanh", text):
        kind = "warranty"
    elif re.search(r"thu tien|co thu|tinh tien", text) and re.search(r"trung binh|binh quan", text):
        kind = "avg_paid"
    if kind is None:
        return None
    staff = [e for e in ctx.ents if e.kind == "staff"]
    if len(P) != 1 or len(staff) != len(ctx.ents) or len(staff) > 1:
        raise E.Abstain("repair question with extra conditions")
    p = P[0]
    a, e = ctx.clip(p)
    O = biz.orders[(biz.orders["date"] >= a) & (biz.orders["date"] <= e)]
    paid_all = O[(O["revenue"] > 0) & (~O["warranty"])]
    if kind == "commission":
        if p.grain != "month" and p.kind != "mtd":
            raise E.Abstain("commission is defined per month")
        O2 = O[O["staff"].notna()]
        com = {n: O2[O2["staff"] == n]["revenue"].sum() * r / 100.0 for n, r in biz.commission.items()}
        if not com:
            raise E.Abstain("no commission rates")
        if staff:
            name = next((n for n in com if n.upper() == staff[0].value), None)
            if name is None:
                raise E.Abstain("mechanic without commission rate")
            return {"type": "number", "answer": E.money(com[name])}
        if re.search(r"\bbang\b|tung|moi tho|cac tho|moi nhan vien|tung nhan vien|xep", text):
            rows = [[n, E.money(v)] for n, v in sorted(com.items(), key=lambda kv: -kv[1])]
            return {"type": "table", "answer": {"columns": ["tho", "hoa_hong"], "rows": rows}}
        if re.search(r"nhat", text):
            g = pd.Series(com)
            return {"type": "string", "answer": E._rank(g, 1, str)[0]}
        raise E.Abstain("commission question not modelled")
    if staff:
        O = O[O["staff"].map(lambda v: str(v).upper() if v else None) == staff[0].value]
    paid = O[(O["revenue"] > 0) & (~O["warranty"])]
    if kind == "regular":
        if staff:
            raise E.Abstain("regular bikes per mechanic not modelled")
        nums = [v for (_op, v) in ctx.nums]
        if nums and nums != [2.0] and nums != [2]:
            raise E.Abstain("threshold other than 2 visits")
        n = paid[paid["plate"].notna()].groupby("plate").size()
        return {"type": "number", "answer": int((n >= 2).sum())}
    if kind == "avg_paid" or (kind == "warranty" and re.search(r"trung binh|binh quan", text)):
        if not len(paid):
            raise E.Abstain("no charged tickets")
        return {"type": "number", "answer": E.money(paid["revenue"].sum() / len(paid))}
    # warranty counts
    w = int(O["warranty"].sum())
    if re.search(r"ti le|ty le|%", text):
        if staff or not len(paid_all):
            raise E.Abstain("warranty rate undefined")
        return {"type": "percent", "answer": E.pct(100.0 * w / len(paid_all))}
    if re.search(r"khong tinh|khong ke|\btru\b|ngoai", text):
        raise E.Abstain("warranty exclusion not modelled")
    if re.search(r"bao nhieu|\bmay\b|dem|nhieu", text):
        return {"type": "number", "answer": w}
    raise E.Abstain("warranty question not modelled")
