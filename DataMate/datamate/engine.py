"""Deterministic question answering over the canonical sales model.

Pipeline: normalise -> entities -> periods -> numbers -> concept tags -> intent rules.
Anything that is not fully understood is answered with ``abstain`` (score 0) rather than guessed.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from .model import Biz
from .question import Period, extract_periods, normalize, strip_spans
from .textnorm import fold, is_blank

ABSTAIN = {"type": "abstain"}


class Abstain(Exception):
    pass


class Unanswerable(Exception):
    pass


def _B(loc: dict):
    if "self" in loc:
        return loc["self"].biz
    if "biz" in loc:
        return loc["biz"]
    return loc["ctx"].biz


def missing(biz, msg: str):
    """A needed attribute is absent: 'unanswerable' only when the schema is documented, else abstain."""
    if getattr(biz, "schema_confident", False):
        return Unanswerable(msg)
    return Abstain("missing (schema not documented): " + msg)


def half_up(x: float, nd: int = 0):
    q = Decimal(1).scaleb(-nd)
    v = Decimal(repr(round(float(x), 8))).quantize(q, rounding=ROUND_HALF_UP)
    return int(v) if nd == 0 else float(v)


def money(x: float) -> int:
    return half_up(x, 0)


def pct(x: float) -> float:
    return half_up(x, 2)


def num(x: float):
    v = half_up(x, 2)
    return int(v) if float(v).is_integer() else v


# ------------------------------------------------------------------ lexicons
def toks(s: str) -> list:
    return re.findall(r"[a-z0-9%]+", s)


CHANNEL_CLASSES = [
    ("zalo", r"zalo"), ("phone", r"dien thoai|phone|hotline|goi dien"), ("store", r"tai quay|tai cua hang|tai quan|tai tiem|tai cho|truc tiep|store|cua hang|quay|tai shop"),
    ("web", r"website|web\b|online"), ("fb", r"facebook|\bfb\b|fanpage"), ("market", r"san tmdt|san thuong mai|shopee|lazada|tiki|tiktok"),
    ("app", r"grab|shopeefood|befood|gofood|app\b|ung dung"),
]
PAY_CLASSES = [
    ("cod", r"\bcod\b|thu ho|xi o di"), ("ewallet", r"vi dien tu|ewallet|e-wallet|momo|zalopay|vnpay|\bvi\b"),
    ("transfer", r"chuyen khoan|transfer|bank|ngan hang"), ("cash", r"tien mat|cash|tien tuoi"),
    ("debt", r"ghi no|\bno\b|debt|cong no|mua chiu|ban chiu|tra sau"), ("card", r"\bthe\b|card|quet the"),
]


def classify(value, classes) -> str | None:
    f = fold(value).replace("_", " ")
    for name, rx in classes:
        if re.search(rx, f):
            return name
    return None


def build_lexicon(biz: Biz) -> None:
    """Token-sequence lexicons for entity matching (cached on the Biz object)."""
    if getattr(biz, "_lex", None) is not None:
        return
    lex = {"prod": {}, "group": {}, "cgroup": {}, "unit": set(), "cust": {}}
    L, O = biz.lines, biz.orders
    for spelling, key in getattr(biz, "spellings", {}).items():
        t = tuple(toks(normalize(spelling)))
        if t:
            lex["prod"].setdefault(t, key)
    for key, info in getattr(biz, "prod", {}).items():
        t = tuple(toks(normalize(info["name"])))
        if t:
            lex["prod"].setdefault(t, key)
        if info.get("group"):
            g = tuple(toks(normalize(info["group"])))
            if g:
                lex["group"].setdefault(g, info["group"])
        if info.get("base_unit"):
            lex["unit"].add(fold(info["base_unit"]))
    for (k, u) in getattr(biz, "conv", {}):
        lex["unit"].add(u)
    if L is not None and "unit_f" in L:
        lex["unit"].update(u for u in L["unit_f"].unique() if u)
    if O is not None and "cust_group" in O:
        for g in O["cust_group"].dropna().unique():
            t = tuple(toks(normalize(str(g))))
            if t:
                lex["cgroup"].setdefault(t, g)
    if O is not None and "cust_label" in O:
        lab = O[["cust", "cust_label"]].dropna().drop_duplicates()
        for k, nm in zip(lab["cust"], lab["cust_label"]):
            nm2 = re.split(r"\s+-\s+\d", str(nm))[0]
            t = tuple(toks(normalize(nm2)))
            if len(t) >= 2:
                lex["cust"].setdefault(t, set()).add(k)
    for kind in ("prod", "group", "cgroup", "cust"):
        idx = {}
        for t in lex[kind]:
            idx.setdefault(t[0], []).append(t)
        for v in idx.values():
            v.sort(key=len, reverse=True)
        lex[kind + "_idx"] = idx
    biz._lex = lex


@dataclass
class Ent:
    kind: str
    value: object
    text: str = ""


def _match_seq(words: list, i: int, idx: dict):
    for cand in idx.get(words[i], []):
        if tuple(words[i:i + len(cand)]) == cand:
            return cand
    return None


def extract_entities(biz: Biz, s: str):
    """Replace entity mentions by placeholders 'zqeN'. Returns (text, [Ent])."""
    build_lexicon(biz)
    lex = biz._lex
    ents: list = []

    def put(kind, value, text):
        ents.append(Ent(kind, value, text))
        return f" zqe{len(ents) - 1} "

    # generic attribute phrases (independent of the shop's vocabulary)
    staff_lex = dict(getattr(biz, "staff_lex", {}) or {})
    for nm in getattr(biz, "staff_names", []) or []:
        staff_lex.setdefault(nm, nm)
    for nm, val in sorted(staff_lex.items(), key=lambda kv: -len(kv[0])):
        tk = toks(normalize(nm))
        if len(tk) < 2:
            continue
        pat = r"\b(?:(?:nhan vien|nguoi ban|tho|ban)(?: ban hang)? )?" + r"\s+".join(re.escape(t) for t in tk) + r"\b"
        s = re.sub(pat, lambda m, val=val: put("staff", str(val).upper(), m.group(0)), s)
    s = re.sub(r"\b(?:(?:nhan vien|nguoi ban)(?: co)?(?: ma)?(?: so)? )?(nv\s?\d{2,4})\b",
               lambda m: put("staff", m.group(1).replace(" ", "").upper(), m.group(0)), s)
    s = re.sub(r"\bca (sang|chieu)\b", lambda m: put("shift", m.group(1), m.group(0)), s)
    s = re.sub(r"\bsize (xxl|xl|nb|m|l|s)\b", lambda m: put("size", m.group(1), m.group(0)), s)
    s = re.sub(r"\b(?:nhan vien|nguoi ban)(?: co)?(?: ma)?(?: so)? (nv\s?\d+)\b", lambda m: put("staff", m.group(1).replace(" ", "").upper(), m.group(0)), s)

    def chan(m):
        c = classify(m.group(2), CHANNEL_CLASSES)
        return put("channel", c, m.group(0)) if c else m.group(0)

    _CH = r"(zalo oa|zalo|dien thoai|website|web|facebook|fb|shopee|san thuong mai dien tu|san thuong mai|san tmdt|quay|cua hang|tiem|cho|grab|app)"
    s = re.sub(r"\b((?:cac )?(?:don )?(?:hang )?(?:ban|dat|mua) (?:hang )?(?:qua|tren|tai|o) )" + _CH + r"\b", chan, s)
    s = re.sub(r"\b((?:cac )?(?:don )?(?:hang )?(?:qua|tren) )(zalo oa|zalo|dien thoai|website|web|facebook|fb|shopee|san tmdt|grab|app)\b", chan, s)
    s = re.sub(r"\b((?:cac )?(?:don )?(?:hang )?tai )(quay)\b", chan, s)

    def pay(m):
        c = classify(m.group(2), PAY_CLASSES)
        return put("pay", c, m.group(0)) if c else m.group(0)

    s = re.sub(r"\b((?:cac )?(?:don )?(?:hang )?(?:khach )?(?:tra|thanh toan|tt)(?: tien)? bang |bang )(tien mat|chuyen khoan|vi dien tu|cod|the|momo)\b", pay, s)
    s = re.sub(r"\b((?:cac )?don (?:hang )?(?:tra |thanh toan )?(?:tien )?(?:bang )?)(tien mat|tien tuoi|chuyen khoan|vi dien tu|cod|xi o di)\b", pay, s)
    s = re.sub(r"\b((?:cac )?don )(cod|ghi no|mua chiu|ban chiu)\b", pay, s)

    s = re.sub(r"\b(?:tru |khong tinh |ngoai |bo )(?:cac )?(?:thuoc )?(?:phai )?(?:ke don|ban theo don)(?: ra)?,? (?:thi )?(?:cac )?(?:mat hang|hang|san pham) con lai\b|"
               r"\b(?:cac )?(?:mat hang |hang |thuoc |san pham )?(?:khong (?:can |phai )?ke don|khong can don(?: bac si)?|ngoai ke don)\b",
               lambda m: put("rx", False, m.group(0)), s)
    s = re.sub(r"\b(?:cac )?(?:thuoc|hang|mat hang|san pham) (?:phai |can |ban theo |thuoc dien )?(?:ke don|ke toa|co don bac si)\b",
               lambda m: put("rx", True, m.group(0)), s)
    ing = {}
    for info in getattr(biz, "prod", {}).values():
        if info.get("ingredient"):
            ing.setdefault(" ".join(toks(normalize(info["ingredient"]))), info["ingredient"])
    if ing:
        def ingr(m):
            rest = " ".join(toks(m.group(2)))
            for cand in sorted(ing, key=len, reverse=True):
                if rest == cand or rest.startswith(cand + " "):
                    tail = rest[len(cand):].strip()
                    return put("ingredient", ing[cand], cand) + " " + tail
            return m.group(0)
        s = re.sub(r"\b((?:cac )?(?:thuoc |hang |san pham )?(?:co |chua )?hoat chat(?: la)? )([a-z0-9 +%]+)", ingr, s)

    words = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[a-z0-9%]+", s)]
    ws = [w for w, _, _ in words]
    spans = []  # (start_char, end_char, replacement)
    i = 0
    while i < len(ws):
        w = ws[i]
        if w.startswith("zqe"):
            i += 1
            continue
        # customer group: 'khach (thuoc) nhom X' / 'nhom khach X'
        if w == "nhom" and i + 1 < len(ws):
            j = i + 1
            if ws[j] in ("hang", "khach") and j + 1 < len(ws):
                j2 = j + 1
            else:
                j2 = j
            prev = ws[max(0, i - 3):i]
            is_cust_ctx = "khach" in prev or ws[j] == "khach"
            cg = _match_seq(ws, j2, lex["cgroup_idx"]) if j2 < len(ws) else None
            pg = _match_seq(ws, j2, lex["group_idx"]) if j2 < len(ws) else None
            if cg and (is_cust_ctx or not pg):
                spans.append((words[i][1], words[j2 + len(cg) - 1][2], put("cgroup", lex["cgroup"][cg], " ".join(cg))))
                i = j2 + len(cg)
                continue
            if pg and not is_cust_ctx:
                spans.append((words[i][1], words[j2 + len(pg) - 1][2], put("group", lex["group"][pg], " ".join(pg))))
                i = j2 + len(pg)
                continue
        pm = _match_seq(ws, i, lex["prod_idx"])
        if pm and (len(pm) >= 2 or len(pm[0]) >= 5):
            spans.append((words[i][1], words[i + len(pm) - 1][2], put("prod", lex["prod"][pm], " ".join(pm))))
            i += len(pm)
            continue
        cm3 = _match_seq(ws, i, lex["cust_idx"])
        if cm3 and len(cm3) >= 3:
            # a full customer name (3+ words) is recognised even without the word 'khách'
            spans.append((words[i][1], words[i + len(cm3) - 1][2], put("cust", lex["cust"][cm3], " ".join(cm3))))
            i += len(cm3)
            continue
        if w in ("khach", "anh", "chi", "co", "chu", "ong", "ba") and i + 2 < len(ws):
            j = i + 2 if (w == "khach" and ws[i + 1] in ("hang", "ten")) else i + 1
            cm = _match_seq(ws, j, lex["cust_idx"]) if j < len(ws) else None
            if cm:
                spans.append((words[i][1], words[j + len(cm) - 1][2], put("cust", lex["cust"][cm], " ".join(cm))))
                i = j + len(cm)
                continue
        i += 1
    out, last = [], 0
    for a, b, rep in spans:
        out.append(s[last:a])
        out.append(rep)
        last = b
    out.append(s[last:])
    res = re.sub(r"\s+", " ", "".join(out)).strip()

    def invert(m):
        i = int(m.group(1))
        if i < len(ents) and ents[i].kind == "rx" and ents[i].value is True:
            ents[i].value = False
            return f" zqe{i} "
        return m.group(0)

    res2 = re.sub(r"\b(?:tru|khong tinh|khong ke|ngoai|bo|loai)(?: di| het)? zqe(\d+)(?: ra| di)?\b,?", invert, res)
    if res2 != res:
        res = re.sub(r"\b(?:thi )?(?:cac |nhung )?(?:mat hang|hang|san pham|thu|mon) con lai\b|\bphan con lai\b", " ", res2)
    return re.sub(r"\s+", " ", res).strip(), ents


# ordered phrase -> concept token
PHRASES = [
    (r"\b(?:cua |o |tai )?(?:cua hang|nha thuoc|cua tiem|tiem thuoc|quan an|quan nuoc|nha hang)(?: minh| ta| nha minh)?\b", "STORE"),
    (r"doanh thu (?:cua |tu )?(?:cac |nhung )?don(?: hang)?\b", "REV"),
    (r"chi tieu|chi nhieu tien|mua nhieu tien|tong tien mua hang|tong tien mua|mua hang nhieu tien|chi tien", "SPEND"),
    (r"khach hang nao|khach nao|nguoi nao", "WHO"),
    (r"khach hang|nguoi mua", "CUSTN"),
    (r"da mua hang|co mua hang|mua hang(?! nhieu tien)", "SOLD"),
    (r"(?:tinh|cong|ke|gom|gop)(?: ca| het| tat ca)? (?:moi |cac |tat ca cac )?(?:loai|mon|mat hang|thu|do uong)(?: do uong| uong| nuoc)?|"
     r"cong tat ca cac mon|tinh tat ca cac mon|(?:cong|gop|tinh) het|tat ca (?:cac )?(?:mat hang|mon|do uong)", "TOTAL"),
    (r"nhieu lan nhat|nhieu luot nhat", "MANYORD"),
    (r"(?<!doanh )\b(?:theo )?(?:dung )?thu tu(?: thoi gian| do)?(?! dau| ngay| thang| khach| cac| don| nhom)|xep tu (?:thang )?cu (?:toi|den) (?:thang )?(?:moi|gan)(?: nhat)?|tung kenh|moi kenh|tung loai|tung ben", "EACH"),
    (r"co mat(?! hang)|xuat hien|duoc mua|mat o|nam trong|duoc goi|duoc dat|gop mat|co trong", "APPEAR"),
    (r"khong ban (?:ra )?duoc (?:cai|mon|don vi|lan|san pham) nao|khong ai mua(?: cai nao)?|khong co ai mua|(?<=hang )e\b(?: han| am)?|bi e|ban khong duoc cai nao|"
     r"khong ban duoc lan nao|khong ban ra duoc cai nao|ban cham den muc|ban cham|den muc", "UNSOLD"),
    (r"lien nhau tinh (?:toi|den)|lien tiep tinh (?:toi|den)|lien nhau ket thuc o|lien tiep ket thuc o|lien nhau ket thuc vao|lien tiep den|lien nhau den|lien tiep|lien nhau|\blien\b(?! truoc| tuc| ke)", "CONSEC"),
    (r"so voi cung ky|so cung ky|voi cung ky|cung ky|(?:so voi |so |voi )?cung so ngay(?: dau)?(?: cua)?|(?:so voi |so |voi )?cung thang(?= <P>)", "SAMEP"),
    (r"thang lien truoc|thang truoc do|thang ngay truoc do|thang ngay truoc|thang ke truoc|thang lien ke truoc", "PREVM"),
    (r"tang lien tuc|lien tuc tang|tang deu|tang dan", "INCR"),
    (r"thang sau (?:co )?(?:luon )?(?:cao hon|lon hon|hon|nhieu hon) thang truoc", "INCR"),
    (r"tang hay giam|tang/giam|tang giam|hon hay kem|cao hay thap hon|cao hon hay thap hon|len hay xuong|thay doi|tang truong|bien dong|"
     r"tang len hay giam xuong|chenh lech|chenh nhau|chenh|len xuong|len hay giam|tang hay xuong|nhieu hay it hon|it hay nhieu hon|"
     r"nhieu hon hay it hon|hon kem|cao hay thap|\bhon\b(?= zqe\d| <P>)", "CHANGE"),
    (r"trung binh|binh quan", "AVG"),
    (r"tren mot don|tren moi don|moi don|mot don|1 don|tung don", "PERORD"),
    (r"gia tri", "VALUE"),
    (r"lai rong|loi nhuan rong|lai thuan|loinhuan rong|loinhuan thuan", "NETPROFIT"),
    (r"lai gop|loi gop|loi nhuan gop|loi nhuan|tien lai|tien loi|\bloi\b|loinhuan gop|loinhuan", "PROFIT"),
    (r"nha cung cap nao|nha cung cap", "SUPPLIER"),
    (r"phieu nhap hang|phieu nhap|nhap hang|hang nhap|nhap vao|nhap ve|\bnhap\b", "IMPORT"),
    (r"gia ban|don gia|ban gia|voi gia|\bgia\b", "PRICE"),
    (r"tong doanh thu ca (?:cua hang|quan|tiem|shop)|doanh thu ca (?:cua hang|quan|tiem|shop)|tong doanh thu", "REVALL"),
    (r"doanh thu|tien ban hang|tien hang|tien ban\b", "REV"),
    (r"ngay nao|hom nao", "WHICHDAY"),
    (r"thang nao", "WHICHMONTH"),
    (r"nguoi ban|nhan vien ban hang|nhan vien ban|nhan vien nao|nhan vien", "STAFFN"),
    (r"khach nao|khach hang nao|\bai\b|nguoi nao", "WHO"),
    (r"cao nhat|nhieu nhat|lon nhat|dat nhat|dat khach nhat|ban gioi nhat|gioi nhat", "MAX"),
    (r"o trang thai huy|bi huy|da huy|huy don|\bhuy\b", "CANCEL"),
    (r"ti le|ty le|ti trong|ty trong", "RATE"),
    (r"%", "PCT"),
    (r"chiem|dong gop|phan cua|\bbang\b(?= «Q| bao| may)", "SHARE"),
    (r"ban chay|chay nhat|\bhot\b|best seller|bestseller", "BEST"),
    (r"liet ke|\bke\b|\btop\b|cho bang", "LIST"),
    (r"moi loai|moi ben|moi thang|tung thang|lan luot|theo thu tu(?: thoi gian| do)?|tu thang cu den thang moi|ghi \d+ so", "EACH"),
    (r"hang nao|mon nao|mat hang nao|san pham nao|hang gi|mon gi|san pham gi|mat hang gi", "WHICHPROD"),
    (r"mat hang|san pham|\bmon\b|\bhang\b", "PRODN"),
    (r"khach hang|\bkhach\b|nguoi mua", "CUSTN"),
    (r"chi tieu|chi nhieu tien|mua nhieu tien|tong tien mua|mua hang nhieu tien|chi tien", "SPEND"),
    (r"bao nhieu|\bmay\b|\bnhieu\b", "Q"),
    (r"tong cong|\btong\b|tat ca|cong tat ca|cong moi|tinh moi", "TOTAL"),
    (r"ban duoc|ban ra|thu ve|thu duoc|mang ve|kiem duoc|dem ve|dem de|dem lai|mang lai|\bban\b|\bmua\b", "SOLD"),
    (r"\btien\b", "MONEY"),
    (r"so don|\bdon\b", "ORDER"),
    (r"dem so|\bdem\b|so luong", "COUNT"),
    (r"\bngay\b", "DAYN"),
    (r"\bnhat\b", "MAX"),
    (r"\bkhong\b", "NEG"),
]
FILLERS = set("""cho hoi minh toi biet xem giup check coi e nhi vay ha ta z do a di trong cua hang quan tiem shop nha thuoc la thi da
co vao nha nhe the tinh rieng duoc het den tu ad oi ah ak nay ben em anh chi ay voi luon ca ke gi bi dat xet theo ra so
toi hien tai dang gio nhu nao ban admin sep bao bao cao thong ke kiem tra tra dum ho cai ma ee
mat ua xiu toan hay mun muon dung thanks tks cam on chu dum vui long tung cong
ghi giai doan van han mot bo ve giua sanh he nghe nghen hen
dong vnd day nua lai va roi thoi de nhung cac j chi ne nho o qua thuoc loai uong do thang ba hai bon nam qua tren khac nhau nguoi tro len lan xu huong phai""".split())


def tag(text: str, units: set):
    t = " " + text + " "
    for rx, tok in PHRASES:
        t = re.sub(rx, f" «{tok}» ", t)
    seq, unknown = [], []
    prev = ""
    for w in re.split(r"[\s,.:()→/\-]+", t):
        if not w:
            continue
        w0, last = w, prev
        prev = w
        if w0 == "con" and "con" in units and last not in ("«Q»", "theo", "so", "moi", "mot", "tung", "«COUNT»"):
            unknown.append(w0)
            continue
        if w.startswith("«"):
            if w.strip("«»") != "STORE":
                seq.append(w.strip("«»"))
        elif w == "zqchain":
            continue
        elif w.startswith("<") or w.startswith("zqe") or w.startswith("zqn") or w.startswith("zqk"):
            seq.append(w)
        elif w in units or w.replace("_", " ") in units:
            seq.append("U:" + w.replace("_", " "))
        elif w in FILLERS:
            continue
        else:
            unknown.append(w)
    return seq, unknown


_CMP_WORDS = [
    (r"(?:tu|dat|dat tu|dat muc|toi thieu|it nhat|>=)\s*(\d+)(?: dong| don| ngay)?(?: tro len| tro di)?", "ge"),
    (r"(\d+)(?: dong| don| dot| lan)?(?: khac nhau)? tro len", "ge"),
    (r"(?:tren|hon|qua|vuot|(?<!P)>)\s*(\d+)\b(?! thang| ngay| tuan| nam| quy| <P>)", "gt"),
    (r"(?:duoi|chua den|chua toi|<(?!P))\s*(\d+)", "lt"),
    (r"(?:toi da|khong qua|<=)\s*(\d+)", "le"),
]


def extract_numbers(s: str):
    """Thresholds -> zqnN (op, value); list sizes -> zqkN."""
    nums: list = []

    def thr(op):
        def f(m):
            nums.append((op, float(m.group(1))))
            return f" zqn{len(nums) - 1} "
        return f

    for rx, op in _CMP_WORDS:
        s = re.sub(rx, thr(op), s)

    def k(m):
        nums.append(("k", int(m.group(1))))
        return f" zqk{len(nums) - 1} "

    s = re.sub(r"(?<![\w.,/])(\d{1,2})(?![\w./]|,\d)", k, s)
    return re.sub(r"\s+", " ", s).strip(), nums


@dataclass
class Filt:
    shift: str | None = None
    prod: object = None
    group: str | None = None
    cgroup: str | None = None
    channel: str | None = None
    pay: str | None = None
    size: str | None = None
    staff: str | None = None
    cust: object = None
    rx: object = None
    ingredient: str | None = None

    def line_level(self) -> bool:
        return bool(self.prod or self.group or self.size or self.rx is not None or self.ingredient)

    def any(self) -> bool:
        return any(v is not None for v in self.__dict__.values())


class Ctx:
    def __init__(self, biz: Biz, question: str, today: dt.date, cal=None, norm_override=None):
        self.biz, self.today, self.cal = biz, today, cal
        self.norm = norm_override or normalize(question)
        self.raw = fold(question)
        self.low = str(question).lower()
        text, self.ents = extract_entities(biz, self.norm)
        if getattr(biz, "phone", False):
            from .phone import phone_pretext
            text = phone_pretext(text)
        if getattr(biz, "repair", False):
            from .repair import repair_pretext
            text = repair_pretext(text)
            text = re.sub(r"\b(?:nhan vien|ban) (zqe\d+)", r"\1", text) if self.ents else text
        text = re.sub(r"thang sau (?:co )?(?:deu |luon |luon luon )?(?:cao hon|lon hon|hon|nhieu hon|vuot) thang truoc", " tang lien tuc ", text)
        text = re.sub(r"\b(trung binh|binh quan)(?: tren| cua)? (?:1|mot|moi) don\b", r"\1 moi don", text)
        text = re.sub(r"\btren (?:1|mot) don\b", "tren moi don", text)
        text = re.sub(r"\b(?:co )?it nhat (?:mot|1) (?=mat hang|san pham|mon\b|hang\b)", " ", text)
        text = re.sub(r"\b(?:1|mot) don (?=.*\b(?:trung binh|binh quan)\b)", "moi don ", text)
        allp = extract_periods(text, today, cal)
        keep, dup = [], []
        for pr in allp:
            between = text[keep[-1].span[1]:pr.span[0]] if keep else ""
            rel = pr.text in ("thang truoc", "nam truoc") or (keep and keep[-1].text in ("thang truoc", "nam truoc"))
            cmpq = re.search(r"\bso\b|hon hay kem|tang|giam|thay doi|chenh", text) is not None
            if keep and (keep[-1].start, keep[-1].end) == (pr.start, pr.end) and not re.search(r"\b(so|voi|hon|kem|thap|cao|va)\b", between) \
                    and not (rel and cmpq):
                dup.append(pr)
            else:
                keep.append(pr)
        if dup:
            text2, last = [], 0
            for pr in allp:
                text2.append(text[last:pr.span[0]])
                text2.append(" " if pr in dup else " <P> ")
                last = pr.span[1]
            text2.append(text[last:])
            text = re.sub(r"\s+", " ", "".join(text2)).strip()
        else:
            text = strip_spans(text, allp, " <P> ")
        mc = re.search(r"<P> (?:cao|lon|nhieu) hon <P>[ ,;]*(?:va|roi|con|thi)?[ ,;]*<P> (?:cung |lai |cung lai |lai cung )?(?:cao|lon|nhieu) hon <P>", text)
        self.chain = None
        if mc and len(keep) >= 4:
            i0 = text[:mc.start()].count("<P>")
            self.chain = [(keep[i0], keep[i0 + 1]), (keep[i0 + 2], keep[i0 + 3])]
            text = re.sub(r"<P>", " ", text[:mc.start()]) + " zqchain " + text[mc.end():]
            keep = [keep[i0], keep[i0 + 1], keep[i0 + 2], keep[i0 + 3]][:0] + [p for j, p in enumerate(keep) if j >= i0 + 4]
        m = re.search(r"\btu <P> (?:den|toi)(?: het)? <P>|<P> (?:den|toi)(?: het)? <P>", text)
        if m and len(keep) >= 2:
            i = text[:m.start()].count("<P>")
            a, b = keep[i], keep[i + 1]
            if a.start <= b.end:
                keep[i:i + 2] = [Period(a.start, b.end, "range")]
                text = text[:m.start()] + " <P> " + text[m.end():]
                text = re.sub(r"\s+", " ", text).strip()
        self.periods = keep
        text, self.nums = extract_numbers(text)
        order = [int(m) for m in re.findall(r"zqe(\d+)", text)]
        if sorted(order) == list(range(len(self.ents))) and order != sorted(order):
            # keep entities in the order they are mentioned in the question
            remap = {old: new for new, old in enumerate(order)}
            self.ents = [self.ents[i] for i in order]
            text = re.sub(r"zqe(\d+)", lambda m: f"zqe{remap[int(m.group(1))]}", text)
        self.text = text
        for u in sorted((u for u in biz._lex["unit"] if " " in u), key=len, reverse=True):
            text = re.sub(r"(?<![a-z0-9])" + re.escape(u) + r"(?![a-z0-9])", u.replace(" ", "_"), text)
        self.text = text
        self.seq, self.unknown = tag(text, biz._lex["unit"])
        self.toks = set(x for x in self.seq if re.fullmatch(r"[A-Z]+", x))
        self.units = list(dict.fromkeys(x[2:] for x in self.seq if x.startswith("U:")))

    def ent(self, kind):
        return [e for e in self.ents if e.kind == kind]

    # ---- period helpers
    def clip(self, p: Period):
        b = self.biz
        start, end = p.start, min(p.end, self.today)
        if b.data_start is None:
            raise Abstain("no data range")
        if end < b.data_start or start > b.data_end or start > end:
            raise Unanswerable("kỳ hỏi nằm ngoài khoảng ngày có dữ liệu")
        return pd.Timestamp(start), pd.Timestamp(end)

    # ---- selections
    def _order_mask(self, O: pd.DataFrame, f: Filt):
        m = pd.Series(True, index=O.index)
        b = self.biz
        if f.shift:
            if "dt" not in O or O["dt"].isna().all():
                raise missing(_B(locals()), "dữ liệu không ghi giờ đặt đơn")
            h = O["dt"].dt.hour
            m &= (h >= 6) & (h < 14) if f.shift == "sang" else (h >= 14) & (h < 22)
        if f.channel:
            if "channel" not in O:
                raise missing(_B(locals()), "dữ liệu không ghi kênh bán")
            cls = _class_series(b, O, "channel", CHANNEL_CLASSES)
            if f.channel not in set(cls.dropna().unique()):
                raise Abstain("channel value not found")
            m &= cls == f.channel
        if f.pay:
            if "pay_method" not in O:
                raise missing(_B(locals()), "dữ liệu không ghi hình thức thanh toán")
            cls = _class_series(b, O, "pay_method", PAY_CLASSES)
            if f.pay not in set(cls.dropna().unique()):
                raise Abstain("pay value not found")
            m &= cls == f.pay
        if f.cgroup:
            if "cust_group" not in O:
                raise missing(_B(locals()), "dữ liệu không ghi nhóm khách")
            m &= O["cust_group"] == f.cgroup
        if f.staff:
            if "staff" not in O:
                raise missing(_B(locals()), "dữ liệu không ghi nhân viên bán")
            st = O["staff"].map(lambda v: str(v).strip().upper() if not is_blank(v) else None)
            if f.staff not in set(st.dropna().unique()):
                raise Abstain("staff not found")
            m &= st == f.staff
        if f.cust is not None:
            m &= O["cust"].isin(f.cust)
        return m

    def orders(self, p: Period, f: Filt | None = None, valid=True) -> pd.DataFrame:
        a, e = self.clip(p)
        O = self.biz.orders
        m = (O["date"] >= a) & (O["date"] <= e)
        if valid is True:
            m &= O["st"].isin(["done", "ship"])
        O = O[m]
        if f is not None and f.any():
            O = O[self._order_mask(O, f)]
            if f.line_level():
                L = self.lines(p, f, valid)
                O = O[O["order_id"].isin(L["order_id"].unique())]
        return O

    def lines(self, p: Period, f: Filt | None = None, valid=True) -> pd.DataFrame:
        L = self.biz.lines
        if L is None:
            raise missing(_B(locals()), "dữ liệu không ghi chi tiết mặt hàng")
        if f is not None and f.line_level() and getattr(self.biz, "combo_names", False):
            raise Abstain("item names are combined or abbreviated in the sales book")
        a, e = self.clip(p)
        m = (L["date"] >= a) & (L["date"] <= e)
        if valid is True:
            m &= L["st"].isin(["done", "ship"])
        L = L[m]
        if f is None:
            return L
        if f.prod is not None:
            if not (self.biz.lines["pkey"] == f.prod).any():
                # the product is in the catalogue but never appears in the sales book (e.g. abbreviated names)
                raise Abstain("product never appears in sales lines")
            L = L[L["pkey"] == f.prod]
            if not len(L):
                # a zero for a named product is more often a naming mismatch than a real zero
                raise Abstain("no sales of that product in the period")
        if f.group:
            if "group" not in L:
                raise missing(_B(locals()), "dữ liệu không ghi nhóm hàng")
            L = L[L["group"] == f.group]
        if f.size:
            if "prod_name" not in L and "pkey" not in L:
                raise Abstain("no names")
            names = L["pkey"].map(lambda k: self.biz.prod.get(k, {}).get("name", ""))
            L = L[names.map(lambda n: fold(n).rstrip().endswith(f"({f.size})"))]
        if f.rx is not None or f.ingredient:
            prod = self.biz.prod
            if not getattr(self.biz, "drug_table", ""):
                raise Abstain("no drug list")
            if f.rx is not None:
                keys = {k for k, v in prod.items() if bool(v.get("rx")) == bool(f.rx)}
            else:
                keys = {k for k, v in prod.items() if v.get("ingredient") == f.ingredient}
            L = L[L["pkey"].isin(keys)]
        of = Filt(shift=f.shift, cgroup=f.cgroup, channel=f.channel, pay=f.pay, staff=f.staff, cust=f.cust)
        if of.any():
            O = self.biz.orders
            ids = O[self._order_mask(O, of)]["order_id"]
            L = L[L["order_id"].isin(set(ids))]
        return L

    def revenue(self, p: Period, f: Filt | None = None) -> float:
        if f is not None and f.line_level():
            return float(self.lines(p, f)["amount"].sum())
        O = self.orders(p, f)
        if len(O) and O["revenue"].isna().all():
            raise Abstain("no revenue column")
        return float(O["revenue"].sum())

    def n_orders(self, p: Period, f: Filt | None = None) -> int:
        need_orders(self)
        return int(len(self.orders(p, f)))


def _class_series(biz: Biz, O: pd.DataFrame, col: str, classes) -> pd.Series:
    cache = biz.__dict__.setdefault("_cls_cache", {})
    if col not in cache:
        vals = biz.orders[col]
        mp = {v: classify(v, classes) for v in vals.dropna().unique()}
        cache[col] = vals.map(mp)
    return cache[col].reindex(O.index)


def need_orders(ctx: Ctx):
    if not ctx.biz.has_order_id:
        if getattr(ctx.biz, "no_order_id_stated", False):
            raise Unanswerable("dữ liệu không ghi mã đơn")
        raise missing(ctx.biz, "dữ liệu không ghi mã đơn")


def only(toks: set, allowed: str) -> bool:
    return toks <= set(allowed.split())


_BASE_VOCAB = None
_CONFUSE = [("l", "n"), ("n", "l"), ("d", "gi"), ("gi", "d"), ("d", "v"), ("v", "d"), ("d", "r"), ("r", "d"), ("x", "s"), ("s", "x"),
            ("ch", "tr"), ("tr", "ch"), ("c", "k"), ("k", "c"), ("gi", "r"), ("r", "gi"), ("z", "d"), ("z", "gi"), ("j", "gi")]


def _vocab(biz: Biz):
    global _BASE_VOCAB
    if _BASE_VOCAB is None:
        content = set()
        for rx, _tok in PHRASES:
            content |= set(re.findall(r"[a-z]{2,}", re.sub(r"\(\?[:=!<]+", "(", rx)))
        content |= set("thang tuan quy nam ngay hom nay qua truoc dau den het tu toi gan nhat".split())
        _BASE_VOCAB = (content - {"zqe", "zqn", "zqk"}, set(FILLERS))
    bw = biz.__dict__.get("_vocab_biz")
    if bw is None:
        build_lexicon(biz)
        bw = set(biz._lex["unit"])
        for kind in ("prod", "group", "cgroup"):
            for t in biz._lex[kind]:
                bw |= {w for w in t if w.isalpha() and len(w) >= 2}
        biz._vocab_biz = bw
    return _BASE_VOCAB[0], _BASE_VOCAB[1], bw


def _edits1(w: str) -> set:
    letters = "abcdefghijklmnopqrstuvwxyz"
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = {a + b[1:] for a, b in sp if b}
    out |= {a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1}
    out |= {a + c + b[1:] for a, b in sp if b for c in letters}
    out |= {a + c + b for a, b in sp for c in letters}
    return out


def _candidates(u: str, content: set, fillers: set, bw: set) -> list:
    """Plausible rewrites of an unknown word, best first (at most three)."""
    known = content | fillers | bw
    out = []
    if len(u) >= 5:
        cuts = [(u[:i], u[i:]) for i in range(2, len(u) - 1) if u[:i] in known and u[i:] in known]
        out += [x + " " + y for x, y in cuts if x in content or y in content or x in bw or y in bw]
    conf = set()
    for a, b in _CONFUSE:
        if u.startswith(a):
            conf.add(b + u[len(a):])
    if u.endswith("ng"):
        conf.add(u[:-2] + "n")
    if u.endswith("n"):
        conf.add(u + "g")
    out += sorted(c for c in conf if c in content or c in bw) + sorted(c for c in conf if c in fillers)
    if len(u) >= 3:
        e1 = _edits1(u)
        out += sorted(c for c in e1 if c in content or c in bw)
        out += sorted(c for c in e1 if c in fillers and len(c) >= 3)
    seen, uniq = set(), []
    for c in out:
        if c not in seen and c != u:
            seen.add(c)
            uniq.append(c)
    return uniq[:3]


def repair_texts(ctx: "Ctx") -> list:
    """Candidate rewritten questions: glued words, consonant mix-ups (l/n, d/gi/v, x/s, ch/tr) and one-letter typos."""
    import itertools
    content, fillers, bw = _vocab(ctx.biz)
    known = content | fillers | bw
    slots = []
    for u in dict.fromkeys(ctx.unknown):
        if not u.isalpha() or len(u) < 2:
            continue
        if u in fillers or u in bw:
            continue  # a real word the grammar has no use for here: never guess
        cands = _candidates(u, content, fillers, bw)
        if u in content:
            cands = [u] + cands[:2]  # part of a known phrase ('trung' of 'trung bình'): it may be fine as typed
        if cands:
            slots.append((u, cands))
    if not slots or len(slots) > 3:
        return []
    texts = []
    for combo in itertools.islice(itertools.product(*[c for _u, c in slots]), 27):
        out = ctx.norm
        for (u, _c), fix in zip(slots, combo):
            out = re.sub(r"(?<![a-z0-9])" + re.escape(u) + r"(?![a-z0-9])", fix, out)
        if out != ctx.norm and out not in texts:
            texts.append(out)
    return texts


def _answer(biz: Biz, question: str, today: dt.date, cal, norm_override=None):
    import sys as _sys

    from . import debts as _debts
    from . import extra as _extra
    E = _sys.modules[__name__]
    ctx = None
    try:
        if biz.orders is None:
            return dict(ABSTAIN, why="no sales model"), None
        if getattr(biz, "unreliable", False):
            return dict(ABSTAIN, why="schema not understood well enough"), None
        ctx = Ctx(biz, question, today, cal, norm_override)
        if getattr(biz, "phone", False):
            from .phone import phone_intent
            ph = phone_intent(ctx, E)
            if ph is not None:
                return ph, ctx
        if getattr(biz, "extras", None) is not None or getattr(biz, "member_dates", None) is not None \
                or any(v.get("brand") or v.get("size") for v in biz.prod.values()):
            from .baby import attr_intent
            at = attr_intent(ctx, E)
            if at is not None:
                return at, ctx
        de = _debts.debt_intent(ctx, E)
        if de is not None:
            return de, ctx
        from . import lots as _lots
        sk = _lots.lots_intent(ctx, E) if hasattr(biz, "lots") else None
        if sk is None:
            sk = _extra.out_of_stock_intent(ctx, E)
        if sk is None:
            sk = _extra.stock_intent(ctx, E)
        if sk is None:
            sk = _extra.stock_value_intent(ctx, E)
        if sk is not None:
            return sk, ctx
        if getattr(biz, "repair", False):
            from .repair import repair_special
            rs = repair_special(ctx, E)
            if rs is not None:
                return rs, ctx
        ua = _extra.unanswerable_attr(ctx, E)
        if ua is None:
            ua = _extra.named_metrics(ctx, E)
        if ua is not None:
            return ua, ctx
        pe = price_effect(ctx)
        if pe is None:
            pe = _extra.import_minus_sales(ctx, E)
        if pe is None:
            pe = _extra.delivery_intent(ctx, E)
        if pe is None:
            pe = _extra.net_profit_intent(ctx, E)
        if pe is None:
            pe = _extra.due_date_intent(ctx, E)
        if pe is None:
            pe = _extra.weekday_avg_intent(ctx, E)
        if pe is not None:
            return pe, ctx
        _extra.group_missing(ctx, E)
        if ctx.unknown:
            return dict(ABSTAIN, why="unknown: " + " ".join(ctx.unknown)), ctx
        res = dispatch(ctx)
        if res is None:
            return dict(ABSTAIN, why="no intent: " + " ".join(ctx.seq)), ctx
        return res, ctx
    except Unanswerable as e:
        return {"type": "unanswerable", "reason": str(e)}, ctx
    except Abstain as e:
        return dict(ABSTAIN, why=str(e)), ctx


def answer_question(biz: Biz, question: str, as_of: str, cal=None) -> dict:
    today = dt.date.fromisoformat(as_of)
    res, ctx = _answer(biz, question, today, cal)
    # a question that was not understood is retried with its unknown words repaired; the answer is kept
    # only when every repair that yields an answer yields the same one
    if res.get("type") == "abstain" and ctx is not None and ctx.unknown:
        found = []
        for fixed in repair_texts(ctx):
            res2, _ = _answer(biz, question, today, cal, fixed)
            if res2.get("type") != "abstain":
                found.append(res2)
        if found and all(json.dumps(f, sort_keys=True, default=str) == json.dumps(found[0], sort_keys=True, default=str) for f in found):
            return found[0]
    # a chatty lead-in ('mình muốn tặng quà khách quen, ...') carries no condition: retry without it
    if res.get("type") == "abstain" and ctx is not None and ctx.unknown and "," in ctx.norm:
        head, tail = ctx.norm.split(",", 1)
        if len(tail.split()) >= 5 and not re.search(r"\d|thang|tuan|quy|nam |hom |ngay|tru |khong|nhom|ca ", head) \
                and any(u in head.split() for u in ctx.unknown):
            res3, _ = _answer(biz, question, today, cal, tail.strip())
            if res3.get("type") != "abstain":
                return res3
    return res


# ------------------------------------------------------------------ intents
def build_filter(ctx: Ctx, allow: str) -> Filt:
    f = Filt()
    allowed = set(allow.split())
    for e in ctx.ents:
        if e.kind not in allowed:
            raise Abstain(f"unexpected entity {e.kind}")
        if getattr(f, e.kind) is not None:
            raise Abstain(f"two entities of kind {e.kind}")
        setattr(f, e.kind, e.value)
    return f


def _pct_change(cur: float, base: float) -> dict:
    if base == 0:
        raise Abstain("zero base")
    return {"type": "percent", "answer": pct(100.0 * (cur - base) / base)}


def _prev_month(p: Period) -> Period:
    first = p.start.replace(day=1)
    last_prev = first - dt.timedelta(days=1)
    return Period(last_prev.replace(day=1), last_prev, "month", grain="month")


def _same_days_prev_month(p: Period, today: dt.date) -> Period:
    """Month-to-date compared with the same number of days of the previous month (rulebook §2)."""
    prev = _prev_month(p)
    day = min(p.end, today).day
    end = prev.start.replace(day=min(day, prev.end.day))
    return Period(prev.start, end, "range")


def _shift_year(p: Period, today: dt.date):
    """Same period one year earlier."""
    import calendar as _c
    try:
        start = p.start.replace(year=p.start.year - 1)
    except ValueError:
        start = p.start.replace(year=p.start.year - 1, day=28)
    if p.grain == "month":
        return Period(start, start.replace(day=_c.monthrange(start.year, start.month)[1]), "month", grain="month")
    end = min(p.end, today)
    try:
        end = end.replace(year=end.year - 1)
    except ValueError:
        end = end.replace(year=end.year - 1, day=28)
    return Period(start, end, "range")


def _measure(ctx: Ctx, T: set):
    """Which base measure the question is about: 'rev' | 'orders' | None."""
    rev = bool(T & {"REV", "REVALL"}) or ("SOLD" in T and "MONEY" in T)
    if rev and "ORDER" not in T:
        return "rev"
    if rev and "ORDER" in T and any(e.kind in ("staff", "cgroup") for e in ctx.ents) and not (T & {"AVG", "PERORD", "CANCEL", "COUNT"}):
        T.discard("ORDER")
        return "rev"
    if "ORDER" in T and not rev:
        return "orders"
    return None


def _val(ctx: Ctx, meas: str, p: Period, f: Filt) -> float:
    return ctx.revenue(p, f) if meas == "rev" else float(ctx.n_orders(p, f))


def dispatch(ctx: Ctx):
    T, P, seq = ctx.toks, ctx.periods, ctx.seq
    kinds = [e.kind for e in ctx.ents]
    ks = [v for (op, v) in ctx.nums if op == "k"]
    thr = [(op, v) for (op, v) in ctx.nums if op != "k"]
    meas = _measure(ctx, T)

    # ---------- comparisons between two shifts
    if kinds.count("shift") == 2 and len(P) == 1 and "CHANGE" in T and "PCT" in T and meas and not thr and not ks \
            and only(T, "CHANGE PCT Q REV ORDER SOLD MONEY"):
        if len(ctx.ents) != 2:
            raise Abstain("extra entities")
        base = Filt(shift="sang")
        cur = Filt(shift="chieu")
        return _pct_change(_val(ctx, meas, P[0], cur), _val(ctx, meas, P[0], base))

    # ---------- two filters, one value each ("mỗi loại bao nhiêu")
    for kind in ("pay", "channel"):
        if kinds.count(kind) == 2 and len(ctx.ents) == 2 and len(P) == 1 and meas == "rev" and not thr and not ks \
                and only(T, "REV Q EACH SOLD MONEY ORDER CMP"):
            vals = [money(ctx.revenue(P[0], Filt(**{kind: e.value}))) for e in ctx.ents]
            return {"type": "list", "answer": vals}

    f = None
    if len(set(kinds)) == len(kinds):
        f = build_filter(ctx, "shift prod group cgroup channel pay size staff cust rx ingredient")
    else:
        return None

    import sys as _sys

    from . import extra as _extra
    _E = _sys.modules[__name__]
    if "NETPROFIT" in T:
        return None
    if "PROFIT" in T:
        return _extra.profit_intent(ctx, f, ks, thr, _E)
    if "IMPORT" in T:
        return _extra.imports_intent(ctx, f, ks, thr, _E)
    if "PRICE" in T:
        return _extra.price_intent(ctx, f, ks, thr, _E)
    if getattr(ctx.biz, "repair", False):
        from .repair import repair_intent
        rr = repair_intent(ctx, f, _E)
        if rr is not None:
            return rr
    r2 = dispatch2(ctx, f, meas, ks, thr)
    if r2 is not None:
        return r2

    # ---------- growth / change in percent
    if "CHANGE" in T and "PCT" in T and meas and not thr and not ks and only(T, "CHANGE PCT Q REV REVALL ORDER SOLD MONEY SAMEP PREVM RATE"):
        cur_p, base_p = _growth_periods(ctx)
        if cur_p is None:
            return None
        return _pct_change(_val(ctx, meas, cur_p, f), _val(ctx, meas, base_p, f))

    # ---------- share of a filtered slice in total revenue
    if ("SHARE" in T or "RATE" in T or ((T & {"REV", "REVALL"}) and "CHANGE" not in T)) and "PCT" in T and len(P) == 1 and f.any() and not thr and not ks \
            and only(T, "SHARE RATE PCT Q REV REVALL SOLD MONEY ORDER CUSTN PRODN TOTAL") and "CANCEL" not in T:
        if "ORDER" in T and not (T & {"REV", "REVALL"}):
            return None
        total = ctx.revenue(P[0], None)
        if total == 0:
            raise Abstain("zero total")
        return {"type": "percent", "answer": pct(100.0 * ctx.revenue(P[0], f) / total)}

    if len(P) == 1 and not ks:
        p = P[0]
        # ----- thresholds: count orders / days above an amount
        if len(thr) == 1 and not f.any():
            op, v = thr[0]
            cmpf = {"ge": lambda s: s >= v, "gt": lambda s: s > v, "lt": lambda s: s < v, "le": lambda s: s <= v}[op]
            if (T & {"REV", "MONEY"}) and (T & {"Q", "COUNT"}) and only(T, "ORDER REV MONEY Q COUNT TOTAL") and "DAYN" not in T \
                    and ("ORDER" in T or v >= 1000):
                need_orders(ctx)
                O = ctx.orders(p)
                return {"type": "number", "answer": int(cmpf(O["revenue"]).sum())}
            if "DAYN" in T and (T & {"REV", "MONEY", "SOLD"}) and "ORDER" not in T and only(T, "DAYN REV MONEY SOLD Q COUNT TOTAL"):
                O = ctx.orders(p)
                g = O.groupby("date")["revenue"].sum()
                return {"type": "number", "answer": int(cmpf(g).sum())}
            return None
        if thr:
            return None
        # ----- average order value
        if "AVG" in T and (("PERORD" in T) or ("VALUE" in T and "ORDER" in T)) and only(T, "AVG PERORD VALUE REV Q SOLD MONEY TOTAL ORDER CUSTN"):
            if f.line_level() or f.cust is not None:
                return None
            need_orders(ctx)
            O = ctx.orders(p, f)
            if not len(O):
                raise Abstain("no orders")
            return {"type": "number", "answer": money(ctx.revenue(p, f) / len(O))}
        # ----- cancellations
        if "CANCEL" in T and only(T, "CANCEL ORDER Q COUNT TOTAL RATE PCT SHARE"):
            if f.any():
                return None
            need_orders(ctx)
            allo = ctx.orders(p, valid=None)
            n_c = int((allo["st"] == "cancel").sum())
            if T & {"RATE", "PCT", "SHARE"}:
                if not len(allo):
                    raise Abstain("no orders")
                return {"type": "percent", "answer": pct(100.0 * n_c / len(allo))}
            return {"type": "number", "answer": n_c}
        # ----- order count
        if "ORDER" in T and (only(T, "ORDER Q COUNT TOTAL SOLD") or (f.line_level() and only(T, "ORDER Q COUNT TOTAL SOLD PRODN"))):
            return {"type": "number", "answer": ctx.n_orders(p, f)}
        # ----- revenue
        if meas == "rev" and only(T, "REV REVALL SOLD MONEY Q TOTAL PRODN CUSTN") and not ctx.units:
            if "CUSTN" in T and not (f.cgroup or f.cust):
                return None
            if "PRODN" in T and not f.line_level():
                return None
            return {"type": "number", "answer": money(ctx.revenue(p, f))}
        # ----- quantity sold in a unit
        if len(ctx.units) == 1 and "SOLD" in T and only(T, "SOLD Q TOTAL COUNT PRODN") and not (T & {"REV", "MONEY"}):
            return _quantity(ctx, p, f, ctx.units[0])
        if not ctx.units and f.prod is not None and "SOLD" in T and only(T, "SOLD Q TOTAL COUNT") and not f.size:
            L = ctx.lines(p, f)
            if "qty_base" not in L or getattr(ctx.biz, "combo_names", False):
                raise Abstain("no quantity")
            return {"type": "number", "answer": num(L["qty_base"].sum())}
        # ----- best revenue day
        if "WHICHDAY" in T and "MAX" in T and meas == "rev" and only(T, "WHICHDAY MAX REV SOLD MONEY Q DAYN") and not f.any():
            O = ctx.orders(p)
            if not len(O):
                raise Abstain("no orders")
            g = O.groupby("date")["revenue"].sum()
            top = g[g == g.max()]
            if len(top) != 1:
                raise Abstain("tie")
            return {"type": "date", "answer": top.index[0].strftime("%Y-%m-%d")}
    return None


def _growth_periods(ctx: Ctx):
    """(current, base) periods for a percentage change question."""
    P, seq, T = ctx.periods, ctx.seq, ctx.toks
    idx = [i for i, x in enumerate(seq) if x == "<P>"]
    if len(P) == 1:
        p = P[0]
        if "PREVM" in T or "SAMEP" in T:
            if p.kind == "mtd":
                return p, _same_days_prev_month(p, ctx.today)
            if p.grain == "month" and "PREVM" in T:
                return p, _prev_month(p)
        return None, None
    if len(P) != 2:
        return None, None
    # the base period follows a comparison marker
    base_i = None
    for n, i in enumerate(idx):
        prev = seq[i - 1] if i > 0 else ""
        prev2 = seq[i - 2] if i > 1 else ""
        if prev in ("SAMEP", "CMP", "CHANGE") or (prev == "REV" and prev2 in ("CMP", "SAMEP")):
            base_i = n
    text = ctx.text
    m = re.search(r"\b(so voi|so|hon hay kem|cao hay thap hon|len hay xuong)\s+(cung ky\s+)?<P>", text)
    if m:
        pos = text[:m.end()].count("<P>") - 1
        base_i = pos
    if base_i is None:
        return None, None
    cur, base = P[1 - base_i], P[base_i]
    if cur.text in ("thang truoc", "nam truoc"):
        return None, None
    if base.text == "thang truoc":
        # 'so với tháng trước' is relative to the month asked about, not to today
        if cur.kind == "mtd":
            base = _same_days_prev_month(cur, ctx.today)
        elif cur.grain == "month":
            base = _prev_month(cur)
        else:
            return None, None
    elif base.text == "nam truoc":
        base = _shift_year(cur, ctx.today)
        if base is None:
            return None, None
    elif cur.kind == "mtd" and base.grain == "month" and base.start == _prev_month(cur).start:
        base = _same_days_prev_month(cur, ctx.today)
    return cur, base


def _quantity(ctx: Ctx, p: Period, f: Filt, unit: str):
    """Quantity sold counted in base units (rulebook §3 'Bán được bao nhiêu <đơn vị>')."""
    biz = ctx.biz
    if getattr(biz, "combo_names", False):
        raise Abstain("items and toppings share one cell")
    L = ctx.lines(p, f)
    if "qty_base" not in L:
        raise Abstain("no quantity")
    base = L["pkey"].map(lambda k: fold(biz.prod.get(k, {}).get("base_unit") or ""))
    if f.prod is not None:
        bu = fold(biz.prod.get(f.prod, {}).get("base_unit") or "")
        if not bu or bu != unit:
            raise Abstain("unit differs from base unit")
        return {"type": "number", "answer": num(L["qty_base"].sum())}
    if (base == "").any():
        raise Abstain("unknown base unit")
    sel = L[base == unit]
    if not (base == unit).any() and unit not in set(fold(v.get("base_unit") or "") for v in biz.prod.values()):
        raise Abstain("unit is not a base unit")
    return {"type": "number", "answer": num(sel["qty_base"].sum())}


# ------------------------------------------------------------------ rankings, series, counts
def _month_series(ctx: Ctx, ks: list):
    """Explicit list / range / 'k consecutive months ending at' -> list of month periods."""
    import calendar as _c
    P, T = ctx.periods, ctx.toks
    if len(P) >= 2 and all(p.grain == "month" for p in P):
        return list(P)
    if len(P) == 1:
        p = P[0]
        if p.kind == "range" and p.start.day == 1 and p.end.day == _c.monthrange(p.end.year, p.end.month)[1]:
            out, y, m = [], p.start.year, p.start.month
            while (y, m) <= (p.end.year, p.end.month) and len(out) < 36:
                out.append(Period(dt.date(y, m, 1), dt.date(y, m, _c.monthrange(y, m)[1]), "month", grain="month"))
                y, m = (y + 1, 1) if m == 12 else (y, m + 1)
            return out if len(out) >= 2 else None
        if p.grain == "month" and "CONSEC" in T and len(ks) == 1 and 2 <= ks[0] <= 12:
            out, cur = [], p
            for _ in range(int(ks[0])):
                out.append(cur)
                cur = _prev_month(cur)
            return out[::-1]
    return None


def _rank(series: pd.Series, k: int, tiekey) -> list:
    """Top-k index values of a metric series; ties broken by code/name; abstain on a tie at the boundary."""
    items = sorted(series.items(), key=lambda kv: (-kv[1], tiekey(kv[0])))
    if not items:
        raise Abstain("nothing to rank")
    if len(items) > k and abs(items[k - 1][1] - items[k][1]) < 1e-9:
        raise Abstain("tie at boundary")
    return [i for i, _ in items[:k]]


def _prod_label(biz: Biz, key) -> str:
    return biz.prod.get(key, {}).get("name") or str(key)


def _top_products(ctx: Ctx, p: Period, f: Filt, k: int):
    biz = ctx.biz
    if getattr(biz, "combo_names", False):
        raise Abstain("items and toppings share one cell")
    L = ctx.lines(p, f)
    if "pkey" not in L or "qty_base" not in L:
        raise Abstain("no product quantities")
    L = L[L["pkey"].notna()]
    names = L["pkey"].map(lambda key: fold(_prod_label(biz, key)))
    L = L[~names.str.startswith("phi phuc vu")]
    g = L.groupby("pkey")["qty_base"].sum()
    top = _rank(g, k, lambda key: str(key) if biz.prod_has_code else fold(_prod_label(biz, key)))
    return [_prod_label(biz, key) for key in top]


def _cust_labels(O: pd.DataFrame) -> dict:
    from .model import _strip_phone
    lab = O.dropna(subset=["cust"]).groupby("cust")["cust_label"].agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None)
    return {k: (_strip_phone(str(v)) if v else str(k)) for k, v in lab.items()}


def _top_customers(ctx: Ctx, p: Period, f: Filt, k: int, by: str):
    need_orders(ctx)
    biz = ctx.biz
    O = ctx.orders(p, f)
    if "cust" not in O or biz.orders["cust"].isna().all():
        raise missing(_B(locals()), "dữ liệu không ghi khách hàng")
    O = O[O["cust"].notna()]
    g = O.groupby("cust")["revenue"].sum() if by == "rev" else O.groupby("cust").size().astype(float)
    labels = _cust_labels(O)
    top = _rank(g, k, lambda key: str(key) if biz.has_cust_code else fold(labels.get(key, str(key))))
    return [labels.get(key, str(key)) for key in top]


def dispatch2(ctx: Ctx, f: Filt, meas, ks: list, thr: list):
    T, P = ctx.toks, ctx.periods
    biz = ctx.biz
    # ---------- series over months
    bare_list = len(P) >= 3 and all(p.grain == "month" for p in P) and only(T, "REV REVALL Q SOLD MONEY TOTAL")
    if getattr(ctx, "chain", None) and meas == "rev" and not thr and not f.any() and only(T, "REV REVALL NEG Q SOLD MONEY INCR MAX"):
        return {"type": "boolean", "answer": bool(all(ctx.revenue(a) > ctx.revenue(b) for a, b in ctx.chain))}
    if meas == "rev" and not thr and not f.any() and ((T & {"EACH", "LIST", "WHICHMONTH", "INCR"}) or bare_list):
        months = _month_series(ctx, ks)
        if months:
            extra_k = [v for v in ks if v != len(months)] if "CONSEC" not in T else []
            if extra_k:
                return None
            if "INCR" in T and only(T, "REV INCR NEG CONSEC Q SOLD MONEY CHANGE"):
                vals = [ctx.revenue(m) for m in months]
                return {"type": "boolean", "answer": bool(all(b > a for a, b in zip(vals, vals[1:])))}
            if "WHICHMONTH" in T and "MAX" in T and only(T, "REV WHICHMONTH MAX SOLD MONEY Q CMP CONSEC"):
                vals = [ctx.revenue(m) for m in months]
                best = [m for m, v in zip(months, vals) if v == max(vals)]
                if len(best) != 1:
                    raise Abstain("tie")
                return {"type": "date", "answer": best[0].start.strftime("%Y-%m")}
            if ((T & {"EACH", "LIST"}) or bare_list) and only(T, "REV REVALL EACH LIST Q SOLD MONEY CONSEC TOTAL") and "WHICHMONTH" not in T:
                return {"type": "list", "answer": [money(ctx.revenue(m)) for m in months]}
        return None
    if len(P) != 1:
        return None
    p = P[0]
    k = int(ks[0]) if len(ks) == 1 else None
    if len(ks) > 1:
        return None
    custy = bool(T & {"CUSTN", "WHO"})
    prody = bool(T & {"PRODN", "WHICHPROD"})
    # ---------- products that used to sell but had no sale in the period
    if "UNSOLD" in T and prody and not custy and not thr and k is None and not f.any() \
            and only(T, "UNSOLD PRODN WHICHPROD Q COUNT TOTAL SOLD NEG") and not ctx.units:
        if getattr(biz, "combo_names", False):
            raise Abstain("items and toppings share one cell")
        a, e = ctx.clip(p)
        L = biz.lines
        if L is None or "pkey" not in L:
            raise Abstain("no line items")
        L = L[L["st"].isin(["done", "ship"]) & L["pkey"].notna()]
        L = L[~L["pkey"].map(lambda key: fold(_prod_label(biz, key)).startswith("phi phuc vu"))]
        before = set(L[L["date"] < a]["pkey"].unique())
        inside = set(L[(L["date"] >= a) & (L["date"] <= e)]["pkey"].unique())
        ck = getattr(biz, "catalog_keys", None)
        if ck:
            before &= ck
        return {"type": "number", "answer": len(before - inside)}
    if "UNSOLD" in T:
        return None
    # ---------- product with the highest revenue
    if prody and not custy and not thr and "MAX" in T and (T & {"MONEY", "REV"}) and not (T & {"ORDER", "BEST"}) \
            and only(T, "MAX PRODN WHICHPROD LIST SOLD Q TOTAL MONEY REV") and not ctx.units and f.prod is None and not f.size:
        if getattr(biz, "combo_names", False):
            raise Abstain("items and toppings share one cell")
        if k is None and "LIST" in T:
            return None
        L = ctx.lines(p, f)
        L = L[L["pkey"].notna()]
        L = L[~L["pkey"].map(lambda key: fold(_prod_label(biz, key)).startswith("phi phuc vu"))]
        g = L.groupby("pkey")["amount"].sum()
        top = _rank(g, k or 1, lambda key: str(key) if biz.prod_has_code else fold(_prod_label(biz, key)))
        names = [_prod_label(biz, key) for key in top]
        return {"type": "list", "answer": names} if k is not None else {"type": "string", "answer": names[0]}
    # ---------- best selling products
    if prody and not custy and not thr and (("BEST" in T) or ("SOLD" in T and "MAX" in T)) and not (T & {"MONEY", "REV", "REVALL", "ORDER"}) \
            and only(T, "BEST MAX PRODN WHICHPROD LIST SOLD Q TOTAL") and len(ctx.units) <= 1:
        if f.prod is not None or f.size:
            return None
        if ctx.units and not f.group:
            return None
        if k is None and "LIST" in T:
            return None
        names = _top_products(ctx, p, f, k or 1)
        return {"type": "list", "answer": names} if k is not None else {"type": "string", "answer": names[0]}
    # ---------- best seller (staff) by revenue
    if "STAFFN" in T and "MAX" in T and not thr and k is None and not f.any() and only(T, "STAFFN WHO MAX REV SOLD MONEY Q"):
        O = ctx.orders(p)
        if "staff" not in O:
            raise missing(_B(locals()), "dữ liệu không ghi nhân viên bán")
        O = O[O["staff"].map(lambda v: not is_blank(v))]
        g = O.groupby(O["staff"].map(lambda v: str(v).strip()))["revenue"].sum()
        return {"type": "string", "answer": _rank(g, 1, str)[0]}
    if "STAFFN" in T:
        return None
    # ---------- top customers
    if custy and not prody and not thr and "MAX" in T | ({"MAX"} if "MANYORD" in T else set()):
        by = None
        if (T & {"ORDER", "MANYORD"}) and not (T & {"MONEY", "REV", "SPEND"}) and only(T, "CUSTN WHO SOLD ORDER MANYORD MAX LIST Q COUNT"):
            by = "orders"
        elif (T & {"SPEND", "MONEY", "REV"}) and not (T & {"ORDER", "MANYORD"}) and only(T, "CUSTN WHO SPEND SOLD MONEY REV MAX LIST Q TOTAL"):
            by = "rev"
        if by:
            if f.cust is not None or f.line_level():
                return None
            if k is None and "LIST" in T:
                return None
            names = _top_customers(ctx, p, f, k or 1, by)
            return {"type": "list", "answer": names} if k is not None else {"type": "string", "answer": names[0]}
    # ---------- customers / products by number of orders (thresholds)
    if len(thr) == 1 and k is None and not f.any():
        op, v = thr[0]
        cmpf = {"ge": lambda s: s >= v, "gt": lambda s: s > v, "lt": lambda s: s < v, "le": lambda s: s <= v}[op]
        if custy and not prody and "SOLD" in T and not (T & {"MONEY", "REV", "REVALL"}) \
                and only(T, "CUSTN WHO SOLD Q COUNT ORDER PCT SHARE RATE TOTAL"):
            need_orders(ctx)
            O = ctx.orders(p)
            if biz.orders["cust"].isna().all():
                raise missing(_B(locals()), "dữ liệu không ghi khách hàng")
            n = O[O["cust"].notna()].groupby("cust").size()
            if T & {"PCT", "SHARE", "RATE"}:
                if not len(n):
                    raise Abstain("no customers")
                return {"type": "percent", "answer": pct(100.0 * int(cmpf(n).sum()) / len(n))}
            return {"type": "number", "answer": int(cmpf(n).sum())}
        if prody and not custy and (T & {"APPEAR", "ORDER"}) and not (T & {"MONEY", "REV", "REVALL", "PCT"}) \
                and only(T, "PRODN APPEAR ORDER Q COUNT TOTAL SOLD"):
            need_orders(ctx)
            if getattr(biz, "combo_names", False):
                raise Abstain("items and toppings share one cell")
            L = ctx.lines(p)
            L = L[L["pkey"].notna()]
            names = L["pkey"].map(lambda key: fold(_prod_label(biz, key)))
            L = L[~names.str.startswith("phi phuc vu")]
            n = L.groupby("pkey")["order_id"].nunique()
            return {"type": "number", "answer": int(cmpf(n).sum())}
    return None


def price_effect(ctx: Ctx):
    """'Hiệu ứng tăng giá': quantity in N days from the change date vs the N days right before (rulebook §5)."""
    n = ctx.norm
    if not re.search(r"tang gia|doi gia|len gia|\blen \d{3,}", n):
        return None
    m = re.search(r"(\d+) ngay (?:ke tu|dau|tu )", n)
    if not m or not re.search(r"truoc", n) or "%" not in n:
        return None
    N = int(m.group(1))
    prods = ctx.ent("prod")
    if len(prods) != 1 or len(ctx.ents) != 1 or not 1 <= N <= 120:
        return None
    if getattr(ctx.biz, "combo_names", False):
        raise Abstain("items and toppings share one cell")
    days = {p.start for p in ctx.periods if p.kind == "day"}
    if len(days) == 1 and all(p.kind == "day" for p in ctx.periods):
        d = days.pop()
    elif not ctx.periods:
        from .ops import price_change_date
        mv = re.search(r"\blen (?:gia )?(?:muc )?(?:thanh )?(\d{3,9})\b", n) or re.search(r"\bgia (?:moi )?(?:la )?(\d{3,9})\b", n)
        if not mv:
            return None
        d = price_change_date(ctx.biz, prods[0].value, float(mv.group(1)))
        if d is None:
            raise Abstain("price change date not found in the price list")
    else:
        return None
    after = Period(d, d + dt.timedelta(days=N - 1), "range")
    before = Period(d - dt.timedelta(days=N), d - dt.timedelta(days=1), "range")
    b = ctx.biz
    if after.end > min(ctx.today, b.data_end) or before.start < b.data_start:
        raise Abstain("window not fully inside the data")
    f = Filt(prod=prods[0].value)
    L = b.lines
    L = L[(L["pkey"] == f.prod) & L["st"].isin(["done", "ship"])]
    qa = float(L[(L["date"] >= pd.Timestamp(after.start)) & (L["date"] <= pd.Timestamp(after.end))]["qty_base"].sum())
    qb = float(L[(L["date"] >= pd.Timestamp(before.start)) & (L["date"] <= pd.Timestamp(before.end))]["qty_base"].sum())
    return _pct_change(qa, qb)
