"""Semantic layer: turn the raw tables of one business into canonical sales facts."""
from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .raw import RawTable, load_raw
from .readme import Readme
from .textnorm import fold, is_blank, norm_answer, parse_datetime, parse_num, parse_time, squash

# ---------------------------------------------------------------- column roles
# README meaning (folded) -> role. First match wins.
MEANING_ROLES = [
    (r"^cot trong", "empty"),
    (r"^(ma don|so hoa don|so thu tu don)\b", "order_id"),
    (r"^thoi diem dat don", "datetime"),
    (r"^gio dat don", "time"),
    (r"^ngay ban", "date"),
    (r"^trang thai don", "status"),
    (r"^(ten hang|hang ban|ten mon)", "prod_name"),
    (r"^ma hang", "prod_code"),
    (r"^so luong da doi ra don vi co ban", "qty_base"),
    (r"^so luong ban lay tu lo", "lot_qty"),
    (r"^so luong khach tra lai", "ret_qty"),
    (r"^so luong nhap", "imp_qty"),
    (r"^so luong", "qty"),
    (r"^don vi tinh", "unit"),
    (r"^don vi co ban", "base_unit"),
    (r"^don vi lon", "big_unit"),
    (r"^don vi nhap", "imp_unit"),
    (r"^he so", "factor"),
    (r"^(don gia|gia ban ap dung)", "price"),
    (r"^(thanh tien sau giam|thanh tien cua dong)", "amount"),
    (r"^ten khach", "cust_name"),
    (r"^ma khach", "cust_id"),
    (r"^nhom hang", "group"),
    (r"^nhom khach", "cust_group"),
    (r"^kenh ban", "channel"),
    (r"^hinh thuc thanh toan", "pay_method"),
    (r"^gia von", "cost"),
    (r"^(giam gia cua ca don|tien bot cho ca don)", "order_discount"),
    (r"^(giam gia|tien giam cua ca dong)", "discount"),
    (r"^(tong tien don|so khach phai tra cho ca don|khach can tra cho ca don|so tien khach tra cho ca don)", "order_total"),
    (r"^(tong tien hang|tien hang cua don)", "order_goods"),
    (r"^(da thanh toan|khach da tra)", "paid"),
    (r"^(con no|so khach con no|khach con no)", "debt"),
    (r"^phi giao hang", "ship_fee"),
    (r"^ghi chu", "note"),
    (r"^ma nhan vien ban", "staff"),
    (r"^ma phieu nhap", "import_id"),
    (r"^ngay nhap", "imp_date"),
    (r"^(ma nha cung cap|nha cung cap)", "supplier"),
    (r"^ngay giao", "ship_date"),
    (r"^ten$", "name"),
    (r"^so thu tu dong", "line_no"),
    (r"^ton kho", "stock"),
    (r"^hoat chat", "ingredient"),
    (r"^dot giao", "batch"),
    (r"^thuoc ke don", "rx"),
]

# folded column name -> role, used when the README does not describe the column
NAME_ROLES = {
    "order_id": ["ma don hang", "ma hoa don", "ma bill", "ma don", "so ct", "ma chung tu", "so hd", "ma gd", "so hoa don",
                 "ma giao dich", "so don", "bill", "order id", "ma phieu", "so phieu", "hd", "so hdon", "ma hd", "don so",
                 "so don hang", "ma dh", "id don", "ma so don"],
    "prod_name": ["ten hang", "ten hang hoa", "ten san pham", "hang", "hang hoa", "product", "hang ban", "noi dung",
                  "ten mat hang", "mat hang", "san pham", "tenhang", "ten sp", "chi tiet hang", "mon", "ten mon", "ten goi",
                  "ten thuoc", "ten", "item"],
    "prod_code": ["ma hang", "ma sku", "ma hang hoa", "ma san pham ban", "ma hang ban", "id sp", "sku", "ma mat hang",
                  "ma hh", "ma h.hoa", "ma sp", "ma san pham", "ma mon", "ma vach"],
    "qty": ["so luong", "so luong san pham", "sl ban", "sl", "sluong", "soluong", "so luong hang", "so luong ban", "s.l", "slg"],
    "date": ["ngay", "ngay dat hang", "thoi gian", "ngay thang", "ngay giao dich", "ngay ban", "ngayban", "ngay mua",
             "ngay lap", "ngay xuat", "date", "ngay dat", "ngay hd", "ngay tao"],
    "datetime": ["thoi gian tao", "gio ban", "thoi gian lap", "gio tao don", "thoi diem tao", "gio tao"],
    "time": ["gio"],
    "status": ["trang thai", "trang thai don hang", "trang thai xu ly", "status", "xu ly", "ghi chu tt", "trang thai don",
               "ket qua", "tinh trang", "tinh trang don", "tien do", "tt don", "trang thai giao", "trang thai giao hang",
               "tinh trang giao", "trang thai hd", "trang thai hoa don", "tinh trang don hang"],
    "price": ["don gia", "gia san pham", "don gia (d)", "gia 1 dv", "gia ban", "gia le", "gia don vi", "don gia ban",
              "gia ban le", "gia", "price", "dg", "gia niem yet"],
    "unit": ["don vi tinh", "dvt", "don vi (tinh)", "donvi", "unit", "dv tinh", "d.vi", "don vi", "dv"],
    "amount": ["thanh tien", "tong tien", "amount", "tri gia", "t.tien", "tong", "tien", "thanh tien (d)", "tien ban", "cong", "tt"],
    "qty_base": ["so luong quy doi", "so luong le", "quy ra dv le", "sl quy doi", "sl le", "sl co ban", "sl quy ra le",
                 "sl (dv co ban)", "sl (dvt co ban)", "so luong (dvt co ban)", "sl dv co ban"],
    "cust_name": ["khach hang", "ten khach hang", "doi tac", "khach", "khach hang (ten)", "ben mua", "ten khach", "tenkh",
                  "ho ten khach", "ten kh"],
    "cust_id": ["ma kh", "ma khach mua", "so kh", "customer id", "ma nguoi mua", "ma the", "ma khach hang", "ma kh mua", "ma khach"],
    "group": ["nhom hang hoa", "nhom hang(3 cap)", "nhom hang", "loai san pham", "nhom", "danh muc", "phan loai", "loai hang"],
    "cust_group": ["nhom khach hang", "loai khach", "hang", "loai khach hang", "nhom kh"],
    "channel": ["kenh ban", "ban qua", "loai ban", "nguon", "kenh dat", "nguon don", "kenh ban hang"],
    "pay_method": ["phuong thuc tt", "tra bang", "hinh thuc thanh toan", "thanh toan bang", "phuong thuc thanh toan",
                   "phuong thuc", "httt", "pttt", "loai tt"],
    "cost": ["gia von", "gia von luc ban", "gia von hang", "gia von hien tai"],
    "discount": ["giam gia", "ck", "tien giam", "chiet khau", "khuyen mai", "giam (d)", "giam"],
    "order_discount": ["bot", "bot gia", "giam gia don", "giam gia hd", "giam gia hoa don"],
    "order_total": ["tong sau giam gia", "tong bill", "tong don", "phai thu", "tong thanh toan", "tongtien"],
    "order_goods": ["tong tien hang", "tien hang", "gia tri hang", "tien hang truoc giam"],
    "paid": ["khach da tra", "khach tra", "da tra", "thanh toan", "da thu"],
    "debt": ["con no", "con lai", "du no", "chua tra", "no", "con phai thu", "cong no"],
    "note": ["ghi chu", "note", "luu y", "dien giai"],
    "staff": ["nhan vien ban", "nguoi ban", "ma nhan vien", "ma nv ban", "nhan vien"],
    "ship_fee": ["phi van chuyen", "phi giao hang", "phi ship"],
    "base_unit": ["don vi co ban", "dv goc", "don vi goc", "dv nho", "don vi co so"],
    "line_no": ["stt", "no.", "thu tu", "tt dong", "so dong", "so thu tu"],
    "ingredient": ["hoat chat", "thanh phan chinh", "hoat chat chinh"],
    "rx": ["thuoc ke don", "ke don", "ban theo don", "thuoc ban theo don"],
}
_NAME2ROLES: dict = {}
for _r, _names in NAME_ROLES.items():
    for _n in _names:
        _NAME2ROLES.setdefault(_n, []).append(_r)

WALKIN = {"", "le", "kl", "khach le", "vang lai", "khach vang lai", "khach", "k le", "kle", "khachle", "none", "nan", "-"}


def split_name(c: str) -> str:
    """Folded column name with CamelCase / snake_case split into words."""
    out = []
    for i, ch in enumerate(c):
        if i > 0 and ch.isupper():
            prev = c[i - 1]
            nxt = c[i + 1] if i + 1 < len(c) else ""
            # 'MaHoaDon' -> 'Ma Hoa Don', 'SLQuyDoi' -> 'SL Quy Doi'; 'ĐVT' stays one word
            if prev.islower() or prev.isdigit() or (prev.isupper() and nxt.islower()):
                out.append(" ")
        out.append(ch)
    return fold("".join(out).replace("_", " "))


def role_from_meaning(meaning: str) -> str | None:
    m = fold(meaning)
    for rx, role in MEANING_ROLES:
        if re.search(rx, m):
            return role
    return None


def classify_status(v) -> str:
    """done | cancel | return | ship"""
    s = fold(v)
    if not s:
        return "done"
    if re.search(r"\bhuy\b|cancel|khong giao duoc|giao that bai|that bai|failed|void|bo don", s):
        return "cancel"
    if re.search(r"tra hang|doi tra|tra lai|hang tra|hoan hang|hoan tra|mang tra|return|refund|\btra ve\b", s) \
            or re.fullmatch(r"(da )?(khach )?(tra|hoan)( don| ve)?", s):
        return "return"
    if re.search(r"^dang |delivering|shipping|cho giao|chua giao|processing|pending|in transit", s):
        return "ship"
    return "done"


@dataclass
class Biz:
    biz_id: str
    rd: Readme
    tables: list
    roles: dict = field(default_factory=dict)  # table name -> {role: column}
    lines: pd.DataFrame | None = None
    orders: pd.DataFrame | None = None
    products: pd.DataFrame | None = None
    data_start: dt.date | None = None
    data_end: dt.date | None = None
    has_order_id: bool = False
    has_cust_code: bool = False
    notes: list = field(default_factory=list)
    extras: object = None
    lines_table: str = ""
    orders_table: str = ""

    def table(self, name: str) -> RawTable | None:
        for t in self.tables:
            if t.name == name:
                return t
        return None


def infer_roles(tb: RawTable) -> dict:
    """role -> column for one raw table."""
    roles: dict = {}
    taken = set()
    # 1) README meanings
    for c in tb.columns:
        if c in tb.meanings:
            r = role_from_meaning(tb.meanings[c])
            taken.add(c)
            if r and r not in roles:
                roles[r] = c
    # 2) column names
    cands: dict = {}
    for c in tb.columns:
        if c in taken:
            continue
        for r in _NAME2ROLES.get(split_name(c), []):
            cands.setdefault(r, []).append(c)
    order = ["order_id", "prod_code", "prod_name", "qty_base", "qty", "datetime", "date", "time", "status", "amount",
             "price", "unit", "cust_id", "cust_name", "group", "channel", "pay_method", "cost", "discount", "order_total",
             "order_goods", "paid", "debt", "note", "staff", "ship_fee", "base_unit", "cust_group", "line_no", "ingredient", "rx",
             "order_discount"]
    used = set(roles.values())
    for r in order:
        if r in roles:
            continue
        for c in cands.get(r, []):
            if c not in used:
                roles[r] = c
                used.add(c)
                break
    if "status" not in roles:
        for c in tb.columns:
            n = split_name(c)
            if c not in used and (n.startswith("trang thai") or n.startswith("tinh trang")) and "hddt" not in n and "hd dt" not in n:
                vals = set(classify_status(v) for v in tb.df[c].dropna().unique()[:200])
                if len(vals) >= 2:
                    roles["status"] = c
                    used.add(c)
                    break
    return roles


def _num_col(s: pd.Series) -> pd.Series:
    return s.map(parse_num).astype("float64")


def _year_hint(sheet) -> int | None:
    if not sheet:
        return None
    m = re.search(r"(20\d{2})", str(sheet))
    return int(m.group(1)) if m else None


def _date_col(df: pd.DataFrame, col: str) -> pd.Series:
    hints = df["_sheet"].map(_year_hint) if "_sheet" in df else pd.Series([None] * len(df))
    out = [parse_datetime(v, h) for v, h in zip(df[col].tolist(), hints.tolist())]
    return pd.to_datetime(pd.Series(out, index=df.index), errors="coerce")


def _frac_dates(tb: RawTable, col: str) -> float:
    vals = [v for v in tb.df[col].head(300).tolist() if not is_blank(v)]
    if not vals:
        return 0.0
    hint = 2025
    return sum(parse_datetime(v, hint) is not None for v in vals) / len(vals)


def _frac_num(tb: RawTable, col: str) -> float:
    vals = [v for v in tb.df[col].head(300).tolist() if not is_blank(v)]
    if not vals:
        return 0.0
    return sum(parse_num(v) is not None for v in vals) / len(vals)


def _is_import_like(tb: RawTable, roles: dict) -> bool:
    if "import_id" in roles or "imp_qty" in roles or "imp_date" in roles or "ship_date" in roles or "lot_qty" in roles:
        return True
    names = " | ".join(split_name(c) for c in tb.columns) + " | " + fold(tb.name)
    if re.search(r"nhap|ncc|nha cung cap|giao hang|ngay cho|lan giao|luot giao|bao hanh|ton kho|ton cuoi|the kho|lich su gia|"
                 r"ngay ap dung|phieu thu|thoi gian thu|han thanh toan|han tra|gop", names):
        return True
    return False


def build(biz_dir: str) -> Biz:
    rd, tables = load_raw(biz_dir)
    from .phone import prepare_abbreviations

    try:
        abbr = prepare_abbreviations(tables)
    except Exception:
        abbr = None
    biz = Biz(biz_id=os.path.basename(biz_dir.rstrip("/")), rd=rd, tables=tables)
    _apply_conventions(rd, tables)
    for tb in tables:
        biz.roles[tb.name] = infer_roles(tb)
    _build_sales(biz)
    from .enrich import enrich

    enrich(biz)
    from .imports import build_imports

    try:
        build_imports(biz)
        from .debts import build_payments

        build_payments(biz)
        from .imports import build_stock

        build_stock(biz)
        from .ops import build_ops

        build_ops(biz)
        from .lots import build_lots

        build_lots(biz)
    except Exception:  # purchases are optional; never break the sales model
        biz.imports = None
        biz.import_slips = None
    # the schema is trusted only when the README documents the columns of the sales tables
    fr = []
    for name in (biz.lines_table, biz.orders_table):
        tb = biz.table(name) if name else None
        if tb is not None and tb.columns:
            fr.append(sum(1 for c in tb.columns if c in tb.meanings) / len(tb.columns))
    biz.schema_confident = bool(fr) and min(fr) >= 0.8
    lr = biz.roles.get(biz.lines_table, {})
    orr = biz.roles.get(biz.orders_table, {})
    has_status = "status" in lr or "status" in orr
    if not has_status and ("note" in lr or "note" in orr):
        # books without a status column keep it in the note (rulebook §8) when the README says so
        has_status = re.search(r"ghi chu[^.\n]*(tra lai|huy|dang giao)", fold(rd.text)) is not None
    biz.unreliable = (not biz.schema_confident) and not (has_status and biz.has_order_id)
    L0 = biz.lines
    biz.combo_names = bool(L0 is not None and "prod_name" in L0 and L0["prod_name"].dropna().astype(str).str.contains(" + ", regex=False).mean() > 0.001)
    biz.abbrev_names = any(fold(tb.name) == "ghi tat" for tb in biz.tables) and abbr is None
    biz.phone = abbr is not None
    if biz.phone:
        from .phone import attach_tradeins

        try:
            attach_tradeins(biz)
            from .phone import build_phone_books
            build_phone_books(biz)
        except Exception:
            pass
    biz.combo_names = biz.combo_names or biz.abbrev_names
    biz.member_dates = None
    for tb in biz.tables:
        mc = next((c for c in tb.columns if fold(tb.meanings.get(c, "")).startswith("ngay dang ky")), None)
        if mc:
            biz.member_dates = _date_col(tb.df, mc).dt.normalize().dropna()
            break
    biz.no_order_id_stated = "khong ghi ma don" in fold(rd.text) or "khong co ma don" in fold(rd.text)
    from .repair import build_repair

    try:
        build_repair(biz)
    except Exception:
        biz.repair = False
    return biz


_CONV_RX = re.compile(r"(?:\*\*([^*`]+)\*\*|`(?:([\w ]+)\.)?([^`.]+)`)\s+là\s+([^;.]+)")
_CONV_OK = re.compile(r"^(so khach phai tra cho ca don|so tien khach tra cho ca don|thanh tien cua dong|so luong da doi ra don vi co ban|gia von mot don vi ban|"
                      r"so khach con no cua don|so tien cua don chua thu|tong tien cua ca phieu|gia nhap mot don vi)")


def _apply_conventions(rd, tables) -> None:
    """Column meanings stated in the README conventions (only for columns the README does not already describe)."""
    for _lb, text in rd.conventions:
        mo = re.search(r"các dòng cùng \*\*([^*]+)\*\* là một đơn", text)
        if mo:
            col = mo.group(1).strip()
            hits = sorted([tb for tb in tables if col in tb.columns], key=lambda t: -len(t.df))
            if hits and col not in hits[0].meanings:
                hits[0].meanings[col] = "mã đơn"
        for m in _CONV_RX.finditer(text):
            col = (m.group(1) or m.group(3) or "").strip()
            tname = (m.group(2) or "").strip()
            meaning = m.group(4).strip()
            f = fold(meaning)
            if not col or not _CONV_OK.match(f):
                continue
            if f.startswith("so tien cua don chua thu") or f.startswith("so khach con no"):
                meaning = "còn nợ"
            elif f.startswith("thanh tien cua dong"):
                meaning = "thành tiền của dòng"
            elif f.startswith("gia von"):
                meaning = "giá vốn"
            elif f.startswith("tong tien cua ca phieu"):
                meaning = "tổng tiền phiếu nhập"
            elif f.startswith("gia nhap mot don vi"):
                meaning = "giá nhập một đơn vị"
            hits = [tb for tb in tables if col in tb.columns and (not tname or tb.name == tname)]
            if len(hits) == 1 and col not in hits[0].meanings:
                hits[0].meanings[col] = meaning


def _pick_lines(biz: Biz):
    best, best_score = None, -1
    for tb in biz.tables:
        r = biz.roles[tb.name]
        if _is_import_like(tb, r):
            continue
        if not ("prod_name" in r or "prod_code" in r):
            continue
        if not ("amount" in r or ("qty" in r and "price" in r)):
            continue
        if "amount" in r and _frac_num(tb, r["amount"]) < 0.8:
            continue
        score = len(tb.df) + (10 ** 9 if "order_id" in r else 0) + (10 ** 8 if "amount" in r else 0)
        if score > best_score:
            best, best_score = tb, score
    return best


def _pick_orders(biz: Biz, lines_tb: RawTable | None):
    """Order header table: one row per order with date/status, no product columns."""
    best, best_score = None, -1
    line_ids = None
    if lines_tb is not None and "order_id" in biz.roles[lines_tb.name]:
        line_ids = set(str(v).strip() for v in lines_tb.df[biz.roles[lines_tb.name]["order_id"]].head(2000) if not is_blank(v))
    for tb in biz.tables:
        if lines_tb is not None and tb.name == lines_tb.name:
            continue
        r = biz.roles[tb.name]
        if "order_id" not in r or "prod_name" in r and "qty" in r:
            continue
        if _is_import_like(tb, r):
            continue
        if not ("date" in r or "datetime" in r):
            continue
        if not ("status" in r or "order_total" in r or "order_goods" in r or "cust_id" in r or "cust_name" in r):
            continue
        ids = tb.df[r["order_id"]]
        uniq = ids.nunique() / max(1, len(ids))
        if uniq < 0.95:
            continue
        if line_ids is not None:
            have = set(str(v).strip() for v in ids if not is_blank(v))
            if len(line_ids & have) < 0.8 * len(line_ids):
                continue
        score = len(tb.df)
        if score > best_score:
            best, best_score = tb, score
    return best


def _ffill_in_order(df: pd.DataFrame, cols: list) -> None:
    """Blank order-level cells on later lines of an order repeat the first line (rulebook §8)."""
    for c in cols:
        if c not in df:
            continue
        s = df[c].where(~df[c].map(is_blank), other=np.nan)
        if s.isna().any():
            df[c] = s.groupby(df["order_id"], sort=False).transform(lambda g: g.ffill().bfill())


_SUMMARY_RX = re.compile(r"^(tong|cong|tong cong|total|subtotal|sum|grand total|tong ngay|tong thang|cong ngay|cong thang)\b")


def _drop_summary_rows(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Remove subtotal lines such as 'TỔNG NGÀY 15/07' that exports interleave with data."""
    if not len(df):
        return df
    sub = df[cols]
    blank = sub.map(is_blank)
    few = (~blank).sum(axis=1) <= max(3, len(cols) // 3)
    if not few.any():
        return df
    cand = sub[few]
    hit = cand.apply(lambda row: any(isinstance(v, str) and _SUMMARY_RX.match(fold(v)) for v in row), axis=1)
    if not hit.any():
        return df
    return df.drop(index=cand.index[hit])


_TOPPING_RX = re.compile(r"^(?:(\d+)\s+)?(.+?)\s*\((\s*[-\d.,]+\s*(?:đ|k|d|vnđ|vnd)?\s*)\)\s*$", re.I)


def _split_toppings(src: pd.DataFrame, r: dict) -> pd.DataFrame:
    """'Trà đào (M) + 2 Trân châu đen (14.000đ)' is two items sold: the drink and the topping (rulebook §6 'Cà phê')."""
    nc, ac = r.get("prod_name"), r.get("amount")
    if not nc or not ac or not len(src):
        return src
    mask = src[nc].map(lambda v: isinstance(v, str) and " + " in v)
    if not mask.any():
        return src
    src = src.copy()
    extra = []
    for idx in src.index[mask]:
        cell = str(src.at[idx, nc])
        main, *tops = [p.strip() for p in cell.split(" + ")]
        total = parse_num(src.at[idx, ac])
        parsed = []
        for t in tops:
            m = _TOPPING_RX.match(t)
            money = parse_num(m.group(3)) if m else None
            if not m or money is None:
                parsed = None
                break
            parsed.append((int(m.group(1) or 1), m.group(2).strip(), money))
        if parsed is None or total is None:
            continue
        src.at[idx, nc] = main
        src.at[idx, ac] = total - sum(mny for _n, _t, mny in parsed)
        for n, tname, mny in parsed:
            row = src.loc[idx].copy()
            row[nc] = tname if fold(tname).startswith("topping") else "Topping " + tname
            row[ac] = mny
            if r.get("qty"):
                row[r["qty"]] = n
            if r.get("price"):
                row[r["price"]] = mny / n if n else mny
            if r.get("unit"):
                row[r["unit"]] = "phần"
            if r.get("qty_base"):
                row[r["qty_base"]] = n
            extra.append(row)
    if extra:
        src = pd.concat([src, pd.DataFrame(extra)], ignore_index=True)
    return src


def _build_sales(biz: Biz) -> None:
    lt = _pick_lines(biz)
    ot = _pick_orders(biz, lt)
    if lt is None and ot is None:
        biz.notes.append("no sales table found")
        return
    hdr = None
    if ot is not None:
        biz.orders_table = ot.name
        r = biz.roles[ot.name]
        h = pd.DataFrame({"order_id": ot.df[r["order_id"]].map(lambda v: str(v).strip() if not is_blank(v) else None)})
        for role in ("status", "cust_id", "cust_name", "channel", "pay_method", "staff", "note", "cust_group"):
            if role in r:
                h[role] = ot.df[r[role]].values
        for role in ("order_total", "order_goods", "paid", "debt", "ship_fee", "order_discount"):
            if role in r:
                h[role] = _num_col(ot.df[r[role]]).values
        if "discount" in r and "order_discount" not in h:
            h["order_discount"] = _num_col(ot.df[r["discount"]]).values
        dcol = r.get("date") or r.get("datetime")
        h["date"] = _date_col(ot.df, dcol).values
        if "datetime" in r:
            h["dt"] = _date_col(ot.df, r["datetime"]).values
        elif "time" in r:
            h["dt"] = _combine_time(h["date"], ot.df[r["time"]])
        h = h.dropna(subset=["order_id"]).drop_duplicates(subset=["order_id"], keep="first")
        hdr = h

    if lt is not None:
        biz.lines_table = lt.name
        r = biz.roles[lt.name]
        src = _drop_summary_rows(lt.df, lt.columns)
        raw_cols = [c for c in lt.columns]
        if "order_id" in r:
            oc = r["order_id"]
            src = src.copy()
            # blank order cells on later lines of an order repeat the first line (rulebook §8)
            src[oc] = src[oc].where(~src[oc].map(is_blank), other=np.nan)
            src[oc] = src.groupby(["_file", "_sheet"], sort=False, dropna=False)[oc].ffill()
            # exact duplicate lines of one order are export errors -> count once
            src = src.drop_duplicates(subset=raw_cols, keep="first")
        src = _split_toppings(src, r)
        L = pd.DataFrame(index=src.index)
        L["_sheet"] = src["_sheet"] if "_sheet" in src else None
        if "order_id" in r:
            L["order_id"] = src[r["order_id"]].map(lambda v: str(v).strip() if not is_blank(v) else None)
            L["order_id"] = L["order_id"].ffill()
            L["order_no"] = L["order_id"]
            biz.has_order_id = True
        elif hdr is None:
            L["order_id"] = [f"_row{i}" for i in range(len(src))]
        for role, name in (("prod_name", "prod_name"), ("prod_code", "prod_code"), ("unit", "unit"), ("status", "status"),
                           ("cust_id", "cust_id"), ("cust_name", "cust_name"), ("channel", "channel"),
                           ("pay_method", "pay_method"), ("staff", "staff"), ("note", "note"), ("group", "group"),
                           ("cust_group", "cust_group")):
            if role in r:
                L[name] = src[r[role]].values
        for role in ("qty", "qty_base", "price", "amount", "cost", "discount", "order_total", "order_goods", "paid", "debt",
                     "ship_fee", "order_discount"):
            if role in r:
                L[role] = _num_col(src[r[role]]).values
        if "date" in r or "datetime" in r:
            L["date"] = _date_col(src, r.get("date") or r["datetime"]).values
            if "datetime" in r:
                L["dt"] = _date_col(src, r["datetime"]).values
            elif "time" in r:
                tcol = src[r["time"]]
                full = _date_col(src, r["time"])
                if full.notna().mean() > 0.8 and (full.dt.hour + full.dt.minute).sum() > 0:
                    L["dt"] = full.values
                else:
                    L["dt"] = _combine_time(L["date"], tcol)
            elif (L["date"].dt.hour + L["date"].dt.minute).sum() > 0:
                L["dt"] = L["date"]
        if "amount" not in L and "qty" in L and "price" in L:
            L["amount"] = L["qty"] * L["price"]
        if biz.has_order_id and L["order_id"].notna().any():
            ids = L["order_id"].dropna()
            if ids.head(500).map(lambda v: bool(re.fullmatch(r"\d+(\.0)?", str(v)))).mean() > 0.9 and L["_sheet"].nunique() > 1:
                L["order_id"] = L["_sheet"].astype(str) + "#" + L["order_id"].astype(str)
        if biz.has_order_id:
            _ffill_in_order(L, ["status", "cust_id", "cust_name", "note", "channel", "pay_method", "staff"])
            if "date" in L:
                L["date"] = L["date"].groupby(L["order_id"], sort=False).transform(lambda g: g.ffill().bfill())
        if hdr is not None and "order_id" in L:
            add = [c for c in hdr.columns if c != "order_id" and c not in L.columns]
            L = L.merge(hdr[["order_id"] + add], on="order_id", how="left")
    else:
        L = None

    if L is not None:
        # stray annotation rows ('Ghi chú: số liệu ghi tay') carry neither an item nor any figure
        has_item = pd.Series(False, index=L.index)
        for c in ("prod_name", "prod_code"):
            if c in L:
                has_item |= L[c].map(lambda v: not is_blank(v))
        has_num = pd.Series(False, index=L.index)
        for c in ("amount", "qty"):
            if c in L:
                has_num |= L[c].notna()
        if ("prod_name" in L or "prod_code" in L) and (~(has_item | has_num)).any():
            L = L[has_item | has_num].copy()
    if L is not None and "date" in L:
        if "dt" not in L:
            L["dt"] = pd.NaT
        L["date"] = L["date"].dt.normalize()
        L = _finish_lines(biz, L)
        biz.lines = L
        O = _orders_from_lines(L)
    elif hdr is not None:
        # invoice list without line items
        O = hdr.copy()
        O["date"] = O["date"].dt.normalize()
        O["st"] = O["status"].map(classify_status) if "status" in O else "done"
        if "order_goods" in O:
            O["revenue"] = O["order_goods"]
        elif "order_total" in O:
            O["revenue"] = O["order_total"]
            biz.notes.append("revenue from order_total")
        else:
            O["revenue"] = np.nan
        biz.has_order_id = True
        _customers(biz, O)
    else:
        biz.notes.append("sales table has no date")
        return
    O = O.dropna(subset=["date"])
    biz.orders = O
    if len(O):
        biz.data_start = O["date"].min().date()
        biz.data_end = rd_end(biz, O)


def rd_end(biz: Biz, O: pd.DataFrame) -> dt.date:
    mx = O["date"].max().date()
    if biz.rd.data_end:
        return biz.rd.data_end
    return mx


def _combine_time(date_s: pd.Series, time_raw: pd.Series) -> pd.Series:
    out = []
    for d, t in zip(pd.to_datetime(date_s).tolist(), time_raw.tolist()):
        tm = parse_time(t)
        if pd.isna(d) or tm is None:
            out.append(pd.NaT)
        else:
            out.append(pd.Timestamp(d.year, d.month, d.day, tm[0], tm[1], tm[2]))
    return pd.Series(out, index=date_s.index)


def _customers(biz: Biz, df: pd.DataFrame) -> None:
    """Add 'cust' (identity key or None for walk-ins) and 'cust_label'."""
    if "cust_id" in df:
        ids = df["cust_id"].map(lambda v: None if (is_blank(v) or fold(v) in WALKIN) else str(v).strip())
        biz.has_cust_code = True
        df["cust"] = ids
        if "cust_name" in df:
            df["cust_label"] = df["cust_name"].map(lambda v: None if is_blank(v) else str(v).strip())
        else:
            df["cust_label"] = ids
    elif "cust_name" in df:
        def key(v):
            if is_blank(v):
                return None
            f = fold(v)
            if f in WALKIN:
                return None
            return squash(_strip_phone(str(v)))

        df["cust"] = df["cust_name"].map(key)
        df["cust_label"] = df["cust_name"].map(lambda v: None if is_blank(v) else str(v).strip())
    else:
        df["cust"] = None
        df["cust_label"] = None


def _strip_phone(v: str) -> str:
    """'Triệu Linh An - 0327 988 481' / 'Châu Hằng Hùng (0932225622)' -> the name alone."""
    out = re.sub(r"[\s\-–(]*\(?\+?\d[\d .\-]{7,}\d\)?\s*$", "", v).strip(" -–(")
    return out or v


def _finish_lines(biz: Biz, L: pd.DataFrame) -> pd.DataFrame:
    L = L.copy()
    st = L["status"].map(classify_status) if "status" in L else pd.Series("done", index=L.index)
    if "note" in L:
        nst = L["note"].map(lambda v: classify_status(v) if not is_blank(v) else "done")
        st = st.where(st != "done", nst) if "status" in L else nst
    L["st_line"] = st
    # order level status: cancel > return > ship > done ; negative quantities mean a return
    rank = {"cancel": 3, "return": 2, "ship": 1, "done": 0}
    inv = {v: k for k, v in rank.items()}
    code = st.map(rank)
    if "qty" in L:
        code = code.where(~((L["qty"] < 0) & (code < 2)), 2)
    g = code.groupby(L["order_id"], sort=False).transform("max")
    L["st"] = g.map(inv)
    _customers(biz, L)
    if "prod_name" in L:
        L["prod_name"] = L["prod_name"].map(lambda v: None if is_blank(v) else str(v).strip())
        # voucher / loyalty points / shipping fee rows are not items and not revenue (rulebook §1)
        non_item = L["prod_name"].map(lambda v: bool(v) and bool(re.match(r"^(voucher|dung \d+ diem|dung diem|phi ship|phi giao hang|phi van chuyen|phi giao)\b", fold(v))))
        if non_item.any():
            from .baby import classify_extra
            X = L[non_item][["order_id", "prod_name", "amount"]].copy()
            X["kind"] = X["prod_name"].map(classify_extra)
            biz.extras = X.dropna(subset=["kind", "amount"])
            L = L[~non_item].copy()
    if "prod_code" in L:
        L["prod_code"] = L["prod_code"].map(lambda v: None if is_blank(v) else str(v).strip())
    return L


def _orders_from_lines(L: pd.DataFrame) -> pd.DataFrame:
    agg = {"date": ("date", "first"), "dt": ("dt", "first"), "st": ("st", "first"), "revenue": ("amount", "sum"),
           "cust": ("cust", "first"), "cust_label": ("cust_label", "first")}
    for c in ("order_no", "channel", "pay_method", "staff", "order_total", "order_goods", "paid", "debt", "ship_fee", "order_discount", "note",
              "cust_group"):
        if c in L:
            agg[c] = (c, "first")
    O = L.groupby("order_id", sort=False).agg(**agg).reset_index()
    return O
