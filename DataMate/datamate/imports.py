"""Purchase (import) slips: one canonical slip table and one line table per business."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .model import Biz, _date_col, _num_col, split_name
from .textnorm import fold, is_blank, squash

# role -> (README meaning regex, column names)
IMP_ROLES = {
    "imp_id": (r"^ma phieu nhap", ["ma nhap hang", "ma don nhap", "so phieu", "ma phieu nhap", "ma phieu", "so phieu nhap"]),
    "imp_date": (r"^ngay nhap", ["ngay nhap", "thoi gian", "ngay", "ngay nhap kho", "ngay phieu", "ngay nhap hang"]),
    "supplier": (r"^(ma nha cung cap|nha cung cap)", ["ma ncc", "nha cung cap", "ma nha cung cap", "ncc", "ten ncc"]),
    "slip_total": (r"^(tong tien phieu nhap|tong tien cua ca phieu)", ["can tra ncc", "tong tien phieu", "tong gia tri nhap", "tong tien", "tong phieu"]),
    "imp_status": (r"^trang thai phieu", ["trang thai", "tinh trang", "trang thai phieu"]),
    "prod_code": (r"^ma hang", ["ma hang", "ma sku", "ma hang hoa", "ma sp", "ma san pham", "id sp", "sku", "ma hh", "ma mon", "ma vach"]),
    "prod_name": (r"^(ten hang|ten$)", ["ten hang", "ten mat hang", "product", "hang hoa", "ten san pham", "san pham", "hang", "mat hang", "ten hang hoa"]),
    "imp_unit": (r"^(don vi nhap|don vi tinh)", ["dvt", "dvt nhap", "don vi tinh", "don vi", "dv nhap", "dv"]),
    "imp_qty": (r"^(so luong nhap|so luong$)", ["sl nhap", "so luong", "sl", "so luong nhap"]),
    "imp_price": (r"^gia nhap mot don vi", ["gia nhap", "don gia", "don gia nhap", "gia"]),
    "imp_amount": (r"^tien nhap cua ca dong", ["thanh tien", "tri gia", "tien"]),
    "note": (r"^ghi chu", ["ghi chu", "noi dung", "mo ta", "dien giai", "ghi chu"]),
    "lot": (r"^so lo", ["lo", "ma lo", "so lo"]),
}


def _roles(tb) -> dict:
    out, used = {}, set()
    for role, (rx, _names) in IMP_ROLES.items():
        for c in tb.columns:
            if c in tb.meanings and c not in used and re.search(rx, fold(tb.meanings[c])):
                out[role] = c
                used.add(c)
                break
    for role, (_rx, names) in IMP_ROLES.items():
        if role in out:
            continue
        for c in tb.columns:
            if c not in used and c not in tb.meanings and split_name(c) in names:
                out[role] = c
                used.add(c)
                break
    return out


def _s(v):
    return None if is_blank(v) else str(v).strip()


def build_imports(biz: Biz) -> None:
    biz.imports = None      # lines: imp_id, date, supplier, pkey, unit_f, qty, price, amount, cancelled
    biz.import_slips = None  # slips: imp_id, date, supplier, total, cancelled
    cands = []
    for tb in biz.tables:
        if tb.name in (biz.lines_table, biz.orders_table, getattr(biz, "catalog_table", "")):
            continue
        names = " | ".join(split_name(c) for c in tb.columns) + " | " + split_name(tb.name)
        if not re.search(r"nhap", names) or re.search(r"lo hang|theo lo|the kho|ton kho", split_name(tb.name)):
            continue
        r = _roles(tb)
        if "imp_id" not in r:
            continue
        cands.append((tb, r))
    head = lines = None
    for tb, r in cands:
        ids = tb.df[r["imp_id"]]
        uniq = ids.nunique() >= 0.98 * len(ids)
        has_prod = "prod_code" in r or "prod_name" in r
        if has_prod and "imp_qty" in r and (lines is None or len(tb.df) > len(lines[0].df)):
            lines = (tb, r)
        if uniq and "imp_date" in r and not has_prod:
            head = (tb, r)
    if lines is None and head is None:
        return
    H = None
    if head is not None:
        tb, r = head
        H = pd.DataFrame({"imp_id": tb.df[r["imp_id"]].map(_s)})
        H["date"] = _date_col(tb.df, r["imp_date"]).dt.normalize().values
        H["supplier"] = tb.df[r["supplier"]].map(_s).values if "supplier" in r else None
        H["total"] = _num_col(tb.df[r["slip_total"]]).values if "slip_total" in r else np.nan
        st = tb.df[r["imp_status"]] if "imp_status" in r else (tb.df[r["note"]] if "note" in r else None)
        H["cancelled"] = st.map(lambda v: bool(re.search(r"\bhuy\b|cancel", fold(v)))).values if st is not None else False
        H = H.dropna(subset=["imp_id"]).drop_duplicates("imp_id")
    L = None
    if lines is not None:
        tb, r = lines
        src = tb.df
        L = pd.DataFrame({"imp_id": src[r["imp_id"]].map(_s)})
        L["imp_id"] = L["imp_id"].ffill()
        code = src[r["prod_code"]].map(_s) if "prod_code" in r else None
        name = src[r["prod_name"]].map(_s) if "prod_name" in r else None
        if code is not None and getattr(biz, "prod_has_code", False):
            L["pkey"] = code.values
        elif name is not None:
            L["pkey"] = name.map(lambda v: biz.name2key.get(squash(v)) if v else None).values
        else:
            L["pkey"] = None
        L["unit_f"] = src[r["imp_unit"]].map(lambda v: fold(v) if not is_blank(v) else "").values if "imp_unit" in r else ""
        L["qty"] = _num_col(src[r["imp_qty"]]).values
        L["lot"] = src[r["lot"]].map(_s).values if "lot" in r else None
        L["price"] = _num_col(src[r["imp_price"]]).values if "imp_price" in r else np.nan
        if "imp_amount" in r:
            L["amount"] = _num_col(src[r["imp_amount"]]).values
        else:
            L["amount"] = L["qty"] * L["price"]
        if H is not None:
            L = L.merge(H[["imp_id", "date", "supplier", "cancelled"]], on="imp_id", how="left")
        else:
            if "imp_date" not in r:
                return
            L["date"] = _date_col(src, r["imp_date"]).dt.normalize().values
            L["supplier"] = src[r["supplier"]].map(_s).values if "supplier" in r else None
            st = src[r["imp_status"]] if "imp_status" in r else (src[r["note"]] if "note" in r else None)
            L["cancelled"] = st.map(lambda v: bool(re.search(r"\bhuy\b|cancel", fold(v)))).values if st is not None else False
            for c in ("date", "supplier"):
                L[c] = L.groupby("imp_id", sort=False)[c].transform(lambda g: g.ffill().bfill())
        L["cancelled"] = L["cancelled"].fillna(False).astype(bool)
        biz.imports = L
    if H is None and L is not None:
        H = L.groupby("imp_id", sort=False).agg(date=("date", "first"), supplier=("supplier", "first"),
                                                total=("amount", "sum"), cancelled=("cancelled", "max")).reset_index()
    elif H is not None and L is not None and H["total"].isna().all():
        tot = L.groupby("imp_id")["amount"].sum()
        H["total"] = H["imp_id"].map(tot)
    biz.import_slips = H
    # supplier display names (code -> name) when a supplier list exists
    biz.supplier_names = {}
    for tb in biz.tables:
        cols = {split_name(c): c for c in tb.columns}
        code_c = cols.get("ma ncc") or cols.get("ma nha cung cap")
        name_c = cols.get("ten goi") or cols.get("ten ncc") or cols.get("ten nha cung cap") or cols.get("nha cung cap") or cols.get("ten")
        if code_c and name_c and tb.df[code_c].nunique() == len(tb.df) and len(tb.df) < 2000:
            biz.supplier_names = {_s(a): _s(b) for a, b in zip(tb.df[code_c], tb.df[name_c])}
            break


STOCK_NAMES = {"ton cuoi ky", "ton cuoi", "ton thuc", "ton kho", "ton", "so luong ton", "ton hien tai", "sl ton"}


def build_stock(biz: Biz) -> None:
    """biz.stock: {pkey: quantity in base units at the end of the data}; biz.stock_ledger: per-movement balances."""
    biz.stock, biz.stock_ledger = None, None
    nprod = max(1, len(getattr(biz, "prod", {})))
    for tb in biz.tables:
        if tb.name in (biz.lines_table, biz.orders_table):
            continue
        sc = None
        for c in tb.columns:
            if fold(tb.meanings.get(c, "")).startswith("ton kho") or (c not in tb.meanings and split_name(c) in STOCK_NAMES):
                sc = c
                break
        if sc is None:
            continue
        r = biz.roles.get(tb.name, {})
        kc_code = r.get("prod_code")
        kc_name = r.get("prod_name") or r.get("name")
        if not (kc_code or kc_name):
            continue

        def key(row):
            if kc_code and getattr(biz, "prod_has_code", False):
                return _s(row[kc_code])
            v = _s(row[kc_name]) if kc_name else None
            return biz.name2key.get(squash(v)) if v else None

        if len(tb.df) > 3 * nprod:
            # movement ledger: balance after every movement
            tc = next((c for c in tb.columns if split_name(c) in ("ngay gio phat sinh", "thoi gian", "ngay", "ngay phat sinh", "thoi diem")), None)
            if tc is None:
                continue
            G = pd.DataFrame({"pkey": [key(row) for _, row in tb.df.iterrows()]})
            G["time"] = _date_col(tb.df, tc).values
            G["bal"] = _num_col(tb.df[sc]).values
            G["seq"] = range(len(G))
            biz.stock_ledger = G.dropna(subset=["pkey", "time", "bal"])
        else:
            st = {}
            for _, row in tb.df.iterrows():
                k, v = key(row), _num_col(pd.Series([row[sc]])).iloc[0]
                if k is not None and not pd.isna(v):
                    st[k] = float(v)
            if st:
                biz.stock = st
