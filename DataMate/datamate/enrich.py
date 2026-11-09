"""Catalogues, unit conversion and entity lexicons built on top of the sales model."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .model import Biz, _is_import_like, split_name
from .textnorm import fold, is_blank, parse_num, squash

FACTOR_NAMES = {"ti le quy doi", "he so", "he so qd", "quy doi", "he so quy doi", "bang", "ty le quy doi"}
BIGUNIT_NAMES = {"don vi quy doi", "dv nhap", "dv lon", "loai dv", "quy cach", "don vi lon"}


def _s(v):
    return None if is_blank(v) else str(v).strip()


def _col(tb, role, roles):
    c = roles.get(role)
    return tb.df[c] if c else None


def _find_catalog(biz: Biz):
    """Product master table: (code?, name, group?, base unit?) with one row per product."""
    L = biz.lines
    line_codes = set(L["prod_code"].dropna().unique()) if L is not None and "prod_code" in L else set()
    line_names = set(squash(v) for v in L["prod_name"].dropna().unique()) if L is not None and "prod_name" in L else set()
    best, best_score = None, -1
    for tb in biz.tables:
        if tb.name in (biz.lines_table, biz.orders_table):
            continue
        r = biz.roles[tb.name]
        if len(tb.df) > 20000 or len(tb.df) == 0:
            continue
        if any(k in r for k in ("import_id", "imp_date", "order_id", "ship_date", "lot_qty", "cust_id", "factor", "big_unit")):
            continue
        name_c = r.get("name") or r.get("prod_name")
        code_c = r.get("prod_code")
        if not name_c:
            continue
        key = tb.df[code_c] if code_c else tb.df[name_c]
        if key.nunique() < 0.9 * len(key):
            continue
        if line_codes and code_c:
            have = set(str(v).strip() for v in tb.df[code_c] if not is_blank(v))
            cover = len(line_codes & have) / len(line_codes)
        elif line_names:
            have = set(squash(v) for v in tb.df[name_c] if not is_blank(v))
            cover = len(line_names & have) / len(line_names)
        else:
            cover = 0.5 if ("group" in r or "base_unit" in r) else 0.0
        if cover < 0.5:
            continue
        score = cover * 10 + (4 if code_c else 0) + (2 if "group" in r else 0) + (1 if ("base_unit" in r or "unit" in r) else 0)
        if score > best_score:
            best, best_score = tb, score
    return best


def _parse_factor(v):
    """'6' -> (6, None); '15 vỉ' -> (15, 'vỉ')"""
    if is_blank(v):
        return None, None
    if isinstance(v, (int, float)):
        return float(v), None
    s = str(v).strip()
    m = re.match(r"^([\d.,]+)\s*(\D.*)?$", s)
    if not m:
        return None, None
    n = parse_num(m.group(1))
    return n, (m.group(2).strip() if m.group(2) else None)


def _readme_conversions(biz: Biz) -> list:
    """[(product name, big unit, n, small unit)] from the README 'Quy đổi đơn vị' list."""
    out = []
    for lb, text in biz.rd.conventions:
        for line in text.split("\n"):
            m = re.match(r"^(.*?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s*(.+?)\s*$", line.strip())
            if m:
                n = parse_num(m.group(3))
                if n:
                    out.append((m.group(1).strip(), m.group(2).strip(), n, m.group(4).strip()))
    # 'A, B, C bán theo bó, 1 bó = 10 cây'
    for lb, text in biz.rd.conventions:
        m = re.match(r"^(.*?) bán theo (\S+), 1 (\S+) = ([\d.,]+) (\S+?)[;.]", text)
        if m:
            for nm in m.group(1).split(", "):
                out.append((nm.strip(), m.group(3), parse_num(m.group(4)), m.group(5)))
    return out


def _table_conversions(biz: Biz) -> list:
    """[(product ref, is_code, big unit, n, small unit or None, base unit or None)]"""
    out = []
    for tb in biz.tables:
        if tb.name in (biz.lines_table, biz.orders_table) or len(tb.df) > 20000:
            continue
        r = biz.roles[tb.name]
        fcol = r.get("factor")
        if not fcol:
            for c in tb.columns:
                if split_name(c) in FACTOR_NAMES:
                    fcol = c
                    break
        if not fcol:
            continue
        ucol = r.get("big_unit") or r.get("imp_unit")
        if not ucol:
            for c in tb.columns:
                if split_name(c) in BIGUNIT_NAMES:
                    ucol = c
                    break
        if not ucol:
            ucol = r.get("unit")
        pcol_code, pcol_name = r.get("prod_code"), (r.get("prod_name") or r.get("name"))
        bcol = r.get("base_unit")
        if bcol == ucol:
            bcol = None
        if not ucol and bcol:
            # catalogue style: 'ĐVT' + free text '1 thùng = 4 bịch'
            ucol = None
        if not (pcol_code or pcol_name):
            continue
        for _, row in tb.df.iterrows():
            ref = _s(row[pcol_code]) if pcol_code else _s(row[pcol_name])
            if not ref:
                continue
            raw = row[fcol]
            base = _s(row[bcol]) if bcol else None
            if isinstance(raw, str) and "=" in raw:
                for part in re.split(r"[;,]", raw):
                    m = re.match(r"^\s*1\s+(.+?)\s*=\s*([\d.,]+)\s*(.+?)\s*$", part)
                    if m and parse_num(m.group(2)):
                        out.append((ref, bool(pcol_code), m.group(1), parse_num(m.group(2)), m.group(3), base))
                continue
            n, small = _parse_factor(raw)
            big = _s(row[ucol]) if ucol else None
            if n and big:
                out.append((ref, bool(pcol_code), big, n, small, base))
    return out


def enrich(biz: Biz) -> None:
    """Attach product keys, groups and base quantities to biz.lines; build lexicons."""
    biz.conv = {}
    biz.base_unit = {}
    biz.prod = {}  # pkey -> dict(code, name, group, base_unit)
    biz.name2key = {}
    cat = _find_catalog(biz)
    biz.catalog_table = cat.name if cat is not None else ""
    code2name, name2code, code2group, name2group, key2base = {}, {}, {}, {}, {}
    key2cost = {}
    if cat is not None:
        r = biz.roles[cat.name]
        name_c, code_c = r.get("name") or r.get("prod_name"), r.get("prod_code")
        gcol, bcol = r.get("group"), (r.get("base_unit") or r.get("unit"))
        ccol = r.get("cost")
        brc = next((c for c in cat.columns if fold(cat.meanings.get(c, "")) == "hang" and "ã" in cat.meanings.get(c, "")), None) \
            or next((c for c in cat.columns if c.strip() == "Hãng" or split_name(c) in ("thuong hieu", "nhan hieu", "brand")), None)
        szc = next((c for c in cat.columns if fold(cat.meanings.get(c, "")).startswith("so cua sua") or split_name(c) in ("so / size", "size", "so/size")), None)
        alc = next((c for c in cat.columns if fold(cat.meanings.get(c, "")).startswith("ten goi khac") or split_name(c) in ("ten goi khac", "ten khac")), None)
        key2attr, aliases = {}, {}
        for _, row in cat.df.iterrows():
            nm, cd = _s(row[name_c]), (_s(row[code_c]) if code_c else None)
            if not nm:
                continue
            if cd:
                code2name[cd] = nm
                name2code[squash(nm)] = cd
            g = _s(row[gcol]) if gcol else None
            b = _s(row[bcol]) if bcol else None
            k = cd or squash(nm)
            if ccol:
                cv = parse_num(row[ccol])
                if cv is not None:
                    key2cost[k] = cv
            key2attr[k] = (_s(row[brc]) if brc else None, _s(row[szc]) if szc else None)
            if alc and not is_blank(row[alc]):
                for al in str(row[alc]).split(","):
                    if al.strip():
                        aliases[al.strip()] = k
            if g:
                (code2group if cd else name2group)[k] = g
            if b:
                key2base[k] = b
    biz.catalog_keys = set(code2name) if code2name else set(name2group) | set(key2base)
    L = biz.lines
    lines_have_code = L is not None and "prod_code" in L
    lines_have_name = L is not None and "prod_name" in L
    use_code = lines_have_code or (bool(name2code) and lines_have_name)
    biz.prod_has_code = bool(use_code)

    def key_of(name, code):
        if use_code:
            if code:
                return code
            if name:
                return name2code.get(squash(name)) or ("~" + squash(name))
            return None
        return squash(name) if name else None

    if L is not None and (lines_have_code or lines_have_name):
        names = L["prod_name"] if lines_have_name else pd.Series([None] * len(L), index=L.index)
        codes = L["prod_code"] if lines_have_code else pd.Series([None] * len(L), index=L.index)
        if use_code and not lines_have_code:
            nk = names.map(lambda v: squash(v) if v else None)
            L["pkey"] = nk.map(lambda k: (name2code.get(k) or ("~" + k)) if k else None)
        elif use_code:
            L["pkey"] = codes
            if lines_have_name:
                miss = L["pkey"].isna() & names.notna()
                if miss.any():
                    L.loc[miss, "pkey"] = names[miss].map(lambda v: name2code.get(squash(v)) or ("~" + squash(v)))
        else:
            L["pkey"] = names.map(lambda v: squash(v) if v else None)
        # display names: catalogue name, else most frequent spelling in the sales lines
        disp = {}
        if lines_have_name:
            vc = L.groupby(["pkey", "prod_name"], sort=False).size().reset_index(name="n").sort_values("n", ascending=False)
            for k, nm in zip(vc["pkey"], vc["prod_name"]):
                disp.setdefault(k, nm)
        for k in L["pkey"].dropna().unique():
            nm = code2name.get(k) if use_code else None
            biz.prod[k] = {"code": k if (use_code and not str(k).startswith("~")) else None,
                           "name": nm or disp.get(k) or str(k),
                           "group": (code2group.get(k) if use_code else name2group.get(k)),
                           "base_unit": key2base.get(k)}
        if "group" not in L or L["group"].isna().all():
            gm = {k: v["group"] for k, v in biz.prod.items()}
            if any(gm.values()):
                L["group"] = L["pkey"].map(gm)
        else:
            # the sales book itself carries the group of each item
            first = L.dropna(subset=["pkey"]).groupby("pkey")["group"].agg(lambda s: next((x for x in s if not is_blank(x)), None))
            for k, g in first.items():
                if k in biz.prod and not biz.prod[k].get("group") and g:
                    biz.prod[k]["group"] = str(g).strip()
    # products that never sold still belong to the catalogue
    if cat is not None:
        for nk, cd in list(name2code.items()):
            if cd not in biz.prod:
                biz.prod[cd] = {"code": cd, "name": code2name[cd], "group": code2group.get(cd), "base_unit": key2base.get(cd)}
        if not name2code:
            r = biz.roles[cat.name]
            name_c = r.get("name") or r.get("prod_name")
            for nm in cat.df[name_c].map(_s).dropna():
                k = squash(nm)
                if not use_code and k not in biz.prod:
                    biz.prod[k] = {"code": None, "name": nm, "group": name2group.get(k), "base_unit": key2base.get(k)}
    for k, cv in key2cost.items():
        if k in biz.prod:
            biz.prod[k]["cost"] = cv
    if cat is not None:
        for k, (br, sz) in key2attr.items():
            if k in biz.prod:
                if br:
                    biz.prod[k]["brand"] = br
                if sz:
                    biz.prod[k]["size"] = sz
    biz.spellings = {}
    for k, v in biz.prod.items():
        biz.name2key[squash(v["name"])] = k
        biz.spellings[v["name"]] = k
    if cat is not None:
        for al, k in aliases.items():
            if k in biz.prod:
                biz.spellings.setdefault(al, k)
    if L is not None and lines_have_name:
        for k, nm in zip(L["pkey"], L["prod_name"]):
            if k is not None and nm:
                biz.name2key.setdefault(squash(nm), k)
                biz.spellings.setdefault(nm, k)

    # ---- unit conversion
    edges = {}  # pkey -> {unit_folded: (n, small_unit_folded or None)}
    for nm, big, n, small in _readme_conversions(biz):
        k = biz.name2key.get(squash(nm))
        if k is None:
            k = key_of(nm, None)
        if k is None:
            continue
        edges.setdefault(k, {})[fold(big)] = (n, fold(small) if small else None)
        biz.prod.setdefault(k, {"code": None, "name": nm, "group": None, "base_unit": None})
    for ref, is_code, big, n, small, base in _table_conversions(biz):
        k = ref if (is_code and use_code) else (biz.name2key.get(squash(ref)) or key_of(ref, None))
        if k is None:
            continue
        edges.setdefault(k, {})[fold(big)] = (n, fold(small) if small else None)
        if base and k in biz.prod and not biz.prod[k].get("base_unit"):
            biz.prod[k]["base_unit"] = base

    def resolve(k, unit, depth=0):
        e = edges.get(k, {})
        if unit not in e or depth > 5:
            return 1.0
        n, small = e[unit]
        if small is None or small not in e:
            return n
        return n * resolve(k, small, depth + 1)

    for k, e in edges.items():
        smalls = [s for (_n, s) in e.values() if s and s not in e]
        if smalls and k in biz.prod and not biz.prod[k].get("base_unit"):
            biz.prod[k]["base_unit"] = smalls[0]
        for u in e:
            biz.conv[(k, u)] = resolve(k, u)
    if L is not None and "pkey" in L and "qty" in L:
        if "unit" in L:
            uf = L["unit"].map(lambda v: fold(v) if not is_blank(v) else "")
            fac = np.array([biz.conv.get((k, u), 1.0) for k, u in zip(L["pkey"], uf)], dtype="float64")
            L["unit_f"] = uf
        else:
            fac = np.ones(len(L))
        calc = L["qty"].to_numpy(dtype="float64") * fac
        if "qty_base" in L and L["qty_base"].notna().mean() > 0.9:
            given = L["qty_base"].to_numpy(dtype="float64")
            ok = np.isclose(given, calc) | np.isnan(given) | np.isnan(calc)
            biz.conv_agree = float(ok.mean())
            L["qty_base"] = np.where(np.isnan(given), calc, given)
        else:
            L["qty_base"] = calc
            biz.conv_agree = None
        # base unit per product = unit of lines with factor 1 when unknown
        if "unit" in L:
            base_rows = L[fac == 1.0]
            for k, u in base_rows.groupby("pkey")["unit"].agg(lambda s: s.dropna().map(lambda x: str(x).strip()).mode().iloc[0] if s.notna().any() else None).items():
                if k in biz.prod and not biz.prod[k].get("base_unit") and u:
                    biz.prod[k]["base_unit"] = u
    biz.lines = L
    _drugs(biz, use_code)
    _customer_groups(biz)
    staff_names(biz)


def _any_group_values(biz: Biz) -> bool:
    """True when some column anywhere holds customer-tier values (lẻ / sỉ / thân thiết ...)."""
    from .extra import GROUP_VALUES
    for tb in biz.tables:
        for c in tb.columns:
            u = tb.df[c].dropna().unique()
            if 2 <= len(u) <= 8:
                vals = {fold(v) for v in u if not is_blank(v)}
                if len(vals) >= 2 and vals <= GROUP_VALUES:
                    return True
    return False


def _customer_groups(biz: Biz) -> None:
    """Join customer master data (group) onto orders."""
    biz.cust_table = ""
    biz.group_col_by_values = _any_group_values(biz)
    O = biz.orders
    if O is None or "cust" not in O:
        return
    for tb in biz.tables:
        if tb.name in (biz.lines_table, biz.orders_table):
            continue
        r = biz.roles[tb.name]
        cid = r.get("cust_id")
        if not cid or "order_id" in r or "prod_name" in r and "qty" in r:
            continue
        if tb.df[cid].nunique() < 0.95 * len(tb.df):
            continue
        biz.cust_table = tb.name
        gcol = r.get("cust_group")
        if not gcol:
            for c in tb.columns:
                if split_name(c) in ("nhom", "nhom khach hang", "loai khach", "hang", "loai khach hang", "nhom kh", "phan loai",
                                     "kieu khach", "kieu khach hang", "hang khach", "phan hang", "loai kh", "hang thanh vien"):
                    gcol = c
                    break
        if not gcol:
            from .extra import GROUP_VALUES
            for c in tb.columns:
                vals = {fold(v) for v in tb.df[c].dropna().unique()[:50] if not is_blank(v)}
                if 2 <= len(vals) <= 8 and vals <= GROUP_VALUES:
                    gcol = c
                    break
        ncol = r.get("name") or r.get("cust_name")
        if not ncol:
            for c in tb.columns:
                if split_name(c) in ("ten goi", "ten", "ho ten", "ten khach hang", "ten kh", "khach hang", "khach hang mua",
                                     "ten khach", "ho va ten", "khach le/si"):
                    ncol = c
                    break
        ids = tb.df[cid].map(_s)
        if gcol:
            gm = dict(zip(ids, tb.df[gcol].map(_s)))
            if biz.has_cust_code:
                if "cust_group" in O and O["cust_group"].map(lambda v: not is_blank(v)).any():
                    return
                O["cust_group"] = O["cust"].map(gm)
                if biz.lines is not None:
                    biz.lines["cust_group"] = biz.lines["cust"].map(gm)
        if ncol and biz.has_cust_code:
            nm = dict(zip(ids, tb.df[ncol].map(_s)))
            lab = O["cust"].map(nm)
            O["cust_label"] = lab.where(lab.notna(), O["cust_label"])
        return


def _truthy(v) -> bool:
    return fold(v) in ("true", "1", "1.0", "x", "co", "yes", "y", "ke don", "thuoc ke don")


def _drugs(biz: Biz, use_code: bool) -> None:
    """Prescription flag and active ingredient per product (pharmacies)."""
    biz.drug_table = ""
    for tb in biz.tables:
        r = biz.roles[tb.name]
        if not ("rx" in r or "ingredient" in r) or len(tb.df) > 20000:
            continue
        kc = r.get("prod_code") or r.get("prod_name") or r.get("name")
        if not kc:
            continue
        biz.drug_table = tb.name
        for _, row in tb.df.iterrows():
            ref = _s(row[kc])
            if not ref:
                continue
            k = ref if (use_code and "prod_code" in r) else biz.name2key.get(squash(ref))
            if k is None or k not in biz.prod:
                continue
            if "rx" in r:
                biz.prod[k]["rx"] = _truthy(row[r["rx"]])
            if "ingredient" in r:
                biz.prod[k]["ingredient"] = _s(row[r["ingredient"]])
        return


def staff_names(biz: Biz) -> None:
    """biz.staff_lex: full staff name -> staff code used on orders (POS exports keep them in a separate list)."""
    biz.staff_lex = {}
    for tb in biz.tables:
        cols = {split_name(c): c for c in tb.columns}
        code_c = cols.get("ma nhan vien") or cols.get("ma nv") or cols.get("ma nhan vien ban")
        name_c = cols.get("ten goi") or cols.get("ho ten") or cols.get("ten nhan vien") or cols.get("ten")
        if code_c and name_c and len(tb.df) < 500 and tb.df[code_c].nunique() == len(tb.df):
            for a, b in zip(tb.df[code_c], tb.df[name_c]):
                if not is_blank(a) and not is_blank(b):
                    biz.staff_lex[str(b).strip()] = str(a).strip()
            return
