"""Text, number and date normalisation helpers."""
from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata

_EXCEL_EPOCH = dt.datetime(1899, 12, 30)


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def fold(s) -> str:
    """Lowercase, strip Vietnamese diacritics, collapse whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFD", str(s).lower().replace("đ", "d").replace("Đ", "d"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


def norm_answer(s) -> str:
    """Official string normalisation used by the scorer (diacritics kept)."""
    s = nfc(str(s)).lower()
    s = "".join(c if (c.isalnum() or c.isspace()) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def squash(s) -> str:
    """Spelling-insensitive key: no diacritics, case, spaces or punctuation."""
    return re.sub(r"[^a-z0-9]", "", fold(s))


def is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


_NUM_RE = re.compile(r"^[-+]?\d[\d.,]*$")


def parse_num(v):
    """Parse messy numbers: '560,000', '25.000đ', '25k', '1tr440', '1 chai', 6.0."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip().lower()
    if not s:
        return None
    s = s.replace(" ", " ").replace("−", "-")
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    if s.startswith("-"):
        neg, s = not neg, s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()
    s = re.sub(r"\s*(vnđ|vnd|đồng|dong|đ|₫)\s*$", "", s).strip()
    s = s.replace("\u202f", " ").replace("\u2009", " ")
    if re.fullmatch(r"\d{1,3}(?: \d{3})+", s):
        s = s.replace(" ", "")  # '531 500': a space between the thousands
    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*(tr|triệu|trieu)\s*(\d{0,3})$", s)
    if m:
        base = float(m.group(1).replace(",", "."))
        tail = m.group(3)
        val = base * 1_000_000
        if tail:
            val += int(tail.ljust(3, "0")) * 1000
        return -val if neg else val
    m = re.match(r"^(\d+(?:[.,]\d+)?)\s*(k|nghìn|ngàn)$", s)
    if m:
        val = float(m.group(1).replace(",", ".")) * 1000
        return -val if neg else val
    m = re.match(r"^(\d[\d.,]*)\s+\D.*$", s)  # '1 chai', '10 hộp'
    if m:
        s = m.group(1)
    if not _NUM_RE.match(s):
        return None
    val = _parse_plain(s)
    if val is None:
        return None
    return -val if neg else val


def _parse_plain(s: str):
    s = s.rstrip(".,")
    has_c, has_d = "," in s, "." in s
    try:
        if has_c and has_d:
            if s.rfind(",") > s.rfind("."):
                return float(s.replace(".", "").replace(",", "."))
            return float(s.replace(",", ""))
        if has_c or has_d:
            sep = "," if has_c else "."
            parts = s.split(sep)
            if all(len(p) == 3 for p in parts[1:]) and 1 <= len(parts[0]) <= 3 and (len(parts) > 2 or parts[0] != "0"):
                return float("".join(parts))
            if len(parts) == 2:
                return float(parts[0] + "." + parts[1])
            return None
        return float(s)
    except ValueError:
        return None


_DATE_PATTERNS = [
    (re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ t](\d{1,2}):(\d{2})(?::(\d{2}))?)?"), "ymd"),
    (re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})(?:[ ,]+(\d{1,2}):(\d{2})(?::(\d{2}))?)?"), "dmy"),
    (re.compile(r"^(\d{1,2})\s*(?:thg|tháng|thang|th)\.?\s*(\d{1,2})[\s,/]+(\d{4})"), "dmy"),
    (re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2})"), "ymd"),
]
_DM_RE = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})$")
_DM_TXT_RE = re.compile(r"^(\d{1,2})\s*(?:thg|tháng|thang|th)\.?\s*(\d{1,2})$")


def parse_datetime(v, year_hint: int | None = None):
    """Return datetime or None. Handles Excel serials, dd/mm/yyyy, ISO, '1 thg 6 2025', 'dd/mm'."""
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return v
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        if 20000 <= v <= 80000:
            return _EXCEL_EPOCH + dt.timedelta(days=float(v))
        return None
    s = str(v).strip().lower()
    if not s:
        return None
    if re.fullmatch(r"\d{5}(\.\d+)?", s):
        f = float(s)
        if 20000 <= f <= 80000:
            return _EXCEL_EPOCH + dt.timedelta(days=f)
    for rx, kind in _DATE_PATTERNS:
        m = rx.match(s)
        if m:
            g = m.groups()
            try:
                if kind == "ymd":
                    y, mo, d = int(g[0]), int(g[1]), int(g[2])
                else:
                    d, mo, y = int(g[0]), int(g[1]), int(g[2])
                hh = int(g[3]) if len(g) > 3 and g[3] else 0
                mi = int(g[4]) if len(g) > 4 and g[4] else 0
                ss = int(g[5]) if len(g) > 5 and g[5] else 0
                return dt.datetime(y, mo, d, hh, mi, ss)
            except ValueError:
                return None
    if year_hint:
        m = _DM_RE.match(s) or _DM_TXT_RE.match(s)
        if m:
            try:
                return dt.datetime(year_hint, int(m.group(2)), int(m.group(1)))
            except ValueError:
                return None
    return None


def parse_time(v):
    """Return (h, m, s) from time-like values or None."""
    if v is None:
        return None
    if isinstance(v, dt.datetime):
        return (v.hour, v.minute, v.second)
    if isinstance(v, dt.time):
        return (v.hour, v.minute, v.second)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None
        frac = float(v) % 1
        secs = int(round(frac * 86400))
        return (secs // 3600 % 24, secs % 3600 // 60, secs % 60)
    s = str(v).strip().lower()
    m = re.search(r"(\d{1,2})[:h](\d{2})(?::(\d{2}))?", s)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
    return None
