"""Question normalisation and period extraction."""
from __future__ import annotations

import calendar as _cal
import datetime as dt
import re
from dataclasses import dataclass

from .textnorm import fold, nfc

_ONSET = r"(?:ngh|ng|nh|gh|gi|kh|ph|th|tr|ch|qu|b|c|d|g|h|k|l|m|n|p|r|s|t|v|x)?"
_NUC = (r"(?:uye|uya|uyu|oai|oay|uay|uoi|uou|ieu|yeu|oeo|uao|uay|uau|ai|ao|au|ay|eo|eu|ia|ie|iu|oa|oe|oi|ua|ue|ui|uo|uu|uy|ya|ye|"
        r"a|e|i|o|u|y)")
_CODA = r"(?:ng|nh|ch|c|m|n|p|t)?"
_SYL = re.compile(rf"^{_ONSET}{_NUC}{_CODA}$")
_TELEX_HINT = re.compile(r"dd|aa|ee|oo|aw|ow|uw|w|[sfrxj]$")


def is_syllable(tok: str) -> bool:
    return bool(_SYL.match(tok))


def telex_fold(tok: str) -> str:
    """Undo raw telex keystrokes: 'toongr' -> 'tong', 'dduwowcj' -> 'duoc'."""
    t = tok
    t = re.sub(r"[sfrxj]+$", "", t) if len(t) > 2 else t
    t = t.replace("dd", "d")
    t = t.replace("uwow", "uo").replace("uow", "uo")
    t = re.sub(r"([aeou])w", r"\1", t)
    t = t.replace("aa", "a").replace("ee", "e").replace("oo", "o")
    t = t.replace("w", "u")
    return t


_ABBR = [
    (r"\bd\.?thu\b", "doanh thu"), (r"\bdoanh so\b", "doanh thu"), (r"\brevenue\b", "doanh thu"),
    (r"\bb\.?nhi(?:e)?u\b", "bao nhieu"), (r"\bbn\b", "bao nhieu"), (r"\bbao nhiu\b", "bao nhieu"),
    (r"\b(bao nhieu|may|nhieu) ten\b", r"\1 tien"),  # 'bao nhiêu tền'
    (r"\bnhiu\b", "nhieu"), (r"\bdc\b", "duoc"), (r"\bdk\b", "duoc"), (r"\bdx\b", "duoc"),
    (r"\bh\.?qua\b", "hom qua"), (r"\bh\.?nay\b", "hom nay"), (r"\bbua nay\b", "hom nay"), (r"\bbua qua\b", "hom qua"),
    (r"\bhom qa\b", "hom qua"), (r"\bngay hom nay\b", "hom nay"), (r"\bo do\b", "don"), (r"\bsop\b(?! nhat)", "shop"), (r"\bbua\b(?! nay\b| qua\b| kia\b)", "ngay"), (r"\bquy(\d)\b", r"quy \1"), (r"\bcat hang\b", "nhap hang"),
    (r"\bdanh hang(?: ve)?\b", "nhap hang"), (r"\blay hang(?: ve)?\b", "nhap hang"),
    (r"\bbua kia\b", "hom kia"), (r"\bngay hom (nay|qua)\b", r"hom \1"),
    (r"\btrc\b", "truoc"), (r"\bvs\b", "voi"), (r"\bsp\b", "san pham"), (r"\bkh\b", "khach hang"),
    (r"\bko\b", "khong"), (r"\bhok\b", "khong"), (r"\bhong\b", "khong"), (r"\bkhg\b", "khong"), (r"\bk\b(?= \?|$)", "khong"),
    (r"\btotal\b", "tong cong"), (r"\bbill\b", "don"), (r"\bhoa don\b", "don"), (r"\bdon hang\b", "don"),
    (r"\border\b", "don"), (r"\bth truoc\b", "thang truoc"), (r"\bthang roi\b", "thang truoc"),
    (r"\bthang vua (?:roi|qua)\b", "thang truoc"), (r"\btuan (?:roi|vua roi|vua qua|qua)\b", "tuan truoc"),
    (r"\bnam (?:ngoai|roi|vua roi)\b", "nam truoc"), (r"\bphan tram\b", "%"), (r"\bgium\b", "giup"),
    (r"\bntn\b", "nhu the nao"), (r"\bcty\b", "cong ty"), (r"\bsl\b", "so luong"), (r"\bnv\b", "nhan vien"),
    (r"\bck\b", "chuyen khoan"), (r"\btm\b", "tien mat"), (r"\bhuy bo\b", "huy"), (r"\bhuỷ\b", "huy"),
]


_TEEN = {
    "mik": "minh", "mk": "minh", "mjk": "minh", "wan": "quan", "dh": "don", "dhang": "don", "ko": "khong", "hok": "khong",
    "hem": "khong", "khom": "khong", "hk": "khong", "hong": "khong", "dc": "duoc", "dk": "duoc", "dx": "duoc",
    "bnhiu": "bao nhieu", "bnhieu": "bao nhieu", "bn": "bao nhieu", "bnh": "bao nhieu", "bnieu": "bao nhieu", "nhiu": "nhieu",
    "gium": "giup", "jum": "giup", "dum": "giup", "zum": "giup", "jup": "giup", "trog": "trong", "ncc": "nha cung cap",
    "ds": "doanh so", "zo": "vao", "wa": "qua", "j": "gi", "ji": "gi", "bik": "biet", "bit": "biet", "rui": "roi", "mi": "may",
    "r": "roi", "zay": "vay", "dzay": "vay", "v": "vay", "za": "a", "dza": "a", "ni": "nay", "hwa": "hom qua",
    "nhe": "", "nhen": "", "hen": "", "ne": "", "nak": "", "ak": "a", "ah": "a", "uh": "a", "tks": "", "thanks": "",
    "sp": "san pham", "kh": "khach hang", "sl": "so luong", "nv": "nhan vien", "cty": "cong ty", "trc": "truoc", "vs": "voi",
    "dthu": "doanh thu", "revenue": "doanh thu", "total": "tong cong", "bill": "don", "order": "don", "thg": "thang",
    "ntn": "nhu the nao", "hnay": "hom nay", "hqua": "hom qua", "pt": "%", "nhiêu": "nhieu",
    "k": "khong", "rua": "vay", "mo": "nao", "ngui": "nguoi", "nguoy": "nguoi", "kon": "con", "bq": "binh quan", "bnhui": "bao nhieu", "doanhthu": "doanh thu", "cko": "cho", "kac": "cac", "ko": "khong",
    "zi": "vay", "nhanh": "", "xin": "", "hsd": "han su dung", "tieng": "tien", "duot": "duoc", "bin": "binh", "chong": "trong", "q": "quy", "nhieudon": "nhieu don", "nhut": "nhat", "kua": "cua", "alo": "", "roo": "roi", "hi": "", "hihi": "",
    "chot": "ban", "nhiu": "nhieu", "dzo": "vao", "hem": "khong", "ui": "", "oi": "", "nhoa": "", "ah": "",
}
_KEYWORDS = set("""nhat len tro doanh thu don hang bao nhieu thang tuan ngay khach tien ban duoc huy nao tong cong trung binh nhom
truoc hom nay qua cao thap tang giam hon kem lien tuc mat san pham chay ti le phan tram mua top liet ke quy nam dau den het""".split())


def _fix_typo(w: str) -> str:
    """Adjacent-letter transposition into a known keyword: 'nhta' -> 'nhat', 'lne' -> 'len'."""
    hits = {w[:i] + w[i + 1] + w[i] + w[i + 2:] for i in range(len(w) - 1)} & _KEYWORDS
    return hits.pop() if len(hits) == 1 else w


def _year4(y: str) -> str:
    return y if len(y) == 4 else "20" + y


# accent-aware rewrites applied before diacritics are stripped: slang whose folded form is ambiguous,
# and North/South names of the same thing (rulebook §8: names for the same thing are the same thing)
_PRE = [
    (r"\bkó\b", "có"), (r"\bkòn\b", "còn"), (r"\bkủa\b", "của"), (r"\bkả\b", "cả"), (r"\bkác\b", "các"), (r"\bzô\b", "vô"),
    (r"\bdzô\b", "vô"), (r"\bgồi\b", "rồi"), (r"\bgòi\b", "rồi"), (r"\bhông\b", "không"), (r"\bhổng\b", "không"),
    (r"\bmỳ\b", "mì"), (r"\blãi\b", "loinhuan"), (r"\bph[oô] mát\b", "phô mai"),
    (r"\bđậu phụ\b", "đậu hũ"), (r"\blạc\b", "đậu phộng"), (r"\brán\b", "chiên"), (r"\blợn\b", "heo"), (r"\bngao\b", "nghêu"),
    (r"\bcá (?:quả|tràu|chuối)\b", "cá lóc"), (r"\bxây trát\b", "xây tô"), (r"\bgăng tay\b", "bao tay"),
    (r"\btất chân\b", "bao chân"), (r"\bhoa quả\b", "trái cây"), (r"\bcốc sạc\b", "củ sạc"),
    (r"\bkệ điện thoại\b", "giá đỡ điện thoại"), (r"\bmì chính\b", "bột ngọt"), (r"\bxà phòng\b", "xà bông"),
    (r"\bngô\b(?= (?:nếp|ngọt|luộc|nướng|rang|xào|hạt))", "bắp"), (r"\bdứa\b", "thơm"), (r"\bthìa\b", "muỗng"),
    (r"\bnem rán\b", "chả giò"), (r"\bgiò lụa\b", "chả lụa"), (r"\bmì tôm\b", "mì gói"), (r"\bruốc\b", "chà bông"),
    (r"\bmộc nhĩ\b", "nấm mèo"), (r"\bvừng\b", "mè"), (r"\bmướp đắng\b", "khổ qua"), (r"\bbí ngô\b", "bí đỏ"),
    (r"\bbánh đa nem\b", "bánh tráng"), (r"\bquẩy\b", "giò cháo quẩy"), (r"\bcủ đậu\b", "củ sắn"),
    (r"\bsăm\b", "ruột"), (r"\blốp\b", "vỏ"), (r"\bphanh\b", "thắng"), (r"\bxích\b", "sên"), (r"\bbóng điện\b", "bóng đèn"),
]
_PRE = [(re.compile(a), b) for a, b in _PRE]
_PRE_HINT = re.compile("|".join(p.pattern for p, _ in _PRE))
_MONTH_WORDS = {"mot": 1, "gieng": 1, "hai": 2, "ba": 3, "tu": 4, "bon": 4, "sau": 6, "bay": 7, "tam": 8, "chin": 9,
                "muoi mot": 11, "muoi hai": 12, "chap": 12, "muoi": 10}
_NUM_WORDS = {"hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}


def normalize(q: str) -> str:
    """Folded, typo-repaired, abbreviation-expanded question text."""
    low = nfc(str(q)).lower()
    if _PRE_HINT.search(low):
        for rx, rep in _PRE:
            low = rx.sub(rep, low)
    low = re.sub(r"\btháng (mười một|mười hai|giêng|chạp|một|hai|ba|tư|bốn|sáu|bảy|tám|chín|mười)\b(?! \d)",
                 lambda m: f"tháng {_MONTH_WORDS[fold(m.group(1))]}", low)
    low = re.sub(r"\btop (hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\b", lambda m: f"top {_NUM_WORDS[fold(m.group(1))]}", low)
    low = re.sub(r"\b(hai|ba|bốn|năm|sáu|bảy)(?= (?:khách|mặt hàng|món |sản phẩm|tháng liên|tháng liền))",
                 lambda m: str(_NUM_WORDS[fold(m.group(1))]), low)
    s = fold(low)
    s = s.replace("≥", " >= ").replace("≤", " <= ").replace("→", " ").replace("–", "-")
    s = re.sub(r"[^a-z0-9 %/.,:\-<>=()+']", " ", s)
    s = re.sub(r"(?<=\d)\.(?=\d{3}(?:\D|$))", "", s)  # 1.000.000 -> 1000000
    s = re.sub(r"\b(\d+)[.,](\d)\s*(?:tr|trieu)\b", lambda m: str(int(m.group(1)) * 1000000 + int(m.group(2)) * 100000), s)
    s = re.sub(r"\b(\d+)\s*tr\s*(\d{1,3})\b", lambda m: str(int(m.group(1)) * 1000000 + int(m.group(2).ljust(3, "0")) * 1000), s)
    s = re.sub(r"\b(\d+)\s*(?:tr|trieu)\b", lambda m: str(int(m.group(1)) * 1000000), s)
    s = re.sub(r"\bt[iy] n[ee]\b", "ti le", s)
    s = re.sub(r"\b(\d+)\s*k\b", lambda m: str(int(m.group(1)) * 1000), s)
    s = re.sub(r"(\d)\s*(?:d|dong|vnd)\b(?![a-z])", r"\1 dong", s)
    # month / year shorthands: th6/2025, t4/25, 6-2025, 8/26, 12/05/26
    s = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b", lambda m: f"{m.group(1)}/{m.group(2)}/20{m.group(3)}", s)
    s = re.sub(r"\b(?:thg|th|t)\.?\s*(\d{1,2})\s*[/\-.]\s*(\d{4}|\d{2})\b", lambda m: f"thang {int(m.group(1))}/{_year4(m.group(2))}", s)
    s = re.sub(r"\b(?:thang|quy)\s*(\d{1,2})\s*[/\-.]\s*(\d{2})\b", lambda m: m.group(0)[:-2] + "20" + m.group(2), s)
    s = re.sub(r"(?<![\d/.\-])(\d{1,2})\s*-\s*(20\d{2})(?![\d/\-])", r"\1/\2", s)
    s = re.sub(r"(?<![\d/.\-])(\d{1,2})/(\d{2})(?![\d/])",
               lambda m: f"{m.group(1)}/20{m.group(2)}" if (1 <= int(m.group(1)) <= 12 and 20 <= int(m.group(2)) <= 35) else m.group(0), s)
    s = re.sub(r"\bq\s*([1-4])\s*[/\-.]\s*(\d{2,4})\b", lambda m: f"quy {m.group(1)}/{_year4(m.group(2))}", s)
    s = re.sub(r"\b(?:thg|th)\s*(\d{1,2})\b", r"thang \1", s)
    s = re.sub(r"\b(\d+)\s*cu\b", lambda m: str(int(m.group(1)) * 1000000), s)
    s = re.sub(r"\bthang (\d{1,2}), ?(\d{1,2})(?:,| va) ?(\d{1,2}) ?/ ?(20\d{2})\b", r"thang \1/\4, thang \2/\4, thang \3/\4", s)
    s = re.sub(r"\b(toi|den) (?:bay )?(?:chu|gio)\b", r"\1 hom nay", s)
    s = re.sub(r"\bbay (?:chu|gio)\b", "hom nay", s)
    s = re.sub(r"\bdem so (?:sanh )?voi\b", "so voi", s)
    s = re.sub(r"\b(?:be|chi|anh|em|ban|ad|admin|shop|sep) oi\b,?", " ", s)
    if re.search(r"thang|tuan|ngay gan|ngay qua|quy|nam nay|dau nam", s):
        s = re.sub(r",? ?\b(?:tinh|ke|gom) (?:ca|luon|toi|den)(?: het)?(?: ngay)? hom nay\b(?: thi)?", " ", s)
    s = re.sub(r"\bthang (\d{1,2}), ?(\d{1,2})(?:,| va) ?(\d{1,2}) nam (20\d{2})\b", r"thang \1/\4, thang \2/\4, thang \3/\4", s)
    s = re.sub(r",? ?(?:tinh|ke|gom) (?:ca|luon) hom nay\b", " ", s)
    s = re.sub(r"\b(?:thg|th)\s+(nay|truoc|roi|sau)\b", r"thang \1", s)
    s = re.sub(r"\b(?:hoa|hue) hong\b", "hoa_hong", s)
    toks = []
    for tok in s.split():
        m = re.fullmatch(r"([a-z]+)([.,:]*)", tok)
        if m:
            w = m.group(1)
            if w in _TEEN:
                w = _TEEN[w]
            elif len(w) > 2 and not is_syllable(w):
                w2 = re.sub(r"(.)\1+$", r"\1", w)  # 'naoo' -> 'nao'
                if w2 != w and (is_syllable(w2) or w2 in _TEEN):
                    w = _TEEN.get(w2, w2)
                elif _TELEX_HINT.search(w) and is_syllable(telex_fold(w)):
                    w = telex_fold(w)
                else:
                    w = _fix_typo(w)
            tok = w + m.group(2)
        toks.append(tok)
    s = " ".join(toks)
    for rx, rep in _ABBR:
        s = re.sub(rx, rep, s)
    s = s.replace("hoa_hong", "hoa hong")
    if re.search(r"thang|tuan|ngay gan|ngay qua|quy|nam nay|dau nam", s):
        s = re.sub(r",? ?\b(?:tinh|ke|gom) (?:ca|luon|toi|den)(?: het)?(?: ngay)? hom nay\b(?: thi)?", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .,:")
    return s


@dataclass
class Period:
    start: dt.date
    end: dt.date
    kind: str  # day | week | month | quarter | year | range | mtd | ytd | lastn | tet | upto
    span: tuple = (0, 0)
    text: str = ""
    grain: str = ""  # 'month' when the mention denotes a whole calendar month


def _month_range(y: int, m: int):
    return dt.date(y, m, 1), dt.date(y, m, _cal.monthrange(y, m)[1])


def _quarter_range(y: int, q: int):
    return dt.date(y, 3 * q - 2, 1), _month_range(y, 3 * q)[1]


def _resolve_year(mm_start: dt.date, today: dt.date, y: int | None, builder):
    return builder(y)


_D = r"(\d{1,2})"
_Y = r"(\d{4})"
_DATE = rf"{_D}\s*[/\-.]\s*{_D}\s*[/\-.]\s*{_Y}"
_DATE_TXT = rf"{_D}\s*thang\s*{_D}\s*(?:nam\s*)?{_Y}"
_MY = rf"thang\s*{_D}\s*(?:[/\-.]|nam)\s*{_Y}"


def _mk(y, m, d):
    try:
        return dt.date(int(y), int(m), int(d))
    except ValueError:
        return None


def extract_periods(s: str, today: dt.date, cal=None) -> list:
    """Find period mentions in a normalised question. Returns non-overlapping Periods in text order."""
    found: list = []

    def add(m, start, end, kind, grain=""):
        if start is None or end is None:
            return
        found.append(Period(start, end, kind, m.span(), m.group(0), grain))

    def free(m):
        a, b = m.span()
        return all(b <= p.span[0] or a >= p.span[1] for p in found)

    def scan(rx, fn):
        for m in re.finditer(rx, s):
            if free(m):
                fn(m)

    # explicit ranges
    scan(rf"tu\s*(?:ngay\s*)?{_DATE}\s*(?:den|toi|-)\s*(?:het\s*)?(?:ngay\s*)?{_DATE}",
         lambda m: add(m, _mk(m[3], m[2], m[1]), _mk(m[6], m[5], m[4]), "range"))
    scan(rf"tu\s*(?:dau\s*)?{_MY}\s*(?:den|toi|-)\s*(?:het\s*)?{_MY}",
         lambda m: add(m, _mk(m[2], m[1], 1), _month_range(int(m[4]), int(m[3]))[1] if 1 <= int(m[3]) <= 12 else None, "range"))
    # to-date forms
    scan(r"tu (?:ngay |mung )?(?:1/1|01/01)(?: nam nay)?(?: (?:den|toi)(?: het)?(?: ngay)? (?:hom nay|nay|gio))",
         lambda m: add(m, dt.date(today.year, 1, 1), today, "ytd"))
    scan(r"tu (?:ngay |mung )(?:1|01)(?: thang nay| dau thang)? (?:den|toi)(?: het)?(?: ngay)? hom nay",
         lambda m: add(m, today.replace(day=1), today, "mtd"))
    scan(r"dau nam(?: nay)? (?:den|toi)(?: het)?(?: ngay)? (?:hom nay|nay|gio)",
         lambda m: add(m, dt.date(today.year, 1, 1), today, "ytd"))
    scan(r"tu dau nam(?: nay)?(?: (?:den|toi)(?: het)?(?: ngay)? (?:hom nay|nay|gio))?",
         lambda m: add(m, dt.date(today.year, 1, 1), today, "ytd"))
    scan(r"tu dau thang(?: nay)?(?: (?:den|toi)(?: het)?(?: ngay)? (?:hom nay|nay|gio))?",
         lambda m: add(m, today.replace(day=1), today, "mtd"))
    scan(r"(\d+) ngay (?:gan nhat|gan day|vua qua|qua|tro lai day)",
         lambda m: add(m, today - dt.timedelta(days=int(m[1]) - 1), today, "lastn"))
    # full dates
    scan(rf"(?:den het ngay|tinh den het ngay|tinh den ngay|den ngay|het ngay)\s*{_DATE}",
         lambda m: add(m, dt.date(1900, 1, 1), _mk(m[3], m[2], m[1]), "upto"))
    scan(rf"(?:ngay|hom|bua)?\s*{_DATE}", lambda m: add(m, _mk(m[3], m[2], m[1]), _mk(m[3], m[2], m[1]), "day"))
    scan(rf"(?:ngay|hom|bua)?\s*{_DATE_TXT}", lambda m: add(m, _mk(m[3], m[2], m[1]), _mk(m[3], m[2], m[1]), "day"))

    def quarter(m):
        q, y = int(m[1]), int(m[2])
        if 1 <= q <= 4:
            a, b = _quarter_range(y, q)
            add(m, a, b, "quarter")

    scan(rf"quy\s*([1-4])\s*(?:[/\-.]|nam)\s*{_Y}", quarter)

    def month(m):
        mo, y = int(m[1]), int(m[2])
        if 1 <= mo <= 12:
            a, b = _month_range(y, mo)
            add(m, a, b, "month", "month")

    scan(_MY, month)
    scan(rf"(?<![\d/]){_D}\s*/\s*{_Y}(?![\d/])", month)

    def quarter_noyear(m):
        q = int(m[1])
        a, b = _quarter_range(today.year, q)
        if a > today:
            a, b = _quarter_range(today.year - 1, q)
        add(m, a, b, "quarter")

    scan(r"quy\s*([1-4])(?!\s*\d)", quarter_noyear)
    scan(rf"(?:ca |trong )?nam\s*{_Y}", lambda m: add(m, dt.date(int(m[1]), 1, 1), dt.date(int(m[1]), 12, 31), "year"))
    # relative
    scan(r"(?:tu dau )?thang nay,?(?: tinh)? (?:toi|den)(?: het)?(?: ngay)? hom nay", lambda m: add(m, today.replace(day=1), today, "mtd"))
    scan(r"hom nay", lambda m: add(m, today, today, "day"))
    scan(r"hom qua", lambda m: add(m, today - dt.timedelta(days=1), today - dt.timedelta(days=1), "day"))
    scan(r"hom kia", lambda m: add(m, today - dt.timedelta(days=2), today - dt.timedelta(days=2), "day"))
    monday = today - dt.timedelta(days=today.weekday())
    scan(r"tuan nay", lambda m: add(m, monday, today, "week"))
    scan(r"tuan truoc", lambda m: add(m, monday - dt.timedelta(days=7), monday - dt.timedelta(days=1), "week"))
    scan(r"thang nay", lambda m: add(m, today.replace(day=1), today, "mtd"))

    def prev_month(m):
        first = today.replace(day=1)
        last_prev = first - dt.timedelta(days=1)
        add(m, last_prev.replace(day=1), last_prev, "month", "month")

    scan(r"thang truoc", prev_month)
    if cal is not None:
        def tet(m):
            y = int(m[1]) if m[1] else None
            if y is None and m.group(0).endswith("nam nay"):
                y = today.year
            elif y is None and m.group(0).endswith("nam truoc"):
                y = today.year - 1
            rng = cal.tet_range(y, today)
            if rng:
                add(m, rng[0], rng[1], "tet")

        scan(r"(?:dip |mua |ky |trong )?tet(?: nguyen dan| am lich| am)?(?: (?:nam )?(\d{4})| nam nay| nam truoc)?", tet)

    def month_lastyear(m):
        mo = int(m[1])
        if 1 <= mo <= 12:
            a, b = _month_range(today.year - 1, mo)
            add(m, a, b, "month", "month")

    scan(r"thang\s*(\d{1,2})\s*(?:cua )?nam truoc", month_lastyear)

    scan(r"nam nay", lambda m: add(m, dt.date(today.year, 1, 1), today, "ytd"))
    scan(r"nam truoc", lambda m: add(m, dt.date(today.year - 1, 1, 1), dt.date(today.year - 1, 12, 31), "year"))

    def month_noyear(m):
        mo = int(m[1])
        if not 1 <= mo <= 12:
            return
        a, b = _month_range(today.year, mo)
        if a > today:
            a, b = _month_range(today.year - 1, mo)
        add(m, a, b, "month", "month")

    scan(r"thang\s*(\d{1,2})(?!\s*[/\d])", month_noyear)

    def day_noyear(m):
        d0 = _mk(today.year, m[2], m[1])
        if d0 and d0 > today:
            d0 = _mk(today.year - 1, m[2], m[1])
        add(m, d0, d0, "day")

    scan(rf"(?:ngay|hom|bua)\s*{_D}\s*/\s*{_D}(?![\d/])", day_noyear)
    tets = [p for p in found if p.kind == "tet"]
    if len(tets) == 1:
        m = re.search(r"(?:(\d+) ngay )?(?:ngay |lien |ngay lien )?truoc tet", s)
        if m and free(m):
            n = (tets[0].end - tets[0].start).days + 1
            if m[1] is None or int(m[1]) == n:
                found.append(Period(tets[0].start - dt.timedelta(days=n), tets[0].start - dt.timedelta(days=1), "range", m.span(), "truoc tet"))
    found.sort(key=lambda p: p.span[0])
    _share_years(found, s)
    return found


_NOYEAR = re.compile(r"(thang|quy)\s*\d{1,2}")
_LINK = re.compile(r"\s*(?:,|;|va|voi|roi|,\s*va|,\s*roi|den|toi|den het|toi het|-)?\s*")


def _share_years(found: list, s: str) -> None:
    """'th7, th8, th9/2025': a year written once in an enumeration belongs to every month (quarter) of it."""
    def unit(p):
        m = _NOYEAR.match(p.text)
        if p.kind == "month" and p.grain == "month" and p.text.startswith("thang") or re.fullmatch(r"\d{1,2}\s*/\s*\d{4}", p.text):
            return "month"
        if p.kind == "quarter":
            return "quarter"
        return None

    def bare(p):
        return _NOYEAR.fullmatch(p.text) is not None

    def idx(p):
        return p.start.month if unit(p) == "month" else (p.start.month - 1) // 3 + 1

    def rebuild(p, y):
        if unit(p) == "month":
            p.start, p.end = _month_range(y, p.start.month)
        else:
            p.start, p.end = _quarter_range(y, (p.start.month - 1) // 3 + 1)

    def linked(a, b):
        return unit(a) is not None and unit(a) == unit(b) and _LINK.fullmatch(s[a.span[1]:b.span[0]]) is not None

    anchored = [unit(p) is not None and not bare(p) and re.search(r"\d{4}", p.text) is not None for p in found]
    for i in range(len(found) - 2, -1, -1):   # '..., th8, th9/2025' : backwards from the year
        p, q = found[i], found[i + 1]
        if bare(p) and anchored[i + 1] and linked(p, q):
            rebuild(p, q.start.year if idx(p) < idx(q) else q.start.year - 1)
            anchored[i] = True
    for i in range(1, len(found)):            # 'th10/2025, th11, th12' : forwards
        p, q = found[i - 1], found[i]
        if bare(q) and not anchored[i] and anchored[i - 1] and linked(p, q):
            rebuild(q, p.start.year if idx(q) > idx(p) else p.start.year + 1)
            anchored[i] = True


def strip_spans(s: str, periods: list, repl: str = " <P> ") -> str:
    out, last = [], 0
    for p in periods:
        out.append(s[last:p.span[0]])
        out.append(repl)
        last = p.span[1]
    out.append(s[last:])
    return re.sub(r"\s+", " ", "".join(out)).strip()
