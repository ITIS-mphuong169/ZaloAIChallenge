"""Parse the per-business README.md (column meanings and conventions)."""
from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass, field

from .textnorm import nfc


@dataclass
class Readme:
    text: str = ""
    intro: str = ""
    # (file_pattern, sheet or None) -> {column: meaning}
    columns: dict = field(default_factory=dict)
    conventions: list = field(default_factory=list)  # (label or '', text)
    data_end: dt.date | None = None

    def meanings_for(self, file: str, sheet: str | None) -> dict:
        """Column meanings for a file/sheet; file patterns may contain <...> wildcards."""
        out = {}
        for (fp, sh), cols in self.columns.items():
            if not _file_match(fp, file):
                continue
            if sh is not None and sheet is not None and nfc(sh) != nfc(sheet):
                continue
            out.update(cols)
        return out

    def convention(self, label: str) -> str:
        return " ".join(t for (lb, t) in self.conventions if lb.lower() == label.lower())


def _file_match(pattern: str, file: str) -> bool:
    pattern, file = nfc(pattern), nfc(file)
    if pattern == file:
        return True
    if "<" in pattern:
        rx = re.escape(pattern)
        rx = re.sub(r"<[^>]*>", ".+", rx.replace(r"\<", "<").replace(r"\>", ">"))
        return re.fullmatch(rx, file) is not None
    # files split with _1/_2/_2026 suffixes share the base name
    base, ext = os.path.splitext(pattern)
    return re.fullmatch(re.escape(base) + r"(_\d+)+" + re.escape(ext), file) is not None


def parse_readme(path: str) -> Readme:
    if not os.path.exists(path):
        return Readme()
    text = nfc(open(path, encoding="utf-8", errors="replace").read())
    rd = Readme(text=text)
    section, key = "intro", None
    intro = []
    for line in text.split("\n"):
        if line.startswith("## "):
            head = line[3:].strip()
            m = re.match(r"Cột trong `([^`]+)`(?:, sheet `([^`]+)`)?", head)
            if m:
                section, key = "cols", (m.group(1), m.group(2))
                rd.columns.setdefault(key, {})
            elif head.lower().startswith("quy ước"):
                section = "conv"
            else:
                section = "other"
            continue
        if line.startswith("# "):
            continue
        if section == "intro":
            intro.append(line)
        elif section == "cols":
            m = re.match(r"- \*\*(.+?)\*\*:\s*(.*)", line)
            if m:
                rd.columns[key][m.group(1).strip()] = m.group(2).strip()
        elif section == "conv":
            m = re.match(r"- (?:\*\*(.+?)\*\*:\s*)?(.*)", line)
            if m and line.startswith("- "):
                rd.conventions.append((m.group(1) or "", m.group(2).strip()))
            elif line.startswith("  - ") and rd.conventions:
                lb, t = rd.conventions[-1]
                rd.conventions[-1] = (lb, t + "\n" + line.strip()[2:])
    rd.intro = "\n".join(intro).strip()
    m = re.search(r"Dữ liệu có đến hết (\d{1,2})/(\d{1,2})/(\d{4})", text)
    if m:
        rd.data_end = dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return rd
