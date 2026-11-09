"""Solar-lunar calendar shipped with each split (calendar.csv)."""
from __future__ import annotations

import csv
import datetime as dt
import os


class LunarCalendar:
    def __init__(self, rows: list):
        self.by_lunar = {}
        for r in rows:
            try:
                d = dt.date.fromisoformat(r["date"])
                leap = str(r.get("lunar_leap", "")).strip().lower() in ("true", "1")
                self.by_lunar[(int(r["lunar_year"]), int(r["lunar_month"]), int(r["lunar_day"]), leap)] = d
            except (KeyError, ValueError):
                continue

    def solar(self, ly: int, lm: int, ld: int):
        return self.by_lunar.get((ly, lm, ld, False))

    def tet_range(self, year: int | None, today: dt.date):
        """Tết of lunar year Y: 23 tháng Chạp (Y-1) .. mùng 10 tháng Giêng (Y)."""
        if year is None:
            return None
        a, b = self.solar(year - 1, 12, 23), self.solar(year, 1, 10)
        return (a, b) if a and b else None


def load_calendar(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as fh:
        return LunarCalendar(list(csv.DictReader(fh)))
