"""公式テキスト（番組表 B／競走成績 K）の解析。

どちらも場ごとに `NNBBGN`〜`NNBEND`（成績は `NNKBGN`〜`NNKEND`）で区切られた
固定幅テキスト。全角数字や半角カナが混ざるため NFKC で正規化してから読む。
読めなかった行は捨てずに件数を数え、`stats` で確認できるようにしている。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date

CLASS_RANK = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}

_BLOCK = re.compile(r"^(\d{2})([BK])BGN")
_BLOCK_END = re.compile(r"^(\d{2})([BK])END")
_RACE_HDR = re.compile(r"^\s*(\d{1,2})R\s+(.*?)\s*H(\d{3,4})m")
_CLOSE = re.compile(r"締切予定\s*(\d{1,2}):(\d{2})")
_WEATHER = re.compile(r"H\d{3,4}m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm")
# 番組表の出走行は固定幅。2連率が 100.00 のときやボート番号が3桁のときは隣の欄とくっつくため、
# 空白区切りではなく桁位置で切り出す（実ファイルで確認済み）。
_B_COLS = {
    "toban": (2, 6), "name": (6, 10), "age": (10, 12), "branch": (12, 14), "weight": (14, 16),
    "racer_class": (16, 18), "nat_win": (18, 23), "nat_2r": (23, 29), "loc_win": (29, 34),
    "loc_2r": (34, 40), "motor_no": (40, 43), "motor_2r": (43, 49), "boat_id": (49, 52),
    "boat_2r": (52, 58), "kon_results": (58, 71),
}
_B_INT = {"age", "weight", "motor_no", "boat_id"}
_B_FLOAT = {"nat_win", "nat_2r", "loc_win", "loc_2r", "motor_2r", "boat_2r"}
_TANSHO = re.compile(r"単勝\s+([1-6])\s+(\d+)")
_SANRENTAN = re.compile(r"3連単\s+([1-6]-[1-6]-[1-6])\s+(\d+)")


def norm(line: str) -> str:
    return unicodedata.normalize("NFKC", line.rstrip("\r\n"))


@dataclass
class ParseStats:
    races: int = 0
    entries: int = 0
    skipped: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return f"races={self.races} entries={self.entries} skipped_lines={len(self.skipped)}"


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _b_entry(raw: str) -> dict | None:
    """出走1行（正規化前の文字列）→ 辞書。形式が合わなければ None。"""
    if len(raw) < 58:
        return None
    row = {"boat_no": int(raw[0])}
    try:
        for k, (a, b) in _B_COLS.items():
            v = unicodedata.normalize("NFKC", raw[a:b]).strip()
            if k in _B_INT:
                v = int(v)
            elif k in _B_FLOAT:
                v = float(v)
            elif k == "name":
                v = re.sub(r"\s+", "", v)
            row[k] = v
    except ValueError:
        return None
    if row["racer_class"] not in CLASS_RANK or not row["toban"].isdigit():
        return None
    return row


def parse_program(text: str, d: date, stats: ParseStats | None = None) -> list[dict]:
    """番組表 B → 出走（1艇1行）の辞書リスト。"""
    stats = stats or ParseStats()
    rows: list[dict] = []
    venue = None
    race = None
    for raw in text.splitlines():
        line = norm(raw)
        if m := _BLOCK.match(line):
            venue, race = m.group(1), None
            continue
        if _BLOCK_END.match(line):
            venue, race = None, None
            continue
        if venue is None:
            continue
        if m := _RACE_HDR.match(line):
            close = _CLOSE.search(line)
            race = {
                "race_no": int(m.group(1)),
                "race_type": re.sub(r"\s+", "", m.group(2)),
                "distance": int(m.group(3)),
                "close_time": f"{int(close.group(1)):02d}:{close.group(2)}" if close else None,
            }
            stats.races += 1
            continue
        if race is None:
            continue
        if re.match(r"^[1-6]\s\d{4}", line):
            row = _b_entry(raw.rstrip("\r\n"))
            if row is None:
                stats.skipped.append(line)
                continue
            rows.append({"date": d.isoformat(), "venue": venue, **race, **row})
            stats.entries += 1
    return rows


def _finish(tok: str):
    t = tok.strip()
    return int(t) if t.isdigit() and 1 <= int(t) <= 6 else None


def parse_results(text: str, d: date, stats: ParseStats | None = None) -> tuple[list[dict], list[dict]]:
    """競走成績 K → (出走結果1艇1行, レース1行[天候・払戻])。"""
    stats = stats or ParseStats()
    entries: list[dict] = []
    races: list[dict] = []
    venue = None
    race = None
    for raw in text.splitlines():
        line = norm(raw)
        if m := _BLOCK.match(line):
            venue, race = m.group(1), None
            continue
        if _BLOCK_END.match(line):
            venue, race = None, None
            continue
        if venue is None:
            continue
        if m := _RACE_HDR.match(line):
            w = _WEATHER.search(line)
            race = {
                "date": d.isoformat(), "venue": venue, "race_no": int(m.group(1)),
                "weather": w.group(1) if w else None,
                "wind_dir": w.group(2) if w else None,
                "wind_speed": int(w.group(3)) if w else None,
                "wave": int(w.group(4)) if w else None,
                "tansho": None, "tansho_pay": None,
                "sanrentan": None, "sanrentan_pay": None,
            }
            races.append(race)
            stats.races += 1
            continue
        if race is None:
            continue
        if m := _TANSHO.search(line):
            race["tansho"], race["tansho_pay"] = int(m.group(1)), int(m.group(2))
            continue
        if m := _SANRENTAN.search(line):
            race["sanrentan"], race["sanrentan_pay"] = m.group(1), int(m.group(2))
            continue
        toks = line.split()
        if len(toks) >= 6 and re.fullmatch(r"[1-6]", toks[1]) and re.fullmatch(r"\d{4}", toks[2]):
            rest = toks[3:]
            i = next((k for k, t in enumerate(rest) if t.isdigit()), None)
            if i is None:
                stats.skipped.append(line)
                continue
            name, nums = "".join(rest[:i]), rest[i:]
            # nums: モーター ボート 展示 進入 ST [レースタイム]
            exh = _f(nums[2]) if len(nums) > 2 else None
            course = int(nums[3]) if len(nums) > 3 and re.fullmatch(r"[1-6]", nums[3]) else None
            st_raw = nums[4] if len(nums) > 4 else ""
            st_flag = "F" if st_raw.startswith("F") else "L" if st_raw.startswith("L") else ""
            st = _f(st_raw.lstrip("FL"))
            entries.append({
                "date": d.isoformat(), "venue": venue, "race_no": race["race_no"],
                "boat_no": int(toks[1]), "toban": toks[2], "name_k": name,
                "finish": _finish(toks[0]), "finish_raw": toks[0],
                "exh_time": exh, "course": course, "st": st, "st_flag": st_flag,
            })
            stats.entries += 1
    return entries, races
