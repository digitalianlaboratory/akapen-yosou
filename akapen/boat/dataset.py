"""日次ファイル → 学習用テーブル（1艇1行）を作り、月ごとの parquet に保存する。"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .download import fetch_text
from .parse import ParseStats, parse_program, parse_results

log = logging.getLogger(__name__)

KEY = ["date", "venue", "race_no", "boat_no"]


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def build_day(d: date, raw_dir: Path, with_results: bool = True) -> pd.DataFrame:
    """1日分。成績がまだ無い日（当日・翌日）は番組表だけで返す。"""
    b = fetch_text("B", d, raw_dir)
    if b is None:
        return pd.DataFrame()
    bs = ParseStats()
    prog = pd.DataFrame(parse_program(b, d, bs))
    log.info("%s B: %s", d, bs.summary())
    if prog.empty:
        return prog
    k = fetch_text("K", d, raw_dir) if with_results else None
    if k is None:
        return prog
    ks = ParseStats()
    ent, races = parse_results(k, d, ks)
    log.info("%s K: %s", d, ks.summary())
    if not ent:
        return prog
    res = pd.DataFrame(ent).drop(columns=["name_k"])
    df = prog.merge(res, on=KEY, how="left", suffixes=("", "_k"))
    bad = df["toban_k"].notna() & (df["toban_k"] != df["toban"])
    if bad.any():
        log.warning("%s 登録番号が番組表と成績で一致しない行: %d（除外）", d, int(bad.sum()))
        df = df[~bad]
    df = df.drop(columns=["toban_k"])
    if races:
        rc = pd.DataFrame(races)
        df = df.merge(rc, on=["date", "venue", "race_no"], how="left")
    return df


def update(start: date, end: date, raw_dir: Path, out_dir: Path) -> int:
    """期間のデータを取得して月別 parquet に上書き保存。保存行数を返す。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [build_day(d, raw_dir) for d in daterange(start, end)]
    frames = [f for f in frames if not f.empty and "finish_raw" in f]
    if not frames:
        return 0
    new = pd.concat(frames, ignore_index=True)
    new = new[new["finish_raw"].notna()]
    n = 0
    for month, part in new.groupby(new["date"].str[:7]):
        path = out_dir / f"{month}.parquet"
        if path.exists():
            old = pd.read_parquet(path)
            part = pd.concat([old, part], ignore_index=True).drop_duplicates(KEY, keep="last")
        part.sort_values(KEY).to_parquet(path, index=False)
        n += len(part)
    return n


def load(out_dir: Path, since: str | None = None) -> pd.DataFrame:
    files = sorted(Path(out_dir).glob("*.parquet"))
    if since:
        files = [f for f in files if f.stem >= since[:7]]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(KEY).reset_index(drop=True)
