"""ボートレース公式（mbrace）が配布する日次ファイルの取得。

- 番組表 B: https://www1.mbrace.or.jp/od2/B/YYYYMM/bYYMMDD.lzh
- 競走成績 K: https://www1.mbrace.or.jp/od2/K/YYYYMM/kYYMMDD.lzh

取得したファイルは raw_dir にキャッシュし、同じ日を二度取りに行かない。
サーバーに負担をかけないよう、1ファイルごとに間隔を空ける。
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import lhafile

BASE = "https://www1.mbrace.or.jp/od2"
UA = "akapen-yosou/0.1 (personal research)"


def url_for(kind: str, d: date) -> str:
    kind = kind.upper()
    return f"{BASE}/{kind}/{d:%Y%m}/{kind.lower()}{d:%y%m%d}.lzh"


def _extract_text(data: bytes) -> str:
    arc = lhafile.LhaFile(io.BytesIO(data))
    names = arc.namelist()
    if not names:
        raise ValueError("空のLZHアーカイブ")
    return arc.read(names[0]).decode("cp932", errors="replace")


def fetch_text(kind: str, d: date, raw_dir: Path, sleep: float = 1.0) -> str | None:
    """その日のファイルの中身（テキスト）を返す。開催なし・未公開なら None。"""
    raw_dir = Path(raw_dir)
    cache = raw_dir / kind.upper() / f"{kind.lower()}{d:%y%m%d}.txt"
    missing = cache.with_suffix(".none")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    if missing.exists() and d < date.today():
        return None
    req = urllib.request.Request(url_for(kind, d), headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            missing.parent.mkdir(parents=True, exist_ok=True)
            missing.touch()
            return None
        raise
    finally:
        time.sleep(sleep)
    text = _extract_text(data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return text
