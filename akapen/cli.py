"""コマンド入口。

  python -m akapen.cli boat-update  --start 2023-01-01 --end 2026-10-20
  python -m akapen.cli boat-train
  python -m akapen.cli boat-predict --date 2026-10-22 --venue 住之江 [--race 1 2 3] [--pre 直前情報.csv]
  python -m akapen.cli boat-score
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "boat" / "raw"
DATA = ROOT / "data" / "boat" / "races"
MODEL = ROOT / "models" / "boat"
PRED = ROOT / "predictions" / "boat"
OUT = ROOT / "output"


def _d(s: str) -> date:
    return date.fromisoformat(s)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="akapen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    u = sub.add_parser("boat-update", help="公式データを取得して学習用テーブルを更新")
    u.add_argument("--start", type=_d)
    u.add_argument("--end", type=_d, default=date.today() - timedelta(days=1))
    u.add_argument("--days", type=int, default=7, help="--start 省略時、何日さかのぼるか")

    t = sub.add_parser("boat-train", help="学習・検証して、基準を上回れば本番モデルを更新")
    t.add_argument("--test-days", type=int, default=28)

    p = sub.add_parser("boat-predict", help="指定レースを予想してPDFに出力")
    p.add_argument("--date", type=_d, default=date.today())
    p.add_argument("--venue", nargs="*", help="場名か場コード（省略で全場）")
    p.add_argument("--race", nargs="*", type=int)
    p.add_argument("--pre", type=Path, help="直前情報CSV（date無し, 列: venue,race_no,boat_no,exh_time,wind_speed,wave）")
    p.add_argument("--out", type=Path)

    sub.add_parser("boat-score", help="記録した予想を結果と突き合わせて実戦成績を更新")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if a.cmd == "boat-update":
        from .boat.dataset import update
        start = a.start or a.end - timedelta(days=a.days - 1)
        n = update(start, a.end, RAW, DATA)
        print(f"更新: {start}〜{a.end}（対象月の合計 {n} 行）")

    elif a.cmd == "boat-train":
        from .boat.dataset import load
        from .boat.model import train
        rep = train(load(DATA), MODEL, test_days=a.test_days)
        print(json.dumps(rep, ensure_ascii=False, indent=2))

    elif a.cmd == "boat-predict":
        from .boat.predict import log_predictions, predict_day
        from .boat.venues import venue_code
        from .pdf import render
        venues = [venue_code(v) for v in a.venue] if a.venue else None
        reps = predict_day(a.date, venues, a.race, RAW, DATA, MODEL, a.pre)
        log_predictions(reps, PRED)
        meta = json.loads((MODEL / "meta.json").read_text(encoding="utf-8"))
        live_p = MODEL / "live.json"
        live = json.loads(live_p.read_text(encoding="utf-8")) if live_p.exists() else None
        tag = "_".join(sorted({r["venue_name"] for r in reps})) if venues else "全場"
        out = a.out or OUT / f"競艇予想_{a.date:%Y%m%d}_{tag}.pdf"
        render(reps, out, meta, live)
        print(f"PDF: {out}（{len(reps)}レース）")

    elif a.cmd == "boat-score":
        from .boat.predict import score
        print(json.dumps(score(PRED, DATA, MODEL / "live.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
