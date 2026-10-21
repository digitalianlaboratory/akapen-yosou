"""指示されたレースの予想を作る。予想は記録しておき、結果が出たら答え合わせする。"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .dataset import build_day, load
from .model import RACE, Predictor, trifecta
from .venues import VENUES

MARKS = ["◎", "○", "▲", "△", "△", ""]


def _num(v, fmt="{:.2f}"):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def reasons(r: pd.Series) -> str:
    """数字にもとづく短い根拠（創作しない）。"""
    out = []
    if r.get("nat_win_rank") == 1:
        out.append(f"全国勝率{r['nat_win']:.2f}はレース内1位")
    if r.get("motor_2r_rank") == 1:
        out.append(f"モーター2連率{r['motor_2r']:.1f}%は1位")
    if r.get("h_lane_n", 0) >= 5:
        out.append(f"この枠で1着率{r['h_lane_win']*100:.0f}%（{int(r['h_lane_n'])}走）")
    if not pd.isna(r.get("h_avg_st")):
        out.append(f"平均ST{r['h_avg_st']:.2f}")
    if r.get("kon_n", 0) >= 2:
        out.append(f"今節平均{r['kon_mean']:.1f}着")
    if not pd.isna(r.get("exh_time")) and r.get("exh_time_rank") == 1:
        out.append(f"展示タイム{r['exh_time']:.2f}は最速")
    return "／".join(out[:4])


def race_reports(pred: pd.DataFrame) -> list[dict]:
    reports = []
    for key, r in pred.groupby(RACE, sort=True):
        r = r.sort_values("p_win", ascending=False).reset_index(drop=True)
        p = dict(zip(r["boat_no"].astype(int), r["p_win"]))
        tri = trifecta(p, top_n=6)
        rows = []
        for i, x in r.iterrows():
            rows.append({
                "mark": MARKS[i], "boat_no": int(x["boat_no"]), "name": x["name"],
                "class": x["racer_class"], "branch": x["branch"],
                "nat_win": _num(x["nat_win"]), "motor_2r": _num(x["motor_2r"], "{:.1f}"),
                "p_win": float(x["p_win"]), "p_top3": float(x["p_top3"]),
                "reason": reasons(x),
            })
        top = rows[0]
        has_pre = bool(r["exh_time"].notna().any())
        reports.append({
            "date": key[0], "venue": key[1], "venue_name": VENUES.get(key[1], key[1]),
            "race_no": int(key[2]), "race_type": r["race_type"].iloc[0],
            "distance": int(r["distance"].iloc[0]), "close_time": r["close_time"].iloc[0],
            "rows": rows,
            "confidence": top["p_win"],
            "trifecta": [{"combo": "-".join(map(str, c)), "p": pr} for c, pr in tri],
            "has_pre_race": has_pre,
        })
    return reports


def predict_day(d: date, venues: list[str] | None, races: list[int] | None,
                raw_dir: Path, data_dir: Path, model_dir: Path, pre_race: Path | None = None) -> list[dict]:
    prog = build_day(d, raw_dir, with_results=False)
    if prog.empty:
        raise RuntimeError(f"{d} の番組表が見つかりません（開催なし、または未公開）")
    if venues:
        prog = prog[prog["venue"].isin(venues)]
    if races:
        prog = prog[prog["race_no"].isin(races)]
    if prog.empty:
        raise RuntimeError("指定の場・レースが番組表にありません")
    if pre_race:
        pre = pd.read_csv(pre_race, dtype={"venue": str})
        pre["date"] = d.isoformat()
        prog = prog.merge(pre, on=["date", "venue", "race_no", "boat_no"], how="left")
    past = load(data_dir)
    past = past[past["date"] < d.isoformat()] if not past.empty else None
    feats = F.build(prog, past)
    pred = Predictor(model_dir).predict(feats)
    return race_reports(pred)


def log_predictions(reports: list[dict], pred_dir: Path) -> None:
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    by_day: dict[str, list] = {}
    for rep in reports:
        by_day.setdefault(rep["date"], []).append(rep)
    for day, reps in by_day.items():
        path = pred_dir / f"{day}.jsonl"
        old = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                o = json.loads(line)
                old[(o["venue"], o["race_no"])] = o
        for rep in reps:
            old[(rep["venue"], rep["race_no"])] = {
                "date": rep["date"], "venue": rep["venue"], "race_no": rep["race_no"],
                "order": [x["boat_no"] for x in rep["rows"]],
                "p_win": [x["p_win"] for x in rep["rows"]],
                "trifecta": [t["combo"] for t in rep["trifecta"]],
            }
        path.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in old.values()), encoding="utf-8")


def score(pred_dir: Path, data_dir: Path, out: Path, days: int = 7) -> dict:
    """記録した予想を確定結果と突き合わせる（実戦での的中率）。"""
    data = load(data_dir)
    rows = []
    for path in sorted(Path(pred_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))
    if not rows or data.empty:
        return {}
    res = data[data["finish"].isin([1, 2, 3])]
    actual = {k: tuple(g.sort_values("finish")["boat_no"].astype(int))
              for k, g in res.groupby(RACE)}
    scored = []
    for o in rows:
        a = actual.get((o["date"], o["venue"], o["race_no"]))
        if not a or len(a) < 3:
            continue
        scored.append({
            "date": o["date"],
            "honmei_win": o["order"][0] == a[0],
            "honmei_top3": o["order"][0] in a,
            "trifecta_hit": "-".join(map(str, a)) in o["trifecta"],
        })
    if not scored:
        return {}
    s = pd.DataFrame(scored)
    recent = s[pd.to_datetime(s["date"]) > pd.to_datetime(s["date"]).max() - pd.Timedelta(days=days)]

    def agg(x):
        return {"races": int(len(x)), "honmei_win": float(x["honmei_win"].mean()),
                "honmei_top3": float(x["honmei_top3"].mean()),
                "trifecta_6pt": float(x["trifecta_hit"].mean())}
    summary = {"all": agg(s), f"last_{days}d": agg(recent)}
    Path(out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
