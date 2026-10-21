"""特徴量づくり。

すべて「そのレースの前に分かっていること」だけから作る（未来の情報を混ぜない）。
- 番組表の値：級別・勝率・2連率・モーター/ボート・今節成績・枠
- 過去成績からの集計：選手のその枠での1着率/3着内率、平均ST、最近の平均着順
  （対象日より前の日付だけを使う）
- 直前情報（展示タイム・風・波）：当日に取得できたときだけ。無い時は欠損として扱う
- レース内の相対値：各指標のレース内順位と平均との差
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .parse import CLASS_RANK

# 全国のコース別1着率の目安（%）。過去データが少ない選手の補正に使う。
LANE_PRIOR_WIN = {1: 0.55, 2: 0.14, 3: 0.12, 4: 0.11, 5: 0.06, 6: 0.02}
LANE_PRIOR_TOP3 = {1: 0.80, 2: 0.55, 3: 0.50, 4: 0.45, 5: 0.35, 6: 0.25}
SMOOTH = 10  # 事前値の重み（このレース数ぶんの事前情報とみなす）
RECENT_DAYS = 20  # 「最近」とみなす出走日数

RACE_TYPES = ["優勝", "準優", "特選", "選抜", "予選", "一般"]
PRE_RACE_OPTIONAL = ["exh_time", "wind_speed", "wave"]
RELATIVE = ["nat_win", "loc_win", "motor_2r", "boat_2r", "class_rank",
            "h_lane_win", "h_recent_finish", "h_avg_st", "kon_mean", "exh_time"]

FEATURES = (
    ["boat_no", "venue_i", "race_type_i", "distance", "age", "weight", "class_rank",
     "nat_win", "nat_2r", "loc_win", "loc_2r", "motor_2r", "boat_2r",
     "kon_n", "kon_mean", "kon_top2",
     "h_starts", "h_lane_n", "h_lane_win", "h_lane_top3", "h_avg_st", "h_recent_finish", "h_recent_n"]
    + PRE_RACE_OPTIONAL
    + [f"{c}_rank" for c in RELATIVE] + [f"{c}_diff" for c in RELATIVE]
)
CATEGORICAL = ["boat_no", "venue_i", "race_type_i"]


def race_type_code(s) -> int:
    s = str(s or "")
    for i, k in enumerate(RACE_TYPES):
        if k in s:
            return i
    return len(RACE_TYPES)


def _kon(s) -> tuple[int, float, int]:
    fin = [int(c) for c in re.findall(r"[1-6]", str(s or ""))]
    if not fin:
        return 0, np.nan, 0
    return len(fin), float(np.mean(fin)), sum(1 for f in fin if f <= 2)


def _history(target: pd.DataFrame, past: pd.DataFrame) -> pd.DataFrame:
    """target の各行に、その日より前の past 成績から作った選手の集計を付ける。"""
    if past is None or past.empty:
        out = target[["date", "toban", "boat_no"]].copy()
        for c in ["h_starts", "h_lane_n", "h_lane_win", "h_lane_top3", "h_avg_st",
                  "h_recent_finish", "h_recent_n"]:
            out[c] = np.nan
        out["h_starts"] = 0
        out["h_lane_n"] = 0
        return _lane_smooth(out)

    p = past[past["finish_raw"].notna()].copy()
    p["started"] = 1
    p["win"] = (p["finish"] == 1).astype(int)
    p["top3"] = (p["finish"] <= 3).astype(int)
    p["fin_ok"] = p["finish"].notna().astype(int)
    p["fin_sum"] = p["finish"].fillna(0)
    p["st_ok"] = (p["st"].notna() & (p["st_flag"] == "")).astype(int)
    p["st_sum"] = p["st"].where(p["st_ok"] == 1, 0)

    # 選手×日 の集計 → 対象日より前の累計と最近 RECENT_DAYS 出走日の合計
    g = p.groupby(["toban", "date"])[["started", "fin_ok", "fin_sum", "st_ok", "st_sum"]].sum().reset_index()
    g = g.sort_values(["toban", "date"])
    cum = g.groupby("toban")[["started", "st_ok", "st_sum"]].cumsum()
    rec = (g.groupby("toban")[["fin_ok", "fin_sum"]]
             .rolling(RECENT_DAYS, min_periods=1).sum().reset_index(level=0, drop=True))
    g["c_started"], g["c_st_ok"], g["c_st_sum"] = cum["started"], cum["st_ok"], cum["st_sum"]
    g["r_fin_ok"], g["r_fin_sum"] = rec["fin_ok"], rec["fin_sum"]

    # 選手×枠×日
    lg = p.groupby(["toban", "boat_no", "date"])[["started", "win", "top3"]].sum().reset_index()
    lg = lg.sort_values(["toban", "boat_no", "date"])
    lc = lg.groupby(["toban", "boat_no"])[["started", "win", "top3"]].cumsum()
    lg["l_n"], lg["l_win"], lg["l_top3"] = lc["started"], lc["win"], lc["top3"]

    t = target[["date", "toban", "boat_no"]].copy()
    t["_i"] = np.arange(len(t))
    # 「対象日より前で最も新しい行」を引く（同日は含めない）
    t["_d"] = pd.to_datetime(t["date"]) - pd.Timedelta(days=1)
    g["_d"] = pd.to_datetime(g["date"])
    lg["_d"] = pd.to_datetime(lg["date"])
    t = t.sort_values("_d")
    a = pd.merge_asof(t, g.sort_values("_d")[["_d", "toban", "c_started", "c_st_ok", "c_st_sum",
                                               "r_fin_ok", "r_fin_sum"]],
                      on="_d", by="toban", direction="backward")
    a = pd.merge_asof(a.sort_values("_d"), lg.sort_values("_d")[["_d", "toban", "boat_no", "l_n", "l_win", "l_top3"]],
                      on="_d", by=["toban", "boat_no"], direction="backward")
    a = a.sort_values("_i")
    out = pd.DataFrame({
        "date": a["date"].values, "toban": a["toban"].values, "boat_no": a["boat_no"].values,
        "h_starts": a["c_started"].fillna(0).values,
        "h_avg_st": (a["c_st_sum"] / a["c_st_ok"]).where(a["c_st_ok"] > 0).values,
        "h_recent_finish": (a["r_fin_sum"] / a["r_fin_ok"]).where(a["r_fin_ok"] > 0).values,
        "h_recent_n": a["r_fin_ok"].fillna(0).values,
        "h_lane_n": a["l_n"].fillna(0).values,
        "_lw": a["l_win"].fillna(0).values, "_lt": a["l_top3"].fillna(0).values,
    })
    return _lane_smooth(out)


def _lane_smooth(out: pd.DataFrame) -> pd.DataFrame:
    pw = out["boat_no"].map(LANE_PRIOR_WIN)
    pt = out["boat_no"].map(LANE_PRIOR_TOP3)
    lw = out.pop("_lw") if "_lw" in out else 0
    lt = out.pop("_lt") if "_lt" in out else 0
    out["h_lane_win"] = (lw + pw * SMOOTH) / (out["h_lane_n"] + SMOOTH)
    out["h_lane_top3"] = (lt + pt * SMOOTH) / (out["h_lane_n"] + SMOOTH)
    return out


def build(target: pd.DataFrame, past: pd.DataFrame | None) -> pd.DataFrame:
    """target（番組表＋分かっていれば直前情報）に特徴量列を足して返す。"""
    df = target.copy().reset_index(drop=True)
    for c in PRE_RACE_OPTIONAL:
        if c not in df:
            df[c] = np.nan
    df["venue_i"] = df["venue"].astype(int)
    df["race_type_i"] = df["race_type"].map(race_type_code)
    df["class_rank"] = df["racer_class"].map(CLASS_RANK)
    kon = df["kon_results"].map(_kon)
    df["kon_n"] = [k[0] for k in kon]
    df["kon_mean"] = [k[1] for k in kon]
    df["kon_top2"] = [k[2] for k in kon]

    h = _history(df, past)
    for c in ["h_starts", "h_lane_n", "h_lane_win", "h_lane_top3", "h_avg_st", "h_recent_finish", "h_recent_n"]:
        df[c] = h[c].values

    grp = df.groupby(["date", "venue", "race_no"])
    for c in RELATIVE:
        asc = c in ("h_recent_finish", "h_avg_st", "kon_mean", "exh_time")  # 小さいほど良い
        df[f"{c}_rank"] = grp[c].rank(ascending=asc, method="min")
        df[f"{c}_diff"] = df[c] - grp[c].transform("mean")
    return df
