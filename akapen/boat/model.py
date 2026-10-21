"""学習・検証・本番モデルの入れ替え。

- 直近 test_days 日を検証用に取り分け（時系列で分割＝未来の情報は混ざらない）
- いくつかの設定で学習し、検証の対数損失が最も小さいものを選ぶ
- 選んだモデルが単純な基準（枠の平均1着率だけ）より悪ければ本番に入れない
- 合格したら全データで学び直して本番モデルとして保存し、検証成績を history に追記
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import features as F

log = logging.getLogger(__name__)

RACE = ["date", "venue", "race_no"]
CONFIGS = [
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 50, "n_estimators": 600},
    {"num_leaves": 63, "learning_rate": 0.03, "min_child_samples": 100, "n_estimators": 900},
    {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 30, "n_estimators": 500},
]
MASK_RATE = 0.3  # 直前情報が無い状況にも慣れさせるため、学習時に一部レースで直前情報を隠す
PRE_COLS = F.PRE_RACE_OPTIONAL + ["exh_time_rank", "exh_time_diff"]


def _mask_pre_race(X: pd.DataFrame, keys: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rid = keys[RACE].astype(str).agg("|".join, axis=1)
    races = rid.unique()
    hide = set(races[rng.random(len(races)) < MASK_RATE])
    X = X.copy()
    X.loc[rid.isin(hide).values, PRE_COLS] = np.nan
    return X


def _fit(df: pd.DataFrame, target: str, cfg: dict, seed: int = 0):
    X = _mask_pre_race(df[F.FEATURES], df, seed)
    y = df[target].astype(int)
    m = lgb.LGBMClassifier(objective="binary", subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=seed, verbose=-1, **cfg)
    m.fit(X, y, categorical_feature=F.CATEGORICAL)
    return m


def normalize(df: pd.DataFrame, raw: np.ndarray, total: float) -> np.ndarray:
    """レース内で合計が total になるよう正規化（1着確率なら1、3着内なら3）。"""
    s = pd.Series(raw, index=df.index).groupby([df[c] for c in RACE]).transform("sum")
    p = raw / s.values * total
    return np.clip(p, 1e-6, 1 - 1e-6) if total == 1 else np.clip(p, 1e-6, 0.99)


def labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y_win"] = (df["finish"] == 1).astype(int)
    df["y_top3"] = (df["finish"] <= 3).astype(int)
    return df


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """1着が確定し6艇そろったレースだけ（不成立・全艇失格などを除く）。"""
    df = df[df["finish_raw"].notna()]
    ok = df.groupby(RACE)["finish"].transform(lambda s: (s == 1).sum() == 1)
    n = df.groupby(RACE)["boat_no"].transform("count")
    return df[ok & (n == 6)]


def evaluate(df: pd.DataFrame, p_win: np.ndarray, p_top3: np.ndarray | None = None) -> dict:
    """1レース単位の成績。◎＝1着確率が最大の艇。"""
    d = df[RACE + ["boat_no", "finish", "tansho", "tansho_pay", "sanrentan", "sanrentan_pay"]].copy()
    d["p"] = p_win
    n_races = d.groupby(RACE).ngroups
    winner_p = d.loc[d["finish"] == 1, "p"]
    top = d.loc[d.groupby(RACE)["p"].idxmax()]
    hit = top["finish"] == 1
    paid = top["tansho_pay"].where(hit & (top["tansho"] == top["boat_no"]), 0).fillna(0)
    out = {
        "races": int(n_races),
        "logloss": float(-np.log(winner_p).mean()),
        "honmei_win": float(hit.mean()),
        "honmei_top3": float((top["finish"] <= 3).mean()),
        "honmei_tansho_roi": float(paid.sum() / (100 * len(top))) if len(top) else None,
    }
    # 3連単：Harville 式で上位1点が当たる率、上位10点に入る率
    hits1 = hits10 = n3 = 0
    for _, r in d.groupby(RACE):
        actual = r.loc[r["finish"].isin([1, 2, 3])].sort_values("finish")["boat_no"].tolist()
        if len(actual) < 3:
            continue
        combos = trifecta(dict(zip(r["boat_no"], r["p"])), top_n=10)
        keys = [c for c, _ in combos]
        n3 += 1
        hits1 += keys[0] == tuple(actual)
        hits10 += tuple(actual) in keys
    out["trifecta_top1"] = hits1 / n3 if n3 else None
    out["trifecta_top10"] = hits10 / n3 if n3 else None
    return out


def trifecta(p: dict, top_n: int = 10) -> list[tuple[tuple[int, int, int], float]]:
    """1着確率から3連単の確率（Harville）。"""
    res = []
    for a, pa in p.items():
        for b, pb in p.items():
            if b == a:
                continue
            for c, pc in p.items():
                if c in (a, b):
                    continue
                pr = pa * pb / (1 - pa) * pc / (1 - pa - pb)
                res.append(((a, b, c), pr))
    res.sort(key=lambda x: -x[1])
    return res[:top_n]


def lane_baseline(te: pd.DataFrame, tr: pd.DataFrame) -> np.ndarray:
    """学習期間のその場の枠別1着率だけで予想した場合（比較の基準）。"""
    rate = tr.groupby(["venue", "boat_no"])["y_win"].mean()
    raw = pd.Series(list(zip(te["venue"], te["boat_no"]))).map(rate).fillna(
        te["boat_no"].map(F.LANE_PRIOR_WIN).reset_index(drop=True)).values.astype(float)
    return normalize(te, raw, 1)


def train(data: pd.DataFrame, model_dir: Path, test_days: int = 56, focus_venue: str | None = "12") -> dict:
    """focus_venue の検証成績で選ぶ。学習データは「全場」と「その場だけ」の両方を試す。"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data = labels(usable(data))
    feats = F.build(data, data)
    end = pd.to_datetime(feats["date"]).max()
    cutoff = (end - timedelta(days=test_days - 1)).strftime("%Y-%m-%d")
    tr, te = feats[feats["date"] < cutoff], feats[feats["date"] >= cutoff]
    if focus_venue:
        te = te[te["venue"] == focus_venue]
    if tr[RACE].drop_duplicates().shape[0] < 500 or te.empty:
        raise RuntimeError(f"学習データが足りません（学習 {len(tr)} 行／検証 {len(te)} 行）")
    log.info("学習 %s〜%s / 検証 %s〜%s", tr["date"].min(), tr["date"].max(), te["date"].min(), te["date"].max())

    base = evaluate(te, lane_baseline(te, tr))
    natwin = te.groupby(RACE)["nat_win"].transform(lambda s: np.exp(s - s.max()))
    base_nat = evaluate(te, normalize(te, natwin.values, 1))
    te_eval = te.copy()
    te_eval[PRE_COLS] = np.nan  # 直前情報なしの成績も別に出す
    pools = {"all": None}
    if focus_venue and (tr["venue"] == focus_venue).any():
        pools["focus"] = focus_venue
    best = None
    for pool, v in pools.items():
        sub = tr if v is None else tr[tr["venue"] == v]
        if sub[RACE].drop_duplicates().shape[0] < 500:
            continue
        for i, cfg in enumerate(CONFIGS):
            m = _fit(sub, "y_win", cfg)
            pw = normalize(te, m.predict_proba(te[F.FEATURES])[:, 1], 1)
            ev = evaluate(te, pw)
            log.info("学習=%s 設定%d: %s", pool, i, ev)
            if best is None or ev["logloss"] < best[2]["logloss"]:
                pw_no = normalize(te_eval, m.predict_proba(te_eval[F.FEATURES])[:, 1], 1)
                best = (pool, cfg, ev, evaluate(te_eval, pw_no))

    pool, cfg, ev, ev_no_pre = best
    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_range": [tr["date"].min(), tr["date"].max()],
        "test_range": [te["date"].min(), te["date"].max()],
        "focus_venue": focus_venue, "train_pool": pool,
        "config": cfg, "test": ev, "test_without_pre_race": ev_no_pre,
        "baseline_lane": base, "baseline_nat_win": base_nat,
        "promoted": ev["logloss"] < base["logloss"],
    }
    if report["promoted"]:
        full = feats if pool == "all" else feats[feats["venue"] == focus_venue]
        full_win = _fit(full, "y_win", cfg)
        full_top3 = _fit(full, "y_top3", cfg)
        full_win.booster_.save_model(str(model_dir / "win.txt"))
        full_top3.booster_.save_model(str(model_dir / "top3.txt"))
        (model_dir / "meta.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        log.warning("基準より悪いため本番モデルは入れ替えません")
    with open(model_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
    return report


class Predictor:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.win = lgb.Booster(model_file=str(model_dir / "win.txt"))
        self.top3 = lgb.Booster(model_file=str(model_dir / "top3.txt"))
        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        out = feats.copy()
        X = feats[F.FEATURES]
        out["p_win"] = normalize(feats, self.win.predict(X), 1)
        out["p_top3"] = normalize(feats, self.top3.predict(X), 3)
        return out
