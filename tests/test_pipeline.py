from datetime import date, timedelta

import pandas as pd
import pytest

from akapen.boat import dataset, model
from akapen.boat.parse import ParseStats, parse_program, parse_results
from akapen.boat.predict import log_predictions, predict_day, score
from akapen.pdf import render

from .synth import make

B_SAMPLE = """12BBGN
ボートレース住之江
　１Ｒ　　予選　　　　　　　　Ｈ１８００ｍ　　電話投票締切予定１０：３３
1 4320峰　　竜太42佐賀51A1 7.85 55.10 8.20 60.00 34 38.46 56 35.29 1 1 3 2
2 4444山田太郎30大阪52B1 5.10 30.00 4.90 28.50 12 30.00 20 31.00 4 5
12BEND
"""
K_SAMPLE = """12KBGN
   1R       予選                 H1800m  晴　  風  北　　 3m  波　  3cm
  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ
-------------------------------------------------------------------------------
  01  1 4320 峰　　竜　太 34   56  6.71   1    0.08     1.48.9
   F  2 4444 山　田　太　郎 12   20  6.80   2   F0.02      .  .
        単勝     1          150
        ３連単   1-3-2      1230  人気     3
12KEND
"""


def test_parse_program():
    st = ParseStats()
    rows = parse_program(B_SAMPLE, date(2026, 10, 20), st)
    assert st.races == 1 and len(rows) == 2 and not st.skipped
    r = rows[0]
    assert (r["venue"], r["race_no"], r["distance"], r["close_time"]) == ("12", 1, 1800, "10:33")
    assert (r["toban"], r["name"], r["age"], r["branch"], r["racer_class"]) == ("4320", "峰竜太", 42, "佐賀", "A1")
    assert (r["nat_win"], r["motor_no"], r["motor_2r"], r["boat_2r"]) == (7.85, 34, 38.46, 35.29)


def test_parse_results():
    ent, races = parse_results(K_SAMPLE, date(2026, 10, 20))
    assert [e["finish"] for e in ent] == [1, None]
    assert ent[0]["exh_time"] == 6.71 and ent[0]["st"] == 0.08 and ent[0]["course"] == 1
    assert ent[1]["st_flag"] == "F" and ent[1]["finish_raw"] == "F"
    assert races[0]["wind_speed"] == 3 and races[0]["wave"] == 3
    assert (races[0]["tansho"], races[0]["tansho_pay"]) == (1, 150)
    assert (races[0]["sanrentan"], races[0]["sanrentan_pay"]) == ("1-3-2", 1230)


def test_history_uses_only_past():
    past = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"], "toban": ["1", "1"], "boat_no": [1, 1],
        "finish": [1, 1], "finish_raw": ["01", "01"], "st": [0.1, 0.1], "st_flag": ["", ""],
    })
    from akapen.boat.features import _history
    tgt = pd.DataFrame({"date": ["2026-01-02"], "toban": ["1"], "boat_no": [1]})
    h = _history(tgt, past)
    assert h["h_lane_n"].iloc[0] == 1  # 当日（01-02）の結果は含めない


def test_end_to_end(tmp_path):
    raw, data, mdl, pred = tmp_path / "raw", tmp_path / "data", tmp_path / "model", tmp_path / "pred"
    start = date(2026, 1, 1)
    make(raw, start, 75, venues=("12", "01"))
    end = start + timedelta(days=73)
    assert dataset.update(start, end, raw, data, ["12"]) > 0
    assert set(dataset.load(data)["venue"]) == {"12"}
    df = dataset.load(data)
    assert df["finish"].notna().mean() > 0.95 and df["exh_time"].notna().all()

    rep = model.train(df, mdl, test_days=14, focus_venue="12")
    assert rep["focus_venue"] == "12" and rep["test"]["races"] < 200
    # 架空データは枠・選手の強さ・モーターで結果が決まるので、枠だけの基準を上回れるはず
    assert rep["promoted"], rep
    assert rep["test"]["logloss"] < rep["baseline_lane"]["logloss"]

    day = start + timedelta(days=74)  # 結果ファイルはあるが、予想時は番組表だけを使う
    reps = predict_day(day, ["12"], [1, 2], raw, data, mdl)
    assert len(reps) == 2
    for r in reps:
        assert abs(sum(x["p_win"] for x in r["rows"]) - 1) < 1e-6
        assert r["rows"][0]["mark"] == "◎"
    log_predictions(reps, pred)
    out = render(reps, tmp_path / "x.pdf", rep, None)
    assert out.stat().st_size > 5000

    dataset.update(day, day, raw, data)
    s = score(pred, data, tmp_path / "live.json")
    assert s["all"]["races"] == 2
