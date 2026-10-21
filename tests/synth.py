"""テスト専用：公式ファイルと同じ書式の「架空データ」を作る。

実データではない。解析→学習→予想→PDF が最後まで通るかを確かめるためだけに使う。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

FW = str.maketrans("0123456789R:mH", "０１２３４５６７８９Ｒ：ｍＨ")
NAMES = ["山田太郎", "佐藤　一", "鈴木健二", "高橋　誠", "田中次郎", "伊藤　翔", "渡辺大輔", "中村　淳"]
BRANCHES = ["大阪", "福岡", "群馬", "東京", "香川", "岡山"]
LANE = np.array([1.6, 0.4, 0.3, 0.2, -0.3, -0.8])


def make(raw_dir: Path, start: date, days: int, venues=("12", "01", "24"), seed: int = 1):
    rng = np.random.default_rng(seed)
    n_racers = 300
    skill = rng.normal(0, 1, n_racers)
    tobans = [f"{4000 + i}" for i in range(n_racers)]
    motor = {v: rng.normal(0, 0.5, 60) for v in venues}
    st_mean = 0.17 - 0.02 * skill + rng.normal(0, 0.01, n_racers)
    for k in range(days):
        d = start + timedelta(days=k)
        b_lines, k_lines = [], []
        for v in venues:
            b_lines += [f"{v}BBGN", "ボートレース場", ""]
            k_lines += [f"{v}KBGN", "成績", ""]
            for r in range(1, 13):
                ids = rng.choice(n_racers, 6, replace=False)
                mot = rng.choice(60, 6, replace=False)
                rtype = "予選" if r < 12 else "一般"
                b_lines.append(f"　{str(r).translate(FW)}Ｒ　　{rtype}　　　　　　　Ｈ１８００ｍ　　電話投票締切予定"
                               + f"{10 + r // 2}:{(r * 7) % 60:02d}".translate(FW))
                b_lines.append("艇 選手 ...")
                strength = skill[ids] + LANE + motor[v][mot] + rng.gumbel(0, 1, 6)
                exh = 6.80 - 0.05 * motor[v][mot] + rng.normal(0, 0.03, 6)
                for lane in range(6):
                    i = ids[lane]
                    cls = "A1" if skill[i] > 1 else "A2" if skill[i] > 0.3 else "B1" if skill[i] > -1 else "B2"
                    nw = np.clip(5.5 + 1.5 * skill[i] + rng.normal(0, 0.3), 1, 9.9)
                    name = NAMES[i % len(NAMES)]
                    m2r = np.clip(35 + 10 * motor[v][mot[lane]] + rng.normal(0, 3), 5, 80)
                    b_lines.append(
                        f"{lane + 1} {tobans[i]}{name}{30 + i % 25:2d}{BRANCHES[i % 6]}{50 + i % 6:2d}{cls}"
                        f" {nw:4.2f} {nw * 6:5.2f} {nw:4.2f} {nw * 6:5.2f}"
                        f" {mot[lane] + 1:2d} {m2r:5.2f} {i % 70 + 1:2d} {30.0:5.2f} {rng.integers(1, 7)} {rng.integers(1, 7)}       ")
                order = list(np.argsort(-strength))
                k_lines.append(f"   {r}R       {rtype}                 H1800m  晴　  風  北　　 {rng.integers(0, 7)}m  波　  {rng.integers(0, 8)}cm")
                k_lines.append("  着 艇 登番 　選　手　名　　ﾓｰﾀｰ ﾎﾞｰﾄ 展示 進入 ｽﾀｰﾄﾀｲﾐﾝｸ ﾚｰｽﾀｲﾑ")
                k_lines.append("-" * 70)
                for pos, lane in enumerate(order, start=1):
                    i = ids[lane]
                    st = max(0.01, st_mean[i] + rng.normal(0, 0.03))
                    fin = f"{pos:02d}"
                    st_s = f"{st:.2f}"
                    if pos == 6 and rng.random() < 0.02:
                        fin, st_s = " F", "F0.01"
                    name = "　".join(NAMES[i % len(NAMES)].replace("　", ""))
                    k_lines.append(f"  {fin}  {lane + 1} {tobans[i]} {name} {mot[lane] + 1:2d}   {i % 70 + 1:2d}  "
                                   f"{exh[lane]:.2f}   {lane + 1}    {st_s}     1.48.9")
                k_lines.append("")
                w = order[0] + 1
                market = np.exp(skill[ids] + LANE)
                pay = int(max(100, round(75 / (market[order[0]] / market.sum()) / 10) * 10))
                k_lines.append(f"        単勝     {w}          {pay}")
                a, b, c = (x + 1 for x in order[:3])
                k_lines.append(f"        ３連単   {a}-{b}-{c}      {pay * 20}  人気     9")
                k_lines.append("")
            b_lines.append(f"{v}BEND")
            k_lines.append(f"{v}KEND")
        for kind, lines in (("B", b_lines), ("K", k_lines)):
            p = Path(raw_dir) / kind / f"{kind.lower()}{d:%y%m%d}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
