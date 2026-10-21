"""予想 PDF（表紙＋1レース1ページ）。"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
RED = colors.HexColor("#C8102E")
INK = colors.HexColor("#222222")
MUTED = colors.HexColor("#666666")
RULE = colors.HexColor("#DDDDDD")
LANE_BG = {1: "#FFFFFF", 2: "#000000", 3: "#E60012", 4: "#0068B7", 5: "#FFD900", 6: "#009944"}
LANE_FG = {1: "#000000", 2: "#FFFFFF", 3: "#FFFFFF", 4: "#FFFFFF", 5: "#000000", 6: "#FFFFFF"}
DISCLAIMER = "この予想は過去データにもとづく統計モデルの推定で、的中や利益を保証しません。舟券は20歳以上・無理のない範囲で。"


def _font() -> str:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            pdfmetrics.registerFont(TTFont("JP", p, subfontIndex=0) if p.endswith(".ttc") else TTFont("JP", p))
            return "JP"
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
    return "HeiseiKakuGo-W5"


def _styles(font):
    def s(name, size, color=INK, lead=None, **kw):
        return ParagraphStyle(name, fontName=font, fontSize=size, leading=lead or size * 1.45, textColor=color, **kw)
    return {
        "title": s("title", 18, lead=24), "h": s("h", 12, RED, lead=18, spaceBefore=6),
        "body": s("body", 9.5), "small": s("small", 8, MUTED), "cell": s("cell", 8.5, lead=11),
        "big": s("big", 26, RED, lead=30, alignment=2),
    }


def _pct(p):
    return f"{p*100:.0f}%"


def _race_page(rep, st, font):
    flow = []
    head = Table([[
        Paragraph(f"{rep['venue_name']}　{rep['race_no']}R　{rep['race_type']}", st["title"]),
        Paragraph(_pct(rep["confidence"]), st["big"]),
    ], [
        Paragraph(f"{rep['date']}　{rep['distance']}m　締切 {rep['close_time'] or '-'}"
                  f"　直前情報 {'あり' if rep['has_pre_race'] else 'なし（番組表のみ）'}", st["small"]),
        Paragraph("◎の想定1着率", st["small"]),
    ]], colWidths=[130 * mm, 40 * mm])
    head.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                              ("LINEBELOW", (0, 1), (-1, 1), 1.2, RED)]))
    flow += [head, Spacer(1, 5 * mm), Paragraph("予想印・評価", st["h"])]

    data = [["印", "艇", "選手", "級", "全国勝率", "モーター", "1着率", "3着内率", "根拠"]]
    for r in rep["rows"]:
        data.append([r["mark"], str(r["boat_no"]), f"{r['name']}（{r['branch']}）", r["class"],
                     r["nat_win"], r["motor_2r"], _pct(r["p_win"]), _pct(r["p_top3"]),
                     Paragraph(r["reason"] or "-", st["cell"])])
    t = Table(data, colWidths=[9*mm, 8*mm, 30*mm, 9*mm, 15*mm, 15*mm, 13*mm, 15*mm, 56*mm], repeatRows=1)
    style = [
        ("FONT", (0, 0), (-1, -1), font, 9), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
        ("FONTSIZE", (0, 0), (-1, 0), 8), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (1, -1), "CENTER"), ("ALIGN", (3, 0), (7, -1), "CENTER"),
        ("TEXTCOLOR", (0, 1), (0, -1), RED), ("FONTSIZE", (0, 1), (0, -1), 12),
        ("TOPPADDING", (0, 1), (-1, -1), 5), ("BOTTOMPADDING", (0, 1), (-1, -1), 5),
    ]
    for i, r in enumerate(rep["rows"], start=1):
        b = r["boat_no"]
        style += [("BACKGROUND", (1, i), (1, i), colors.HexColor(LANE_BG[b])),
                  ("TEXTCOLOR", (1, i), (1, i), colors.HexColor(LANE_FG[b])),
                  ("BOX", (1, i), (1, i), 0.3, RULE)]
    t.setStyle(TableStyle(style))
    flow += [t, Spacer(1, 4 * mm), Paragraph("買い目候補（3連単・確率の高い順）", st["h"])]

    tri = [["組み合わせ", "想定確率"]] + [[x["combo"], f"{x['p']*100:.1f}%"] for x in rep["trifecta"]]
    tt = Table(tri, colWidths=[35 * mm, 25 * mm], hAlign="LEFT")
    tt.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), font, 10), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
                            ("FONTSIZE", (0, 0), (-1, 0), 8), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
                            ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE), ("ALIGN", (1, 0), (1, -1), "RIGHT")]))
    flow += [tt, Spacer(1, 3 * mm), Paragraph(
        "確率はモデルの1着率から計算した目安です。オッズは考慮していないため、"
        "購入時はオッズと見比べ、確率×オッズが1を十分上回るものに絞ってください。", st["small"])]
    return flow


def _cover(reports, st, font, meta, live):
    days = sorted({r["date"] for r in reports})
    flow = [Paragraph("赤ペン予想　競艇", st["title"]),
            Paragraph(f"{'、'.join(days)}　{len(reports)}レース", st["body"]),
            Spacer(1, 4 * mm), Paragraph("本命一覧", st["h"])]
    data = [["場", "R", "◎", "選手", "1着率", "3連単 本線"]]
    for r in reports:
        top = r["rows"][0]
        data.append([r["venue_name"], str(r["race_no"]), str(top["boat_no"]), top["name"],
                     _pct(top["p_win"]), r["trifecta"][0]["combo"]])
    t = Table(data, colWidths=[22*mm, 10*mm, 10*mm, 40*mm, 18*mm, 28*mm], hAlign="LEFT", repeatRows=1)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), font, 9.5), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED),
                           ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK), ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
                           ("ALIGN", (1, 0), (2, -1), "CENTER"), ("ALIGN", (4, 0), (4, -1), "RIGHT")]))
    flow += [t, Spacer(1, 6 * mm), Paragraph("モデルの成績（数字はすべて実データでの結果）", st["h"])]
    lines = []
    if meta:
        te, base = meta["test"], meta["baseline_lane"]
        lines.append(
            f"検証期間 {meta['test_range'][0]}〜{meta['test_range'][1]}（学習に使っていない {te['races']} レース）："
            f"◎1着 {_pct(te['honmei_win'])}、◎3着内 {_pct(te['honmei_top3'])}、"
            f"3連単上位10点内 {_pct(te['trifecta_top10'] or 0)}、◎単勝回収率 {_pct(te['honmei_tansho_roi'] or 0)}。"
            f"比較：いつも1号艇を◎にした場合の◎1着 {_pct(base['honmei_win'])}。")
    if live:
        for k, v in live.items():
            label = "通算" if k == "all" else f"直近{k.split('_')[1].rstrip('d')}日"
            lines.append(f"実戦（{label}・{v['races']}レース）：◎1着 {_pct(v['honmei_win'])}、"
                         f"◎3着内 {_pct(v['honmei_top3'])}、3連単6点内 {_pct(v['trifecta_6pt'])}")
    if not lines:
        lines.append("まだ成績データがありません。")
    flow += [Paragraph(x, st["body"]) for x in lines]
    return flow


def render(reports: list[dict], path: Path, meta: dict | None = None, live: dict | None = None) -> Path:
    font = _font()
    st = _styles(font)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(15 * mm, 10 * mm, DISCLAIMER)
        canvas.drawRightString(195 * mm, 10 * mm, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=15*mm, bottomMargin=18*mm, title="赤ペン予想 競艇")
    flow = _cover(reports, st, font, meta, live)
    for rep in reports:
        flow += [PageBreak()] + _race_page(rep, st, font)
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    return path
