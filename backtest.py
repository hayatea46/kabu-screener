"""6条件の過去検証（AI不使用・トークン消費なし）
ルール: シグナル翌日に買い指値で約定したら、損切り・利確のIFDOCO。30営業日で時間切れ決済。
同じ日に損切りと利確の両方に届いた場合は、損切りとして厳しめに数える。
"""
import json
import random
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yfinance as yf

from screener import CHECKS, MIN_TURNOVER, STRATEGIES, fetch, load_universe

SAMPLE = 600        # 検証する銘柄数（売買代金の条件を満たす中からランダム）
YEARS = 6           # 検証期間
MAX_HOLD = 30       # 最大保有営業日
WINDOW = 300        # 判定に使う過去日数
OUT = Path("docs")


def mood_series():
    t = yf.download("1306.T", period=f"{YEARS + 2}y", progress=False, auto_adjust=True)["Close"].squeeze().dropna()
    return t > t.rolling(25).mean()


def simulate(d, i, s):
    """i日目の引けでシグナル→i+1日目に指値で約定するか→決済まで追う"""
    if i + 1 >= len(d):
        return None
    nxt = d.iloc[i + 1]
    if nxt["Low"] > s["entry"]:
        return {"filled": False}
    buy = min(s["entry"], nxt["Open"])
    risk = buy - s["stop"]
    for k in range(i + 1, min(i + 1 + MAX_HOLD, len(d))):
        bar = d.iloc[k]
        if bar["Low"] <= s["stop"]:
            px = min(s["stop"], bar["Open"])
            return {"filled": True, "exit": k, "r": (px - buy) / risk, "ret": px / buy - 1, "days": k - i, "why": "損切り"}
        if bar["High"] >= s["target"]:
            px = max(s["target"], bar["Open"]) if k > i + 1 else s["target"]
            return {"filled": True, "exit": k, "r": (px - buy) / risk, "ret": px / buy - 1, "days": k - i, "why": "利確"}
    k = min(i + MAX_HOLD, len(d) - 1)
    if k <= i + 1:
        return None
    px = d["Close"].iloc[k]
    return {"filled": True, "exit": k, "r": (px - buy) / risk, "ret": px / buy - 1, "days": k - i, "why": "期限切れ"}


def run_stock(args):
    d, start = args
    trades = []
    busy = {k: -1 for k in CHECKS}
    for i in range(max(start, WINDOW), len(d) - 1):
        win = d.iloc[i - WINDOW + 1:i + 1]
        if (win["Close"] * win["Volume"]).iloc[-20:].mean() < MIN_TURNOVER:
            continue
        for k, f in CHECKS.items():
            if i <= busy[k]:
                continue
            try:
                s = f(win)
            except Exception:
                s = None
            if not s:
                continue
            res = simulate(d, i, s)
            if res is None:
                continue
            res.update(strategy=k, date=d.index[i])
            trades.append(res)
            if res["filled"]:
                busy[k] = res["exit"]
    return trades


def summarize(trades, mood):
    rows = []
    for k, (label, _) in STRATEGIES.items():
        t = [x for x in trades if x["strategy"] == k]
        f = [x for x in t if x["filled"]]
        row = {"条件": label, "シグナル": len(t), "約定": len(f)}
        for tag, sub in (("全体", f), ("追い風", [x for x in f if mood.get(x["date"], False)]),
                         ("向かい風", [x for x in f if not mood.get(x["date"], True)])):
            if not sub:
                row[tag] = None
                continue
            r = pd.Series([x["r"] for x in sub])
            row[tag] = {"件数": len(sub), "勝率": round((r > 0).mean() * 100, 1),
                        "期待値R": round(r.mean(), 2),
                        "平均損益%": round(pd.Series([x["ret"] for x in sub]).mean() * 100, 2),
                        "平均日数": round(pd.Series([x["days"] for x in sub]).mean(), 1),
                        "損切り%": round(sum(x["why"] == "損切り" for x in sub) / len(sub) * 100, 1),
                        "利確%": round(sum(x["why"] == "利確" for x in sub) / len(sub) * 100, 1),
                        "期限切れ%": round(sum(x["why"] == "期限切れ" for x in sub) / len(sub) * 100, 1)}
        rows.append(row)
    return rows


def cell(v):
    if not v:
        return "<td colspan=8>なし</td>"
    good = "good" if v["期待値R"] > 0.1 else "bad" if v["期待値R"] < 0 else ""
    return (f"<td>{v['件数']}</td><td>{v['勝率']}%</td><td class='{good}'>{v['期待値R']:+.2f}</td>"
            f"<td>{v['平均損益%']:+.2f}%</td><td>{v['平均日数']}</td>"
            f"<td>{v['損切り%']}%</td><td>{v['利確%']}%</td><td>{v['期限切れ%']}%</td>")


def build(rows, n):
    head = "<th>件数</th><th>勝率</th><th>期待値R</th><th>平均損益</th><th>平均日数</th><th>損切り</th><th>利確</th><th>期限切れ</th>"
    body = ""
    for tag in ("全体", "追い風", "向かい風"):
        body += f"<h2>{tag}</h2><div class=w><table><tr><th>条件</th>{head}</tr>"
        body += "".join(f"<tr><th>{r['条件']}</th>{cell(r[tag])}</tr>" for r in rows) + "</table></div>"
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>過去検証</title>
<link href="https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&display=swap" rel="stylesheet">
<style>body{{margin:0;background:#F3F5F4;color:#18201F;font-family:'BIZ UDPGothic',sans-serif;font-variant-numeric:tabular-nums}}
main{{max-width:720px;margin:0 auto;padding:16px}}h1{{font-size:1.25rem}}h2{{font-size:1rem;margin:20px 0 6px}}
.w{{overflow-x:auto}}table{{border-collapse:collapse;background:#fff;width:100%;font-size:.9rem}}
th,td{{border:1px solid #D5DBD9;padding:6px 8px;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}
.good{{color:#C62E2E;font-weight:700}}.bad{{color:#2A5DA8;font-weight:700}}p{{color:#66706E;font-size:.85rem}}</style></head>
<body><main><h1>過去{YEARS}年の検証（{n}銘柄）</h1>
<p>期待値Rは「1回の取引で、損切り幅の何倍を平均で稼げたか」。+0.1以上が実用の目安、マイナスは使わない方がよい条件。
損切り・利確・期限切れは決済理由の割合（期限切れは最大保有日数で引け売り）。翌日に指値が約定した取引のみ集計。同日に損切りと利確の両方に届いた日は損切り扱い。</p>{body}
<p>注意：現在上場している銘柄だけで検証しているため、実際より成績がやや良く出る傾向があります。</p></main></body></html>"""
    OUT.mkdir(exist_ok=True)
    (OUT / "backtest.html").write_text(page, encoding="utf-8")
    (OUT / "backtest.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def main():
    names = load_universe()
    tickers = list(names)
    random.seed(0)
    random.shuffle(tickers)
    data = fetch(tickers[:SAMPLE * 2], period=f"{YEARS + 2}y")
    data = dict(list({t: d for t, d in data.items()
                      if (d["Close"] * d["Volume"]).iloc[-250:].mean() >= MIN_TURNOVER}.items())[:SAMPLE])
    print(f"検証対象 {len(data)} 銘柄")
    mood = mood_series().to_dict()
    jobs = [(d, len(d) - YEARS * 245) for d in data.values()]
    with Pool() as p:
        trades = [x for ts in p.map(run_stock, jobs) for x in ts]
    rows = summarize(trades, mood)
    build(rows, len(data))
    for r in rows:
        print(r["条件"], r["全体"])


if __name__ == "__main__":
    main()
