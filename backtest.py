"""6条件の過去検証（AI不使用・トークン消費なし）
ルール: シグナル翌日に買い指値で約定したら、損切り・利確のIFDOCO。30営業日で時間切れ決済。
同じ日に損切りと利確の両方に届いた場合は、損切りとして厳しめに数える。
会社員ルール（比較用）: 指値・逆指値を使わず、夜に注文して翌朝の寄付（始値）に成行で約定する前提。
シグナル翌朝に買い、引けで損切り価格以下・利確価格以上なら翌朝寄付で売る。株数は前夜に買値予定と損切り価格から決める。
"""
import json
import random
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import yfinance as yf

from screener import BUFFER, CAPITAL, CHECKS, LOSS_LIMIT, LOT, MAX_RANGE, MIN_TURNOVER, STRATEGIES, day_range, fetch, load_universe, shares, similar_lookup

SAMPLE = 600        # 検証する銘柄数（売買代金の条件を満たす中からランダム）
YEARS = 6           # 検証期間
MAX_HOLD = 15       # 最大保有営業日
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
            return {"filled": True, "exit": k, "r": (px - buy) / risk, "ret": px / buy - 1, "days": k - i, "why": "損切り", "buy": buy, "risk": risk}
        if bar["High"] >= s["target"]:
            px = max(s["target"], bar["Open"]) if k > i + 1 else s["target"]
            return {"filled": True, "exit": k, "r": (px - buy) / risk, "ret": px / buy - 1, "days": k - i, "why": "利確", "buy": buy, "risk": risk}
    k = min(i + MAX_HOLD, len(d) - 1)
    if k <= i + 1:
        return None
    px = d["Close"].iloc[k]
    return {"filled": True, "exit": k, "r": (px - buy) / risk, "ret": px / buy - 1, "days": k - i, "why": "期限切れ", "buy": buy, "risk": risk}


def simulate_open(d, i, s):
    """会社員ルール：i日目の引けでシグナル→翌朝寄付に成行買い→毎日の引けを見て翌朝寄付に成行売り"""
    risk = s["entry"] - s["stop"]   # 前夜に株数を決めたときの損切り幅（1R）
    if i + 1 >= len(d) or risk <= 0:
        return None
    buy = d["Open"].iloc[i + 1]
    for k in range(i + 1, i + 1 + MAX_HOLD):
        if k + 1 >= len(d):
            return None
        c = d["Close"].iloc[k]
        why = "損切り" if c <= s["stop"] else "利確" if c >= s["target"] else None
        if why is None and k < i + MAX_HOLD:
            continue
        px = d["Open"].iloc[k + 1]
        return {"filled": True, "exit": k + 1, "r": (px - buy) / risk, "ret": px / buy - 1, "days": k + 1 - i,
                "why": why or "期限切れ", "buy": buy, "risk": risk, "rule": "寄付"}


def features(win, s):
    """シグナル時点で分かる特徴（大負けの傾向分析用）"""
    c, v = win["Close"], win["Volume"]
    return {"損切り幅%": (s["entry"] - s["stop"]) / s["entry"] * 100,
            "値動きの荒さ%": ((win["High"] - win["Low"]) / c).iloc[-14:].mean() * 100,
            "25日線との差%": (c.iloc[-1] / c.iloc[-25:].mean() - 1) * 100,
            "直近5日の騰落%": (c.iloc[-1] / c.iloc[-6] - 1) * 100,
            "出来高倍率": v.iloc[-1] / max(v.iloc[-21:-1].mean(), 1),
            "株価": c.iloc[-1]}


def run_stock(args):
    d, start, code = args
    trades = []
    busy = {(k, rule): -1 for k in CHECKS for rule in ("指値", "寄付")}
    for i in range(max(start, WINDOW), len(d) - 1):
        win = d.iloc[i - WINDOW + 1:i + 1]
        if (win["Close"] * win["Volume"]).iloc[-20:].mean() < MIN_TURNOVER or day_range(win) > MAX_RANGE:
            continue
        for k, f in CHECKS.items():
            if i <= busy[k, "指値"] and i <= busy[k, "寄付"]:
                continue
            try:
                s = f(win)
            except Exception:
                s = None
            if not s:
                continue
            for rule, sim in (("指値", simulate), ("寄付", simulate_open)):
                if i <= busy[k, rule]:
                    continue
                res = sim(d, i, s)
                if res is None:
                    continue
                res.update(strategy=k, date=d.index[i], rule=rule, stop=s["stop"], rr=s["rr"], code=code, **features(win, s))
                if res["filled"]:
                    res.update(buy_date=d.index[i + 1], exit_date=d.index[res["exit"]])
                trades.append(res)
                if res["filled"]:
                    busy[k, rule] = res["exit"]
    return trades


def stats(sub):
    if not sub:
        return None
    r = pd.Series([x["r"] for x in sub])
    n = len(sub)
    return {"件数": n, "勝率": round((r > 0).mean() * 100, 1),
            "期待値R": round(r.mean(), 2),
            "実効R": round(pd.Series([x["ret"] * x["buy"] / (x["risk"] + x["buy"] * BUFFER) for x in sub]).mean(), 2),
            "平均損益%": round(pd.Series([x["ret"] for x in sub]).mean() * 100, 2),
            "平均日数": round(pd.Series([x["days"] for x in sub]).mean(), 1),
            "損切り%": round(sum(x["why"] == "損切り" for x in sub) / n * 100, 1),
            "利確%": round(sum(x["why"] == "利確" for x in sub) / n * 100, 1),
            "期限切れ%": round(sum(x["why"] == "期限切れ" for x in sub) / n * 100, 1),
            "最悪R": round(r.min(), 2),
            "1.5R超%": round((r < -1.5).mean() * 100, 1), "2R超%": round((r < -2).mean() * 100, 1),
            "超過95%": round(pd.Series([max(0, x["stop"] - x["buy"] * (1 + x["ret"])) / x["stop"]
                                        for x in sub if x["why"] == "損切り"] or [0]).quantile(.95) * 100, 1)}


def affordable(x, lot=LOT):
    """lot株単位で1単位以上、資金内・損切り損失が上限以内で買える（実際に買える）"""
    return shares(x["buy"], x["buy"] - x["risk"], lot) > 0


def summarize(trades, mood):
    rows = []
    for k, (label, _) in STRATEGIES.items():
        t = [x for x in trades if x["strategy"] == k and x["rule"] == "指値"]
        f = [x for x in t if x["filled"]]
        o = [x for x in trades if x["strategy"] == k and x["rule"] == "寄付" and affordable(x)]
        row = {"条件": label, "シグナル": len(t), "約定": len(f), "買える": {}, "100株単位": {}, "寄付成行": {}}
        for tag, sub in (("全体", f), ("追い風", [x for x in f if mood.get(x["date"], False)]),
                         ("向かい風", [x for x in f if not mood.get(x["date"], True)])):
            row[tag] = stats(sub)
            row["買える"][tag] = stats([x for x in sub if affordable(x)])
            row["100株単位"][tag] = stats([x for x in sub if affordable(x, 100)])
        for tag, sub in (("全体", o), ("追い風", [x for x in o if mood.get(x["date"], False)]),
                         ("向かい風", [x for x in o if not mood.get(x["date"], True)])):
            row["寄付成行"][tag] = stats(sub)
        rows.append(row)
    return rows


def cell(v):
    if not v:
        return "<td colspan=13>なし</td>"
    good = "good" if v["期待値R"] > 0.1 else "bad" if v["期待値R"] < 0 else ""
    return (f"<td>{v['件数']}</td><td>{v['勝率']}%</td><td class='{good}'>{v['期待値R']:+.2f}</td><td>{v['実効R']:+.2f}</td>"
            f"<td>{v['平均損益%']:+.2f}%</td><td>{v['平均日数']}</td>"
            f"<td>{v['損切り%']}%</td><td>{v['利確%']}%</td><td>{v['期限切れ%']}%</td><td>{v['最悪R']:+.2f}</td>"
            f"<td>{v['1.5R超%']}%</td><td>{v['2R超%']}%</td><td>{v['超過95%']}%</td>")


def bigloss(trades, mood):
    """大負け（損失上限の1.5倍超）の傾向：特徴ごとに5段階に分けて、大負けの割合を比べる"""
    html = ("<h1>大負けしやすい特徴（1株単位で買える取引）</h1><p>買う時点で分かる特徴を、小さい順に5つの組に同数ずつ分け、"
            "各組で損失上限の1.5倍を超えて負けた割合（大負け率）と期待値Rを比べた表。大負け率が高く期待値Rが低い組は避ける候補。</p>")
    for rule, title in (("寄付", "会社員ルール"), ("指値", "指値＋逆指値")):
        t = pd.DataFrame([x for x in trades if x["rule"] == rule and x["filled"] and affordable(x)])
        if t.empty:
            continue
        t["big"] = t["r"] < -1.5
        t["地合い"] = t["date"].map(lambda d: "追い風" if mood.get(d, False) else "向かい風")
        html += f"<h2>{title}（全体の大負け率 {t['big'].mean() * 100:.1f}%）</h2><div class=w><table>"
        html += "<tr><th>特徴</th><th>組</th><th>範囲</th><th>件数</th><th>大負け率</th><th>期待値R</th></tr>"
        for col in ("損切り幅%", "値動きの荒さ%", "25日線との差%", "直近5日の騰落%", "出来高倍率", "株価", "地合い"):
            g = t.groupby(t[col] if col == "地合い" else pd.qcut(t[col], 5, duplicates="drop"), observed=True)
            for j, (key, sub) in enumerate(g):
                rng = key if col == "地合い" else f"{key.left:.1f}〜{key.right:.1f}"
                cls = " class=bad" if sub["big"].mean() > t["big"].mean() * 1.5 else ""
                html += (f"<tr><th>{col if j == 0 else ''}</th><td>{j + 1}</td><td>{rng}</td><td>{len(sub)}</td>"
                         f"<td{cls}>{sub['big'].mean() * 100:.1f}%</td><td>{sub['r'].mean():+.2f}</td></tr>")
        html += "</table></div>"
    return html


MIN_SIMILAR = 30    # 似た取引がこれ未満なら、区分を粗くして数え直す
MAX_POS = 3         # 運用シミュレーション：同時に持つ最大銘柄数
NEW_PER_DAY = 1     # 運用シミュレーション：1日に新しく買う最大銘柄数


def similar_table(t):
    """会社員ルールの取引から「似た取引」の成績表を作る。区分＝条件×損切り幅（5段階）×地合い。
    件数が少ない区分は、条件×損切り幅 → 条件だけ、と粗くした区分の値を使う"""
    edges = list(t["損切り幅%"].quantile([.2, .4, .6, .8]))
    t = t.assign(band=t["損切り幅%"].map(lambda v: sum(v > e for e in edges)))
    table = {}
    for keys in (["strategy"], ["strategy", "band"], ["strategy", "band", "地合い"]):
        for key, g in t.groupby(keys):
            key = key if isinstance(key, tuple) else (key,)
            if len(g) >= MIN_SIMILAR:
                table["|".join(map(str, key))] = {"件数": len(g), "勝率": round((g["r"] > 0).mean() * 100, 1),
                                                  "実効R": round(g["eff"].mean(), 3)}
    return {"edges": edges, "table": table}


def portfolio(trades, mood):
    """前半4年で「似た取引の成績」を作り、後半2年で ①その成績は当たるか ②選び方で運用成績が変わるか を確かめる"""
    t = pd.DataFrame([x for x in trades if x["rule"] == "寄付" and x["filled"] and affordable(x)])
    if t.empty:
        return "", None
    t["地合い"] = t["date"].map(lambda d: "追い風" if mood.get(d, False) else "向かい風")
    t["eff"] = t["ret"] * t["buy"] / (t["risk"] + t["buy"] * BUFFER)
    t["shares"] = [shares(b, b - r) for b, r in zip(t["buy"], t["risk"])]
    t["yen"] = t["shares"] * t["buy"] * t["ret"]
    split = t["date"].min() + (t["date"].max() - t["date"].min()) * 4 / 6
    train, test = t[t["date"] < split], t[t["date"] >= split].copy()
    model = similar_table(train)
    hits = [similar_lookup(model, s, w, m) for s, w, m in zip(test["strategy"], test["損切り幅%"], test["地合い"])]
    test["score"] = [h["実効R"] if h else float("nan") for h in hits]
    test["pwin"] = [h["勝率"] if h else float("nan") for h in hits]
    test = test.dropna(subset=["score"])
    html = (f"<h1>似た取引の成績は当たるか（{split:%Y年%m月}以降の取引で確認）</h1>"
            f"<p>{split:%Y年%m月}より前の取引だけで「条件×損切り幅×地合い」ごとの勝率と期待値を作り、それ以降の取引に当てはめた。"
            "予想の高い組ほど実際の成績も高ければ、候補の並べ替えに使える。</p><div class=w><table>"
            "<tr><th>予想の組</th><th>件数</th><th>予想勝率</th><th>実際の勝率</th><th>予想の期待値</th><th>実際の期待値</th></tr>")
    for j, (_, g) in enumerate(test.groupby(pd.qcut(test["score"].rank(method="first"), 5, labels=False))):
        html += (f"<tr><th>{['低い', 'やや低い', '中くらい', 'やや高い', '高い'][j]}</th><td>{len(g)}</td>"
                 f"<td>{g['pwin'].mean():.1f}%</td><td>{(g['r'] > 0).mean() * 100:.1f}%</td>"
                 f"<td>{g['score'].mean() * LOSS_LIMIT:+.0f}円</td><td>{g['eff'].mean() * LOSS_LIMIT:+.0f}円</td></tr>")
    html += "</table></div><p>期待値は損失上限2,000円・余裕幅込みの株数で1回あたり平均何円残るか。</p>"
    days = sorted(test["date"].unique())
    by_day = {d: g for d, g in test.groupby("date")}
    html += (f"<h1>運用シミュレーション（資金{CAPITAL // 10000}万円・同時{MAX_POS}銘柄まで・1日{NEW_PER_DAY}銘柄まで、{split:%Y年%m月}以降）</h1>"
             "<p>毎日の候補から選び方を変えて買い続けた場合の比較。ランダムは20回の平均。</p><div class=w><table>"
             "<tr><th>選び方</th><th>取引数</th><th>勝率</th><th>合計損益</th><th>1回平均</th><th>最大の落ち込み</th></tr>")
    methods = [("リスクリワード順（今のサイト）", lambda g, rnd: g.sort_values("rr", ascending=False)),
               ("似た取引の期待値順", lambda g, rnd: g.sort_values("score", ascending=False)),
               ("ランダム", lambda g, rnd: g.sample(frac=1, random_state=rnd))]
    for name, order in methods:
        res = []
        for rnd in range(20 if name == "ランダム" else 1):
            held, done = [], []
            for d in days:
                held = [h for h in held if h["exit_date"] > d]
                if d not in by_day:
                    continue
                new = 0
                for _, c in order(by_day[d], rnd).iterrows():
                    if len(held) >= MAX_POS or new >= NEW_PER_DAY:
                        break
                    cost = c["shares"] * c["buy"]
                    if any(h["code"] == c["code"] for h in held) or cost + sum(h["cost"] for h in held) > CAPITAL:
                        continue
                    held.append({"code": c["code"], "exit_date": c["exit_date"], "cost": cost})
                    done.append((c["exit_date"], c["yen"]))
                    new += 1
            pnl = pd.Series([y for _, y in sorted(done)])
            curve = pnl.cumsum()
            res.append((len(pnl), (pnl > 0).mean() * 100, pnl.sum(), pnl.mean(), (curve.cummax().clip(lower=0) - curve).max()))
        n, w, tot, avg, dd = [sum(x[k] for x in res) / len(res) for k in range(5)]
        html += (f"<tr><th>{name}</th><td>{n:.0f}</td><td>{w:.1f}%</td><td>{tot:+,.0f}円</td>"
                 f"<td>{avg:+,.0f}円</td><td>-{dd:,.0f}円</td></tr>")
    html += "</table></div><p>最大の落ち込みは、損益の合計が一番高かった時点からどれだけ減ったか（実現損益ベース）。</p>"
    return html, similar_table(t)


def build(rows, n, extra=""):
    head = "<th>件数</th><th>勝率</th><th>期待値R</th><th>実効R</th><th>平均損益</th><th>平均日数</th><th>損切り</th><th>利確</th><th>期限切れ</th><th>最悪R</th><th>1.5R超</th><th>2R超</th><th>損切り超過95%</th>"
    body = ""
    for tag in ("全体", "追い風", "向かい風"):
        body += f"<h2>{tag}</h2><div class=w><table><tr><th>条件</th>{head}</tr>"
        body += "".join(f"<tr><th>{r['条件']}</th>{cell(r[tag])}</tr>" for r in rows) + "</table></div>"
    for key, lot, note in (("買える", LOT, "今の運用条件。サイトの条件の外す・残すはこの成績で決める"),
                           ("100株単位", 100, "比較用（従来の100株単位）")):
        body += (f"<h1>買える銘柄だけ（{lot}株単位・{CAPITAL // 10000}万円以内・損切り損失{LOSS_LIMIT:,}円以内）</h1>"
                 f"<p>{note}。件数が少ない区分は参考程度に。</p>")
        for tag in ("全体", "追い風", "向かい風"):
            body += f"<h2>{lot}株単位・{tag}</h2><div class=w><table><tr><th>条件</th>{head}</tr>"
            body += "".join(f"<tr><th>{r['条件']}</th>{cell(r[key].get(tag))}</tr>" for r in rows) + "</table></div>"
    body += (f"<h1>会社員ルール（指値・逆指値なし、朝の寄付に成行だけで売買・{LOT}株単位）</h1>"
             "<p>前の晩に注文し、翌朝の始値で買う。毎日の終値が損切り価格以下・利確価格以上なら、翌朝の始値で売る。"
             "最悪Rが-1より小さいほど、損切りが遅れて損失上限（1R）を超えた取引があったことを表す。</p>")
    for tag in ("全体", "追い風", "向かい風"):
        body += f"<h2>会社員ルール・{tag}</h2><div class=w><table><tr><th>条件</th>{head}</tr>"
        body += "".join(f"<tr><th>{r['条件']}</th>{cell(r['寄付成行'].get(tag))}</tr>" for r in rows) + "</table></div>"
    body += extra
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
損切り・利確・期限切れは決済理由の割合（期限切れは最大保有日数で引け売り）。実効Rは余裕幅（株価の{BUFFER:.0%}）込みで株数を決めたときの、損失上限（2,000円）あたりの平均損益。値動きの荒い銘柄（1日の値幅の平均が{MAX_RANGE}%超）は除外。最悪Rは一番大きく負けた取引（-1より小さいと損失上限超え）。1.5R超・2R超は損失上限の1.5倍・2倍を超えて負けた取引の割合。
損切り超過95%は、損切りした取引の95%が「損切り価格から株価の何%以内の下で売れたか」で、株数計算の余裕幅の目安。翌日に指値が約定した取引のみ集計（会社員ルールを除く）。同日に損切りと利確の両方に届いた日は損切り扱い。</p>{body}
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
    jobs = [(d, len(d) - YEARS * 245, t.replace(".T", "")) for t, d in data.items()]
    with Pool() as p:
        trades = [x for ts in p.map(run_stock, jobs) for x in ts]
    rows = summarize(trades, mood)
    sim_html, model = portfolio(trades, mood)
    build(rows, len(data), sim_html + bigloss(trades, mood))
    if model:
        (OUT / "similar.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(r["条件"], r["全体"], r["買える"]["全体"])


if __name__ == "__main__":
    main()
