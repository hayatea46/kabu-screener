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

from patterns import NEW_CHECKS, NEW_LABELS
from screener import BUFFER, CAPITAL, CHECKS, LOSS_LIMIT, LOT, MAX_RANGE, MIN_TURNOVER, STRATEGIES, day_range, fetch, load_universe, shares, similar_lookup

SAMPLE = 600        # 検証する銘柄数（売買代金の条件を満たす中からランダム）
YEARS = 30          # 検証期間（検証用：30年）
MAX_HOLD = 30       # 最大保有営業日
WINDOW = 300        # 判定に使う過去日数
OUT = Path("docs")
RUN_NEW = ("random_buy",)   # 追加パターンのうち今回回すもの（他は2026-10-14の全14行検証で負け・検証不能と分かったので外す。全部回すと約3時間）
ALL_CHECKS = {**CHECKS, **{k: v for k, v in NEW_CHECKS.items() if k in RUN_NEW}}   # 本番の6条件＋検証専用の追加パターン＋ランダム買い基準


def mood_series():
    sym = "1306.T" if YEARS <= 20 else "^N225"   # 1306は2001年上場。長期検証は日経平均で判定
    t = yf.download(sym, period=f"{YEARS + 2}y", progress=False, auto_adjust=True)["Close"].squeeze().dropna()
    t.index = t.index.tz_localize(None) if t.index.tz is not None else t.index
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


EXIT_TEST = ("oversold", "random_buy")   # 出口ルールを比べる型（売られすぎ反発とランダム買い基準）
EXITS = {"20日保有のみ": (20, None, False), "20日保有＋損切り-8%": (20, 0.08, False), "20日保有＋損切り-5%": (20, 0.05, False),
         "10日保有＋損切り-5%": (10, 0.05, False), "20日保有＋逆指値-8%": (20, 0.08, True), "20日保有＋逆指値-5%": (20, 0.05, True)}


def simulate_exit(d, i, hold, stop_pct, intraday=False):
    """翌朝寄付に成行で買う。損切りは2通り：引けが買値の-stop_pct以下なら翌朝寄付に成行（intraday=False）、
    前夜に逆指値を入れておき日中に届いたらその場で売る（intraday=True。窓開けなら始値）。保有日数の満了は翌朝寄付に成行"""
    buy = d["Open"].iloc[i + 1]
    stop = buy * (1 - stop_pct) if stop_pct is not None else None
    for k in range(i + 1, i + 1 + hold):
        if k + 1 >= len(d):
            return None
        if intraday and d["Low"].iloc[k] <= stop:
            return {"ret": min(stop, d["Open"].iloc[k]) / buy - 1, "days": k - i, "stopped": True, "exit_date": d.index[k]}
        stopped = stop is not None and not intraday and d["Close"].iloc[k] <= stop
        if stopped or k == i + hold:
            return {"ret": d["Open"].iloc[k + 1] / buy - 1, "days": k + 1 - i, "stopped": stopped, "exit_date": d.index[k + 1]}


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
    busy = {(k, rule): -1 for k in ALL_CHECKS for rule in ("指値", "寄付")}
    for i in range(max(start, WINDOW), len(d) - 1):
        win = d.iloc[i - WINDOW + 1:i + 1]
        if (win["Close"] * win["Volume"]).iloc[-20:].mean() < MIN_TURNOVER or day_range(win) > MAX_RANGE:
            continue
        for k, f in ALL_CHECKS.items():
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
                if rule == "寄付" and k in EXIT_TEST and res["filled"]:
                    res["exits"] = {n: simulate_exit(d, i, h, sp, iday) for n, (h, sp, iday) in EXITS.items()}
                if i + 20 < len(d):   # 型の比較用：翌朝寄付から20営業日後の終値までの騰落
                    res["fwd20"] = d["Close"].iloc[i + 20] / d["Open"].iloc[i + 1] - 1
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


def walkforward(t):
    """毎年「その年より前の取引だけ」で成績表を作り、その年を2銘柄まで・1日1銘柄で運用する（全期間が未来を見ない検証）"""
    years = sorted(t["date"].dt.year.unique())
    html = ("<h1>毎年ローリング検証（前年までの取引で成績表を作り、翌年を運用）</h1>"
            "<p>2銘柄まで・1日1銘柄まで。最初の3年は成績表づくりに使うため運用しない。複利は決済のたびに株数を資金に比例させる。税金・手数料なし。</p><div class=w><table>"
            "<tr><th>年</th><th>取引数</th><th>期待値順 単利</th><th>期待値順 複利の資金</th><th>ランダム 単利</th><th>RR順 単利</th></tr>")
    eq = {"期待値順": CAPITAL, "ランダム": CAPITAL}
    tot = {"期待値順": 0, "ランダム": 0, "RR順": 0}
    low = {"期待値順": CAPITAL, "ランダム": CAPITAL}
    curve = {"期待値順": [], "ランダム": []}
    def sim(test, order, compound, equity, rnd=0):
        held, pnl = [], 0.0
        for d in sorted(test["date"].unique()):
            for h in [h for h in held if h["exit_date"] <= d]:
                equity += h["yen"]
                pnl += h["yen"]
            held = [h for h in held if h["exit_date"] > d]
            new = 0
            for _, c in order(test[test["date"] == d], rnd).iterrows():
                if len(held) >= 2 or new >= NEW_PER_DAY:
                    break
                f = equity / CAPITAL if compound else 1.0
                cost = c["shares"] * c["buy"] * f
                if any(h["code"] == c["code"] for h in held) or cost + sum(h["cost"] for h in held) > equity:
                    continue
                held.append({"code": c["code"], "exit_date": c["exit_date"], "cost": cost, "yen": c["yen"] * f})
                new += 1
        for h in held:
            equity += h["yen"]
            pnl += h["yen"]
        return pnl, equity
    for y in years[3:]:
        train, test = t[t["date"].dt.year < y], t[t["date"].dt.year == y].copy()
        model = similar_table(train)
        hits = [similar_lookup(model, s_, w, m) for s_, w, m in zip(test["strategy"], test["損切り幅%"], test["地合い"])]
        test["score"] = [h["実効R"] if h else float("nan") for h in hits]
        test = test.dropna(subset=["score"])
        if test.empty:
            continue
        p_ev, _ = sim(test, lambda g, r: g.sort_values("score", ascending=False), False, CAPITAL)
        _, eq["期待値順"] = sim(test, lambda g, r: g.sort_values("score", ascending=False), True, eq["期待値順"])
        rs = [sim(test, lambda g, r: g.sample(frac=1, random_state=r), False, CAPITAL, r)[0] for r in range(10)]
        p_rd = sum(rs) / len(rs)
        p_rr, _ = sim(test, lambda g, r: g.sort_values("rr", ascending=False), False, CAPITAL)
        tot["期待値順"] += p_ev
        tot["ランダム"] += p_rd
        tot["RR順"] += p_rr
        html += (f"<tr><th>{y}</th><td>{len(test)}</td><td>{p_ev:+,.0f}円</td><td>{eq['期待値順']:,.0f}円</td>"
                 f"<td>{p_rd:+,.0f}円</td><td>{p_rr:+,.0f}円</td></tr>")
    n = max(len(years) - 3, 1)
    html += (f"<tr><th>合計（{n}年）</th><td></td><td>{tot['期待値順']:+,.0f}円</td><td>{eq['期待値順']:,.0f}円"
             f"（{(eq['期待値順'] / CAPITAL - 1) * 100:+.0f}%）</td><td>{tot['ランダム']:+,.0f}円</td><td>{tot['RR順']:+,.0f}円</td></tr>")
    html += "</table></div><p>複利の年率は {:+.1f}% （期待値順）。</p>".format(((max(eq['期待値順'], 1) / CAPITAL) ** (1 / n) - 1) * 100)
    return html


def unit_compare(t):
    """売買単位の比較：毎年ローリング（前年までの取引で成績表→翌年を運用、2銘柄まで・1日1銘柄・期待値順）で、
    10万円・1株単位 と 100万円・100株単位 を並べる。株数は決済のたびに「そのときの資金」で計算し直す（複利）。単利は元手固定"""
    years = sorted(t["date"].dt.year.unique())
    scen = [("10万円・1株単位", 100000, 1), ("100万円・1株単位", 1000000, 1), ("100万円・100株単位", 1000000, 100)]
    rows = ""
    def sim(test, cap0, lot, compound, st):
        held = []
        eq = st["eq"]
        for d in sorted(test["date"].unique()):
            for h in sorted([h for h in held if h["exit_date"] <= d], key=lambda h: h["exit_date"]):
                eq += h["yen"]
                st["curve"].append(eq)
            held = [h for h in held if h["exit_date"] > d]
            new = 0
            for _, c in test[test["date"] == d].sort_values("score", ascending=False).iterrows():
                if len(held) >= 2 or new >= NEW_PER_DAY:
                    break
                cap = eq if compound else cap0
                n = shares(c["buy"], c["buy"] - c["risk"], lot, cap, cap * 0.02)
                if n <= 0 or any(h["code"] == c["code"] for h in held):
                    continue
                cost = n * c["buy"]
                if cost + sum(h["cost"] for h in held) > cap:
                    continue
                held.append({"code": c["code"], "exit_date": c["exit_date"], "cost": cost, "yen": n * c["buy"] * c["ret"]})
                st["n"] += 1
                st["cost"].append(cost)
                new += 1
        for h in sorted(held, key=lambda h: h["exit_date"]):
            eq += h["yen"]
            st["curve"].append(eq)
        st["eq"] = eq
    for compound in (True, False):
        state = {n: {"eq": c, "curve": [c], "n": 0, "cand": 0, "ok": 0, "cost": []} for n, c, _ in scen}
        for y in years[3:]:
            train, test = t[t["date"].dt.year < y], t[t["date"].dt.year == y].copy()
            model = similar_table(train)
            hits = [similar_lookup(model, s_, w, m) for s_, w, m in zip(test["strategy"], test["損切り幅%"], test["地合い"])]
            test["score"] = [h["実効R"] if h else float("nan") for h in hits]
            test = test.dropna(subset=["score"])
            if test.empty:
                continue
            for name, cap0, lot in scen:
                st = state[name]
                st["cand"] += len(test)
                st["ok"] += sum(shares(b, b - r, lot, cap0, cap0 * 0.02) > 0 for b, r in zip(test["buy"], test["risk"]))
                sim(test, cap0, lot, compound, st)
        n_years = max(len(years) - 3, 1)
        label = "複利" if compound else "単利（元手固定）"
        rows += f"<tr><th colspan=8>{label}</th></tr>"
        for name, cap0, lot in scen:
            st = state[name]
            c = pd.Series(st["curve"])
            dd = ((c.cummax() - c) / c.cummax()).max() * 100
            fin = st["eq"]
            ann = ((max(fin, 1) / cap0) ** (1 / n_years) - 1) * 100
            avgc = sum(st["cost"]) / max(len(st["cost"]), 1)
            rows += (f"<tr><th>{name}</th><td>{st['ok'] / max(st['cand'], 1) * 100:.0f}%</td><td>{st['n']}</td><td>{avgc:,.0f}円</td>"
                    f"<td>{fin:,.0f}円</td><td>{ann:+.1f}%</td><td>-{dd:.1f}%</td><td>{(fin - cap0) / cap0 * 100:+.0f}%</td></tr>")
    return ("<h1>売買単位の比較（毎年ローリング・2銘柄まで・期待値順）</h1>"
            "<p>損失上限は常に資金の2%（余裕幅7%込み）。100株単位は、損切り幅＋株価7%の100株分が上限に収まる銘柄しか買えない。"
            "買える候補の割合は、その年の候補のうち1株でも（100株単位なら100株）買える割合。税金・手数料なし。</p><div class=w><table>"
            "<tr><th>設定</th><th>買える候補</th><th>取引数</th><th>1回の投入額</th><th>最終資金</th><th>年率</th><th>最大の落ち込み</th><th>増減率</th></tr>"
            + rows + "</table></div>")


def pattern_compare(trades):
    """型の比較：ランダム買い（基準）と並べる。20日後の騰落は、同じ月のランダム買いの平均を引いた「超過」で見る（相場全体の上げ下げを除くため）"""
    t = pd.DataFrame([x for x in trades if x["rule"] == "寄付" and x["filled"]])
    t["month"] = t["date"].dt.to_period("M")
    base = t[t["strategy"] == "random_buy"].groupby("month")["fwd20"].mean()
    t["excess"] = t["fwd20"] - t["month"].map(base)
    labels = {**{k: v[0] for k, v in STRATEGIES.items()}, **NEW_LABELS}
    html = ("<h1>型の比較（ランダム買いを基準に）</h1>"
            "<p>寄付成行ルール・損切りの余裕幅なし。超過＝シグナル翌朝から20営業日後の騰落率から、同じ月のランダム買いの平均を引いた値。"
            "t値は2以上なら偶然では出にくい目安（同じ日の取引は似た動きをするため、実際はこれより甘い）。</p><div class=w><table>"
            "<tr><th>型</th><th>件数</th><th>勝率</th><th>期待値R</th><th>20日後の騰落</th><th>超過</th><th>超過のt値</th></tr>")
    for k in list(labels):
        g = t[t["strategy"] == k]
        if g.empty:
            continue
        e = g["excess"].dropna()
        tv = e.mean() / (e.std() / len(e) ** 0.5) if len(e) > 2 and e.std() > 0 else float("nan")
        html += (f"<tr><th>{labels[k]}</th><td>{len(g)}</td><td>{(g['r'] > 0).mean() * 100:.1f}%</td><td>{g['r'].mean():+.2f}</td>"
                 f"<td>{g['fwd20'].mean() * 100:+.2f}%</td><td>{e.mean() * 100:+.2f}%</td><td>{tv:+.1f}</td></tr>")
    return html + "</table></div>" + robustness(t, labels)


def exit_compare(trades):
    """出口ルールの比較：売られすぎ反発とランダム買いに同じ出口を当て、同じ月のランダム買いとの差（超過）で見る"""
    rows = []
    for x in trades:
        if x["rule"] == "寄付" and x["filled"] and x["strategy"] in EXIT_TEST and x.get("exits"):
            base = {"現行の損切り・利確（30日まで）": {"ret": x["ret"], "days": x["days"], "stopped": x["why"] == "損切り"}}
            for n, e in {**base, **x["exits"]}.items():
                if e:
                    rows.append({"strategy": x["strategy"], "exit": n, "month": x["date"].to_period("M"), **e})
    t = pd.DataFrame(rows)
    html = ("<h1>出口ルールの比較（売られすぎ反発）</h1>"
            "<p>同じシグナルに、出口だけ変えて当てた。翌朝寄付に成行で買い、毎日の引けで判断して翌朝寄付に成行で売る。"
            "超過＝売られすぎ反発の1回の損益率から、同じ出口・同じ月のランダム買いの平均を引いた値。"
            "月単位のt値は同じ月の取引を1つにまとめた値。株数の違いは調整していない。</p><div class=w><table>"
            "<tr><th>出口</th><th>件数</th><th>勝率</th><th>1回平均</th><th>損切り率</th><th>平均日数</th>"
            "<th>最悪</th><th>ランダム買い1回平均</th><th>超過</th><th>月単位のt値</th></tr>")
    for n in t["exit"].unique():
        g, r = t[(t["strategy"] == "oversold") & (t["exit"] == n)], t[(t["strategy"] == "random_buy") & (t["exit"] == n)]
        base = r.groupby("month")["ret"].mean()
        ex = (g["ret"] - g["month"].map(base)).dropna()
        mo = ex.groupby(g.loc[ex.index, "month"]).mean()
        tv = mo.mean() / (mo.std() / len(mo) ** 0.5) if len(mo) > 2 else float("nan")
        html += (f"<tr><th>{n}</th><td>{len(g)}</td><td>{(g['ret'] > 0).mean() * 100:.1f}%</td><td>{g['ret'].mean() * 100:+.2f}%</td>"
                 f"<td>{g['stopped'].mean() * 100:.0f}%</td><td>{g['days'].mean():.1f}</td><td>{g['ret'].min() * 100:.0f}%</td>"
                 f"<td>{r['ret'].mean() * 100:+.2f}%</td><td>{ex.mean() * 100:+.2f}%</td><td>{tv:+.1f}</td></tr>")
    return html + "</table></div>"


def _prep(trades, exit_name, sp):
    """売られすぎ反発の取引を、指定した出口の結果で並べ直す。余裕幅（損切りの滑りの95%点）も返す"""
    tr = []
    for x in trades:
        if x["rule"] == "寄付" and x["filled"] and x["strategy"] == "oversold" and x.get("exits"):
            e = x["exits"].get(exit_name)
            if e:
                tr.append({"date": x["date"], "exit_date": e["exit_date"], "code": x["code"], "buy": x["buy"], "ret": e["ret"], "stopped": e["stopped"]})
    slip = pd.Series([max(0.0, -e["ret"] - sp) for e in tr if e["stopped"]])
    by_day = {}
    for e in tr:
        by_day.setdefault(e["date"], []).append(e)
    return by_day, sorted(by_day), (float(slip.quantile(.95)) if len(slip) else 0.0)


def _run_port(by_day, days, cap0, lot, sp, buf, lim, maxpos, seed):
    """運用シミュレーション：1日1銘柄まで、同時に maxpos 銘柄まで、候補はランダム、複利。(取引数, 最終資金, 最大の落ち込み, 最悪の1回, 上限超え率, 平均同時保有)"""
    rnd = random.Random(seed)
    cap, held, curve, worst, over, n, occ = cap0, [], [cap0], 0.0, 0, 0, 0
    for d in days:
        for h in sorted([h for h in held if h["exit_date"] <= d], key=lambda h: h["exit_date"]):
            cap += h["yen"]
            curve.append(cap)
            worst = min(worst, h["yen"] / h["cap"])
            over += h["yen"] < -h["cap"] * lim
            n += 1
        held = [h for h in held if h["exit_date"] > d]
        c = rnd.choice(by_day[d])
        if len(held) < maxpos and not any(h["code"] == c["code"] for h in held):
            k = shares(c["buy"], c["buy"] * (1 - sp), lot, cap, cap * lim, buf)
            cost = k * c["buy"]
            if k > 0 and cost + sum(h["cost"] for h in held) <= cap:
                held.append({"code": c["code"], "exit_date": c["exit_date"], "cost": cost, "yen": cost * c["ret"], "cap": cap})
        occ += len(held)
    for h in held:
        cap += h["yen"]
        curve.append(cap)
    s = pd.Series(curve)
    return n, cap, ((s.cummax() - s) / s.cummax()).max(), -worst, over / max(n, 1), occ / max(len(days), 1)


def risk_sweep(trades):
    """損失上限×余裕幅×注文方法の比較：売られすぎ反発だけを、同時2銘柄・1日1銘柄・複利で運用した場合（候補はランダムに選ぶ。10回平均）。
    余裕幅は、その出口で損切りが名目より何%余計に滑ったかの95%点をデータから求めて使う"""
    scen = [("10万円・1株・成行の損切り", 100000, 1, "20日保有＋損切り-8%", 0.08),
            ("100万円・100株・成行の損切り", 1000000, 100, "20日保有＋損切り-8%", 0.08),
            ("100万円・100株・逆指値", 1000000, 100, "20日保有＋逆指値-8%", 0.08)]
    base = [x for x in trades if x["rule"] == "寄付" and x["filled"] and x["strategy"] == "oversold" and x.get("exits")]
    days_all = sorted({x["date"] for x in base})
    years = max((days_all[-1] - days_all[0]).days / 365.25, 0.1)
    html = ("<h1>損失上限と余裕幅の比較（売られすぎ反発・同時2銘柄・複利）</h1>"
            "<p>出口は20日保有＋損切り-8%。株数＝損失上限÷（損切り幅8%＋余裕幅）。余裕幅は損切りの滑りの95%点（成行は翌朝寄付での滑り、逆指値は窓開けでの滑り）。"
            "損失上限超え＝1回の損失が上限（資金の何%）を超えた割合。税金・手数料なし。候補はランダムに選び、10回の平均。</p><div class=w><table>"
            "<tr><th>設定</th><th>余裕幅</th><th>損失上限</th><th>取引数/年</th><th>年率</th><th>最大の落ち込み</th><th>最悪の1回</th><th>上限超え</th></tr>")
    for label, cap0, lot, exit_name, sp in scen:
        by_day, days, buf = _prep(base, exit_name, sp)
        for lim in (0.01, 0.02, 0.03, 0.05):
            res = [_run_port(by_day, days, cap0, lot, sp, buf, lim, 2, seed) for seed in range(10)]
            m = [sum(r[i] for r in res) / len(res) for i in range(6)]
            html += (f"<tr><th>{label}</th><td>{buf * 100:.1f}%</td><td>{lim * 100:.0f}%</td><td>{m[0] / years:.0f}</td>"
                     f"<td>{((max(m[1], 1) / cap0) ** (1 / years) - 1) * 100:+.1f}%</td><td>-{m[2] * 100:.0f}%</td><td>-{m[3] * 100:.1f}%</td><td>{m[4] * 100:.1f}%</td></tr>")
    return html + "</table></div>" + diversify_sweep(base, years)


def diversify_sweep(base, years):
    """分散の比較：同時保有数を増やし、1銘柄あたりの損失上限を下げると、年率÷落ち込みは上がるか"""
    html = ("<h1>分散の比較（売られすぎ反発・複利・1日1銘柄まで）</h1>"
            "<p>同時保有数を増やし、1銘柄あたりの損失上限を下げた場合。出口は20日保有＋損切り-8%（100万円・100株は逆指値）。"
            "年率÷落ち込みが大きいほど効率が良い（前回の2銘柄では約0.1）。資金不足で買えない分は見送り。10回平均。</p><div class=w><table>"
            "<tr><th>設定</th><th>同時保有</th><th>1銘柄の損失上限</th><th>平均の保有数</th><th>取引数/年</th><th>年率</th><th>最大の落ち込み</th><th>年率÷落ち込み</th></tr>")
    for label, cap0, lot, exit_name in (("10万円・1株・成行", 100000, 1, "20日保有＋損切り-8%"), ("100万円・100株・逆指値", 1000000, 100, "20日保有＋逆指値-8%")):
        by_day, days, buf = _prep(base, exit_name, 0.08)
        for maxpos in (2, 5, 10, 20):
            for lim in (0.0025, 0.005, 0.01, 0.02):
                res = [_run_port(by_day, days, cap0, lot, 0.08, buf, lim, maxpos, seed) for seed in range(6)]
                m = [sum(r[i] for r in res) / len(res) for i in range(6)]
                ann = (max(m[1], 1) / cap0) ** (1 / years) - 1
                html += (f"<tr><th>{label}</th><td>{maxpos}</td><td>{lim * 100:.2f}%</td><td>{m[5]:.1f}</td><td>{m[0] / years:.0f}</td>"
                         f"<td>{ann * 100:+.1f}%</td><td>-{m[2] * 100:.0f}%</td><td>{ann / max(m[2], 1e-9):.2f}</td></tr>")
    return html + "</table></div>"


def robustness(t, labels):
    """勝った型が安定しているか：年別・前半後半・月単位のt値・暴落月を除いた超過"""
    keys = ["oversold", "hammer", "engulfing", "box_bottom", "pullback"]
    crash = set(t[t["strategy"] == "random_buy"].groupby("month")["fwd20"].mean().loc[lambda s: s < -0.05].index)
    t = t.assign(year=t["date"].dt.year)
    years = sorted(t["year"].unique())
    half = years[len(years) // 2]
    html = ("<h1>勝った型は安定しているか</h1><p>超過（同じ月のランダム買い比）を、年別・前半後半・月ごとに集計し直した値で見る。"
            "月単位のt値は、同じ月の取引を1つにまとめるので、時期の集中の影響が小さい。"
            "暴落月は、その月のランダム買いの20日後平均が-5%以下の月。</p><div class=w><table><tr><th>型</th><th>全期間</th>"
            f"<th>{years[0]}〜{half - 1}年</th><th>{half}〜{years[-1]}年</th><th>プラスの年</th><th>月単位のt値</th><th>暴落月を除く</th><th>暴落月だけ</th></tr>")
    for k in keys:
        g = t[(t["strategy"] == k)].dropna(subset=["excess"])
        if g.empty:
            continue
        yr = g.groupby("year")["excess"].mean()
        mo = g.groupby("month")["excess"].mean()
        tv = mo.mean() / (mo.std() / len(mo) ** 0.5) if len(mo) > 2 else float("nan")
        nc, cr = g[~g["month"].isin(crash)], g[g["month"].isin(crash)]
        f = lambda s: f"{s['excess'].mean() * 100:+.2f}%" if len(s) else "-"
        html += (f"<tr><th>{labels[k]}</th><td>{f(g)}</td><td>{f(g[g['year'] < half])}</td><td>{f(g[g['year'] >= half])}</td>"
                 f"<td>{(yr > 0).sum()}/{len(yr)}年</td><td>{tv:+.1f}</td><td>{f(nc)}</td><td>{f(cr)}</td></tr>")
    html += "</table></div><h2>売られすぎ反発の年別の超過</h2><div class=w><table><tr>"
    g = t[t["strategy"] == "oversold"].dropna(subset=["excess"])
    yr = g.groupby("year")["excess"].agg(["mean", "size"])
    html += "".join(f"<th>{y}</th>" for y in yr.index) + "</tr><tr>"
    html += "".join(f"<td>{r['mean'] * 100:+.1f}%<br>({int(r['size'])}件)</td>" for _, r in yr.iterrows()) + "</tr></table></div>"
    return html


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
    def run(order, rnd, max_pos, thr=None, compound=False):
        held, done, equity = [], [], CAPITAL
        for d in days:
            for h in [h for h in held if h["exit_date"] <= d]:
                equity += h["yen"]   # 決済済みの損益を資金に足す（複利のときだけ株数に効く）
            held = [h for h in held if h["exit_date"] > d]
            if d not in by_day:
                continue
            new = 0
            for _, c in order(by_day[d], rnd).iterrows():
                if len(held) >= max_pos or new >= NEW_PER_DAY:
                    break
                if thr is not None and len(held) >= 2 and not c["pwin"] > thr:
                    continue   # 基本2銘柄。3銘柄目以降は似た取引の勝率がしきい値超えの銘柄だけ
                f = equity / CAPITAL if compound else 1.0   # 複利：資金に比例して株数（損失上限も）を増減
                cost = c["shares"] * c["buy"] * f
                if any(h["code"] == c["code"] for h in held) or cost + sum(h["cost"] for h in held) > equity:
                    continue
                held.append({"code": c["code"], "exit_date": c["exit_date"], "cost": cost, "yen": c["yen"] * f})
                done.append((c["exit_date"], c["yen"] * f))
                new += 1
        pnl = pd.Series([y for _, y in sorted(done)])
        curve = CAPITAL + pnl.cumsum()
        return (len(pnl), (pnl > 0).mean() * 100, pnl.sum(), pnl.mean(), (curve.cummax().clip(lower=CAPITAL) - curve).max(), curve.iloc[-1])

    def average(name, order, max_pos, thr=None, compound=False, full=False):
        res = [run(order, rnd, max_pos, thr, compound) for rnd in range(20 if name == "ランダム" else 1)]
        return [sum(x[k] for x in res) / len(res) for k in range(6 if full else 5)]

    for name, order in methods:
        n, w, tot, avg, dd = average(name, order, MAX_POS)
        html += (f"<tr><th>{name}</th><td>{n:.0f}</td><td>{w:.1f}%</td><td>{tot:+,.0f}円</td>"
                 f"<td>{avg:+,.0f}円</td><td>-{dd:,.0f}円</td></tr>")
    html += "</table></div><p>最大の落ち込みは、損益の合計が一番高かった時点からどれだけ減ったか（実現損益ベース）。</p>"
    html += ("<h1>同時に持つ銘柄数の比較（1日1銘柄まで、同じ後半期間）</h1>"
             "<p>同時保有の上限だけ変えて、選び方ごとの成績を比べた。</p><div class=w><table>"
             "<tr><th>選び方</th><th>同時上限</th><th>取引数</th><th>勝率</th><th>合計損益</th><th>1回平均</th><th>最大の落ち込み</th></tr>")
    for name, order in methods:
        for max_pos in (1, 2, 3, 5, 8):
            n, w, tot, avg, dd = average(name, order, max_pos)
            html += (f"<tr><th>{name}</th><td>{max_pos}銘柄</td><td>{n:.0f}</td><td>{w:.1f}%</td>"
                     f"<td>{tot:+,.0f}円</td><td>{avg:+,.0f}円</td><td>-{dd:,.0f}円</td></tr>")
    html += "</table></div>"
    html += ("<h1>基本2銘柄＋勝率の高い銘柄だけ追加（同時3銘柄まで）</h1>"
             "<p>2銘柄までは普通に買い、3銘柄目は「似た取引の勝率」がしきい値を超える銘柄だけ買う。</p><div class=w><table>"
             "<tr><th>選び方</th><th>3銘柄目の条件</th><th>取引数</th><th>勝率</th><th>合計損益</th><th>1回平均</th><th>最大の落ち込み</th></tr>")
    for name, order in methods:
        for label, max_pos, thr in (("追加なし（2銘柄まで）", 2, None), ("勝率55%超", 3, 55), ("勝率60%超", 3, 60), ("勝率70%超", 3, 70), ("条件なし（3銘柄まで）", 3, None)):
            n, w, tot, avg, dd = average(name, order, max_pos, thr)
            html += (f"<tr><th>{name}</th><td>{label}</td><td>{n:.0f}</td><td>{w:.1f}%</td>"
                     f"<td>{tot:+,.0f}円</td><td>{avg:+,.0f}円</td><td>-{dd:,.0f}円</td></tr>")
    html += "</table></div>"
    yrs = max((days[-1] - days[0]).days / 365.25, 0.1)
    html += (f"<h1>複利の場合（{split:%Y年%m月}以降、約{yrs:.1f}年、元手{CAPITAL // 10000}万円）</h1>"
             "<p>決済のたびに資金が増減し、次の買いの株数（損失上限は資金の2%）も資金に比例して変える。1株未満の端数は無視。税金・手数料は引いていない。</p><div class=w><table>"
             "<tr><th>選び方</th><th>ルール</th><th>取引数</th><th>最終資金</th><th>増減率</th><th>年率</th><th>最大の落ち込み</th></tr>")
    for name, order in methods:
        for label, max_pos, thr in (("2銘柄まで", 2, None), ("3銘柄まで", 3, None), ("2銘柄＋勝率60%超で3銘柄目", 3, 60)):
            for comp in (False, True):
                n, w, tot, avg, dd, fin = average(name, order, max_pos, thr, comp, True)
                html += (f"<tr><th>{name}</th><td>{label}（{'複利' if comp else '単利'}）</td><td>{n:.0f}</td><td>{fin:,.0f}円</td>"
                         f"<td>{(fin / CAPITAL - 1) * 100:+.0f}%</td><td>{((max(fin, 1) / CAPITAL) ** (1 / yrs) - 1) * 100:+.1f}%</td><td>-{dd:,.0f}円</td></tr>")
    html += "</table></div>"
    html += walkforward(t) + unit_compare(t)
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
    base_trades = [x for x in trades if x["strategy"] in CHECKS]   # 本番の6条件だけ（追加パターンはサイトの成績表に混ぜない）
    rows = summarize(base_trades, mood)
    sim_html, model = portfolio(base_trades, mood)
    build(rows, len(data), risk_sweep(trades) + exit_compare(trades) + pattern_compare(trades) + sim_html + bigloss(base_trades, mood))
    if model:
        (OUT / "similar.json").write_text(json.dumps(model, ensure_ascii=False), encoding="utf-8")
    for r in rows:
        print(r["条件"], r["全体"], r["買える"]["全体"])


if __name__ == "__main__":
    main()
