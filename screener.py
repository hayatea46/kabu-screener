"""日本株スイング候補スクリーナー（日足・1日1回）
GitHub Actionsで毎営業日の大引け後に実行し、docs/ にサイトを出力する。
AIは使わないのでトークン消費なし。
"""
import datetime as dt
import html
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
OUT = Path("docs")
MIN_TURNOVER = 1e8   # 20日平均売買代金 1億円未満は除外
MIN_RR = 1.0         # リスクリワード1未満は載せない
TOP_N = 30           # 条件ごとの最大表示数（買える銘柄のみ）
COUNCIL_N = 10       # エージェント協議に回す銘柄数
MIN_EDGE = 0.1       # 過去検証の期待値Rがこれ未満の条件は、タブと協議候補から外す
MIN_N = 100          # 過去検証の件数がこれ未満の区分は、外す・残すの判断に使わない（参考表示）
CAPITAL = 100000    # 運用資金（円）
LOSS_LIMIT = 2000   # 1銘柄の損失上限（円）＝資金の2%
LOT = 1             # 売買単位（株）。1＝単元未満株（1株から）、100＝従来の100株単位
BUFFER = 0.07       # 損切りが翌朝の寄付になって遅れる分の余裕（株価の7%）。株数計算で損切り幅に足す
MAX_RANGE = 3.7     # 直近14日の1日の値幅の平均（株価の%）がこれを超える荒い銘柄は除外
CHUNK = 100

STRATEGIES = {
    "box_bottom": ("ボックス下限", "60日間一定の幅で往復している銘柄が下限付近"),
    "box_break": ("ボックス上抜け", "ボックス上限を出来高2倍以上で突破"),
    "inv_hs": ("逆三尊", "逆三尊のネックラインを直近3日以内に突破"),
    "pullback": ("押し目買い", "上向き25日線への押し目（グランビル②）＋高値安値の切り上げ（ダウ）"),
    "oversold": ("売られすぎ反発", "RSI30以下または25日線乖離-10%以下から陽線で反発"),
    "new_high": ("52週高値更新", "1年来高値を出来高1.5倍以上で更新"),
}


# ---------- データ ----------
def load_universe():
    import io, re, urllib.request
    ua = {"User-Agent": "Mozilla/5.0"}
    get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers=ua), timeout=60).read()
    page = get(JPX_PAGE).decode("utf-8", "ignore")
    m = re.search(r'href="([^"]+\.xlsx?)"', page)
    if not m:
        raise RuntimeError("JPXの銘柄一覧ファイルが見つかりません")
    url = m.group(1) if m.group(1).startswith("http") else "https://www.jpx.co.jp" + m.group(1)
    df = pd.read_excel(io.BytesIO(get(url)), dtype=str)
    df = df[df["市場・商品区分"].str.contains("内国株式", na=False)]
    return dict(zip(df["コード"].str.strip() + ".T", df["銘柄名"].str.strip()))


def fetch(tickers, period="14mo"):
    data = {}
    for i in range(0, len(tickers), CHUNK):
        part = tickers[i:i + CHUNK]
        try:
            raw = yf.download(part, period=period, interval="1d", group_by="ticker",
                              auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print("取得エラー:", e)
            continue
        if not isinstance(raw.columns, pd.MultiIndex):
            raw = pd.concat({part[0]: raw}, axis=1)
        for t in part:
            if t not in raw.columns.get_level_values(0):
                continue
            d = raw[t].dropna()
            if len(d) >= 130:
                data[t] = d
        time.sleep(1)
    return data


# ---------- 指標 ----------
def rsi(c, n=14):
    diff = c.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def pivots(s, w, kind):
    roll = s.rolling(2 * w + 1, center=True)
    ext = roll.min() if kind == "low" else roll.max()
    return list(s.index[s == ext])


def tick(p):
    return 1 if p <= 3000 else 5 if p <= 5000 else 10 if p <= 30000 else 50 if p <= 50000 else 100


def rnd(p):
    t = tick(p)
    return round(p / t) * t


def sig(entry, target, stop, note):
    entry, target, stop = rnd(entry), rnd(target), rnd(stop)
    risk = entry - stop
    if risk <= 0 or target <= entry:
        return None
    rr = (target - entry) / risk
    if rr < MIN_RR:
        return None
    return {"entry": entry, "target": target, "stop": stop, "rr": round(rr, 2), "note": note}


# ---------- 判定条件 ----------
def box_bottom(d):
    w = d.iloc[-60:]
    H, L = w["High"].max(), w["Low"].min()
    width = (H - L) / L
    if not 0.06 <= width <= 0.20:
        return None
    c = d["Close"].iloc[-1]
    if c > L * 1.03:
        return None
    pl = [i for i in pivots(w["Low"], 3, "low") if w["Low"][i] <= L * 1.03]
    ph = [i for i in pivots(w["High"], 3, "high") if w["High"][i] >= H * 0.97]
    if len(pl) < 2 or len(ph) < 2:
        return None
    return sig(c, H * 0.98, L * 0.97, f"幅{width:.0%}のボックス {L:,.0f}〜{H:,.0f}円")


def box_break(d):
    prev = d.iloc[-61:-1]
    H, L = prev["High"].max(), prev["Low"].min()
    if (H - L) / L > 0.20:
        return None
    c, pc = d["Close"].iloc[-1], d["Close"].iloc[-2]
    vratio = d["Volume"].iloc[-1] / d["Volume"].iloc[-21:-1].mean()
    if c <= H or pc > H or vratio < 2:
        return None
    return sig(c, c + (H - L), H * 0.97, f"上限{H:,.0f}円を出来高{vratio:.1f}倍で突破")


def inv_hs(d):
    w = d.iloc[-120:]
    lows = pivots(w["Low"], 5, "low")
    if len(lows) < 3:
        return None
    a, b, e = lows[-3:]
    la, lb, le = w["Low"][a], w["Low"][b], w["Low"][e]
    if not (lb < la and lb < le) or abs(la - le) / min(la, le) > 0.06:
        return None
    neck = max(w["High"][a:b].max(), w["High"][b:e].max())
    c = d["Close"].iloc[-1]
    if not (neck < c <= neck * 1.05) or d["Close"].iloc[-4:-1].min() > neck:
        return None
    return sig(c, neck + (neck - lb), le * 0.98, f"ネックライン{neck:,.0f}円を突破")


def pullback(d):
    c = d["Close"]
    ma5, ma25, ma75 = (c.rolling(n).mean() for n in (5, 25, 75))
    m = ma25.iloc[-1]
    if not ma25.iloc[-1] > ma25.iloc[-11]:
        return None
    if not (d["Low"].iloc[-1] <= m * 1.01 and c.iloc[-1] > m):
        return None
    w = d.iloc[-120:]
    hs, ls = pivots(w["High"], 5, "high"), pivots(w["Low"], 5, "low")
    if len(hs) < 2 or len(ls) < 2:
        return None
    if not (w["High"][hs[-1]] > w["High"][hs[-2]] and w["Low"][ls[-1]] > w["Low"][ls[-2]]):
        return None
    po = ma5.iloc[-1] > m > ma75.iloc[-1]
    stop = min(m * 0.97, w["Low"][ls[-1]] * 0.99)
    note = "25日線で反発" + ("・パーフェクトオーダー" if po else "")
    return sig(c.iloc[-1], d["High"].iloc[-60:].max(), stop, note)


def oversold(d):
    c = d["Close"]
    ma25 = c.rolling(25).mean()
    r = rsi(c)
    dev = c / ma25 - 1
    if not (r.iloc[-4:-1].min() <= 30 or dev.iloc[-2] <= -0.10):
        return None
    if not (c.iloc[-1] > d["Open"].iloc[-1] and c.iloc[-1] > c.iloc[-2]):
        return None
    return sig(c.iloc[-1], ma25.iloc[-1], d["Low"].iloc[-5:].min() * 0.99,
               f"RSI{r.iloc[-2]:.0f}・乖離{dev.iloc[-2]:+.0%}から反発")


def new_high(d):
    c = d["Close"].iloc[-1]
    if len(d) < 251 or c <= d["High"].iloc[-251:-1].max():
        return None
    vratio = d["Volume"].iloc[-1] / d["Volume"].iloc[-21:-1].mean()
    if vratio < 1.5:
        return None
    m = d["Close"].rolling(25).mean().iloc[-1]
    stop = max(m * 0.99, c * 0.92)
    return sig(c, c + 2 * (c - stop), stop, f"出来高{vratio:.1f}倍で高値更新")


CHECKS = {"box_bottom": box_bottom, "box_break": box_break, "inv_hs": inv_hs,
          "pullback": pullback, "oversold": oversold, "new_high": new_high}


# ---------- 実行 ----------
def market_mood():
    try:
        t = yf.download("1306.T", period="6mo", progress=False, auto_adjust=True)["Close"].squeeze().dropna()
        up = t.iloc[-1] > t.rolling(25).mean().iloc[-1]
        return ("追い風", "TOPIX連動ETFが25日線の上") if up else ("向かい風", "TOPIX連動ETFが25日線の下。新規買いは控えめに")
    except Exception:
        return ("不明", "地合いデータを取得できませんでした")


def load_edge(mood):
    """docs/backtest.json から、条件ごとの期待値Rを返す（{条件名: {"r": R, "ref": 参考か}}）。
    会社員ルール（翌朝寄付の成行で売買）で実際に買える銘柄の成績を使う。無ければ指値ルールの買える銘柄の成績。
    今の地合いの件数が MIN_N 未満なら「全体」に戻し、それも少なければ参考表示（判断に使わない）。
    ファイルが無い・読めない場合は空。古い形式（買える集計なし）は全銘柄の値で同じ判断をする。"""
    try:
        rows = json.loads((OUT / "backtest.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    edge = {}
    for r in rows:
        src = r.get("寄付成行") or r.get("買える") or {t: r.get(t) for t in ("全体", "追い風", "向かい風")}
        order = [mood[0], "全体"] if mood[0] in ("追い風", "向かい風") else ["全体"]
        cands = [src.get(t) for t in order if src.get(t) and "期待値R" in src[t]]
        enough = [v for v in cands if v["件数"] >= MIN_N]
        if enough:
            edge[r["条件"]] = {"r": enough[0]["期待値R"], "ref": False}
        elif cands:
            edge[r["条件"]] = {"r": cands[-1]["期待値R"], "ref": True}
    return edge


def active(edge):
    """期待値Rが基準未満の条件を除いた条件キー（STRATEGIESの順）。値が無い条件・参考表示の条件は残す。"""
    return [k for k, (label, _) in STRATEGIES.items()
            if label not in edge or edge[label]["ref"] or edge[label]["r"] >= MIN_EDGE]


def shares(entry, stop, lot=LOT):
    """買える株数。損切り幅＋余裕（株価×BUFFER）での損失が LOSS_LIMIT 以内、買い代金が CAPITAL 以内になる最大株数（lot 単位で切り捨て）。0なら買えない"""
    if entry - stop <= 0 or entry <= 0:
        return 0
    n = min(LOSS_LIMIT // (entry - stop + entry * BUFFER), CAPITAL // entry)
    return int(n // lot * lot)


def load_similar():
    """docs/similar.json（backtest.py が作る「似た取引」の成績表）。無ければ None"""
    try:
        return json.loads((OUT / "similar.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def similar_lookup(model, strategy, width, mood):
    """条件×損切り幅（株価の%）×地合いが似た過去の取引の成績 {"件数","勝率","実効R"}。件数が少なければ区分を粗くする"""
    band = sum(width > e for e in model["edges"])
    for key in (f"{strategy}|{band}|{mood}", f"{strategy}|{band}", strategy):
        if key in model["table"]:
            return model["table"][key]
    return None


def day_range(d):
    """直近14日の1日の値幅（高値−安値）の平均。株価の%"""
    return float(((d["High"] - d["Low"]) / d["Close"]).iloc[-14:].mean() * 100)


def screen(data, names, mood=("不明",), model=None):
    results = {k: [] for k in CHECKS}
    for t, d in data.items():
        if (d["Close"] * d["Volume"]).iloc[-20:].mean() < MIN_TURNOVER or day_range(d) > MAX_RANGE:
            continue
        for k, f in CHECKS.items():
            try:
                s = f(d)
            except Exception:
                s = None
            if s and shares(s["entry"], s["stop"]) > 0:  # 実際に買える銘柄だけ
                s.update(shares=shares(s["entry"], s["stop"]), code=t.replace(".T", ""), name=names.get(t, ""),
                         turnover=round((d["Close"] * d["Volume"]).iloc[-20:].mean() / 1e8, 1),
                         rsi=round(float(rsi(d["Close"]).iloc[-1]), 0))
                h = model and similar_lookup(model, k, (s["entry"] - s["stop"]) / s["entry"] * 100, mood[0])
                if h:  # 期待値は損失上限あたり（余裕幅込みの株数）で1回平均何円残るか
                    s.update(similar_n=h["件数"], win_pct=h["勝率"], ev_yen=round(h["実効R"] * LOSS_LIMIT))
                results[k].append(s)
    for k in results:  # 似た取引の期待値が高い順（無ければリスクリワード順）
        results[k] = sorted(results[k], key=lambda x: (-x.get("ev_yen", -1e9), -x["rr"]))[:TOP_N]
    return results


def council_input(results, mood, keys):
    pool = [dict(r, strategy=STRATEGIES[k][0]) for k in keys for r in results[k]]
    pool = sorted(pool, key=lambda x: (-x.get("ev_yen", -1e9), -x["rr"]))
    seen, top = set(), []
    for r in pool:
        if r["code"] not in seen:
            seen.add(r["code"])
            top.append({k: r[k] for k in ("code", "name", "strategy", "entry", "target", "stop", "shares", "rr", "rsi", "turnover", "note")}
                       | {k: r[k] for k in ("similar_n", "win_pct", "ev_yen") if k in r})
        if len(top) >= COUNCIL_N:
            break
    return {"date": str(dt.date.today()), "market": mood[0], "candidates": top}


# ---------- サイト ----------
CSS = """
:root{--bg:#F3F5F4;--ink:#18201F;--mute:#66706E;--line:#D5DBD9;--up:#C62E2E;--down:#2A5DA8;--card:#FFFFFF}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:'BIZ UDPGothic',sans-serif;line-height:1.6;font-variant-numeric:tabular-nums}
main{max-width:720px;margin:0 auto;padding:16px}
h1{font-size:1.25rem;margin:4px 0 2px}.date{color:var(--mute);font-size:.85rem}
.mood{margin:14px 0;padding:10px 14px;border-left:4px solid var(--mute);background:var(--card)}
.mood.good{border-color:var(--up)}.mood.bad{border-color:var(--down)}.mood b{font-size:1.05rem}
nav{display:flex;gap:6px;overflow-x:auto;padding-bottom:6px;margin:8px 0}
nav button{flex:none;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:6px 12px;border-radius:999px;font:inherit;font-size:.9rem}
nav button[aria-selected=true]{background:var(--ink);color:#fff;border-color:var(--ink)}
nav button:focus-visible{outline:2px solid var(--up);outline-offset:2px}
.desc{color:var(--mute);font-size:.85rem;margin:4px 0 12px}
.row{background:var(--card);border:1px solid var(--line);padding:12px 14px;margin-bottom:8px}
.head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.name{font-weight:700}.code{color:var(--mute);font-size:.85rem;margin-right:6px}
.rr{font-size:1.4rem;font-weight:700}.rr small{font-size:.75rem;color:var(--mute);font-weight:400}
.note{font-size:.85rem;color:var(--mute)}
.ladder{position:relative;height:8px;margin:12px 0 4px;background:var(--line)}
.ladder .r{position:absolute;top:0;bottom:0;background:var(--down)}.ladder .g{position:absolute;top:0;bottom:0;background:var(--up)}
.order{display:grid;grid-template-columns:repeat(3,1fr);font-size:.9rem}
.order div:nth-child(2){text-align:center}.order div:nth-child(3){text-align:right}
.order span{display:block;font-size:.75rem;color:var(--mute)}
.stop{color:var(--down)}.tgt{color:var(--up)}
.empty{color:var(--mute);padding:20px 0}
footer{color:var(--mute);font-size:.75rem;margin:24px 0}
"""

JS = """
const b=[...document.querySelectorAll('nav button')],p=[...document.querySelectorAll('section')];
b.forEach((x,i)=>x.onclick=()=>{b.forEach(y=>y.setAttribute('aria-selected',y===x));p.forEach((s,j)=>s.hidden=i!==j)});
"""


def row_html(r):
    span = r["target"] - r["stop"]
    loss = (r["entry"] - r["stop"]) / span * 100
    return f"""<div class="row"><div class="head"><div><span class="code">{r['code']}</span><span class="name">{html.escape(r['name'])}</span></div>
<div class="rr">{r['rr']}<small> RR</small></div></div><div class="note">{html.escape(r['note'])}／{r['shares']:,}株まで（損失{(r['entry'] - r['stop']) * r['shares']:,.0f}円）・売買代金{r['turnover']}億円・RSI{r['rsi']:.0f}</div>
{f'<div class="note">似た取引{r["similar_n"]:,}件：勝率{r["win_pct"]:.0f}%・1回平均{r["ev_yen"]:+,}円</div>' if "ev_yen" in r else ""}
<div class="ladder"><div class="r" style="left:0;width:{loss:.1f}%"></div><div class="g" style="left:{loss:.1f}%;right:0"></div></div>
<div class="order"><div class="stop"><span>損切り</span>{r['stop']:,}円</div><div><span>買値の目安</span>{r['entry']:,}円</div><div class="tgt"><span>利確</span>{r['target']:,}円</div></div></div>"""


def build_site(results, mood, council, keys, edge):
    OUT.mkdir(exist_ok=True)
    cls = {"追い風": "good", "向かい風": "bad"}.get(mood[0], "")
    nav, secs = [], []
    for i, k in enumerate(keys):
        label, desc = STRATEGIES[k]
        rs = results[k]
        e = edge.get(label)
        er = f" {'参考' if e['ref'] else ''}R{e['r']:+.2f}" if e else ""
        nav.append(f'<button aria-selected="{str(i == 0).lower()}">{label}{er} {len(rs)}</button>')
        body = "".join(row_html(r) for r in rs) or '<p class="empty">今日は該当なし。条件を満たす銘柄が出るまで待つのも戦略です。</p>'
        secs.append(f'<section{"" if i == 0 else " hidden"}><p class="desc">{desc}。RRの高い順。</p>{body}</section>')
    page = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>スイング候補 {council['date']}</title>
<link href="https://fonts.googleapis.com/css2?family=BIZ+UDPGothic:wght@400;700&display=swap" rel="stylesheet"><style>{CSS}</style></head>
<body><main><h1>スイング候補</h1><div class="date">{council['date']} 大引け時点・日足</div>
<div class="mood {cls}"><b>地合い：{mood[0]}</b><br>{mood[1]}</div>
{'' if council['candidates'] else f'<div class="mood"><b>今日は買える候補なし</b><br>{LOT}株以上を{CAPITAL // 10000}万円以内・損切り損失{LOSS_LIMIT:,}円以内で買える銘柄が、残っている条件に見つかりませんでした。見送りです。</div>'}
<nav>{''.join(nav)}</nav>{''.join(secs) or '<p class="empty">今の地合いで過去検証の成績が基準を満たす条件がありません。見送りが妥当です。</p>'}
<footer>注文はIFDOCO（買い指値→利確・損切りを同時セット）を想定。表示は機械的な判定で、売買の判断はご自身で。</footer></main>
<script>{JS}</script></body></html>"""
    (OUT / "index.html").write_text(page, encoding="utf-8")
    (OUT / "council_input.json").write_text(json.dumps(council, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main():
    names = load_universe()
    print(f"対象 {len(names)} 銘柄")
    data = fetch(list(names))
    print(f"取得 {len(data)} 銘柄")
    mood = market_mood()
    results = screen(data, names, mood, load_similar())
    edge = load_edge(mood)
    keys = active(edge)
    print(f"地合い {mood[0]} / 期待値R {edge} / 残す条件 {[STRATEGIES[k][0] for k in keys]}")
    build_site(results, mood, council_input(results, mood, keys), keys, edge)
    print({STRATEGIES[k][0]: len(v) for k, v in results.items()})


if __name__ == "__main__":
    main()
