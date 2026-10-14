"""検証専用の追加パターン（サイト・毎日の判定には使わない）。判定条件は screener.py の CHECKS と同じ形式で、backtest.py だけが読む。
基準の random は「型と無関係に、約10%の日にランダムで買う」比較用。"""
import numpy as np

from screener import pivots, sig


def cup_handle(d):
    w = d.iloc[-120:]
    left = w["High"].iloc[:-30]
    a = left.idxmax()
    ia = w.index.get_loc(a)
    if ia > len(w) - 45:
        return None
    rim2 = w["High"].iloc[-30:-1].max()
    base = w["Low"].iloc[ia:-1]
    low = base.min()
    depth = (left.max() - low) / left.max()
    if not 0.12 <= depth <= 0.35 or not 0.95 <= rim2 / left.max() <= 1.05:
        return None
    pos = (w.index.get_loc(base.idxmin()) - ia) / max(len(w) - 1 - ia, 1)
    c = d["Close"].iloc[-1]
    if not 0.2 <= pos <= 0.8 or not (c > rim2 >= d["Close"].iloc[-2]):
        return None
    stop = d["Low"].iloc[-10:].min() * 0.99
    return sig(c, c + (left.max() - low), stop, f"カップ深さ{depth:.0%}・右縁{rim2:,.0f}円を突破")


def double_bottom(d):
    w = d.iloc[-120:]
    lows = []
    for i in pivots(w["Low"], 5, "low"):   # 同値で並ぶ安値は最初の1つだけ使う
        if not lows or w.index.get_loc(i) - w.index.get_loc(lows[-1]) > 5:
            lows.append(i)
    if len(lows) < 2:
        return None
    a, b = lows[-2:]
    la, lb = w["Low"][a], w["Low"][b]
    if abs(la - lb) / min(la, lb) > 0.03 or (w.index.get_loc(b) - w.index.get_loc(a)) < 15:
        return None
    neck = w["High"][a:b].max()
    c = d["Close"].iloc[-1]
    if not (neck < c <= neck * 1.05) or d["Close"].iloc[-4:-1].min() > neck:
        return None
    low = min(la, lb)
    return sig(c, neck + (neck - low), low * 0.98, f"W底 ネック{neck:,.0f}円を突破")


def asc_triangle(d):
    w = d.iloc[-60:]
    R = w["High"].iloc[:-1].max()
    ph = [i for i in pivots(w["High"].iloc[:-1], 3, "high") if w["High"][i] >= R * 0.985]
    pl = pivots(w["Low"].iloc[:-1], 3, "low")
    if len(ph) < 2 or len(pl) < 2 or not w["Low"][pl[-1]] > w["Low"][pl[-2]]:
        return None
    c, pc = d["Close"].iloc[-1], d["Close"].iloc[-2]
    if not (c > R >= pc):
        return None
    return sig(c, c + (R - w["Low"][pl[0]]), w["Low"][pl[-1]] * 0.99, f"上昇三角形 上限{R:,.0f}円を突破")


def _rr2(c, stop, note):
    return sig(c, c + 2 * (c - stop), stop, note) if stop < c else None


def engulfing(d):
    o, c, lo = d["Open"], d["Close"], d["Low"]
    ma25 = c.rolling(25).mean().iloc[-1]
    if not (c.iloc[-2] < o.iloc[-2] and c.iloc[-1] > o.iloc[-1] and o.iloc[-1] <= c.iloc[-2] and c.iloc[-1] >= o.iloc[-2]):
        return None
    if not (c.iloc[-1] < ma25 and c.iloc[-1] < c.iloc[-6]):
        return None
    return _rr2(c.iloc[-1], min(lo.iloc[-1], lo.iloc[-2]) * 0.99, "下落中の陽線包み足")


def hammer(d):
    o, c, h, lo = (d[k].iloc[-1] for k in ("Open", "Close", "High", "Low"))
    body = abs(c - o)
    rng = h - lo
    if rng <= 0 or body <= 0 or (min(o, c) - lo) < 2 * body or (h - max(o, c)) > body or (c - lo) / rng < 0.66:
        return None
    if not (c < d["Close"].rolling(25).mean().iloc[-1] * 0.97):
        return None
    return _rr2(c, lo * 0.99, "安値圏のハンマー")


def _ichimoku(d):
    h, lo, c = d["High"], d["Low"], d["Close"]
    ten = (h.rolling(9).max() + lo.rolling(9).min()) / 2
    kij = (h.rolling(26).max() + lo.rolling(26).min()) / 2
    sa = ((ten + kij) / 2).shift(26)
    sb = ((h.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    return ten, kij, sa.combine(sb, max), sa.combine(sb, min)


def sanyaku(d):
    c = d["Close"]
    ten, kij, top, bot = _ichimoku(d)
    ok = (ten > kij) & (c > top) & (c > c.shift(26))
    if np.isnan(top.iloc[-1]) or not (ok.iloc[-1] and not ok.iloc[-2]):
        return None
    return _rr2(c.iloc[-1], min(kij.iloc[-1], top.iloc[-1]) * 0.98, "一目 三役好転")


def cloud_break(d):
    c = d["Close"]
    _, _, top, _ = _ichimoku(d)
    if np.isnan(top.iloc[-1]) or not (c.iloc[-1] > top.iloc[-1] and c.iloc[-2] <= top.iloc[-2]):
        return None
    return _rr2(c.iloc[-1], top.iloc[-1] * 0.98, "一目 雲の上抜け")


def random_buy(d):
    """比較基準：型と無関係。約10%の日に、損切り4.5%・利確は損切り幅の2倍で買う"""
    if d.index[-1].toordinal() % 10 != 0:
        return None
    c = d["Close"].iloc[-1]
    return sig(c, c * 1.09, c * 0.955, "ランダム買い（基準）")


NEW_CHECKS = {"cup_handle": cup_handle, "double_bottom": double_bottom, "asc_triangle": asc_triangle,
              "engulfing": engulfing, "hammer": hammer, "sanyaku": sanyaku, "cloud_break": cloud_break,
              "random_buy": random_buy}
NEW_LABELS = {"cup_handle": "カップウィズハンドル", "double_bottom": "ダブルボトム", "asc_triangle": "上昇三角形",
              "engulfing": "陽線包み足", "hammer": "ハンマー", "sanyaku": "一目 三役好転", "cloud_break": "一目 雲の上抜け",
              "random_buy": "ランダム買い（基準）"}
