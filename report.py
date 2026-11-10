"""
تقرير الفوركس والذهب: يعمل مرة واحدة ثم ينتهي، وتشغّله GitHub Actions.
الأسعار من Yahoo Finance وقد تختلف قليلاً عن MT5، فاعتمد على النسب المئوية.
تحليل آلي وليس نصيحة مالية.
"""
import os
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# الاسم كما يظهر في MT5 : رموز Yahoo (يجرّب بالترتيب)
MARKETS = {
    "XAUUSD": ["XAUUSD=X", "GC=F"],
    "EURUSD": ["EURUSD=X"],
    "GBPUSD": ["GBPUSD=X"],
    "USDJPY": ["USDJPY=X"],
    "AUDUSD": ["AUDUSD=X"],
    "USDCAD": ["USDCAD=X"],
    "USDCHF": ["USDCHF=X"],
    "NZDUSD": ["NZDUSD=X"],
    "EURJPY": ["EURJPY=X"],
    "GBPJPY": ["GBPJPY=X"],
}
CANDLE_MIN = 15
HOLD_CANDLES = 4
MIN_CONFIDENCE = 35
STALE_SECONDS = 3600          # أقدم من ساعة = السوق مغلق
TZ = ZoneInfo("Africa/Algiers")
SNAPSHOT_OFFSETS = [0, 1]     # شمعتان = 30 دقيقة
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def fetch_one(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    r = requests.get(url, params={"interval": "15m", "range": "10d"}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "ts": res["timestamp"], "open": q["open"], "high": q["high"],
        "low": q["low"], "close": q["close"], "volume": q["volume"],
    })
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    df["volume"] = df["volume"].fillna(0)
    return df


def fetch_candles(tickers):
    last_err = None
    for t in tickers:
        try:
            df = fetch_one(t)
            if len(df) >= 60:
                return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"no data: {last_err}")


def add_indicators(df):
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["mhist"] = macd - macd.ewm(span=9, adjust=False).mean()
    df["sma20"] = c.rolling(20).mean()
    df["std20"] = c.rolling(20).std()
    pc = c.shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["obv"] = (np.sign(c.diff()).fillna(0) * df["volume"]).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()
    ratio = df["volume"] / df["volume"].rolling(20).mean()
    df["vol_ratio"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    return df


def persona_trend(df):
    l = df.iloc[-1]
    s = (1 if l.ema20 > l.ema50 else -1) + (1 if l.ema50 > l.ema200 else -1) + (1 if l.close > l.ema20 else -1)
    return s / 3, "الاتجاه صاعد" if s > 0 else "الاتجاه هابط"


def persona_momentum(df):
    l, p = df.iloc[-1], df.iloc[-2]
    strength = float(np.clip(l.mhist / l.atr * 4, -1, 1))
    score = 0.7 * strength + 0.3 * (1 if l.mhist > p.mhist else -1)
    return float(np.clip(score, -1, 1)), "الزخم صاعد" if score > 0 else "الزخم هابط"


def persona_reversion(df):
    l = df.iloc[-1]
    bb = (l.close - l.sma20) / (2 * l.std20) if l.std20 else 0
    raw = 0.5 * float(np.clip(-bb, -1, 1)) + 0.5 * float(np.clip((50 - l.rsi) / 25, -1, 1))
    score = raw if (abs(bb) > 0.8 or l.rsi < 30 or l.rsi > 70) else raw * 0.3
    if l.rsi < 30:
        why = f"تشبع بيعي RSI={l.rsi:.0f}"
    elif l.rsi > 70:
        why = f"تشبع شرائي RSI={l.rsi:.0f}"
    else:
        why = f"لا تطرف RSI={l.rsi:.0f}"
    return float(np.clip(score, -1, 1)), why


def persona_breakout(df):
    hi = df["high"].iloc[-21:-1].max()
    lo = df["low"].iloc[-21:-1].min()
    c = df["close"].iloc[-1]
    if c > hi:
        return 1.0, "اختراق أعلى 20 شمعة"
    if c < lo:
        return -1.0, "كسر أدنى 20 شمعة"
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    return float(np.clip((c - mid) / half, -1, 1)) * 0.3, "داخل النطاق"


def persona_volume(df):
    if df["volume"].iloc[-20:].sum() == 0:
        return 0.0, "لا توجد بيانات حجم لهذا السوق"
    l = df.iloc[-1]
    obv_dir = 1 if l.obv > l.obv_ema else -1
    vr = min(float(l.vol_ratio), 2) / 2
    score = 0.5 * obv_dir + 0.5 * np.sign(l.close - l.open) * vr
    return float(score), "السيولة تدعم الشراء" if score > 0 else "السيولة تدعم البيع"


def persona_candles(df):
    a, b = df.iloc[-2], df.iloc[-1]
    body = abs(b.close - b.open)
    rng = (b.high - b.low) or 1e-9
    lower = min(b.open, b.close) - b.low
    upper = b.high - max(b.open, b.close)
    if b.close > b.open and a.close < a.open and b.close > a.open and b.open < a.close:
        return 1.0, "شمعة ابتلاعية صاعدة"
    if b.close < b.open and a.close > a.open and b.close < a.open and b.open > a.close:
        return -1.0, "شمعة ابتلاعية هابطة"
    if lower > 2 * body and lower / rng > 0.6:
        return 0.7, "مطرقة"
    if upper > 2 * body and upper / rng > 0.6:
        return -0.7, "نجمة السقوط"
    return float(np.sign(df["close"].iloc[-1] - df["close"].iloc[-4])) * 0.2, "لا نموذج واضح"


PERSONAS = [
    ("🧭 المتتبع للترند", persona_trend),
    ("⚡ خبير الزخم", persona_momentum),
    ("🔄 صائد الارتداد", persona_reversion),
    ("🚀 قناص الاختراق", persona_breakout),
    ("💧 محلل السيولة", persona_volume),
    ("🕯️ قارئ الشموع", persona_candles),
]


def analyze(name, tickers):
    """يرجع (الحالة، البيانات): ok أو stale أو error"""
    try:
        df = fetch_candles(tickers)
        df = df.iloc[:-1]  # احذف الشمعة غير المكتملة
        if time.time() - float(df["ts"].iloc[-1]) > STALE_SECONDS:
            return "stale", None
        df = add_indicators(df).dropna().reset_index(drop=True)
        if len(df) < 40:
            return "error", None
        snaps = []
        for k in SNAPSHOT_OFFSETS:
            sub = df.iloc[: len(df) - k]
            snaps.append({n: fn(sub) for n, fn in PERSONAS})
        last = df.iloc[-1]
        names = [n for n, _ in PERSONAS]
        avg = {n: float(np.mean([s[n][0] for s in snaps])) for n in names}
        mean = float(np.mean(list(avg.values())))
        direction = 1 if mean > 0 else -1
        agree = sum(1 for v in avg.values() if np.sign(v) == direction) / len(avg)
        per_snap = [np.mean([s[n][0] for n in names]) for s in snaps]
        stability = sum(1 for x in per_snap if np.sign(x) == direction) / len(per_snap)
        atr_pct = float(last.atr / last.close * 100)
        quality = 1.0 if 0.02 <= atr_pct <= 2.0 else 0.6
        conf = min(abs(mean) * 1.6, 1) * agree * stability * quality * 100
        return "ok", {
            "symbol": name, "direction": direction, "confidence": conf, "agree": agree,
            "price": float(last.close), "atr": float(last.atr),
            "votes": [(n, snaps[0][n][0], snaps[0][n][1]) for n in names],
        }
    except Exception as e:
        print("فشل", name, e)
        return "error", None


def fmt(p):
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 100:
        return f"{p:.3f}"
    return f"{p:.5f}"


def build_report():
    res, stale, errors = [], 0, 0
    for name, tickers in MARKETS.items():
        status, data = analyze(name, tickers)
        if status == "ok":
            res.append(data)
        elif status == "stale":
            stale += 1
        else:
            errors += 1
    if not res:
        if errors == 0:
            return None  # الأسواق مغلقة، لا نرسل شيئاً
        return "⚠️ تعذر جلب بيانات الأسواق هذه المرة."
    res.sort(key=lambda r: r["confidence"], reverse=True)
    best = res[0]
    now = datetime.now(TZ)
    lines = [f"📊 تقرير الـ30 دقيقة — {now:%Y-%m-%d %H:%M}\n"]
    if best["confidence"] < MIN_CONFIDENCE:
        lines.append("⚠️ لا توجد فرصة واضحة الآن. الأفضل الانتظار.\n")
    else:
        entry = now.replace(second=0, microsecond=0) + timedelta(minutes=CANDLE_MIN - now.minute % CANDLE_MIN)
        exit_t = entry + timedelta(minutes=CANDLE_MIN * HOLD_CANDLES)
        d, p, atr = best["direction"], best["price"], best["atr"]
        tp, sl = p + 2.5 * atr * d, p - 1.5 * atr * d
        tp_pct, sl_pct = 2.5 * atr / p * 100, 1.5 * atr / p * 100
        side = "شراء 🟢 (Buy)" if d == 1 else "بيع 🔴 (Sell)"
        lines += [
            f"🏆 أفضل سوق: {best['symbol']}",
            f"الاتجاه: {side}",
            f"الثقة: {best['confidence']:.0f}% | اتفاق المحللين: {best['agree']*100:.0f}%",
            f"⏰ الدخول: {entry:%H:%M}",
            f"⏰ الخروج: {exit_t:%H:%M}",
            f"💰 السعر: {fmt(p)}",
            f"🎯 الهدف: {fmt(tp)} ({'+' if d == 1 else '-'}{tp_pct:.2f}%)",
            f"🛑 وقف الخسارة: {fmt(sl)} ({'-' if d == 1 else '+'}{sl_pct:.2f}%)",
            "ℹ️ أسعار MT5 قد تختلف قليلاً، فطبّق النسب على سعر منصتك.\n",
            "🗣️ آراء المحللين:",
        ]
        for n, s, why in best["votes"]:
            icon = "🟢" if s > 0.15 else "🔴" if s < -0.15 else "⚪"
            lines.append(f"{icon} {n}: {why}")
    lines.append("\n📋 ترتيب الأسواق:")
    for r in res[:6]:
        lines.append(f"{'▲' if r['direction'] == 1 else '▼'} {r['symbol']} — ثقة {r['confidence']:.0f}%")
    lines.append("\n⚠️ تحليل آلي وليس نصيحة مالية.")
    return "\n".join(lines)


if __name__ == "__main__":
    text = build_report()
    if text is None:
        print("الأسواق مغلقة الآن، لا يوجد تقرير.")
    else:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text},
            timeout=30,
        )
        print(r.status_code, r.text[:200])
        r.raise_for_status()
