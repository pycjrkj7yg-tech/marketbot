"""
تقرير سوق يعمل مرة واحدة ثم ينتهي، وتشغّله GitHub Actions كل 30 دقيقة.
تحليل آلي وليس نصيحة مالية.
"""
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import ccxt
import numpy as np
import pandas as pd
import requests

TOKEN = os.environ["TELEGRAM_TOKEN"]
CHAT_ID = os.environ["CHAT_ID"]

# Kraken لأنه يعمل من سيرفرات أمريكا (Binance يحظرها)
SYMBOLS = [
    "BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD",
    "ADA/USD", "DOGE/USD", "AVAX/USD", "LINK/USD",
]
TIMEFRAME = "15m"
CANDLE_MIN = 15
HOLD_CANDLES = 4
MIN_CONFIDENCE = 35
TZ = ZoneInfo("Asia/Riyadh")  # غيّرها حسب بلدك
SNAPSHOT_OFFSETS = [0, 1]     # شمعتان = 30 دقيقة

exchange = ccxt.kraken({"enableRateLimit": True})


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
    df["vol_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
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
    why = f"تشبع بيعي RSI={l.rsi:.0f}" if l.rsi < 30 else f"تشبع شرائي RSI={l.rsi:.0f}" if l.rsi > 70 else f"لا تطرف RSI={l.rsi:.0f}"
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
    l = df.iloc[-1]
    obv_dir = 1 if l.obv > l.obv_ema else -1
    vr = min(float(l.vol_ratio) if not np.isnan(l.vol_ratio) else 1, 2) / 2
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


def analyze(symbol):
    try:
        raw = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=400)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.iloc[:-1]  # احذف الشمعة غير المكتملة
        df = add_indicators(df).dropna().reset_index(drop=True)
        if len(df) < 40:
            return None
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
        quality = 1.0 if 0.08 <= atr_pct <= 2.0 else 0.6
        conf = min(abs(mean) * 1.6, 1) * agree * stability * quality * 100
        return {
            "symbol": symbol, "direction": direction, "confidence": conf, "agree": agree,
            "price": float(last.close), "atr": float(last.atr),
            "votes": [(n, snaps[0][n][0], snaps[0][n][1]) for n in names],
        }
    except Exception as e:
        print("فشل", symbol, e)
        return None


def fmt(p):
    return f"{p:,.2f}" if p >= 1 else f"{p:.5f}"


def build_report():
    res = [r for r in (analyze(s) for s in SYMBOLS) if r]
    if not res:
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
        side = "شراء 🟢 (Long)" if d == 1 else "بيع 🔴 (Short)"
        lines += [
            f"🏆 أفضل سوق: {best['symbol']}",
            f"الاتجاه: {side}",
            f"الثقة: {best['confidence']:.0f}% | اتفاق المحللين: {best['agree']*100:.0f}%",
            f"⏰ الدخول: {entry:%H:%M}",
            f"⏰ الخروج: {exit_t:%H:%M}",
            f"💰 السعر: {fmt(p)}",
            f"🎯 الهدف: {fmt(p + 2.5 * atr * d)}",
            f"🛑 وقف الخسارة: {fmt(p - 1.5 * atr * d)}\n",
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
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text},
        timeout=30,
    )
    print(r.status_code, r.text[:200])
    r.raise_for_status()
