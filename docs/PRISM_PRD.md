# PRISM — Product Requirements Document
**Predictive Regime Intelligence & Signal Model**
*Version 1.2 | Author: Ada Sandpaw | Date: 2026-04-20*
*Changelog: v1.1 — Multi-timeframe top-down architecture; v1.2 — FVG confluence, Exness broker recommendation*
*Prepared for: Brian | Confidential*

---

## 1. Executive Summary

PRISM is an AI/ML-powered trading intelligence system that predicts directional bias across Forex pairs and Gold (XAU/USD), generates precise entry signals, and calculates optimized Stop Loss (SL) and Take Profit (TP) levels. Built with a Python-first ML stack, backtested with institutional-grade rigor, and deployed as a native MT5 Expert Advisor (EA) via Python ↔ MT5 bridge.

PRISM learns *why* markets move (macro regime, institutional flow, sentiment) — not just *that* they moved.

**Tagline:** *"Trade the signal. Not the noise."*

---

## 2. Model Name Options

| Name | Meaning | Vibe |
|------|---------|------|
| **PRISM** *(recommended)* | Predictive Regime Intelligence & Signal Model | Premium, analytical |
| **MIDAS** | Market Intelligence Decision & Analysis System | Gold connotation, memorable |
| **ORACLE** | Optimized Real-time Algorithmic Currency & Levels Engine | Predictive, powerful |
| **APEX** | Adaptive Predictive EXecution | Clean, marketable |

**Recommendation:** PRISM. It signals that you see through complexity to the clear trading signal — and it works across all markets, not just gold.

---

## 3. Vision & Goals

### 3.1 Vision
Build the most signal-accurate, risk-aware ML trading model for retail traders — starting with Forex and Gold, expandable to any MT5-compatible instrument.

### 3.2 Primary Goals
- **G1:** Predict directional bias (Long/Short/Neutral) with ≥62% win rate on backtests
- **G2:** Generate SL/TP levels based on volatility-adjusted ATR + structural zones
- **G3:** Full MT5 EA integration — signals execute automatically on user's broker
- **G4:** Multi-instrument: EUR/USD, GBP/USD, USD/JPY, XAU/USD (Gold), expandable
- **G5:** Production-ready within 8 weeks (2 sprints of 4 weeks each)

### 3.3 Success Metrics
| Metric | Target |
|--------|--------|
| Directional accuracy (backtest) | ≥62% |
| Sharpe Ratio (backtest, 1Y) | ≥1.5 |
| Max Drawdown | ≤15% |
| Risk:Reward ratio average | ≥1:2 |
| MT5 latency (signal → order) | ≤2 seconds |
| False signal rate | ≤20% |

---

## 4. Target Instruments

### Phase 1 (Build)
| Instrument | Type | Why |
|-----------|------|-----|
| XAU/USD | Gold | Highest volatility, trending instrument |
| EUR/USD | Forex | Most liquid pair, richest data history |
| GBP/USD | Forex | High volatility, clear institutional footprint |

### Phase 2 (Expand)
| Instrument | Type |
|-----------|------|
| USD/JPY | Forex (safe haven dynamics) |
| NAS100 / SPX500 | Index CFDs |
| BTC/USD | Crypto (via MT5 broker support) |

---

## 5. Data Architecture

### 5.1 Price Data
| Source | What | Tier |
|--------|------|------|
| **Tiingo API** | Clean adjusted OHLCV, intraday + EOD, news sentiment | Free/Paid |
| **MetaTrader5 Python** | Live tick data, M1/M5/M15/H1/H4/D1 bars | Free (broker) |
| **yfinance** | Fallback / validation cross-check | Free |

> *Note: Tiingo is the primary price source — as @mar_antaya calls it: "the easiest upgrade from yfinance most beginners never make."*

### 5.2 Macro / Regime Data
| Source | What | Tier |
|--------|------|------|
| **FRED API** | Interest rates (Fed Funds, SOFR, IORB), CPI, GDP, unemployment — back to 1950s | Free |
| **Trading Economics API** | Central bank rates (ECB, BOE, BOJ, Fed), PMI, NFP | Freemium |
| **Investing.com scraper** | Economic calendar events | Free |

> *FRED gives us the "why" — rates environment tells us whether markets are in risk-on, risk-off, or transitional regime. @mar_antaya's core thesis: "This is how you know why markets move, not just that they moved."*

### 5.3 Alternative / Sentiment Data
| Source | What | Tier |
|--------|------|------|
| **Quiver Quantitative API** | Congressional trades, lobbying, Reddit sentiment, web traffic | Free tier |
| **Tiingo News Sentiment** | Real-time news sentiment scores per ticker | Included with Tiingo |
| **Fear & Greed Index** | CNN F&G (market-wide sentiment) | Free scrape |
| **CFTC COT Reports** | Institutional positioning (commitment of traders) | Free (CFTC.gov) |

### 5.4 Feature Engineering Pipeline

```
Raw Data Inputs
    ├── Price Features (OHLCV)
    │   ├── Returns: 1-bar, 5-bar, 20-bar, 50-bar
    │   ├── Volatility: ATR(14), ATR(50), HV(20)
    │   ├── Technical: EMA(9), EMA(21), EMA(50), EMA(200)
    │   ├── Momentum: RSI(14), MACD, Stochastic(14,3,3)
    │   ├── Volume: OBV, VWAP deviation
    │   └── Structure: Session highs/lows, daily range position
    │
    ├── Macro Features (FRED)
    │   ├── Fed Funds rate delta (week-over-week)
    │   ├── CPI YoY change
    │   ├── Yield curve spread (10Y - 2Y)
    │   └── DXY regime (USD strength index)
    │
    ├── Sentiment Features
    │   ├── News sentiment score (Tiingo, 24h rolling)
    │   ├── COT net positioning (large speculators)
    │   ├── Fear & Greed Index
    │   └── Reddit/social sentiment (Quiver)
    │
    └── Calendar Features
        ├── Session (London/NY/Tokyo/overlap)
        ├── Day of week
        ├── Days to/from major events (FOMC, NFP, CPI)
        └── Volatility event proximity flag
```

**Key data quality rule (from @mar_antaya):** *"Models can't handle messy data. The better the quality data you have, the better prediction you'll make."*
- Fill/drop NaN systematically
- Cap outliers at 3σ (winsorize)
- Feature normalization: StandardScaler per instrument
- Train/test split: chronological ONLY — never shuffle time-series data

---

## 6. ML Architecture

### 6.1 Model Stack

PRISM uses a **three-layer ensemble** approach:

```
Layer 1: Direction Classifier
    XGBoost (primary) + LightGBM (ensemble)
    Output: Long / Short / Neutral (3-class)
    Target: Next N-bar directional bias

Layer 2: Magnitude Regressor
    XGBoost Regressor
    Output: Expected pip move (used to size TP)
    Target: Max favorable excursion over next N bars

Layer 3: Risk Level Classifier
    Random Forest
    Output: High / Medium / Low confidence
    Target: Signal confidence score → position sizing
```

### 6.2 Signal Timeframes — Top-Down Multi-Timeframe Architecture

PRISM uses a **3-tier top-down approach**: higher timeframes define *what* the market is doing; lower timeframes define *where* to get in.

| Tier | Timeframe | Role | ML Layer | Data Source |
|------|-----------|------|----------|-------------|
| 1 — Regime | D1 / H4 | Trend direction, macro bias, FVG zones | Layer 1 XGB+LGB classifier | Tiingo D1 (training) + MT5 live |
| 2 — Structure | H1 | ICC pattern detection (Indication/Correction) | ICC detector | MT5 via Python bridge |
| 3 — Entry | M15 / M5 | FVG retest + ICC Continuation trigger | FVG detector + entry logic | MT5 via Python bridge |

> *The regime layer trains on D1/H4 where macro data is meaningful. Entry precision comes from M5/M15 off the broker's live feed — the exact prices you'll trade on.*

**Top-Down Logic:**
```
D1/H4  → "LONG bias" (XGBoost + macro confirms uptrend)
  H1   → ICC forming: Indication swept liquidity, now in Correction
    M15 → Price enters H4 FVG zone → break & retest → ENTRY
    M5  → Entry refinement: tighter SL, better RR
```

### 6.3 Training Protocol
- **Lookback window:** 3 years of data per instrument (minimum)
- **Walk-forward validation:** 6-month rolling windows
- **Cross-validation:** TimeSeriesSplit (5 folds) — never shuffle
- **Overfitting check:** If train accuracy > test accuracy by >15%, flag for review
- **Retraining cadence:** Weekly (Sunday 02:00) with last 4 weeks of new data

### 6.4 Feature Importance Loop
After each training run:
1. SHAP analysis on top 20 features
2. Drop features with SHAP importance < 0.01
3. Log feature importance to `feature_importance_log.json`
4. Alert if a previously important feature drops >50% — market regime may have shifted

---

## 7. Signal Generation: Entry, SL & TP

### 7.1 Signal Logic Flow — Top-Down with FVG Confluence

```
Step 1: D1/H4 Regime Filter
   → XGBoost Layer 1 predicts directional bias
   → Catalogue open FVG zones on H4 (store in signals/fvg_zones.json)
   → If NEUTRAL confidence > 40%: NO TRADE
   → If Long/Short confidence ≥ 60%: proceed

Step 2: H1 ICC Structure Check
   → Is ICC in INDICATION phase? (liquidity sweep / displacement)
   → Is ICC entering CORRECTION phase? (30–65% retracement)
   → EMA 9/21 alignment must match regime bias direction
   → If no ICC setup: WAIT

Step 3: M15/M5 Entry Trigger — FVG + ICC Continuation
   → Price returns to H4 FVG zone (imbalance retest)
   → ICC enters CONTINUATION phase (Correction low holds)
   → Break & retest: M5 candle closes ABOVE FVG lower boundary (LONG)
                              closes BELOW FVG upper boundary (SHORT)
   → Both conditions required: ENTRY CONFIRMED

Step 4: SL Calculation
   → Place SL below ICC Correction low (LONG) / above Correction high (SHORT)
   → Must be below/above FVG zone (zone acts as structural support/resistance)
   → Minimum SL: 10 pips (forex) / $5 per oz (gold)
   → ATR(14) × 1.5 as maximum SL cap

Step 5: TP Calculation
   → TP1 = prior swing high (LONG) / prior swing low (SHORT) — 50% close
   → TP2 = measured move from entry (ICC leg 1 magnitude projected) — 50% close
   → Minimum RR = 1:1.5; target RR ≥ 1:2. If RR < 1.5: NO TRADE

Step 6: News Filter (Phase 2)
   → Tiingo news sentiment check (last 4 hours)
   → Economic calendar: high-impact event within 30 min → SKIP
   → Geopolitical spike on gold keywords (Iran, war, Fed emergency) → LONG XAU bias
```

### 7.1a Fair Value Gap (FVG) Detection Logic

FVG = 3-candle imbalance where price moves so fast it leaves an unfilled gap:

```
Bullish FVG (demand zone — LONG confluence):
  Candle[i-2].high < Candle[i].low
  → Unfilled gap between top of candle[i-2] and bottom of candle[i]
  → Price tends to return to this zone and bounce LONG
  → Entry: at 50% midline of FVG (or zone boundary for conservative entry)

Bearish FVG (supply zone — SHORT confluence):
  Candle[i-2].low > Candle[i].high
  → Unfilled gap between bottom of candle[i-2] and top of candle[i]
  → Price tends to return to zone and reject SHORT
  → Entry: at 50% midline or zone boundary

Mitigation rules:
  → FVG is "mitigated" (invalidated) when price closes through the full zone
  → Partially mitigated FVGs (price touched midline, closed back out) remain valid
  → Stale FVGs (> 20 bars old without retest) deprioritised
```

FVG zones stored per instrument per timeframe in `signals/fvg_zones.json`. Refreshed on each H4 close. Phase 2 implementation: `prism/signal/fvg.py`.

### 7.2 Signal Output Schema
```json
{
  "instrument": "XAUUSD",
  "signal_time": "2026-04-20T14:30:00Z",
  "direction": "LONG",
  "confidence": 0.74,
  "entry": 2385.50,
  "sl": 2371.20,
  "tp1": 2406.95,
  "tp2": 2428.40,
  "rr_ratio": 2.1,
  "risk_level": "MEDIUM",
  "regime": "BULLISH_MACRO",
  "features_used": ["ATR14", "RSI", "FRED_RATE_DELTA", "COT_NET_LONG"],
  "model_version": "prism_v1.2"
}
```

---

## 8. MT5 Integration Architecture

### 8.1 Stack

```
PRISM Python Core
    ↕ (MetaTrader5 Python library)
MT5 Terminal (Windows/Wine on Linux)
    ↕ (MQL5 EA bridge)
Broker (any MT5-compatible)
```

### 8.2 MT5 Python Bridge
```python
# Core components
import MetaTrader5 as mt5
import pandas as pd
import schedule

# Signal executor
def execute_signal(signal: dict):
    mt5.initialize()
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": signal["instrument"],
        "volume": calculate_lot_size(signal["sl"], risk_pct=1.0),
        "type": mt5.ORDER_TYPE_BUY if signal["direction"] == "LONG" else mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(signal["instrument"]).ask,
        "sl": signal["sl"],
        "tp": signal["tp2"],  # Full TP, manage TP1 via trailing logic
        "comment": f"PRISM_{signal['model_version']}",
        "magic": 20260420,
        "type_time": mt5.ORDER_TIME_GTC,
    }
    result = mt5.order_send(request)
    return result
```

### 8.3 Position Sizing (Kelly-adjusted)
```
Risk per trade: 1% of account balance (configurable)
Lot size = (Account Balance × 0.01) / (SL_pips × pip_value)
Max concurrent trades: 3 (configurable)
Max exposure per currency: 2% of balance
```

### 8.4 Broker Requirements

**Recommended: Exness** (for retail/algo traders starting out)
- $10 minimum deposit — start live trading while PRISM is still calibrating
- Best XAUUSD spreads in the market (~0.08 pips)
- Official Python MT5 library: github.com/exness-pymt5 — built for algo traders
- Instant withdrawals, no holding periods
- Account setup: exness.com → Open MT5 account → note Login ID + Server + Password

**Upgrade path** (when account > $10K):
- IC Markets: ASIC/CySEC regulated, EUR/USD 0.62 pips raw, $200 min
- Pepperstone: ASIC/FCA regulated, no minimum, good for larger volume

**Infrastructure:**
- VPS: Windows Server or Wine on Linux (latency < 50ms to broker)
- Python 3.10+ on same machine as MT5 terminal

---

## 9. OpenClaw Automation Layer

Based on the OpenClaw Live Trading Bot pattern (Hyperliquid tutorial), PRISM will integrate with OpenClaw for:

| Function | Implementation |
|---------|----------------|
| Signal delivery | Slack DM alert with signal JSON + chart image |
| Manual override | Brian can respond "SKIP" or "CONFIRM" before execution |
| Daily briefing | Morning PnL summary, open positions, model confidence |
| Model retraining | Triggered via Slack command `/retrain prism` |
| Risk alerts | Immediate Slack DM if drawdown > 5% in a session |
| Performance tracking | Weekly Sharpe, win rate, avg R:R to Slack |

**Execution mode options:**
- `AUTO` — signals execute automatically with no confirmation
- `CONFIRM` — Slack DM with 5-min window to approve/skip (default for Phase 1)
- `NOTIFY` — signals delivered as alerts only, Brian executes manually

---

## 10. Backtesting Framework

### 10.1 Engine
- **Primary:** `backtesting.py` (Python) for rapid strategy testing
- **Secondary:** MT5's built-in Strategy Tester for exact broker simulation
- **Validation:** Walk-forward on out-of-sample data (never seen during training)

### 10.2 Metrics Tracked
| Metric | Target | Red Line |
|--------|--------|----------|
| Win Rate | ≥62% | <50% |
| Profit Factor | ≥1.8 | <1.2 |
| Sharpe Ratio | ≥1.5 | <0.8 |
| Max Drawdown | ≤15% | >25% |
| Avg R:R | ≥1:2 | <1:1.5 |
| Trades/Month | 20-60 | <10 |

### 10.3 Backtest Data Requirements
- EUR/USD: M15 data from 2018 (includes COVID crash, 2022 rate cycle)
- XAU/USD: M15 data from 2019 (includes pandemic safe-haven spike)
- Ensure data includes: trending, ranging, high-volatility regimes

---

## 11. Phased Development Plan

### Phase 0: Research & Infrastructure (Week 1-2)
- [ ] Pull all mar_antaya IG posts + reels (2026 content) — requires IG login
- [ ] Analyze @richkuo7 methodology — likely SMC/ICT (Order Blocks, FVG, CHOCH)
- [ ] Review all 14 Google Drive files Brian shared
- [ ] Set up data pipeline: Tiingo + FRED + Quiver APIs
- [ ] Set up MT5 on dev machine + Python bridge test
- [ ] Define train/test datasets for EUR/USD + XAU/USD

### Phase 1: Core Model (Week 3-5)
- [ ] Feature engineering pipeline (Jupyter → production Python)
- [ ] Train XGBoost direction classifier (EUR/USD first)
- [ ] Train magnitude regressor
- [ ] Backtest on 2018-2023 data, validate on 2024-2025
- [ ] SHAP analysis — understand what features actually drive predictions
- [ ] Codex review of all model code
- [ ] SL/TP calculation module

### Phase 2: MT5 Integration (Week 5-6)
- [ ] MT5 Python bridge (connect, read data, send orders)
- [ ] Signal executor with position sizing
- [ ] Risk management layer (max drawdown kill switch)
- [ ] Paper trading: 2 weeks on demo account

### Phase 3: OpenClaw Integration (Week 6-7)
- [ ] Signal delivery via Slack DM (Ada → Brian)
- [ ] CONFIRM mode (5-min approval window)
- [ ] Daily/weekly performance reports
- [ ] `/retrain` and `/status` slash commands

### Phase 4: Gold + Multi-Pair (Week 7-8)
- [ ] Retrain model on XAU/USD (gold has different volatility profile)
- [ ] Add GBP/USD
- [ ] Cross-instrument risk management (don't be long USD everywhere)
- [ ] Live trading: start with 0.01 lots, scale after 4 weeks of live results

---

## 12. Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Framework | scikit-learn, XGBoost, LightGBM |
| Data Manipulation | pandas, numpy |
| Feature Importance | SHAP |
| Backtesting | backtesting.py + MT5 Strategy Tester |
| MT5 Bridge | MetaTrader5 Python library (official) |
| Scheduling | `schedule` library + OpenClaw heartbeat |
| Price Data | Tiingo API |
| Macro Data | FRED API (fredapi Python library) |
| Alt Data | Quiver Quantitative API |
| Notifications | OpenClaw → Slack |
| Storage | SQLite (signal log, backtest results, performance) |
| Version Control | GitHub (private repo) |
| Model Versioning | MLflow (lightweight) |
| Dev Environment | Jupyter for research, production Python scripts |
| Primary AI for build | Claude Opus (claude-opus-4-5) |
| Code review | Codex |

---

## 13. Repository Structure

```
prism/
├── README.md
├── PRD.md                          # This document
├── requirements.txt
├── config/
│   ├── instruments.yaml            # Pairs, lot sizes, risk params
│   ├── model_params.yaml           # Hyperparameters
│   └── api_keys.yaml               # (gitignored, use .env)
├── data/
│   ├── raw/                        # Downloaded OHLCV, macro, sentiment
│   ├── processed/                  # Feature-engineered, train/test splits
│   └── credentials.db              # API keys (local SQLite)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_backtesting.ipynb
│   └── 05_shap_analysis.ipynb
├── prism/
│   ├── __init__.py
│   ├── data/
│   │   ├── tiingo.py               # Price data fetcher
│   │   ├── fred.py                 # Macro data fetcher
│   │   ├── quiver.py               # Alt data fetcher
│   │   └── pipeline.py             # Feature engineering
│   ├── model/
│   │   ├── train.py                # Training loop
│   │   ├── predict.py              # Inference
│   │   ├── evaluate.py             # Backtesting + metrics
│   │   └── retrain.py              # Weekly retraining
│   ├── signal/
│   │   ├── generator.py            # Signal logic (entry/SL/TP)
│   │   ├── risk.py                 # Position sizing, risk management
│   │   └── validator.py            # Signal quality checks
│   ├── mt5/
│   │   ├── bridge.py               # MT5 connection + order execution
│   │   ├── monitor.py              # Open position monitoring
│   │   └── reporter.py             # PnL, performance reporting
│   └── openclaw/
│       ├── notifier.py             # Slack signal delivery
│       ├── commands.py             # /retrain, /status, /skip
│       └── briefing.py             # Daily/weekly summaries
├── scripts/
│   ├── backtest_all.sh
│   ├── retrain.sh
│   └── live_trade.sh
└── tests/
    ├── test_signal_generator.py
    ├── test_risk.py
    └── test_mt5_bridge.py
```

---

## 14. Risk & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Overfitting | High | Walk-forward validation, SHAP audit, 3-regime test |
| Look-ahead bias | High | Strict chronological splits, no future data in features |
| Black Swan events (COVID, Flash Crash) | High | Max drawdown kill switch, low initial lot size |
| Broker slippage | Medium | Use ECN broker, test execution in paper first |
| API rate limits | Low | Cache FRED/Tiingo data locally, rate limit guards |
| Model drift | Medium | Weekly retraining + feature importance monitoring |
| MT5 connection drop | Medium | Auto-reconnect logic + Slack alert |
| Instagram data quality | Low | Cross-reference with GitHub repos and newsletter |

---

## 15. Important Notes & Open Questions for Brian

1. *MT5 broker:* Which broker are you trading on? (Need to test Python bridge compatibility)
2. *Account size:* Defines lot sizing — model calibrates risk% per trade
3. *Execution mode:* AUTO vs CONFIRM for Phase 1?
4. *Google Drive files:* I can't access those 14 Drive links — please share them differently (email, Slack file, or make them "anyone with link" if they're not already)
5. *@richkuo7:* I'll need the IG login to review their content — looks like a smaller private account with no public web footprint
6. *VPS for MT5:* Do you have a Windows VPS, or do we need to set one up?

---

## Appendix: mar_antaya's Core Methodology (from content analysis)

Extracted from her Instagram posts and GitHub repos:

| Principle | Detail |
|-----------|--------|
| Data quality first | "Models can't handle messy data" — prep > algorithm |
| Free data sources | FRED, Tiingo, Quiver Quantitative (her top 3) |
| Model hierarchy | Linear Reg → Logistic Reg → XGBoost (complexity ladder) |
| Overfitting watch | Train 99% / Test 60% = bad. Always check gap |
| Sentiment signals | News sentiment as a feature, not a noise filter |
| Monte Carlo | Portfolio risk simulation before going live |
| Evaluation | F1, Precision, Recall over raw accuracy |

*@richkuo7 methodology: To be completed after IG access is confirmed.*

## Appendix B: @tradesbysci ICC Method (Indication → Correction → Continuation)

**Who:** "Sci" — 429K IG / 460K YouTube, verified trader. Pure price action, no indicators.

### The ICC Framework

| Phase | What Happens | What PRISM Detects |
|-------|-------------|--------------------|
| *I — Indication* | Price prints new Higher High (bullish) or Lower Low (bearish) — trend direction confirmed | Swing high/low detection, new extreme print |
| *C — Correction* | Price pulls back / retraces after the indication — don't enter yet, let it breathe | Retracement depth: 38-61.8% of the indication move |
| *C — Continuation* | Price resumes in the original direction, breaks back through the key level — *THIS IS THE ENTRY* | Candle close above/below correction low/high |

### Why ICC Maps to PRISM Perfectly

ICC is the *entry trigger mechanism* for PRISM. The ML layers tell you the macro direction and regime. ICC tells you *when* to pull the trigger:

```
PRISM Layer 1 (XGBoost) → Direction: LONG, Confidence: 74%
    ↓
PRISM Layer 3 → Risk: MEDIUM — proceed
    ↓
ICC Phase: Wait for Indication (new HH confirmed)
    ↓
ICC Phase: Wait for Correction (pullback to 38-50% zone)
    ↓
ICC Phase: Continuation candle closes above correction low → ENTRY
    ↓
ATR(14) × 1.5 → SL below correction low
SL × 2.0-3.0 → TP levels
```

### ICC Detection Algorithm
```python
def detect_icc_phase(df: pd.DataFrame, lookback: int = 20):
    """
    Returns: 'NONE', 'INDICATION', 'CORRECTION', 'CONTINUATION'
    """
    # Step 1: Find swing high/low (Indication)
    recent_high = df['high'].rolling(lookback).max()
    recent_low = df['low'].rolling(lookback).min()
    new_hh = df['high'].iloc[-1] > recent_high.iloc[-2]  # New Higher High
    new_ll = df['low'].iloc[-1] < recent_low.iloc[-2]    # New Lower Low

    # Step 2: Measure retracement (Correction)
    indication_range = abs(df['high'].iloc[-1] - df['low'].iloc[-1-lookback])
    pullback = abs(df['close'].iloc[-1] - df['high'].iloc[-1])
    retracement_pct = pullback / indication_range
    in_correction = 0.30 <= retracement_pct <= 0.65

    # Step 3: Continuation candle
    if new_hh and in_correction:
        # Check if price broke back above correction zone
        correction_low = df['low'].iloc[-5:-1].min()
        continuation = df['close'].iloc[-1] > correction_low
        if continuation:
            return 'CONTINUATION_LONG'
        return 'CORRECTION'
    elif new_ll and in_correction:
        correction_high = df['high'].iloc[-5:-1].max()
        continuation = df['close'].iloc[-1] < correction_high
        if continuation:
            return 'CONTINUATION_SHORT'
        return 'CORRECTION'
    elif new_hh or new_ll:
        return 'INDICATION'
    return 'NONE'
```

### ICC-Specific SL Rules (from Sci's method)
- SL placed *below the correction low* (long) or *above the correction high* (short)
- Never place SL below the full indication swing — too wide
- If correction is too shallow (< 30%), wait — no quality setup
- If correction is too deep (> 65%), trend may be broken — skip

### Areas of Interest (AOI) — Sci's Key Concept
- Mark daily/weekly highs and lows as *Areas of Interest*
- ICC setups at AOI confluence = highest probability trades
- PRISM will mark AOI levels automatically and weight signals that occur at AOI

---

*PRISM PRD v1.1 — Ada Sandpaw for Brian | 2026-04-20*
*Sources: @mar_antaya (data + ML methodology), @tradesbysci (ICC entry framework), @richkuo7 (TBD post IG login)*
*Next step: Brian provides remaining Drive file access + broker/account details → Phase 0 begins.*
