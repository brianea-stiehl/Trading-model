# PRD2 — PRISM Intelligence Layer

**Version:** 2.0  
**Author:** Ada Sandpaw  
**Date:** 2026-04-26  
**Status:** DRAFT

---

## Vision

PRISM v2.0 transforms from a signal generator into a **smart money trading system**. When PRD2 is complete:

- **HTF Bias Engine** locks direction from 1H/4H swing structure before any entry scan
- **Smart Money Layer** detects Order Blocks, Liquidity Sweeps, CHOCH/BOS, Po3 phases, and Displacement
- **ML Features** include FVG quality scores, sweep confirmation, OB proximity, OTE zone, and Po3 phase labels
- **Trade Quality Filter** gates signals on FVG quality, OTE alignment, R:R minimum, and candle confirmation

**Expected Outcome:** Win rate improvement from ~55% to ~70-75% by filtering out low-quality setups and aligning with institutional order flow.

---

## Foundation: Phases 1-4 (Reference Only)

PRD2 builds on PRISM v1.0 (PRD1 Phases 1-4), assumed complete and merged to main:

| Phase | Capability | Status |
|-------|------------|--------|
| 1 | ML Direction Model (XGBoost/LightGBM) | ✅ Done |
| 2 | FVG Detection + Break-and-Retest | ✅ Done |
| 3 | Session Filter + Slack Notification + Poll Flow | ✅ Done |
| 4 | Tiingo Live Bars + MT5 Execution + Drawdown Guard | ✅ Done |

**Test Baseline:** 360 tests green (pytest).

---

## Phase 5: HTF Bias Engine

### Objective
Lock trade direction from 1H and 4H swing structure **before** any 5M signal generation. A signal is only valid if it aligns with both 1H and 4H bias.

### Implementation

#### New File: `prism/signal/htf_bias.py`

```python
"""
HTF Bias Engine — Higher Timeframe Direction Lock.
Implements finastictrading's "1H is your compass" rule extended to 4H.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd

class Bias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"

@dataclass
class HTFBiasResult:
    bias_1h: Bias
    bias_4h: Bias
    bias_daily: Optional[Bias]  # Optional — not required for gating
    swing_points_1h: list       # [{"price": float, "type": "HH"|"HL"|"LH"|"LL", "bar_idx": int}, ...]
    swing_points_4h: list
    aligned: bool               # True if 1H and 4H agree (both BULLISH or both BEARISH)
    allowed_direction: Optional[str]  # "LONG" | "SHORT" | None (if ranging or misaligned)

def detect_swing_structure(df: pd.DataFrame, lookback: int = 3) -> list:
    """
    Detect the last N swing highs and lows.
    Returns list of swing points with type: HH, HL, LH, LL.
    """
    pass  # Implementation: rolling window for local max/min

def classify_bias(swing_points: list) -> Bias:
    """
    Classify trend bias from swing points.
    - HH + HL = BULLISH
    - LH + LL = BEARISH
    - Mixed or insufficient = RANGING
    """
    pass

def get_htf_bias(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_daily: Optional[pd.DataFrame] = None,
    min_swing_points: int = 3,
) -> HTFBiasResult:
    """
    Main entry point: compute HTF bias from 1H and 4H bars.
    """
    pass

class HTFBiasEngine:
    """
    Stateful wrapper for HTF bias — caches results per session.
    """
    def __init__(self, lookback_bars: int = 100):
        self.lookback_bars = lookback_bars
        self._cache: dict = {}
    
    def refresh(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> HTFBiasResult:
        """Recompute bias (call once per kill zone start or hourly)."""
        pass
    
    def gate_signal(self, direction: str) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        - allowed=True if direction matches HTF bias
        - allowed=False with reason if misaligned or ranging
        """
        pass
```

#### Key Function Signatures

| Function | Parameters | Returns |
|----------|------------|---------|
| `detect_swing_structure(df, lookback=3)` | OHLCV DataFrame, lookback bars | `list[dict]` of swing points |
| `classify_bias(swing_points)` | Swing point list | `Bias` enum |
| `get_htf_bias(df_1h, df_4h, df_daily=None, min_swing_points=3)` | 1H, 4H, optional Daily DataFrames | `HTFBiasResult` |
| `HTFBiasEngine.refresh(df_1h, df_4h)` | Fresh 1H/4H bars | `HTFBiasResult` |
| `HTFBiasEngine.gate_signal(direction)` | "LONG" or "SHORT" | `tuple[bool, str]` |

#### Integration with `generator.py`

```python
# In SignalGenerator.__init__
from prism.signal.htf_bias import HTFBiasEngine
self.htf_engine = HTFBiasEngine()

# In SignalGenerator.generate()
# After Layer 1 (ML direction), before Layer 2 (ICC):
htf_result = self.htf_engine.refresh(h1_df, h4_df)
allowed, htf_reason = self.htf_engine.gate_signal(direction_str)
if not allowed:
    logger.info(f"HTF gate blocked: {htf_reason}")
    return None
```

#### Data Requirements

- **Source:** Tiingo via `MT5Bridge.get_bars()` (already implemented for H4)
- **New Fetches:** 1H bars (add to runner scan loop)
- **Lookback:** `PRISM_HTF_LOOKBACK_BARS` env var (default: 100)

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRISM_HTF_ENABLED` | `1` | Enable/disable HTF bias gating |
| `PRISM_HTF_LOOKBACK_BARS` | `100` | Bars to analyze for swing structure |
| `PRISM_HTF_MIN_SWING_POINTS` | `3` | Minimum swing points to classify bias |

#### Slack Format Changes

Add HTF bias to signal block:

```
📊 HTF Bias
• 1H: BULLISH (HH → HL → HH)
• 4H: BULLISH (HH → HL)
• Alignment: ✅ LONG only
```

#### Test File: `tests/test_htf_bias.py`

| Test | Description |
|------|-------------|
| `test_detect_swing_structure_uptrend` | HH/HL sequence returns correct swing points |
| `test_detect_swing_structure_downtrend` | LH/LL sequence returns correct swing points |
| `test_detect_swing_structure_ranging` | Mixed sequence returns RANGING |
| `test_classify_bias_bullish` | HH+HL → BULLISH |
| `test_classify_bias_bearish` | LH+LL → BEARISH |
| `test_classify_bias_ranging_mixed` | Mixed swings → RANGING |
| `test_classify_bias_insufficient_points` | <3 points → RANGING |
| `test_get_htf_bias_aligned_bullish` | 1H+4H both BULLISH → aligned=True, allowed="LONG" |
| `test_get_htf_bias_aligned_bearish` | 1H+4H both BEARISH → aligned=True, allowed="SHORT" |
| `test_get_htf_bias_misaligned` | 1H BULLISH, 4H BEARISH → aligned=False |
| `test_get_htf_bias_ranging_blocks` | Either ranging → allowed=None |
| `test_htf_engine_gate_allows_aligned` | LONG signal with BULLISH bias → allowed |
| `test_htf_engine_gate_blocks_misaligned` | SHORT signal with BULLISH bias → blocked |
| `test_htf_engine_caches_within_session` | Multiple calls don't recompute if bars unchanged |
| `test_htf_env_disabled` | PRISM_HTF_ENABLED=0 bypasses gate |
| `test_htf_lookback_configurable` | PRISM_HTF_LOOKBACK_BARS honored |
| `test_generator_integration_htf_gate` | generator.py respects HTF gate |
| ... (25+ tests total) |

**Test Count:** 25+

---

## Phase 6: Smart Money Entry Layer

### Objective
Detect and classify smart money concepts: Order Blocks, Liquidity Sweeps, CHOCH/BOS, Po3 phases, and Displacement. Gate signal generation on sweep confirmation and OB proximity.

### New Files

#### `prism/signal/order_blocks.py`

```python
"""
Order Block Detection — Last opposing candle before displacement.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class OrderBlock:
    instrument: str
    timeframe: str
    direction: str       # "BULLISH" or "BEARISH"
    high: float
    low: float
    midpoint: float
    formed_at: str       # ISO datetime
    formed_bar: int
    displacement_size: float  # Pips of displacement that followed
    mitigated: bool
    age_bars: int

class OrderBlockDetector:
    def __init__(self, instrument: str, timeframe: str = "H4"):
        self.instrument = instrument
        self.timeframe = timeframe
        self.blocks: list[OrderBlock] = []
    
    def detect(self, df: pd.DataFrame, min_displacement_pips: float = 10.0) -> list[OrderBlock]:
        """
        Scan for Order Blocks:
        - Bullish OB: Last bearish candle before bullish displacement (2+ pips up)
        - Bearish OB: Last bullish candle before bearish displacement (2+ pips down)
        """
        pass
    
    def get_active_blocks(self, max_age_bars: int = 50) -> list[OrderBlock]:
        """Return unmitigated OBs within age limit."""
        pass
    
    def get_nearest_ob(self, price: float, direction: str) -> Optional[OrderBlock]:
        """Return the nearest relevant OB for entry (bullish OB for LONG, etc.)."""
        pass
    
    def distance_to_ob(self, price: float, direction: str) -> Optional[float]:
        """Return distance in pips to nearest OB, or None if no relevant OB."""
        pass
```

#### `prism/signal/sweeps.py`

```python
"""
Liquidity Sweep Detection — Price takes out swing high/low then reverses.
"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class LiquiditySweep:
    instrument: str
    type: str            # "HIGH_SWEEP" or "LOW_SWEEP"
    swept_level: float   # The swing high/low that was taken
    sweep_bar: int       # Bar where sweep occurred
    close_inside: bool   # True if candle closed back inside (confirmation)
    timestamp: str
    displacement_followed: bool  # True if displacement confirmed direction

class SweepDetector:
    def __init__(self, instrument: str, lookback: int = 20):
        self.instrument = instrument
        self.lookback = lookback
        self.sweeps: list[LiquiditySweep] = []
    
    def detect(self, df: pd.DataFrame) -> list[LiquiditySweep]:
        """
        Detect liquidity sweeps:
        - HIGH_SWEEP: Price wicks above recent swing high, closes below it
        - LOW_SWEEP: Price wicks below recent swing low, closes above it
        """
        pass
    
    def has_recent_sweep(self, direction: str, bars_back: int = 5) -> bool:
        """
        Check if there's a confirmed sweep supporting the given direction:
        - LONG: needs a LOW_SWEEP (manipulation took out lows, now ready to go up)
        - SHORT: needs a HIGH_SWEEP
        """
        pass
    
    def last_sweep(self, direction: str) -> Optional[LiquiditySweep]:
        """Return the most recent relevant sweep."""
        pass
```

#### `prism/signal/po3.py`

```python
"""
Power of Three (Po3) Phase Detection — Accumulation / Manipulation / Distribution.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd

class Po3Phase(str, Enum):
    ACCUMULATION = "ACCUMULATION"
    MANIPULATION = "MANIPULATION"
    DISTRIBUTION = "DISTRIBUTION"
    UNKNOWN = "UNKNOWN"

@dataclass
class Po3State:
    phase: Po3Phase
    session: str          # "LONDON" or "NY"
    session_open: float   # Price at session open
    session_high: float
    session_low: float
    range_size_pips: float
    sweep_detected: bool
    displacement_detected: bool

class Po3Detector:
    def __init__(self, instrument: str):
        self.instrument = instrument
    
    def detect_phase(self, df: pd.DataFrame, session: str) -> Po3State:
        """
        Classify current Po3 phase within the session:
        - ACCUMULATION: First 30-60 min, tight range, building positions
        - MANIPULATION: Sweep of session high/low (take out stops)
        - DISTRIBUTION: Strong directional move after manipulation
        """
        pass
    
    def is_entry_phase(self, state: Po3State) -> bool:
        """
        Return True only when:
        - Manipulation has completed (sweep_detected=True)
        - Distribution is starting (displacement_detected=True)
        """
        return state.sweep_detected and state.displacement_detected
```

#### Integration with `generator.py`

```python
# New imports
from prism.signal.order_blocks import OrderBlockDetector
from prism.signal.sweeps import SweepDetector
from prism.signal.po3 import Po3Detector, Po3Phase

# In SignalGenerator.__init__
self.ob_detector = OrderBlockDetector(instrument, "H4")
self.sweep_detector = SweepDetector(instrument)
self.po3_detector = Po3Detector(instrument)

# In SignalGenerator.generate(), after Layer 3 (FVG):

# --- Smart Money Layer ---
# 1. Detect Order Blocks
self.ob_detector.detect(h4_df)

# 2. Check for recent liquidity sweep
if not self.sweep_detector.has_recent_sweep(direction_str, bars_back=5):
    logger.info(f"No recent sweep for {direction_str} — skipping")
    return None

# 3. Check Po3 phase
po3_state = self.po3_detector.detect_phase(entry_df, session_label)
if not self.po3_detector.is_entry_phase(po3_state):
    logger.info(f"Po3 phase {po3_state.phase} — not entry phase")
    return None

# 4. Check OB proximity (optional confluence boost)
ob_distance = self.ob_detector.distance_to_ob(current_price, direction_str)
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRISM_SWEEP_REQUIRED` | `1` | Require sweep confirmation before entry |
| `PRISM_OB_MAX_DISTANCE_PIPS` | `30` | Max distance to OB for "in range" |
| `PRISM_MIN_DISPLACEMENT_PIPS` | `10` | Minimum displacement size for OB detection |

#### Slack Format Changes

```
🎯 Smart Money
• Sweep: ✅ LOW_SWEEP at 2340.50 (3 bars ago)
• Po3: DISTRIBUTION (manipulation complete)
• OB: BULLISH at 2338.00-2339.50 (12.5 pips away)
• Displacement: ✅ 18 pip up-move confirmed
```

#### Test Files

**`tests/test_order_blocks.py`** (15+ tests)
- `test_detect_bullish_ob_after_displacement`
- `test_detect_bearish_ob_after_displacement`
- `test_ob_mitigation_tracking`
- `test_ob_age_filtering`
- `test_get_nearest_ob_long`
- `test_get_nearest_ob_short`
- `test_distance_to_ob_calculation`
- ... etc.

**`tests/test_sweeps.py`** (10+ tests)
- `test_detect_high_sweep`
- `test_detect_low_sweep`
- `test_sweep_requires_close_inside`
- `test_has_recent_sweep_long`
- `test_has_recent_sweep_short`
- ... etc.

**`tests/test_po3.py`** (5+ tests)
- `test_detect_accumulation_phase`
- `test_detect_manipulation_phase`
- `test_detect_distribution_phase`
- `test_is_entry_phase_requires_both`
- `test_po3_session_awareness`

**Total Test Count:** 30+

---

## Phase 7: ML Feature Enhancement

### Objective
Add ICT-derived features to the ML training pipeline, retrain models, and A/B test old vs. new feature sets.

### New File: `prism/data/feature_engineering.py`

```python
"""
ICT-Derived Feature Engineering for PRISM ML Models.
"""
from typing import Optional
import pandas as pd
import numpy as np

def compute_fvg_quality_score(
    fvg_zone,
    in_kill_zone: bool,
    post_displacement: bool,
    ob_confluence: bool,
    size_vs_atr: float,
    age_bars: int,
) -> float:
    """
    Compute FVG quality score (0.0 - 1.0).
    
    Scoring:
    - In kill zone: +0.25
    - Post displacement: +0.25
    - OB confluence (OB within 10 pips): +0.20
    - Size > 0.5× ATR: +0.15
    - Age < 10 bars: +0.15
    """
    score = 0.0
    if in_kill_zone:
        score += 0.25
    if post_displacement:
        score += 0.25
    if ob_confluence:
        score += 0.20
    if size_vs_atr > 0.5:
        score += 0.15
    if age_bars < 10:
        score += 0.15
    return min(1.0, score)

def compute_ote_zone(
    current_price: float,
    swing_high: float,
    swing_low: float,
) -> tuple[bool, float]:
    """
    Check if price is in OTE zone (61.8% - 78.6% fib retracement).
    Returns (in_ote, fib_level).
    """
    range_size = swing_high - swing_low
    if range_size <= 0:
        return False, 0.0
    retracement = (swing_high - current_price) / range_size  # For bullish
    in_ote = 0.618 <= retracement <= 0.786
    return in_ote, retracement

def compute_kill_zone_strength(hour_utc: int) -> int:
    """
    Score kill zone strength (0-3).
    - 0: Off-session
    - 1: Asian (avoid)
    - 2: London/NY edges
    - 3: London/NY core (highest activity)
    """
    if 8 <= hour_utc <= 10:  # London core
        return 3
    if 14 <= hour_utc <= 16:  # NY core
        return 3
    if 7 <= hour_utc <= 11:  # London extended
        return 2
    if 13 <= hour_utc <= 17:  # NY extended
        return 2
    if 0 <= hour_utc <= 6:  # Asian
        return 1
    return 0

def compute_htf_alignment(bias_1h: str, bias_4h: str, direction: str) -> int:
    """
    Score HTF alignment (0-3).
    - 0: Both against
    - 1: One against
    - 2: One aligned, one neutral
    - 3: Both aligned
    """
    aligned_1h = (bias_1h == "BULLISH" and direction == "LONG") or \
                 (bias_1h == "BEARISH" and direction == "SHORT")
    aligned_4h = (bias_4h == "BULLISH" and direction == "LONG") or \
                 (bias_4h == "BEARISH" and direction == "SHORT")
    neutral_1h = bias_1h == "RANGING"
    neutral_4h = bias_4h == "RANGING"
    
    if aligned_1h and aligned_4h:
        return 3
    if (aligned_1h and neutral_4h) or (neutral_1h and aligned_4h):
        return 2
    if aligned_1h or aligned_4h:
        return 1
    return 0

class ICTFeatureEngineer:
    """
    Adds ICT-derived features to the PRISM feature matrix.
    """
    def __init__(self, instrument: str):
        self.instrument = instrument
    
    def enrich_features(
        self,
        df: pd.DataFrame,
        fvg_zones: list,
        ob_zones: list,
        sweeps: list,
        htf_bias: dict,
        po3_phase: str,
    ) -> pd.DataFrame:
        """
        Add ICT columns to feature DataFrame:
        - fvg_quality_score (0-1)
        - sweep_confirmed (bool)
        - ob_distance_pips (float)
        - po3_phase (enum encoded)
        - ote_zone (bool)
        - kill_zone_strength (0-3)
        - htf_alignment (0-3)
        """
        pass
```

#### Integration with `pipeline.py`

```python
# In PRISMFeaturePipeline._engineer_features()

# After existing features, add ICT features:
from prism.data.feature_engineering import ICTFeatureEngineer

ict_eng = ICTFeatureEngineer(self.instrument)
# ... compute FVG zones, OB zones, sweeps, HTF bias, Po3 for each bar
# ... call ict_eng.enrich_features()
```

#### New Features Summary

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| `fvg_quality_score` | float (0-1) | Composite quality based on kill zone, displacement, OB confluence, size, age | Justin Werlein |
| `sweep_confirmed` | bool | True if sweep of swing high/low within last N bars | GatieTrades Po3 |
| `ob_distance_pips` | float | Distance to nearest relevant Order Block | GatieTrades GEMS |
| `po3_phase` | int (0-2) | 0=ACCUMULATION, 1=MANIPULATION, 2=DISTRIBUTION | GatieTrades Po3 |
| `ote_zone` | bool | True if price within 61.8%-78.6% fib retracement | ICT OTE |
| `kill_zone_strength` | int (0-3) | Session activity level | Multiple sources |
| `htf_alignment` | int (0-3) | HTF bias agreement with signal direction | finastictrading |

#### Retraining Process

1. Run `python -m prism.model.retrain --instrument XAUUSD` with new features
2. Compare test-set accuracy: old features vs. new features
3. Log feature importance rankings
4. A/B test in NOTIFY mode for 2 weeks before enabling AUTO

#### Test File: `tests/test_feature_engineering.py`

| Test | Description |
|------|-------------|
| `test_fvg_quality_score_max` | All factors present → 1.0 |
| `test_fvg_quality_score_min` | No factors → 0.0 |
| `test_fvg_quality_partial` | Some factors → proportional score |
| `test_ote_zone_inside` | 61.8%-78.6% → True |
| `test_ote_zone_outside` | 50% → False |
| `test_kill_zone_strength_london_core` | 09:00 UTC → 3 |
| `test_kill_zone_strength_asian` | 03:00 UTC → 1 |
| `test_htf_alignment_both_aligned` | BULLISH+BULLISH+LONG → 3 |
| `test_htf_alignment_misaligned` | BULLISH+BEARISH → 0 |
| `test_enrich_features_all_columns` | All 7 columns added |
| ... (15+ tests total) |

---

## Phase 8: Trade Quality Filter + Dynamic Targets

### Objective
Add final gating layer: FVG quality minimum, OTE zone preference, dynamic TP targeting, R:R minimum increase, and candle confirmation.

### New File: `prism/signal/quality_filter.py`

```python
"""
Trade Quality Filter — Final gate before signal emission.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

@dataclass
class QualityCheckResult:
    passed: bool
    fvg_quality: float
    in_ote: bool
    rr_ratio: float
    candle_confirmed: bool
    ema_confirmed: bool
    reasons: list[str]  # List of failing checks if passed=False

def check_fvg_quality(fvg_zone, min_quality: float = 0.6) -> Tuple[bool, float]:
    """
    Check if FVG meets minimum quality threshold.
    Returns (passed, quality_score).
    """
    pass

def check_ote_zone(
    entry_price: float,
    swing_high: float,
    swing_low: float,
    direction: str,
) -> bool:
    """
    Check if entry is in OTE zone (61.8%-78.6% fib).
    """
    pass

def check_rr_ratio(
    entry: float,
    sl: float,
    tp: float,
    min_rr: float = 2.0,
) -> Tuple[bool, float]:
    """
    Check if R:R meets minimum threshold.
    Returns (passed, actual_rr).
    """
    pass

def check_candle_confirmation(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 3,
) -> bool:
    """
    Check for reversal candle pattern:
    - Engulfing
    - Hammer / Inverted Hammer
    - Doji
    - Morning Star / Evening Star
    """
    pass

def check_ema_confirmation(
    df: pd.DataFrame,
    direction: str,
    ema_period: int = 8,
) -> bool:
    """
    Check if last candle closed above (LONG) or below (SHORT) the 8 EMA.
    """
    pass

def calculate_dynamic_tp(
    entry: float,
    direction: str,
    df: pd.DataFrame,
    lookback: int = 50,
) -> float:
    """
    Calculate TP as next liquidity pool (unconsumed swing high/low).
    Scans lookback bars for swing points not yet taken.
    """
    pass

class QualityFilter:
    def __init__(
        self,
        min_fvg_quality: float = 0.6,
        min_rr: float = 2.0,
        require_candle_confirm: bool = True,
        require_ema_confirm: bool = True,
        prefer_ote: bool = True,
    ):
        self.min_fvg_quality = min_fvg_quality
        self.min_rr = min_rr
        self.require_candle_confirm = require_candle_confirm
        self.require_ema_confirm = require_ema_confirm
        self.prefer_ote = prefer_ote
    
    def check(
        self,
        fvg_zone,
        entry: float,
        sl: float,
        tp: float,
        direction: str,
        swing_high: float,
        swing_low: float,
        df: pd.DataFrame,
    ) -> QualityCheckResult:
        """
        Run all quality checks and return result.
        """
        pass
```

#### Integration with `generator.py`

```python
# In SignalGenerator.__init__
from prism.signal.quality_filter import QualityFilter, calculate_dynamic_tp
self.quality_filter = QualityFilter(
    min_fvg_quality=float(os.environ.get("PRISM_FVG_MIN_QUALITY", "0.6")),
    min_rr=float(os.environ.get("PRISM_MIN_RR", "2.0")),
)

# In SignalGenerator.generate(), after SL/TP calculation:

# --- Quality Filter ---
# 1. Calculate dynamic TP (next liquidity pool)
dynamic_tp = calculate_dynamic_tp(entry, direction_str, entry_df)
tp2 = dynamic_tp if dynamic_tp else tp2  # Fall back to calculated TP2

# 2. Run quality checks
quality = self.quality_filter.check(
    fvg_zone=fvg_zone,
    entry=entry,
    sl=sl,
    tp=tp2,
    direction=direction_str,
    swing_high=df["high"].iloc[-20:].max(),
    swing_low=df["low"].iloc[-20:].min(),
    df=entry_df,
)

if not quality.passed:
    logger.info(f"Quality filter failed: {quality.reasons}")
    return None
```

#### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRISM_FVG_MIN_QUALITY` | `0.6` | Minimum FVG quality score (0-1) |
| `PRISM_MIN_RR` | `2.0` | Minimum R:R ratio (was 1.5 in Phase 1-4) |
| `PRISM_REQUIRE_CANDLE_CONFIRM` | `1` | Require reversal candle pattern |
| `PRISM_REQUIRE_EMA_CONFIRM` | `1` | Require close above/below 8 EMA |
| `PRISM_USE_DYNAMIC_TP` | `1` | Calculate TP from next liquidity pool |

#### Slack Format Changes

```
✅ Quality Check
• FVG Quality: 0.85 ✅ (min: 0.60)
• OTE Zone: ✅ (fib 68.2%)
• R:R: 2.4 ✅ (min: 2.0)
• Candle: ✅ Bullish Engulfing
• 8 EMA: ✅ Closed above

🎯 Targets
• TP1: 2355.00 (1:1 partial)
• TP2: 2368.50 (next liquidity pool — unconsumed HH)
• Dynamic TP: ✅ Enabled
```

#### Test File: `tests/test_quality_filter.py`

| Test | Description |
|------|-------------|
| `test_fvg_quality_passes_above_min` | 0.7 >= 0.6 → pass |
| `test_fvg_quality_fails_below_min` | 0.5 < 0.6 → fail |
| `test_ote_zone_passes_in_range` | 65% → pass |
| `test_ote_zone_fails_outside` | 50% → fail |
| `test_rr_passes_above_min` | 2.5 >= 2.0 → pass |
| `test_rr_fails_below_min` | 1.5 < 2.0 → fail |
| `test_candle_confirms_engulfing` | Bullish engulfing → True |
| `test_candle_confirms_hammer` | Hammer → True |
| `test_candle_rejects_no_pattern` | Doji → False (unless at structure) |
| `test_ema_confirms_above` | Close > 8 EMA for LONG → True |
| `test_ema_confirms_below` | Close < 8 EMA for SHORT → True |
| `test_ema_rejects_wrong_side` | Close < 8 EMA for LONG → False |
| `test_dynamic_tp_finds_liquidity` | Next swing high → correct TP |
| `test_dynamic_tp_fallback` | No swing → uses fixed TP |
| `test_quality_filter_all_pass` | All checks pass → QualityCheckResult.passed=True |
| `test_quality_filter_partial_fail` | One check fails → passed=False with reasons |
| `test_quality_filter_env_overrides` | Env vars honored |
| ... (25+ tests total) |

**Total Test Count:** 25+

---

## Summary: PRD2 Deliverables

### New Files

| Phase | File | Purpose |
|-------|------|---------|
| 5 | `prism/signal/htf_bias.py` | HTF swing structure + bias lock |
| 5 | `tests/test_htf_bias.py` | 25+ tests |
| 6 | `prism/signal/order_blocks.py` | OB detection |
| 6 | `prism/signal/sweeps.py` | Liquidity sweep detection |
| 6 | `prism/signal/po3.py` | Po3 phase tagging |
| 6 | `tests/test_order_blocks.py` | 15+ tests |
| 6 | `tests/test_sweeps.py` | 10+ tests |
| 6 | `tests/test_po3.py` | 5+ tests |
| 7 | `prism/data/feature_engineering.py` | ICT-derived ML features |
| 7 | `tests/test_feature_engineering.py` | 15+ tests |
| 8 | `prism/signal/quality_filter.py` | Final quality gate |
| 8 | `tests/test_quality_filter.py` | 25+ tests |

### Modified Files

| File | Changes |
|------|---------|
| `prism/signal/generator.py` | HTF gate, sweep gate, Po3 gate, quality filter integration |
| `prism/delivery/runner.py` | Fetch 1H bars, pass to generator |
| `prism/delivery/slack_notifier.py` | New blocks: HTF Bias, Smart Money, Quality Check |
| `prism/data/pipeline.py` | Call ICTFeatureEngineer for enrichment |

### Environment Variables (All New)

| Variable | Default | Phase |
|----------|---------|-------|
| `PRISM_HTF_ENABLED` | `1` | 5 |
| `PRISM_HTF_LOOKBACK_BARS` | `100` | 5 |
| `PRISM_HTF_MIN_SWING_POINTS` | `3` | 5 |
| `PRISM_SWEEP_REQUIRED` | `1` | 6 |
| `PRISM_OB_MAX_DISTANCE_PIPS` | `30` | 6 |
| `PRISM_MIN_DISPLACEMENT_PIPS` | `10` | 6 |
| `PRISM_FVG_MIN_QUALITY` | `0.6` | 8 |
| `PRISM_MIN_RR` | `2.0` | 8 |
| `PRISM_REQUIRE_CANDLE_CONFIRM` | `1` | 8 |
| `PRISM_REQUIRE_EMA_CONFIRM` | `1` | 8 |
| `PRISM_USE_DYNAMIC_TP` | `1` | 8 |

### Test Counts

| Phase | Tests |
|-------|-------|
| 5 (HTF Bias) | 25+ |
| 6 (Smart Money) | 30+ |
| 7 (Features) | 15+ |
| 8 (Quality) | 25+ |
| **Total PRD2** | **95+** |
| **PRD1 Baseline** | 360 |
| **Combined** | **455+** |

---

## Timeline Estimate

| Phase | Effort | Depends On |
|-------|--------|------------|
| 5 (HTF Bias) | 3-4 days | — |
| 6 (Smart Money) | 5-7 days | Phase 5 |
| 7 (Features) | 3-4 days | Phase 6 |
| 8 (Quality) | 3-4 days | Phase 7 |
| **Total** | **14-19 days** | Sequential |

*Assumes single Claude Code agent or equivalent coding capacity.*

---

*End of PRD2*
