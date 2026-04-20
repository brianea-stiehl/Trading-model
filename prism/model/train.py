"""
prism/model/train.py
PRISM 3-Layer ML Training Pipeline.

Layer 1: XGBClassifier + LGBMClassifier — direction (-1/0/1 → 0/1/2)
Layer 2: XGBRegressor — magnitude in pips
Layer 3: RandomForestClassifier — confidence tier (0/1/2)
"""
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
OVERFIT_THRESHOLD = 0.15
N_CV_SPLITS = 5
DIRECTION_MAP = {-1: 0, 0: 1, 1: 2}          # model label space
DIRECTION_RMAP = {0: -1, 1: 0, 2: 1}          # inverse


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrainingResult:
    instrument: str
    layer: str                        # "layer1_xgb" | "layer1_lgb" | "layer2_magnitude" | "layer3_confidence"
    train_accuracy: float
    test_accuracy: float
    f1_macro: float
    feature_importance: dict[str, float]
    model_path: str
    overfit_flag: bool                # True if train-test > 0.15


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _map_labels(y: pd.Series) -> pd.Series:
    """Map direction {-1,0,1} → {0,1,2} for multi-class classifiers."""
    return y.map(DIRECTION_MAP).fillna(1).astype(int)


def _build_confidence_labels(proba: np.ndarray) -> np.ndarray:
    """
    Derive a 3-tier confidence label from XGB class probabilities.
    Uses prediction entropy — fully available at inference, no ground-truth needed.

    entropy = -sum(p * log(p + 1e-9))  per row
    HIGH (2): entropy < 0.5
    MED  (1): entropy < 0.9
    LOW  (0): else
    """
    entropy = -np.sum(proba * np.log(proba + 1e-9), axis=1)
    labels = np.where(entropy < 0.5, 2, np.where(entropy < 0.9, 1, 0))
    return labels.astype(int)


def _run_shap(model, X: pd.DataFrame, instrument: str, layer: str) -> None:
    """Run SHAP TreeExplainer, save top-20 importances to JSON."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        # Use up to 500 rows for speed
        sample = X.iloc[:500] if len(X) > 500 else X
        shap_values = explainer.shap_values(sample)

        # For multi-class shap_values is a list; take mean across classes
        if isinstance(shap_values, list):
            sv = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            sv = np.abs(shap_values)

        mean_abs = np.mean(sv, axis=0)
        importance = dict(zip(X.columns.tolist(), mean_abs.tolist()))
        top20 = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])

        out_path = MODELS_DIR / f"shap_importance_{instrument}_{layer}.json"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(top20, f, indent=2)
        logger.info(f"SHAP top-20 saved → {out_path}")
    except Exception as e:
        logger.warning(f"SHAP analysis skipped: {e}")




# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PRISMTrainer:
    """
    Orchestrates training of all 3 PRISM model layers for a given instrument.

    Usage:
        trainer = PRISMTrainer("EURUSD")
        results = trainer.train_all_layers("2022-01-01", "2025-01-01")
    """

    def __init__(self, instrument: str = "EURUSD", timeframe: str = "H1"):
        self.instrument = instrument
        self.timeframe = timeframe
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all_layers(
        self, start_date: str, end_date: str
    ) -> list[TrainingResult]:
        """
        Build features, train all 3 layers, save models, run SHAP.
        Returns a list of TrainingResult (one per sub-model).
        """
        logger.info(f"[PRISM] Training {self.instrument} | {start_date} → {end_date}")

        # ---- Feature pipeline ----
        from prism.data.pipeline import PRISMFeaturePipeline
        pipeline = PRISMFeaturePipeline(self.instrument, self.timeframe)
        df = pipeline.build_features(start_date, end_date)

        # Gracefully fill missing macro/sentiment columns
        for col in ["cot_net_speculative", "fear_greed", "dxy", "vix",
                    "us_10y_yield", "sentiment_score"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        df = df.dropna(subset=["direction_fwd_4", "magnitude_pips"]).reset_index(drop=True)

        X_train_df, X_test_df, y_train_dir, y_test_dir = pipeline.split_train_test(df)

        feature_cols = pipeline._feature_cols
        X_train = X_train_df.values.astype(np.float32)
        X_test = X_test_df.values.astype(np.float32)

        y_train_cls = _map_labels(y_train_dir).values
        y_test_cls = _map_labels(y_test_dir).values

        y_train_reg = df.loc[X_train_df.index, "magnitude_pips"].fillna(0).values
        y_test_reg = df.loc[X_test_df.index, "magnitude_pips"].fillna(0).values

        results: list[TrainingResult] = []

        # ---- Layer 1a: XGBClassifier ----
        results.append(self._train_layer1_xgb(
            X_train, X_test, y_train_cls, y_test_cls, X_train_df, feature_cols
        ))

        # ---- Layer 1b: LGBMClassifier ----
        results.append(self._train_layer1_lgbm(
            X_train, X_test, y_train_cls, y_test_cls, X_train_df, feature_cols
        ))

        # ---- Layer 2: XGBRegressor ----
        results.append(self._train_layer2_magnitude(
            X_train, X_test, y_train_reg, y_test_reg, X_train_df, feature_cols
        ))

        # ---- Layer 3: RF Confidence ----
        results.append(self._train_layer3_confidence(
            X_train, X_test, y_train_cls, y_test_cls, X_train_df, feature_cols
        ))

        # Report
        for r in results:
            flag = " ⚠ OVERFIT" if r.overfit_flag else ""
            logger.info(
                f"[{r.layer}] train={r.train_accuracy:.4f} test={r.test_accuracy:.4f} "
                f"f1={r.f1_macro:.4f}{flag}"
            )

        return results

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def _train_layer1_xgb(
        self, X_train, X_test, y_train, y_test, X_df, feature_cols
    ) -> TrainingResult:
        logger.info("Training Layer 1a — XGBClassifier (direction)")
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            num_class=3,
            random_state=42,
            n_jobs=-1,
        )
        val_size = max(1, int(len(X_train) * 0.15))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
        overfit = (train_acc - test_acc) > OVERFIT_THRESHOLD

        path = str(MODELS_DIR / f"layer1_xgb_{self.instrument}.joblib")
        joblib.dump(model, path)
        logger.info(f"Saved → {path}")

        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        _run_shap(model, pd.DataFrame(X_train, columns=feature_cols), self.instrument, "layer1_xgb")

        return TrainingResult(
            instrument=self.instrument,
            layer="layer1_xgb",
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            f1_macro=f1,
            feature_importance=importance,
            model_path=path,
            overfit_flag=overfit,
        )

    def _train_layer1_lgbm(
        self, X_train, X_test, y_train, y_test, X_df, feature_cols
    ) -> TrainingResult:
        logger.info("Training Layer 1b — LGBMClassifier (direction)")
        model = LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            num_class=3,
            objective="multiclass",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        val_size = max(1, int(len(X_train) * 0.15))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
        )
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        f1 = f1_score(y_test, test_pred, average="macro", zero_division=0)
        overfit = (train_acc - test_acc) > OVERFIT_THRESHOLD

        # NB: filename must match PRISMPredictor._load("layer1_lgb") — the
        # predictor loads layer1_lgb_{instrument}.joblib, not layer1_lgbm_.
        path = str(MODELS_DIR / f"layer1_lgb_{self.instrument}.joblib")
        joblib.dump(model, path)
        logger.info(f"Saved → {path}")

        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        _run_shap(model, pd.DataFrame(X_train, columns=feature_cols), self.instrument, "layer1_lgb")

        return TrainingResult(
            instrument=self.instrument,
            layer="layer1_lgb",
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            f1_macro=f1,
            feature_importance=importance,
            model_path=path,
            overfit_flag=overfit,
        )

    def _train_layer2_magnitude(
        self, X_train, X_test, y_train, y_test, X_df, feature_cols
    ) -> TrainingResult:
        logger.info("Training Layer 2 — XGBRegressor (magnitude pips)")
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        val_size = max(1, int(len(X_train) * 0.15))
        X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
        y_tr, y_val = y_train[:-val_size], y_train[-val_size:]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Use R² as "accuracy" proxy for regressor
        from sklearn.metrics import r2_score
        train_acc = max(0.0, float(r2_score(y_train, train_pred)))
        test_acc = max(0.0, float(r2_score(y_test, test_pred)))
        overfit = (train_acc - test_acc) > OVERFIT_THRESHOLD

        # NB: filename must match PRISMPredictor._load("layer2_magnitude").
        path = str(MODELS_DIR / f"layer2_magnitude_{self.instrument}.joblib")
        joblib.dump(model, path)
        logger.info(f"Saved → {path}")

        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))
        _run_shap(model, pd.DataFrame(X_train, columns=feature_cols), self.instrument, "layer2_magnitude")

        return TrainingResult(
            instrument=self.instrument,
            layer="layer2_magnitude",
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            f1_macro=0.0,          # N/A for regression
            feature_importance=importance,
            model_path=path,
            overfit_flag=overfit,
        )

    def _train_layer3_confidence(
        self, X_train, X_test, y_train, y_test, X_df, feature_cols
    ) -> TrainingResult:
        logger.info("Training Layer 3 — RandomForestClassifier (confidence tier)")
        # Need layer1_xgb predictions to build confidence labels
        xgb_path = MODELS_DIR / f"layer1_xgb_{self.instrument}.joblib"
        xgb_model = joblib.load(xgb_path)

        train_proba = xgb_model.predict_proba(X_train)
        test_proba = xgb_model.predict_proba(X_test)

        conf_train = _build_confidence_labels(train_proba)
        conf_test = _build_confidence_labels(test_proba)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, conf_train)
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_acc = accuracy_score(conf_train, train_pred)
        test_acc = accuracy_score(conf_test, test_pred)
        f1 = f1_score(conf_test, test_pred, average="macro", zero_division=0)
        overfit = (train_acc - test_acc) > OVERFIT_THRESHOLD

        # NB: filename must match PRISMPredictor._load("layer3_confidence").
        path = str(MODELS_DIR / f"layer3_confidence_{self.instrument}.joblib")
        joblib.dump(model, path)
        logger.info(f"Saved → {path}")

        importance = dict(zip(feature_cols, model.feature_importances_.tolist()))

        return TrainingResult(
            instrument=self.instrument,
            layer="layer3_confidence",
            train_accuracy=train_acc,
            test_accuracy=test_acc,
            f1_macro=f1,
            feature_importance=importance,
            model_path=path,
            overfit_flag=overfit,
        )
