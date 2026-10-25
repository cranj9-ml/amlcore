"""The LightGBM model: fitting and scoring.

Why LightGBM: gradient-boosted trees are the standard for tabular fraud/AML data. They
handle skewed amounts, missing values (our NaN ratios) and interactions without feature
scaling, and train on millions of rows in minutes. A linear model trailed by about 30
points of recall here: the signal is in thresholds and interactions (docs/experiments.md).

Class imbalance (~0.1% positives): positives are NOT up-weighted by default. We only need
a good ranking of the top of the list, and ranking doesn't need balanced classes.
Up-weighting by sqrt(neg/pos) cut validation average precision to about a third: trees
chase the rare positives with large leaf values, which scrambles the order at the top
(docs/talking_points.md #4). The "sqrt" option is kept so that can be reproduced.

Early stopping on validation average precision: the number of trees is chosen on data
the model did not train on. That makes validation results slightly optimistic; test is
the honest estimate.
"""

from __future__ import annotations

import math

import lightgbm as lgb
import numpy as np
import pandas as pd

from amlcore.config import ModelConfig
from amlcore.features import CATEGORICAL_FEATURES


def positive_weight(y: pd.Series | np.ndarray, class_weighting: str) -> float:
    """LightGBM's scale_pos_weight: 1 for "none", sqrt(negatives / positives) for "sqrt"."""
    y = np.asarray(y)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("the training split has no positives; cannot train a model")
    return 1.0 if class_weighting == "none" else math.sqrt((len(y) - n_pos) / n_pos)


def fit_lightgbm(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series,
                 config: ModelConfig) -> lgb.LGBMClassifier:
    """Fit on train, stopping when validation average precision stops improving."""
    model = lgb.LGBMClassifier(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=1,  # subsample only takes effect when bagging runs every N trees
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        scale_pos_weight=positive_weight(y_train, config.class_weighting),
        metric="average_precision",
        random_state=config.seed,
        deterministic=True,
        verbose=-1,
    )
    model.fit(
        x_train, y_train,
        eval_X=(x_val,), eval_y=(y_val,),
        categorical_feature=[col for col in CATEGORICAL_FEATURES if col in x_train.columns],
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, first_metric_only=True, verbose=False)],
    )
    return model


def predict_scores(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    """Laundering scores for ranking (uses the best iteration found by early stopping)."""
    return model.predict_proba(x)[:, 1]
