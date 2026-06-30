"""
ML Quality Model - Quality prediction models
"""


import numpy as np
import pandas as pd
import pickle
from typing import Dict, Any

# ================================
# scikit-learn imports
# ================================
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error
)

# ================================
# XGBoost (Regression)
# ================================
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# ======================================================================
# Base Class
# ======================================================================

class QualityPredictionModel:
    """
    A flexible ML model wrapper that supports:
    - Random Forest
    - XGBoost
    - Linear Regression
    Provides training, prediction, evaluation, saving, loading, and CV.
    """

    def __init__(self, model_type: str = "random_forest", params: Dict[str, Any] = None):
        """
        Initialize model based on type.
        """
        self.model_type = model_type.lower()
        self.params = params or {}

        # ----------------------------------------------------------
        # Initialize model instance based on type
        # ----------------------------------------------------------
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", None),
                random_state=42
            )

        elif self.model_type == "linear_regression":
            self.model = LinearRegression()

        elif self.model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed in this environment.")
            self.model = XGBRegressor(
                n_estimators=self.params.get("n_estimators", 300),
                learning_rate=self.params.get("learning_rate", 0.05),
                max_depth=self.params.get("max_depth", 6),
                subsample=self.params.get("subsample", 0.8),
                colsample_bytree=self.params.get("colsample_bytree", 0.8),
                objective="reg:squarederror",
                random_state=42
            )
        elif self.model_type == "svr":
            self.model = SVR(
                kernel=self.params.get("kernel", "rbf"),
                C=self.params.get("C", 1.0),
                epsilon=self.params.get("epsilon", 0.1),
                gamma=self.params.get("gamma", "scale")
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=self.params.get("n_estimators", 300),
                learning_rate=self.params.get("learning_rate", 0.05),
                max_depth=self.params.get("max_depth", 3),
                subsample=self.params.get("subsample", 1.0),
                loss=self.params.get("loss", "squared_error"),
                random_state=self.params.get("random_state", 42)
            )

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    # ======================================================================
    # TRAIN
    # ======================================================================
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model.
        """
        self.model.fit(X_train, y_train)

    # ======================================================================
    # PREDICT
    # ======================================================================
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict values for test set.
        """
        return np.array(self.model.predict(X_test))

    # ======================================================================
    # PROBABILITY / CONFIDENCE (if supported)
    # ======================================================================
    def predict_proba(self, X_test: pd.DataFrame):
        """
        Predict probability score or confidence (if supported).
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        else:
            # For regression, return variance if available
            if hasattr(self.model, "predict"):
                preds = self.model.predict(X_test)
                return np.abs(preds - np.mean(preds))  # simple uncertainty estimate
            return None

    # ======================================================================
    # EVALUATE
    # ======================================================================
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate using R2, RMSE, MAE, MAPE.
        """
        preds = self.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1e-9))) * 100

        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "mape": mape
        }

    # ======================================================================
    # FEATURE IMPORTANCE
    # ======================================================================
    def get_feature_importance(self, feature_names: list = None) -> Dict[str, float]:
        """
        Returns a dictionary of feature importances.
        Works for RandomForest & XGBoost.
        Returns None for Linear Regression.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            return None

        if feature_names:
            return {name: float(imp) for name, imp in zip(feature_names, importances)}

        return {f"feature_{i}": float(v) for i, v in enumerate(importances)}

    # ======================================================================
    # SAVE MODEL
    # ======================================================================
    def save(self, filepath: str) -> None:
        """
        Save model to disk using pickle.
        """
        with open(filepath, "wb") as f:
            pickle.dump({
                "model_type": self.model_type,
                "params": self.params,
                "model": self.model
            }, f)

    # ======================================================================
    # LOAD MODEL
    # ======================================================================
    @staticmethod
    def load(filepath: str) -> "QualityPredictionModel":
        """
        Load model from disk.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        obj = QualityPredictionModel(model_type=data["model_type"], params=data["params"])
        obj.model = data["model"]
        return obj

    # ======================================================================
    # MODEL TYPE
    # ======================================================================
    def get_model_type(self) -> str:
        """
        Return model type.
        """
        return self.model_type

    # ======================================================================
    # CROSS VALIDATION
    # ======================================================================
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> float:
        """
        Perform k-fold cross validation.
        """
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="r2")
        return float(np.mean(scores))

    # ======================================================================
    # HYPERPARAMETER TUNING
    # ======================================================================
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, param_grid: Dict[str, list]):
        """
        Grid search hyperparameter tuning.
        """
        grid = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring="r2",
            cv=3,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        self.model = grid.best_estimator_
        return grid.best_params_

