"""
ML Model Training Pipeline
Trains multiple model types on synthetic HLD dataset
"""


import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor


class LargeScaleMLTrainer:
    """Train, evaluate, and save multiple ML regression models on synthetic HLD dataset."""

    def __init__(self):
        # Model registry
        self.models: Dict[str, Any] = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=200, random_state=42),
            "SVR": SVR(kernel="rbf"),
            "LinearRegression": LinearRegression(),
        }
        self.results: Dict[str, Dict[str, float]] = {}

        # Data placeholders
        self.df: pd.DataFrame | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.scaler = StandardScaler()

    # -----------------------------
    # Dataset Loading & Preparation
    # -----------------------------
    def load_dataset(self, filepath: str) -> None:
        """Load and validate dataset from CSV file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found at {filepath}")

        self.df = pd.read_csv(filepath)
        if self.df.empty:
            raise ValueError("Loaded dataset is empty.")

        if "quality_score" not in self.df.columns:
            raise ValueError("Dataset must include a 'quality_score' column.")

    def prepare_data(self, df: pd.DataFrame | None = None) -> None:
        """Split dataset into features and target sets."""
        if df is None:
            df = self.df
        if df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        X = df.drop(columns=["quality_score"])
        y = df["quality_score"]

        X_scaled = self.scaler.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

    # -----------------------------
    # Model Training & Evaluation
    # -----------------------------
    def train_models(self) -> None:
        """Train all models on the prepared dataset."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        for name, model in self.models.items():
            try:
                model.fit(self.X_train, self.y_train)
                print(f"[INFO] Trained model: {name}")
            except Exception as e:
                print(f"[ERROR] Failed to train {name}: {e}")

    def evaluate_models(self) -> None:
        """Evaluate models and store performance metrics."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data not prepared.")

        for name, model in self.models.items():
            try:
                preds = model.predict(self.X_test)
                metrics = {
                    "R2_Train": r2_score(self.y_train, model.predict(self.X_train)),
                    "R2_Test": r2_score(self.y_test, preds),
                    "RMSE": np.sqrt(mean_squared_error(self.y_test, preds)),
                    "MAE": mean_absolute_error(self.y_test, preds),
                    "MAPE": np.mean(np.abs((self.y_test - preds) / (self.y_test + 1e-8))) * 100,
                }
                self.results[name] = metrics
                print(f"[EVAL] {name}: {metrics}")
            except Exception as e:
                print(f"[ERROR] Evaluation failed for {name}: {e}")

    # -----------------------------
    # Model Persistence & Insights
    # -----------------------------
    def save_models(self, output_dir: str = "ml/models") -> None:
        """Save trained models to disk."""
        os.makedirs(output_dir, exist_ok=True)
        for name, model in self.models.items():
            path = os.path.join(output_dir, f"{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(model, f)
            print(f"[SAVED] {name} -> {path}")

    def get_feature_importance(self, model_name: str) -> Dict[str, float]:
        """Return feature importances for supported models."""
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.models[model_name]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return {
                f"feature_{i}": imp for i, imp in enumerate(importances)
            }
        else:
            raise ValueError(f"Model '{model_name}' does not support feature importances.")

    def cross_validation_score(self, model_name: str, cv: int = 5) -> float:
        """Compute cross-validation score for a given model."""
        if self.df is None:
            raise ValueError("Dataset not loaded.")
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in trainer.")

        X = self.df.drop(columns=["quality_score"])
        y = self.df["quality_score"]
        model = self.models[model_name]
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        mean_score = np.mean(scores)
        print(f"[CV] {model_name}: {mean_score:.4f}")
        return mean_score

