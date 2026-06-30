"""
Terminal Script: HLD Quality Prediction using trained ML models
Loads RandomForest, GradientBoosting, and XGBoost models and compares predictions
"""

import os
import numpy as np
import pandas as pd
from ml.training.inference import HLDQualityPredictor
models_dir = "Project/ml/models"


def classify_quality(score: float) -> str:
    """Classify HLD quality into Excellent, Average, or Poor."""
    if score >= 80:
        return "🏆 Excellent HLD Quality"
    elif score >= 60:
        return "⚙️ Average HLD Quality"
    else:
        return "🚨 Poor HLD Quality"

def load_models_from_disk():
    import pickle, os
    model_files = {
        "RandomForest": "RandomForest.pkl",
        "GradientBoosting": "GradientBoosting.pkl",
        "XGBoost": "XGBoost.pkl",
        # "SVR": "SVR.pkl",
        # "LinearRegression": "LinearRegression.pkl",
    }

    models = {}
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            print(f"⚠️ Model not found: {path}")
    # print(f"[INFo] Model loaded {list(self.models.keys())}")
    return models

def run_quality_prediction_cli():
    """Run the quality prediction pipeline via terminal."""
    models_dir = "Project/ml/models"
    predictor = HLDQualityPredictor(model_dir=models_dir)

    print("🔍 Loading trained models from:", models_dir)
    models = load_models_from_disk()

    if not models:
        print("❌ No trained models found. Please train them first.")
        return
    predictor.models = models
    print(f"✅ Loaded models: {list(models.keys())}\n")

    # --- Define Quick Scenarios ---
    scenarios = {
        "Excellent HLD": {k: np.mean(v) * 0.9 for k, v in predictor.feature_ranges.items()},
        "Average HLD": {k: np.mean(v) * 0.6 for k, v in predictor.feature_ranges.items()},
        "Poor HLD": {k: np.mean(v) * 0.35 for k, v in predictor.feature_ranges.items()},
    }

    print("⚡ Available Scenarios:")
    for i, s in enumerate(scenarios.keys(), start=1):
        print(f"  {i}. {s}")
    choice = input("\nSelect a scenario (1/2/3): ").strip()

    try:
        choice = int(choice)
        scenario_name = list(scenarios.keys())[choice - 1]
    except (ValueError, IndexError):
        print("❌ Invalid choice. Exiting.")
        return

    print(f"\n🚀 Running prediction for: {scenario_name}")
    features = scenarios[scenario_name]

    # --- Run Prediction ---
    preds = predictor.predict(features)

    if "error" in preds:
        print("❌ Error:", preds["error"])
        return

    # --- Extract model predictions ---
    rf = preds.get("RandomForest", None)
    gb = preds.get("GradientBoosting", None)
    xgb = preds.get("XGBoost", None)
    ensemble = preds.get("ensemble_average", None)
    conf = preds.get("confidence", None)
    unc = preds.get("uncertainty", None)

    print("\n📊 Model Predictions:")
    df = pd.DataFrame([
        {"Model": "Random Forest", "Prediction": round(rf, 3)},
        {"Model": "Gradient Boosting", "Prediction": round(gb, 3)},
        {"Model": "XGBoost", "Prediction": round(xgb, 3)},
        {"Model": "Ensemble (Avg)", "Prediction": round(ensemble, 3)},
    ])
    print(df.to_string(index=False))

    print("\n🔎 Confidence:", f"{conf*100:.2f}%")
    print("🎯 Uncertainty (σ):", round(unc, 3))
    print("💡 Quality Verdict:", classify_quality(ensemble))


if __name__ == "__main__":
    run_quality_prediction_cli()
