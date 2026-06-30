from ml.training.inference import HLDQualityPredictor
import numpy as np 

models_dir = "Project/ml/models"
def load_models_from_disk():
    import pickle, os
    model_files = {
        "RandomForest": "RandomForest.pkl",
        "GradientBoosting": "GradientBoosting.pkl",
        "XGBoost": "XGBoost.pkl",
        "SVR": "SVR.pkl",
        "LinearRegression": "LinearRegression.pkl",
    }

    models = {}
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            print(f"⚠️ Model not found: {path}")
    print(f"[INFo] Model loaded {list(self.models.keys())}")
    return models

predictor = HLDQualityPredictor(model_dir=models_dir)
predictor.load_models_from_disk()

# print(list(models.keys()))

feature = {f: np.mean(rng) for f,rng in predictor.feature_ranges.items()}
result = predictor.predict(feature)

print(result)