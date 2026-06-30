import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional


class HLDQualityPredictor:

    # ----------------------------------------------------------------------
    def __init__(self, model_dir: str = "Project/ml/models"):
        """
        Initialize:
        - empty model dictionary
        - feature names list (empty until first loaded)
        - model directory
        """
        self.model_dir = model_dir
        self.models: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.feature_ranges = self.get_feature_ranges()
        self.scaler=None

    # ----------------------------------------------------------------------
    def load_models_from_disk(self) -> bool:
        """
        Load all trained models from disk.
        Expected filenames:
            rf_model.pkl
            xgb_model.pkl
            gb_model.pkl
            svr_model.pkl
            linear_model.pkl
        Returns True if at least one model loads successfully.
        """
        if not os.path.exists(self.model_dir):
            print(f"[WARN] Model directory not found: {self.model_dir}")
            return False

        model_files = {
            "rf_model" : "RandomForest.pkl",
       "gb_model" : "GradientBoosting.pkl",
       "xgb_model" : "XGBoost.pkl",
        "svr_model" : "SVR.pkl",
        # "LinearRegression.pkl"
        }

        loaded_any = False

        for model_name, filename in model_files.items():
            path = os.path.join(self.model_dir, filename)

            if not os.path.exists(path):
                print(f"[WARN]Model file not found:{filename}")
                continue

            try:
                with open(path, "rb") as f:
                    model_obj = pickle.load(f)

                self.models[model_name] = model_obj
                loaded_any = True

                # Extract feature names stored inside model object
                if hasattr(model_obj, "feature_names_in_"):
                    self.feature_names = list(model_obj.feature_names_in_)
                elif hasattr(model_obj,"feature_names"):
                    self.feature_names=model_obj.feature_names

                print(f"[INFO] Loaded {model_name} from {filename}")

            except Exception as e:
                print(f"[ERROR] Could not load model {filename}: {e}")
        scaler_path=os.path.join(self.model_dir,"scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path,"rb")as f:
                    self.scaler=pickle.load(f)
                print(f"[INFO] Loaded feature scaler")
            except Exception as e:
                print(f"Could not loadscaler:{e}")
        if loaded_any:
            print(f"[INFO]Successfully loaded{len(self.models)}models")
        return loaded_any
    # ----------------------------------------------------------------------
    # 🔥 NEW METHOD - train_models_from_scratch
    # ----------------------------------------------------------------------
    def train_models_from_scratch(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        save_models: bool = True
        ) -> Dict[str, Any]:
        """
        Train all models from scratch using provided training data.
        
        Args:
            X: Feature DataFrame (n_samples x 37 features)
            y: Target variable (quality_score)
            save_models: Whether to save trained models to disk
            
        Returns:
            Dictionary containing trained models and training metrics
        """
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.svm import SVR
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
        except ImportError as e:
            raise ImportError(f"Required ML libraries not installed: {e}")
        
        # Try to import XGBoost (optional)
        try:
            import xgboost as xgb
            has_xgb = True
        except ImportError:
            print("[WARN] XGBoost not installed, skipping XGBoost model")
            has_xgb = False
        
        print(f"[INFO] Training models with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        training_results = {}
        
        # 1. Random Forest
        print("[INFO] Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_val)
        self.models["rf_model"] = rf_model
        training_results["rf_model"] = {
            "rmse": np.sqrt(mean_squared_error(y_val, rf_pred)),
            "r2": r2_score(y_val, rf_pred)
        }
        
        # 2. Gradient Boosting
        print("[INFO] Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_val)
        self.models["gb_model"] = gb_model
        training_results["gb_model"] = {
            "rmse": np.sqrt(mean_squared_error(y_val, gb_pred)),
            "r2": r2_score(y_val, gb_pred)
        }
        
        # 3. XGBoost (if available)
        if has_xgb:
            print("[INFO] Training XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_val)
            self.models["xgb_model"] = xgb_model
            training_results["xgb_model"] = {
                "rmse": np.sqrt(mean_squared_error(y_val, xgb_pred)),
                "r2": r2_score(y_val, xgb_pred)
            }
        
        # 4. SVR
        print("[INFO] Training Support Vector Regressor...")
        svr_model = SVR(kernel='rbf', C=10, gamma='scale')
        svr_model.fit(X_train, y_train)
        svr_pred = svr_model.predict(X_val)
        self.models["svr_model"] = svr_model
        training_results["svr_model"] = {
            "rmse": np.sqrt(mean_squared_error(y_val, svr_pred)),
            "r2": r2_score(y_val, svr_pred)
        }
        
        # 5. Linear Regression (baseline)
        print("[INFO] Training Linear Regression...")
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        linear_pred = linear_model.predict(X_val)
        self.models["linear_model"] = linear_model
        training_results["linear_model"] = {
            "rmse": np.sqrt(mean_squared_error(y_val, linear_pred)),
            "r2": r2_score(y_val, linear_pred)
        }
        
        # Save models if requested
        if save_models:
            self._save_models()
        
        print("[INFO] ✅ Training complete!")
        print("\n=== Model Performance ===")
        for model_name, metrics in training_results.items():
            print(f"{model_name:20} RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}")
        
        return training_results

    # ----------------------------------------------------------------------
    # 🔥 NEW HELPER METHOD - Save models to disk
    # ----------------------------------------------------------------------
    def _save_models(self) -> None:
        """Save all trained models to disk"""
        os.makedirs(self.model_dir, exist_ok=True)
        
        for model_name, model_obj in self.models.items():
            filepath = os.path.join(self.model_dir, f"{model_name}.pkl")
            try:
                with open(filepath, "wb") as f:
                    pickle.dump(model_obj, f)
                print(f"[INFO] Saved {model_name} to {filepath}")
            except Exception as e:
                print(f"[ERROR] Could not save {model_name}: {e}")

    # ----------------------------------------------------------------------
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Predict quality score based on 37 input features.
        Returns:
        - individual model predictions
        - ensemble_average
        - confidence
        - uncertainty (std)
        """
        # Validate features
        self._validate_features(features)

        # Convert to DataFrame (single row)
        df = pd.DataFrame([features], columns=list(features.keys()))
        df=self._scale_features(df)
        # Run predictions for each model
        preds = {}
        ensemble_predictions=[]
        for model_name, model_obj in self.models.items():
            try:
                pred = model_obj.predict(df)[0]
                # if model_name == "LinearRegression":
                #     pred = min(pred,70)
                #     pred = max(pred,50)
                preds[model_name] = float(pred)
                # print("Models expect ", list(list(features.keys())))
                # print("You provided", list(df.columns))
            except Exception as e:
                preds[model_name] = None
                print(f"[ERROR] Error predicting with model {model_name}: {e}")

        if len(ensemble_predictions)==0:
            return{
                "error":"No valid model predictions available",
                "ensemble_average":0,
                "confidence":0,
                "uncertainity":0
            }

       
        ensemble_avg = float(np.mean(ensemble_predictions))
        uncertainty = float(np.std(ensemble_predictions))

        # Confidence = 1 - (std/max_range)
        confidence = max(0,min(100,100-(uncertainty*2)))

        preds["ensemble_average"] = ensemble_avg
        preds["confidence"] = round(confidence, 3)
        preds["uncertainty"] = round(uncertainty, 3)

        return preds
    def _scale_features(self,df:pd.DataFrame)->pd.DataFrame:
        if hasattr(self,'scaler') and self.scaler is not None:
            return pd.DataFrame(
                self.scaler.transform(df),
                columns=df.columns
            )
        return df
    # ----------------------------------------------------------------------
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[Dict]:
        """
        Predict multiple samples.
        """
        results = []
        for f in features_list:
            results.append(self.predict(f))
        return results

    # ----------------------------------------------------------------------
    def print_feature_guide(self) -> None:
        """
        Prints recommended ranges for all features.
        """
        print("\n=== FEATURE GUIDE (Expected Ranges) ===\n")
        for k, v in self.feature_ranges.items():
            print(f"{k:<30} min={v[0]}, max={v[1]}")

        print("\nExamples:")
        print("- Good readability: 10–40")
        print("- Strong security_mentions: 5+")
        print("- Good completeness_score: 40–80")
        print("- Too many duplicate_headers: >3")

    # ----------------------------------------------------------------------
    def get_feature_ranges(self) -> Dict[str, tuple]:
        """
        Returns expected min/max for each of the 37 features.
        🔥 UPDATED - Added 2 new features: consistency_score, structure_quality
        """
        return {
            "word_count": (300, 5000),
            "sentence_count": (10, 300),
            "avg_sentence_length": (5, 50),
            "header_count": (3, 40),
            "code_block_count": (0, 20),
            "table_count": (0, 10),
            "list_count": (0, 30),
            "diagram_count": (0, 10),

            "completeness_score": (0, 100),
            "security_mentions": (0, 20),
            "scalability_mentions": (0, 20),
            "api_mentions": (0, 30),
            "database_mentions": (0, 30),
            "performance_mentions": (0, 20),
            "monitoring_mentions": (0, 20),

            "duplicate_headers": (0, 10),
            "header_coverage": (0, 1),
            "code_coverage": (0, 1),
            "keyword_density": (0, 0.05),
            "section_density": (0, 1),

            "has_architecture_section": (0, 1),
            "has_security_section": (0, 1),
            "has_scalability_section": (0, 1),
            "has_deployment_section": (0, 1),
            "has_monitoring_section": (0, 1),
            "has_api_spec": (0, 1),
            "has_data_model": (0, 1),

            "service_count": (0, 25),
            "entity_count": (0, 40),
            "api_endpoint_count": (0, 100),

            "readability": (10, 50),
            "documentation_quality": (20, 100),
            "technical_depth": (10, 100),
            "formatting_quality": (20, 100),
            "examples_count": (0, 20),
            "consistency_score": (30, 100),    # 🔥 NEW
            "structure_quality": (40, 100)     # 🔥 NEW
        }

    # ----------------------------------------------------------------------
    def _validate_features(self, features: Dict[str, float]) -> None:
        """
        Validate:
        - All 37 features present
        - Values inside expected range
        """
        expected = set(self.feature_ranges.keys())

        missing = expected - set(features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Validate ranges
        for key, val in features.items():
            min_v, max_v = self.feature_ranges[key]
            if not (min_v <= float(val) <= max_v):
                print(f"[WARN] Feature {key}={val} outside expected range {min_v}-{max_v}")

    # ----------------------------------------------------------------------
    # Confidence & Uncertainty helpers
    # ----------------------------------------------------------------------
    def compute_confidence(self, predictions: List[float]) -> float:
        """
        Confidence = 1 - normalized std deviation.
        """
        std = np.std(predictions)
        confidence = max(0, 1 - std / 50)
        return confidence

    def compute_uncertainty(self, predictions: List[float]) -> float:
        return float(np.std(predictions))
