import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from utils import load_config

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

# =========================
# LOAD DATA
# =========================
def load_data(path):
    return pd.read_csv(path)


# =========================
# CREATE MODELS
# =========================
def create_models(config):
    return {
        "ridge": Ridge(**config["models"]["ridge"]),
        "rf": RandomForestRegressor(**config["models"]["random_forest"]),
        "xgb": XGBRegressor(**config["models"]["xgboost"]),
        "lgbm": LGBMRegressor(**config["models"]["lightgbm"]),
    }


# =========================
# TRAIN + EVALUATE
# =========================
def train_models(
    X_train,
    y_train,
    X_test,
    y_test,
    config
):
    models = create_models(config)

    results = {}

    print("\n===== MODEL EVALUATION =====")

    for name, model in models.items():

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {
            "model": model,
            "mae": mae,
            "r2": r2
        }

        print(
            f"{name.upper():<6}"
            f" MAE: {mae:.3f}"
            f" | R2: {r2:.3f}"
        )

    return results


# =========================
# SELECT BEST MODEL
# =========================
def select_best(results):

    best_name = min(results,key=lambda x: results[x]["mae"])

    print("\n===== BEST MODEL =====")
    print(f"{best_name.upper()} "f"(MAE={results[best_name]['mae']:.3f}, "f"R2={results[best_name]['r2']:.3f})")

    return best_name


# =========================
# RETRAIN ON FULL HISTORY
# =========================
def retrain_best_model(best_name,X_full,y_full,config):

    models = create_models(config)

    best_model = models[best_name]

    print(f"\nRetraining {best_name.upper()} "f"on full historical dataset...")

    best_model.fit(X_full, y_full)

    return best_model


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    config = load_config()

    load_dotenv()
    engine = create_engine(os.getenv("SUPABASE_DB_URL"))

    # ---------------------
    # Load Data
    # ---------------------
    df = pd.read_sql(f"SELECT * FROM features_table",engine)

    df = df.replace([np.inf, -np.inf],np.nan)
    df = df.dropna()

    # Historical data only
    train_df = df[df["is_new"] == 0].copy()

    train_df["GAME_DATE"] = pd.to_datetime(train_df["GAME_DATE"])
    train_df = train_df.sort_values("GAME_DATE")

    target = "PTS"
    feature_cols = [col for col in train_df.columns if col not in [target,"Game_ID","GAME_DATE","Team_ID","OPP_Team_ID", "OPP_PTS","is_new"]]

    # ---------------------
    # Time-series split
    # ---------------------
    split_idx = int(len(train_df) * 0.8)

    train_set = train_df.iloc[:split_idx]
    test_set = train_df.iloc[split_idx:]

    X_train = train_set[feature_cols]
    y_train = train_set[target]

    X_test = test_set[feature_cols]
    y_test = test_set[target]

    print(f"Train rows: {len(train_set)}")
    print(f"Test rows : {len(test_set)}")

    # ---------------------
    # Train & Evaluate
    # ---------------------
    results = train_models(X_train,y_train,X_test,y_test,config)

    # ---------------------
    # Select Best Model
    # ---------------------
    best_name = select_best(results)

    # ---------------------
    # Retrain on all history
    # ---------------------
    X_full = train_df[feature_cols]
    y_full = train_df[target]

    best_model = retrain_best_model(best_name,X_full,y_full,config)

    # ---------------------
    # Save Model
    # ---------------------
    model_path = "models/best_model.pkl"

    joblib.dump(best_model,model_path)

    print(f"\nModel saved to: {model_path}")