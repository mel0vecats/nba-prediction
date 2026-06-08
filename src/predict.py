import pandas as pd
import yaml
import joblib

from utils import load_config

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

# =========================
# LOAD MODEL
# =========================
def load_model(path):
    return joblib.load(path)

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    config = load_config()

    model = load_model("models/best_model.pkl")

    load_dotenv()
    engine = create_engine(os.getenv("SUPABASE_DB_URL"))

    df = pd.read_sql(f"SELECT * FROM features_table",engine)

    feature_cols = [
        col for col in df.columns
        if col not in ["PTS", "Game_ID", "GAME_DATE", "Team_ID", "OPP_Team_ID", "OPP_PTS", "is_new"]
    ]

    df["Predicted_PTS"] = model.predict(df[feature_cols])

    preds = df[["GAME_DATE", "Game_ID", "Team_ID", "OPP_Team_ID", "PTS", "OPP_PTS", "Predicted_PTS"]]
    
    preds.to_sql(
        name="predictions",
        con=engine,
        if_exists="replace",
        index=False
    )

    print("All predictions computed and stored in Supabase.")