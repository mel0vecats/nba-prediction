import pandas as pd
import streamlit as st

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

query = """
SELECT
    p."GAME_DATE" as "DATE",
    p."Game_ID",
    t1.abbreviation AS "TEAM",
    t2.abbreviation AS "OPP",
    p."PTS",
    p."Predicted_PTS" as "PREDICTED"
FROM predictions p
LEFT JOIN nba_teams t1
    ON p."Team_ID" = t1.id
LEFT JOIN nba_teams t2
    ON p."OPP_Team_ID" = t2.id
"""

df = pd.read_sql(query, engine)

print(df.columns)

df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
df["PREDICTED"] = df["PREDICTED"].round(0).astype(int)

df = df.sort_values(by=["DATE", "Game_ID"], ascending=[False, True])

# Split once globally
next_games_all = df[df["PTS"].isna()]
completed_games_all = df[df["PTS"].notna()]

# Drop Game_ID for display
next_games_display = next_games_all.drop(columns=["Game_ID", "PTS"])
completed_games_display = completed_games_all.drop(columns=["Game_ID"])

# Show all completed games
st.subheader("NEXT GAMES")
st.dataframe(next_games_display, use_container_width=True)

st.subheader("FINAL RESULTS")
st.dataframe(completed_games_display, use_container_width=True)