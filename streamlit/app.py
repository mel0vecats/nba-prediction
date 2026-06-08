import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

# --- Load environment variables ---
load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

# --- SQL query ---
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

# --- Load data ---
df = pd.read_sql(query, engine)

# --- Process data ---
df["DATE"] = pd.to_datetime(df["DATE"])
df["PREDICTED"] = df["PREDICTED"].round(0).astype('Int64')

df = df.sort_values(by=["DATE", "Game_ID"], ascending=[False, True])

# --- Split games ---
next_games_all = df[df["PTS"].isna()]
completed_games_all = df[df["PTS"].notna()]

completed_games_all["DIFFERENCE"] = (
    completed_games_all["PTS"] - completed_games_all["PREDICTED"]
).abs()

# Month filter (MULTI SELECT)
completed_games_all["MONTH"] = completed_games_all["DATE"].dt.to_period("M").astype(str)
months = sorted(completed_games_all["MONTH"].dropna().unique())

selected_months = st.sidebar.multiselect(
    "Select Months",
    options=months
)

# Team filter (MULTI SELECT)
teams = sorted(completed_games_all["TEAM"].dropna().unique())

selected_team = st.sidebar.multiselect(
    "Select Teams",
    options=teams
)

# --- Apply filters ---
filtered_completed = completed_games_all.copy()

# Month filter (empty = all)
if selected_months:
    filtered_completed = filtered_completed[
        filtered_completed["MONTH"].isin(selected_months)
    ]

# Team filter (empty = all)
if selected_team:
    filtered_completed = filtered_completed[
        filtered_completed["TEAM"].isin(selected_team)
    ]

# --- Fix DATE display ---
filtered_completed["DATE"] = filtered_completed["DATE"].dt.date
next_games_all["DATE"] = next_games_all["DATE"].dt.date



# --- Display ---
next_games_display = next_games_all.drop(columns=["Game_ID", "PTS"])
completed_games_display = filtered_completed.drop(columns=["Game_ID", "MONTH"])

completed_games_display = completed_games_display.reset_index(drop=True)
next_games_display = next_games_display.reset_index(drop=True)

st.subheader("NEXT GAMES")
st.dataframe(next_games_display, use_container_width=True)

st.subheader("FINAL RESULTS")
st.dataframe(completed_games_display, use_container_width=True)

# --- MAE ---
mae = filtered_completed["DIFFERENCE"].mean()

st.subheader("MODEL PERFORMANCE")
st.metric(
    label="MAE (Mean Absolute Error)",
    value=round(mae, 2) if not pd.isna(mae) else None
)