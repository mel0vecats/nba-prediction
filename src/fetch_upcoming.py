import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2int
from utils import load_config
from datetime import datetime, timedelta

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

def fetch_upcoming_games(season, playoff_rounds):
    schedule = scheduleleaguev2int.ScheduleLeagueV2Int(season=season)
    games_df = schedule.get_data_frames()[0]

    games_df["gameDate"] = pd.to_datetime(games_df["gameDate"])

    today = pd.Timestamp(datetime.now().date())

    filtered = pd.DataFrame()
    days_ahead = 0
    max_days_ahead = 10

    while filtered.empty and days_ahead <= max_days_ahead:
        target_date = today + timedelta(days=days_ahead)

        filtered = games_df[games_df["gameDate"].dt.date == target_date.date()].copy()

        # Remove unwanted games
        filtered = filtered[~filtered["gameLabel"].str.contains("star|preseason", case=False, na=False)]

        if filtered.empty:
            print(f"No games found on {target_date.date()}. Trying next day...")
            days_ahead += 1
        else:
            print(f"Found {len(filtered)} game(s) on {target_date.date()}")

    filtered = filtered[
        [
            "seasonYear",
            "gameDate",
            "gameId",
            "homeTeam_teamId",
            "awayTeam_teamId",
            "gameLabel"
        ]
    ]

    filtered["is_playoffs"] = filtered["gameLabel"].apply(
        lambda x: int(any(r in str(x) for r in playoff_rounds))
    )

    # =========================
    # HOME TEAM ROWS
    # =========================
    home_df = filtered.copy()
    home_df["Team_ID"] = home_df["homeTeam_teamId"]
    home_df["OPP_Team_ID"] = home_df["awayTeam_teamId"]
    home_df["is_home"] = 1

    # =========================
    # AWAY TEAM ROWS
    # =========================
    away_df = filtered.copy()
    away_df["Team_ID"] = away_df["awayTeam_teamId"]
    away_df["OPP_Team_ID"] = away_df["homeTeam_teamId"]
    away_df["is_home"] = 0

    cols = [
        "seasonYear",
        "gameDate",
        "gameId",
        "Team_ID",
        "OPP_Team_ID",
        "is_playoffs",
        "is_home"
    ]

    team_games_df = pd.concat(
        [home_df[cols], away_df[cols]],
        ignore_index=True
    )

    team_games_df = team_games_df.rename(columns={
        "seasonYear": "SEASON",
        "gameDate": "GAME_DATE",
        "gameId": "Game_ID"
    })

    return team_games_df


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    config = load_config()

    load_dotenv()
    engine = create_engine(os.getenv("SUPABASE_DB_URL"))

    season = config["prediction"]["season"]
    playoff_rounds = config["prediction"]["playoff_rounds"]

    upcoming_games = fetch_upcoming_games(
        season,
        playoff_rounds
    )

    upcoming_games.to_sql(
        name="upcoming_table",
        con=engine,
        if_exists="replace",
        index=False
    )

    print(f"Saved {len(upcoming_games)//2} upcoming games to Supabase table 'upcoming_table'")