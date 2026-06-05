import pandas as pd
import time
from nba_api.stats.endpoints import teamgamelog
from utils import load_config

import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

import pandas as pd
import time
from nba_api.stats.endpoints import teamgamelog

def fetch_historical_games(nba_teams, seasons, season_types):
    all_games = []
    
    for season in seasons:
        print(f"\n===== Processing season {season} =====")
        
        for _, team in nba_teams.iterrows():
            team_id = team["id"]
            team_name = team["full_name"]

            for season_type in season_types:
                try:
                    gamelog = teamgamelog.TeamGameLog(
                        team_id=team_id,
                        season=season,
                        season_type_all_star=season_type
                    )

                    df = gamelog.get_data_frames()[0]
                    df["SEASON"] = season
                    df["is_playoffs"] = 1 if season_type.lower() == "playoffs" else 0

                    all_games.append(df)
                    print(f"Appended {team_name} {season_type}")
                    time.sleep(1)  # Avoid rate limiting

                except Exception as e:
                    print(f"Error with {team_name} ({season}): {e}")

    if all_games:  # Make sure there is at least one DataFrame
        league_games = pd.concat(all_games, ignore_index=True)
    else:
        league_games = pd.DataFrame()
    
    return league_games

def transform(df):
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
    df["is_win"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)
    df["is_home"] = df["MATCHUP"].apply(lambda x: 1 if " vs. " in x else 0)
    df = df.sort_values(by=["GAME_DATE", "Game_ID"])
    return df


def build_opponent_columns(df):
    df_opp = df.add_prefix("OPP_")
    df_merged = df.merge(df_opp, left_on="Game_ID", right_on="OPP_Game_ID")

    df_merged = df_merged[df_merged["Team_ID"] != df_merged["OPP_Team_ID"]]

    drop_cols = [
        "OPP_SEASON",
        "OPP_Game_ID",
        "OPP_GAME_DATE",
        "OPP_MATCHUP",
        "OPP_is_home",
        "OPP_is_playoffs"
    ]
    df_merged = df_merged.drop(columns=drop_cols)

    return df_merged



if __name__ == "__main__":
    # Load config
    config = load_config()

    load_dotenv()
    engine = create_engine(os.getenv("SUPABASE_DB_URL"))

    try:
        nba_teams_df = pd.read_sql("SELECT * FROM nba_teams", engine)
    except Exception as e:
        raise Exception(
            "nba_teams table not found in Supabase. "
            "You must first load teams into the database using get_teams.py."
        )

    seasons = config["data"]["seasons"]
    season_types = config["data"]["season_types"]

    # Step 1: Download raw game logs
    games = fetch_historical_games(nba_teams_df,seasons,season_types)

    # Step 2: Clean basic columns
    games = transform(games)

    # Step 3: Merge opponent stats
    full_historical = build_opponent_columns(games)

    full_historical.to_sql(
        name="historical_table",
        con=engine,
        if_exists="replace",
        index=False
    )

    print(f"Saved {len(full_historical)//2} historical games to Supabase table 'historical_table'")