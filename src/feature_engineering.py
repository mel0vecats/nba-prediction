import pandas as pd
import numpy as np
from utils import load_config

import os
from sqlalchemy import create_engine
from dotenv import load_dotenv


# =========================
# BASIC STATS & POSSESSIONS
# =========================
def compute_basic_stats(df):
    # Total points
    df['TOTAL_PTS'] = df['PTS'] + df['OPP_PTS']

    # Possessions
    df['POSS'] = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']
    df['OPP_POSS'] = df['OPP_FGA'] + 0.44 * df['OPP_FTA'] - df['OPP_OREB'] + df['OPP_TOV']

    # Pace
    df['PACE'] = df['POSS']
    df['OPP_PACE'] = df['OPP_POSS']

    # Offensive/Defensive ratings
    df['ORTG'] = df['PTS'] / df['POSS'] * 100
    df['DRTG'] = df['OPP_PTS'] / df['POSS'] * 100
    df['OPP_ORTG'] = df['OPP_PTS'] / df['OPP_POSS'] * 100
    df['OPP_DRTG'] = df['PTS'] / df['OPP_POSS'] * 100

    # Shooting efficiency
    df['eFG'] = (df['FGM'] + 0.5 * df['FG3M']) / df['FGA']
    df['OPP_eFG'] = (df['OPP_FGM'] + 0.5 * df['OPP_FG3M']) / df['OPP_FGA']
    df['TS'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    df['OPP_TS'] = df['OPP_PTS'] / (2 * (df['OPP_FGA'] + 0.44 * df['OPP_FTA']))

    # Four Factors
    df['TOV_RATE'] = df['TOV'] / df['POSS']
    df['OPP_TOV_RATE'] = df['OPP_TOV'] / df['OPP_POSS']
    df['OREB_PCT'] = df['OREB'] / (df['OREB'] + df['OPP_DREB'])
    df['OPP_OREB_PCT'] = df['OPP_OREB'] / (df['OPP_OREB'] + df['DREB'])
    df['FTR'] = df['FTA'] / df['FGA']
    df['OPP_FTR'] = df['OPP_FTA'] / df['OPP_FGA']

    return df


# =========================
# ROLLING & SEASON STATS
# =========================
def compute_rolling_stats(df, rolling_windows=[2,5,10]):
    rolling_stats = ['ORTG','DRTG','PACE','eFG','TS','TOV_RATE','OREB_PCT','FTR']

    for stat in rolling_stats:
        for window in rolling_windows:
            # Team rolling mean & std
            df[f'{stat}_last{window}'] = df.groupby(['Team_ID','SEASON'])[stat].transform(
                lambda x: x.shift().rolling(window, min_periods=1).mean()
            )
            df[f'{stat}_std{window}'] = df.groupby(['Team_ID','SEASON'])[stat].transform(
                lambda x: x.shift().rolling(window, min_periods=1).std()
            )

            # Opponent rolling mean & std
            df[f'OPP_{stat}_last{window}'] = df.groupby(['OPP_Team_ID','SEASON'])[f'OPP_{stat}'].transform(
                lambda x: x.shift().rolling(window, min_periods=1).mean()
            )
            df[f'OPP_{stat}_std{window}'] = df.groupby(['OPP_Team_ID','SEASON'])[f'OPP_{stat}'].transform(
                lambda x: x.shift().rolling(window, min_periods=1).std()
            )

        # Season averages
        df[f'{stat}_season'] = df.groupby(['Team_ID','SEASON'])[stat].transform(
            lambda x: x.shift().expanding().mean()
        )
        df[f'OPP_{stat}_season'] = df.groupby(['OPP_Team_ID','SEASON'])[f'OPP_{stat}'].transform(
            lambda x: x.shift().expanding().mean()
        )

    return df


# =========================
# OFFENSIVE ADJUSTED STATS
# =========================
def compute_adjusted_stats(df):
    off_stats = ['ORTG','eFG','TS','TOV_RATE','OREB_PCT','FTR']
    for stat in off_stats:
        df[f'{stat}_ADJ'] = df[stat] / df['OPP_DRTG_season']
        df[f'OPP_{stat}_ADJ'] = df[f'OPP_{stat}'] / df['DRTG_season']

    return df


# =========================
# REST & WIN FEATURES
# =========================
def compute_rest_win_features(df, rolling_windows=[2,5,10]):
    # Rest days
    df['DAYS_REST'] = df.groupby(['Team_ID','SEASON'])['GAME_DATE'].diff().dt.days.fillna(4)
    df['OPP_DAYS_REST'] = df.groupby(['OPP_Team_ID','SEASON'])['GAME_DATE'].diff().dt.days.fillna(4)
    df['REST_DIFF'] = df['DAYS_REST'] - df['OPP_DAYS_REST']

    # Back-to-back
    df['B2B'] = (df['DAYS_REST']==1).astype(int)
    df['OPP_B2B'] = (df['OPP_DAYS_REST']==1).astype(int)

    # Rolling win %
    for window in rolling_windows:
        df[f'WIN_PCT_last{window}'] = df.groupby(['Team_ID','SEASON'])['is_win'].transform(
            lambda x: x.shift().rolling(window, min_periods=1).mean()
        )
        df[f'OPP_WIN_PCT_last{window}'] = df.groupby(['OPP_Team_ID','SEASON'])['OPP_is_win'].transform(
            lambda x: x.shift().rolling(window, min_periods=1).mean()
        )

    # Win streak
    def calc_streak(series):
        streak,count = [],0
        for win in series:
            streak.append(count)
            count = count+1 if win==1 else 0
        return streak

    df['WIN_STREAK'] = df.groupby(['Team_ID','SEASON'])['is_win'].transform(calc_streak)
    df['OPP_WIN_STREAK'] = df.groupby(['OPP_Team_ID','SEASON'])['OPP_is_win'].transform(calc_streak)

    return df


# =========================
# MATCHUP HISTORY
# =========================
def compute_matchup_features(df, rolling_stats=['ORTG','DRTG','PACE','eFG','TS','TOV_RATE','OREB_PCT','FTR']):
    for stat in rolling_stats:
        df[f'{stat}_MATCHUP'] = df.groupby(['Team_ID','OPP_Team_ID','SEASON'])[stat].transform(
            lambda x: x.shift().expanding().mean()
        ).fillna(0)
    return df


# =========================
# COMPOSITE FEATURES
# =========================
def compute_adjusted_stats(df, rolling_windows=[2,5,10], off_stats=['ORTG','eFG','TS','TOV_RATE','OREB_PCT','FTR']):
    for off_stat in off_stats:
        df[f'{off_stat}_ADJ'] = (
            df[f'{off_stat}']
            / df['OPP_DRTG_season']
        )
        
        df[f'OPP_{off_stat}_ADJ'] = (
            df[f'OPP_{off_stat}']
            / df['DRTG_season']
        )

    # Adjusted rolling features
    for stat in off_stats:
        for window in rolling_windows:
            df[f'{stat}_ADJ_last{window}'] = df.groupby(['Team_ID','SEASON'])[f'{stat}_ADJ'].transform(
                lambda x: x.shift().rolling(window, min_periods=1).mean()
            )
            df[f'OPP_{stat}_ADJ_last{window}'] = df.groupby(['OPP_Team_ID','SEASON'])[f'OPP_{stat}_ADJ'].transform(
                lambda x: x.shift().rolling(window, min_periods=1).mean()
            )
    return df


def merge_and_build_dataset(hist_df, upc_df):

    upc_df['is_new'] = 1
    hist_df['is_new'] = 0

    df = pd.concat([upc_df, hist_df], ignore_index=True, sort=False)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    df = compute_basic_stats(df)
    df = compute_rolling_stats(df)
    df = compute_rest_win_features(df)
    df = compute_matchup_features(df)
    df = compute_adjusted_stats(df)

    # composite features
    df["EXPECTED_PACE"] = (df["PACE_last5"] + df["OPP_PACE_last5"]) / 2

    df["NET_RTG"] = df["ORTG_last5"] - df["DRTG_last5"]
    df["OPP_NET_RTG"] = df["OPP_ORTG_last5"] - df["OPP_DRTG_last5"]

    return df

# =========================
# FINAL FEATURES
# =========================
def select_features(df, feature_cols, target="PTS"):
    cols = feature_cols + [target, "Game_ID", "GAME_DATE", "Team_ID", "OPP_Team_ID", "is_new"]
    model_df = df[cols].replace([np.inf, -np.inf], np.nan)

    model_df = model_df[
        (model_df["is_new"] == 1) |
        ((model_df["is_new"] == 0) & model_df.notna().all(axis=1))
    ]

    return model_df

def generate_feature_cols():
    rolling_stats = ['ORTG','DRTG','PACE','eFG','TS','TOV_RATE','OREB_PCT','FTR']
    rolling_windows = [2,5,10]
    off_stats = ['ORTG','eFG','TS','TOV_RATE','OREB_PCT','FTR']

    feature_cols = []

    # =========================
    # ROLLING + STD + SEASON
    # =========================
    for stat in rolling_stats:
        for w in rolling_windows:
            feature_cols += [
                f'{stat}_last{w}',
                f'{stat}_std{w}',
                f'OPP_{stat}_last{w}',
                f'OPP_{stat}_std{w}'
            ]

        feature_cols += [
            f'{stat}_season',
            f'OPP_{stat}_season'
        ]

    # =========================
    # ADJUSTED FEATURES
    # =========================
    for stat in off_stats:
        for w in rolling_windows:
            feature_cols += [
                f'{stat}_ADJ_last{w}',
                f'OPP_{stat}_ADJ_last{w}'
            ]

    # =========================
    # COMPOSITES
    # =========================
    feature_cols += [
        'EXPECTED_PACE',
        'NET_RTG',
        'OPP_NET_RTG'
    ]

    # =========================
    # REST FEATURES
    # =========================
    feature_cols += [
        'DAYS_REST',
        'OPP_DAYS_REST',
        'REST_DIFF',
        'B2B',
        'OPP_B2B'
    ]

    # =========================
    # WIN FEATURES
    # =========================
    feature_cols += [
        'WIN_PCT_last2', 'OPP_WIN_PCT_last2',
        'WIN_PCT_last5', 'OPP_WIN_PCT_last5',
        'WIN_PCT_last10', 'OPP_WIN_PCT_last10',
        'WIN_STREAK', 'OPP_WIN_STREAK'
    ]

    # =========================
    # MATCHUP FEATURES
    # =========================
    feature_cols += [
        'ORTG_MATCHUP',
        'DRTG_MATCHUP',
        'PACE_MATCHUP',
        'eFG_MATCHUP',
        'TS_MATCHUP',
        'TOV_RATE_MATCHUP',
        'OREB_PCT_MATCHUP',
        'FTR_MATCHUP'
    ]

    # =========================
    # CONTEXT FEATURES
    # =========================
    feature_cols += [
        'is_home',
        'is_playoffs'
    ]

    return feature_cols

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    config = load_config()

    load_dotenv()
    engine = create_engine(os.getenv("SUPABASE_DB_URL"))

    # LOAD FROM Supabase
    hist_df = pd.read_sql(f"SELECT * FROM historical_table",engine)
    upc_df = pd.read_sql(f"SELECT * FROM upcoming_table",engine)

    # BUILD DATASET
    df = merge_and_build_dataset(hist_df, upc_df)

    # FEATURE LIST (keep your existing function if unchanged)
    feature_cols = generate_feature_cols()

    features = select_features(df, feature_cols)

    # SAVE BACK TO Supabase
    features.to_sql(
        name="features_table",
        con=engine,
        if_exists="replace",
        index=False
    )

    print("Saved features to Supabase table: features_table")