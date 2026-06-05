import pandas as pd
from nba_api.stats.static import teams

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.getenv("SUPABASE_DB_URL"))

nba_teams = teams.get_teams()
teams_df = pd.DataFrame(nba_teams)
teams_df.to_sql("nba_teams", engine, if_exists="replace", index=False)

print(f"Saved {len(teams_df)} teams to Supabase table 'nba_teams'")