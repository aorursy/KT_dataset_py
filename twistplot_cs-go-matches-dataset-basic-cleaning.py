import pandas as pd
import numpy as np
df = pd.read_csv('../input/reordered_hltv_db.csv', low_memory=False)

df.info()
df.shape
df.head()
df.columns.values
# changing 'map' to 'maps'
df.columns = ['match_id', 'match_link', 'match_date', 'match_time',
       'home_team_country', 'away_team_country', 'home_team_name',
       'away_team_name', 'home_team_link', 'away_team_link', 'match_notes',
       'match_demo_link', 'tournament_id', 'tournament_link', 'vote_ban',
       'maps', 'home_team_id', 'away_team_id', 'home_team_side',
       'away_team_side', 'home_first_score', 'away_first_score',
       'home_second_score', 'away_second_score', 'home_team_score',
       'away_team_score', 'stats_link', 'home_first_kills',
       'away_first_kills', 'home_clutches_won', 'away_clutches_won',
       'ht_p1_id', 'ht_p1_name', 'ht_p1_link', 'ht_p1_kills',
       'ht_p1_headshots', 'ht_p1_asists', 'ht_p1_deaths', 'ht_p1_kast',
       'ht_p1_adr', 'ht_p1_fk_diff', 'ht_p1_rating20', 'ht_p2_id',
       'ht_p2_name', 'ht_p2_link', 'ht_p2_kills', 'ht_p2_headshots',
       'ht_p2_asists', 'ht_p2_deaths', 'ht_p2_kast', 'ht_p2_adr',
       'ht_p2_fk_diff', 'ht_p2_rating20', 'ht_p3_id', 'ht_p3_name',
       'ht_p3_link', 'ht_p3_kills', 'ht_p3_headshots', 'ht_p3_asists',
       'ht_p3_deaths', 'ht_p3_kast', 'ht_p3_adr', 'ht_p3_fk_diff',
       'ht_p3_rating20', 'ht_p4_id', 'ht_p4_name', 'ht_p4_link',
       'ht_p4_kills', 'ht_p4_headshots', 'ht_p4_asists', 'ht_p4_deaths',
       'ht_p4_kast', 'ht_p4_adr', 'ht_p4_fk_diff', 'ht_p4_rating20',
       'ht_p5_id', 'ht_p5_name', 'ht_p5_link', 'ht_p5_kills',
       'ht_p5_headshots', 'ht_p5_asists', 'ht_p5_deaths', 'ht_p5_kast',
       'ht_p5_adr', 'ht_p5_fk_diff', 'ht_p5_rating20', 'at_p1_id',
       'at_p1_name', 'at_p1_link', 'at_p1_kills', 'at_p1_headshots',
       'at_p1_asists', 'at_p1_deaths', 'at_p1_kast', 'at_p1_adr',
       'at_p1_fk_diff', 'at_p1_rating20', 'at_p2_id', 'at_p2_name',
       'at_p2_link', 'at_p2_kills', 'at_p2_headshots', 'at_p2_asists',
       'at_p2_deaths', 'at_p2_kast', 'at_p2_adr', 'at_p2_fk_diff',
       'at_p2_rating20', 'at_p3_id', 'at_p3_name', 'at_p3_link',
       'at_p3_kills', 'at_p3_headshots', 'at_p3_asists', 'at_p3_deaths',
       'at_p3_kast', 'at_p3_adr', 'at_p3_fk_diff', 'at_p3_rating20',
       'at_p4_id', 'at_p4_name', 'at_p4_link', 'at_p4_kills',
       'at_p4_headshots', 'at_p4_asists', 'at_p4_deaths', 'at_p4_kast',
       'at_p4_adr', 'at_p4_fk_diff', 'at_p4_rating20', 'at_p5_id',
       'at_p5_name', 'at_p5_link', 'at_p5_kills', 'at_p5_headshots',
       'at_p5_asists', 'at_p5_deaths', 'at_p5_kast', 'at_p5_adr',
       'at_p5_fk_diff', 'at_p5_rating20']
df['maps']

df.maps.unique()
df['maps'] = df.maps.str.replace("Nke", "Nuke")
df['maps'] = df.maps.str.replace("Dst2", "Dust2")
df['maps'] = df.maps.str.replace("\\n", "")
df['maps'] = df.maps.str.replace(" ", "")
df.maps.unique()
df['fist_half_home_side'] = np.where(df['home_team_side'] == 'ct-color', 0, 1 )
df = df.drop('home_team_side', axis=1)
df = df.drop('away_team_side', axis=1)
df.shape
df.match_notes.unique()
df['best_of'] = df.match_notes.str[8]
df.best_of.unique()
df.head()
df.match_link = "http://www.hltv.org" + df.match_link
df.home_team_link = "http://www.hltv.org" + df.home_team_link
df.away_team_link = "http://www.hltv.org" + df.away_team_link
df.head()
df.ht_p1_headshots.head()
type(df.match_date[2])
df.match_date = pd.to_datetime(df.match_date)
df['year'] = df.match_date.dt.year
df['month'] = df.match_date.dt.month
df['day'] = df.match_date.dt.day
df = df.drop('match_date', axis=1)
# df.to_csv('cs_go_matches_cleaned.csv', index=False)