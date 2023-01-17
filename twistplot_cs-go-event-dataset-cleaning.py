import pandas as pd
import numpy as np
import re
LINK = '../input/events_db.csv'

# Getting year, month day
df = pd.read_csv(LINK)
df.event_date = pd.to_datetime(df.event_date, unit='ms')
df['year'] = df.event_date.dt.year
df['month'] = df.event_date.dt.month
df['day'] = df.event_date.dt.day
df = df.drop('event_date', axis=1)

# Parsing prize pool
def parse_prize_pool(prize_string):
    regex = r"\d{1,3}(,\d{3})?(,\d{3})"
    matches = re.finditer(regex, prize_string)
    prize = 0
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1
        prize = match.group()
        prize = prize.replace(",", "")
    return prize

def parse_prize_pool_second(prize_string):
    regex = r"\$\d{3,5}"
    matches = re.finditer(regex, prize_string)
    prize = 0
    for matchNum, match in enumerate(matches):
        matchNum = matchNum + 1
        prize = match.group()
        prize = prize.replace("$", "")
    return prize 

df['prize_money'] = df.prize_pool.apply(lambda x: parse_prize_pool(x))
df.prize_money = df.prize_money.astype(int)

for i in range(0, len(df.prize_money) - 1):
    if (df.prize_money[i] == 0):
        parsed_money = parse_prize_pool_second(df.prize_pool[i])
        df.loc[i, 'prize_money'] = parsed_money

# Fixing wrong data
df.prize_money = df.prize_money.astype(int)
df.year.unique()
df[df.year == 1999].event_link
df = df[df.year != 1999]
df.event_teams_count.unique()
df[df.event_teams_count == "TBA"].event_link
df.loc[111, 'event_teams_count'] = 12
df.loc[146, 'event_teams_count'] = 4
df.loc[155, 'event_teams_count'] = 8
df.loc[625, 'event_teams_count'] = 16
df.event_teams_count.unique()
df[df.event_teams_count == "-"].event_link
df.loc[709, 'event_teams_count'] = 4
df.event_teams_count.unique()
df[df.event_teams_count == "3+"].event_link
df.loc[2, 'event_teams_count'] = 4
df.loc[650, 'event_teams_count'] = 3

# fixing team cont on tournament
def team_count_from_placement(index_location, placement_list):
    len_list = placement_list.split("]")
    len_list = len(len_list)
    team_count = (len_list - 2)/6
    df.loc[index_location, 'event_teams_count'] = team_count

index_list = df[df.event_teams_count == "4+"].index

for element in index_list:
    team_count_from_placement(element, df.loc[element].teams_placement_list)

index_list = df[df.event_teams_count == "8+"].index

for element in index_list:
    team_count_from_placement(element, df.loc[element].teams_placement_list)

index_list = df[df.event_teams_count == "2+"].index

for element in index_list:
    team_count_from_placement(element, df.loc[element].teams_placement_list)

index_list = df[df.event_teams_count == "16+"].index

for element in index_list:
    team_count_from_placement(element, df.loc[element].teams_placement_list)

df[df.event_teams_count == "114"].event_link

element = 637
team_count_from_placement(element, df.loc[element].teams_placement_list)
df[df.teams_placement_list.isnull()].event_link
df.loc[99, 'event_teams_count'] = 7
df.event_teams_count = df.event_teams_count.astype('int')

# splitting country and city (if avalible)
def get_city(location):
    location = location.split(',')
    if len(location) > 1:
        city = location[0]
        coutry = location[1].replace("|", "").strip()
        return city
    else:
        city = "no_data"
        coutry = location[0].replace("|", "").strip()
        return city

def get_country(location):
    location = location.split(',')
    if len(location) > 1:
        city = location[0]
        country = location[1].replace("|", "").strip()
        return country
    else:
        city = "no_data"
        country = location[0].replace("|", "").strip()
        return country

df['country'] = df.location.apply(lambda x: get_country(x))
df['city'] = df.location.apply(lambda x: get_city(x))
df[df.city == "no_data"].shape
df.country.unique()
df[df.country == "United Kingdom (Online)"]
df = df.drop(index=740)
# df.to_csv('../input/cs_go_events_cleaned.csv', index=False) - cant write data like that on kaggle