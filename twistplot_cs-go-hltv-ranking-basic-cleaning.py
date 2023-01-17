import pandas as pd
df_rank = pd.read_csv('../input//all_ranks_hltv.csv')
df = df_rank
df.head()
def take_date(rank_link, date_part):
    day = rank_link.split("/")
    day = day[date_part]
    return day
day_split = 7
month_split = 6
year_split = 5
df['day'] = df.ranking_day.apply(lambda x: take_date(x, day_split))
df['month'] = df.ranking_day.apply(lambda x: take_date(x, month_split))
df['year'] = df.ranking_day.apply(lambda x: take_date(x, year_split))
df.month.unique()
def to_int(name):
    if name == "january": return 1
    elif name == "february": return 2
    elif name == "march": return 3
    elif name == "april": return 4
    elif name == "may": return 5
    elif name == "june": return 6
    elif name == "july": return 7
    elif name == "august": return 8
    elif name == "september": return 9
    elif name == "october": return 10
    elif name == "november": return 11
    elif name == "december": return 12
    else: raise ValueError
df['month'] = df.month.apply(lambda x: to_int(x))
df.head()
rank = df.ranking[0]
rank
rank = rank.replace("'", "").replace("[", "").split(']]')
rank
# df.to_csv('cs_go_hltv_ranking_clean.csv')