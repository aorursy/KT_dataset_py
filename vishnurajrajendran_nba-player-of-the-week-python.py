import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import datetime
from time import gmtime, strftime

%matplotlib inline
nba_df = pd.read_csv("../input/NBA_player_of_the_week.csv")
abbr_to_num = {name: num for num, name in enumerate(calendar.month_abbr) if num}
east_teams = nba_df[nba_df['Conference'] == 'East']['Team'].unique()
west_teams = nba_df[nba_df['Conference'] == 'West']['Team'].unique()
def missing_conf(myteam,myConf):
    if myConf is not None :
        if ((myteam in east_teams) | (myteam == 'Washington Bullets')):
            return 'East'
        elif ((myteam in west_teams) | (myteam == 'Washington Bullets')):
            return 'West'
    else:
        return myConf
nba_df['Conference_2'] = nba_df.apply(lambda row : missing_conf(row['Team'],row['Conference']), axis = 1)
nba_df.head()
def makeHeightNum(x):
    if 'cm' in x:
        return int(x.replace('cm',''))
    else:
        foot = int(x.split('-')[0])
        inches = int(x.split('-')[1])
        return ((foot*12 + inches)*2.54)
def makeWeightNum(x):
    if 'kg' in x:
        return int(x.replace('kg',''))
    else:
        return (int(x)*0.453592)
nba_df['Height_cm'] = nba_df.apply(lambda row : makeHeightNum(row['Height']),axis = 1)
nba_df['Weight_kg'] = nba_df.apply(lambda row : makeWeightNum(row['Weight']),axis = 1)
nba_df.head()
nba_df['BMI'] = nba_df['Weight_kg']/((nba_df['Height_cm']/100)**2)
nba_df.head()
pos = ["PG","SG","F","C" ,"SF","PF","G","FC","GF","F-C","G-F"]
posNames = ["point guard","shooting guard","forward","center","small forward","power forward", "guard", "forward center","guard forward", "forward center", "guard forward"]
weekday = ["Mon", "Tue", "Wed", "Thu","Fri", "Sat", "Sun"]
def get_position(y):
    pos_index = pos.index(y)
    return posNames[pos_index]
def makeDate(x):
    current_date = x.split(',')[0]
    current_year = x.split(',')[1]
    current_day = current_date.split(' ')[1]
    current_month = abbr_to_num[current_date.split(' ')[0]]
    return (current_year.strip()+'-'+str(current_month)+'-'+current_day)
def getWeek(x):
    date_ymd = makeDate(x)
    week_num = datetime.date(int(date_ymd.split('-')[0]), int(date_ymd.split('-')[1]),int(date_ymd.split('-')[2])).strftime("%V")
    return week_num
def getWeekDay(x):
    date_ymd = makeDate(x)
    week_day = datetime.date(int(date_ymd.split('-')[0]), int(date_ymd.split('-')[1]),int(date_ymd.split('-')[2])).weekday()
    weekday_name = weekday[week_day]
    return weekday_name
nba_df['Position_name'] = nba_df.apply(lambda row : get_position(row['Position']), axis = 1)
nba_df["Date_formatted"] = nba_df.apply(lambda row : makeDate(row["Date"]), axis = 1)
nba_df["Week_number"] = nba_df.apply(lambda row : getWeek(row["Date"]), axis = 1)
nba_df["Week_Day"] = nba_df.apply(lambda row : getWeekDay(row["Date"]), axis = 1)
nba_df["Month"] = nba_df.apply(lambda row : calendar.month_name[int(row['Date_formatted'].split('-')[1])], axis = 1)
nba_df.head(5)
plt.figure(figsize=(16,8))
plt.title('BMI v Position')
print(sns.boxplot(x='Position_name',y='BMI',data=nba_df))
plt.figure(figsize=(16,8))
plt.title('Height_cm v Position')
print(sns.boxplot(x='Position_name',y='Height_cm',data=nba_df))
plt.figure(figsize=(16,8))
plt.title('Weight_kg v Position')
print(sns.boxplot(x='Position_name',y='Weight_kg',data=nba_df))
grouped_byPlayer = nba_df[['Real_value']].groupby(nba_df['Player']).sum()
nba_df = nba_df.join(grouped_byPlayer, on='Player', how='left', lsuffix='_groupby')
plt.figure(figsize = (16,8))
sns.scatterplot(x = "Height_cm", y = "Weight_kg",hue = "Position_name", size = "Real_value", data=nba_df)
sns.lineplot(x = "Height_cm", y = "Weight_kg", data = nba_df)
grouped_by_Conference = nba_df.groupby(['Conference_2','Season short','Real_value']).sum()
grouped_by_Conference.reset_index(inplace=True)
plt.figure(figsize=(16,8))
sns.barplot(x='Season short',y='Real_value',hue='Conference_2',ci =None,data=grouped_by_Conference)
grouped_by_teams = nba_df.groupby(['Team','Conference_2'])['Real_value'].sum().to_frame()
grouped_by_teams.reset_index(inplace=True)
plt.figure(figsize=(16,8))
g = sns.barplot(x="Team",y="Real_value",hue="Conference_2",data=grouped_by_teams)
plt.xticks(rotation=90)
