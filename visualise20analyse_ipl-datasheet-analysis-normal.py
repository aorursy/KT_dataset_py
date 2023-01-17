# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series,DataFrame
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/matches.csv')
data.head()
data.info()
#Cleaning of Data as umpire 3 values are null and deleting any null values.


data.drop(data.columns[[17]],axis=1,inplace=True)
data.dropna(inplace=True)
data.shape
#Characterising Data by each season
name_dict = {}
season_2008 = data[data["season"]==2008]
name_dict["season_2008"]=season_2008

season_2009 = data[data["season"]==2009]
name_dict["season_2009"]=season_2009

season_2010 = data[data["season"]==2010]
name_dict["season_2010"]=season_2010

season_2011 = data[data["season"]==2011]
name_dict["season_2011"]=season_2011

season_2012 = data[data["season"]==2012]
name_dict["season_2012"]=season_2012

season_2013 = data[data["season"]==2013]
name_dict["season_2013"]=season_2013

season_2014 = data[data["season"]==2014]
name_dict["season_2014"]=season_2014

season_2015 = data[data["season"]==2015]
name_dict["season_2015"]=season_2015

season_2016 = data[data["season"]==2016]
name_dict["season_2016"]=season_2016

season_2017 = data[data["season"]==2017]
name_dict["season_2017"]=season_2017

#No.of matches won by each team over 10 seasons
data.winner.value_counts()
#Plotting this winner data
(data.winner.value_counts(normalize =True)*100).plot(kind = 'barh' , title='Match winning percentages by each team',figsize = (20,10))
# No. of matches won by team who wins the toss

toss_match_winner = data[data["toss_winner"]==data["winner"]]
toss_match_winner.winner.value_counts()
(toss_match_winner.winner.value_counts(normalize =True)*100).plot(kind = 'barh' , title='Match and toss winning percentages by each team',figsize = (12,10))
pd.crosstab(toss_match_winner.winner,toss_match_winner.toss_decision).plot(kind='bar',title='Winning match w.r.t winning toss decisions')
# No.of matches won in each city for season 2017

pd.crosstab(season_2017.winner,season_2017.city)
pd.crosstab(season_2017.winner,season_2017.city).plot(kind='bar',title='Winning w.r.t cities in 2017',figsize=(12,10))

data.player_of_match.value_counts()
symbol = ["2008","2009","2010","2011","2012","2013","2014","2015","2016","2017"]

for i in symbol:
    season = "season_" +i
    pd.crosstab(name_dict[season].season,name_dict[season].player_of_match).plot(kind='bar', title='Player of match in '+ i).legend(bbox_to_anchor=(1.2, 0.5))

sns.barplot(data.result.value_counts().index,data.result.value_counts(),data=data)
# DL applied
data.dl_applied.value_counts()
(data.dl_applied.value_counts(normalize =True)*100).plot(kind = 'barh' , title='Percentage of matches where DL Rule was applied',figsize = (6,4))
#Matches won by team batting second
df = data[data["win_by_wickets"]!=0]
df.head()
df.dl_applied.value_counts().plot(kind='barh')
print(df.dl_applied.value_counts())
data.toss_winner.value_counts()
data.toss_winner.value_counts().plot(kind='bar')
sns.barplot(x=data.toss_winner.value_counts().values,y=data.toss_winner.value_counts().index)
symbol = ["2008","2009","2010","2011","2012","2013","2014","2015","2016","2017"]

for i in symbol:
    season = "season_" +i
    print("Season:"+i+"\n")
    print("Winner:"+name_dict[season].winner.iloc[-1])
    print("Runner-up:"+(name_dict[season].team2.iloc[-1] if(name_dict[season].team1.iloc[-1]==name_dict[season].winner.iloc[-1]) else (name_dict[season].team1.iloc[-1])))
    print("\n")
