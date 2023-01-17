# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/meta-kaggle/Users.csv")

df = df.rename(columns={'Id':'UserId',})

df
df1 = pd.read_csv('/kaggle/input/meta-kaggle/UserAchievements.csv').drop(['Id','Tier','TierAchievementDate'],axis=1)

df1
full_data = df.merge(df1, on = ['UserId'],how = 'right')

full_data
full_data = full_data.sort_values(by='CurrentRanking', ascending=True)

full_data
ranked_user_data = full_data.dropna(subset=['CurrentRanking'])

ranked_user_data
user_names = set(ranked_user_data.UserName.unique())

len(user_names)
print(type(user_names))

user_names = pd.DataFrame(user_names)

print(type(user_names))
user_names = user_names.rename(columns={0:'UserName',})

user_names
!conda install -y gdown
import gdown

gdown.download('https://drive.google.com/uc?id=1c9ksxzAtOLeGF-VEmu6x7HYJhvzAjQc-', 'Kaggler_Countries.csv', quiet=False)
countries = pd.read_csv("/kaggle/working/Kaggler_Countries.csv")

countries
countries = countries.merge(user_names, on = ['UserName'],how = 'right')

countries.isna().sum()
countries.to_csv(r'/kaggle/working/Kaggler_Countries.csv', index = False)
from bs4 import BeautifulSoup

import re

import requests

from lxml import html

import time

import sys

import numpy as np

import pandas as pd

from datetime import datetime

import pytz



headers = {'user-agent': 'my-app/0.0.1'}
countries = pd.read_csv('/kaggle/working/Kaggler_Countries.csv')

nan_countries = countries[(countries.loc[:,['Country']].isnull()).any(axis=1)]

nan_countries
len(nan_countries)
count=0

temp=0

print('Started at',datetime.now(pytz.timezone('Asia/Seoul')).strftime("%H:%M:%S %d/%m/%Y"))

while True:

    try:

        for idx, row in enumerate(nan_countries.values):

            i = nan_countries.index[idx]

            name,country = row

            response = requests.get('http://www.kaggle.com/{}'.format(name),headers=headers)

            raw_text = str(BeautifulSoup(response.text, "lxml")).replace('"', '')

            c = re.findall('(country:[A-Za-z\s\n]+)', raw_text)[0][8:]

            print(i,'http://www.kaggle.com/{}'.format(name),c)

            if c=='null':

                countries.Country[i]='Unknown'

            else:

                countries.Country[i]= c

            temp+=1

    except:

        print('Found =',temp)

        dt_string = datetime.now(pytz.timezone('Asia/Seoul')).strftime("%H:%M:%S %d/%m/%Y")

        print("Oops!", sys.exc_info()[0], "occurred.")

        print("Finished at", dt_string)	

        print("Stopped! Try again!")

        print()

        countries.to_csv(r'/kaggle/working/Kaggler_Countries.csv', index = False)

        countries = pd.read_csv('/kaggle/working/Kaggler_Countries.csv')

        nan_countries = countries[(countries.loc[:,['Country']].isnull()).any(axis=1)]

        print(len(nan_countries))

        time.sleep(300)

        count+=1

        print('Try count =',count)

        continue

    break

countries.to_csv(r'/kaggle/working/Kaggler_Countries.csv', index = False)
df = ranked_user_data.merge(countries, on = ['UserName'],how = 'right').sort_values(by='Points', ascending=False).reset_index(drop=True)

labels = {5:"Kaggleteams",4:"Grandmasters",3:"Masters",2:"Experts"}

df["PerformanceTier"]= df["PerformanceTier"].apply(lambda x :np.array(labels[x]))

df['CurrentRanking'] = df['CurrentRanking'].astype(int)

df['HighestRanking'] = df['HighestRanking'].astype(int)

df
stats_countries = pd.DataFrame(df['Country'])

unknowns = int(stats_countries.isin(['Unknown']).sum())

unknowns
rankCountries = pd.DataFrame(stats_countries.groupby(stats_countries.columns.tolist(),as_index=False).size())

rankCountries = rankCountries.rename(columns={0:'Total'})

rankCountries = rankCountries.sort_values(by=['Total'],ascending=False)

print('Total ranked users =',rankCountries.Total.sum())

print('Unknown country users =',unknowns)

print('Total user countries =',len(rankCountries)-1)
pd.set_option('display.max_rows', None)

rankCountries
mains = rankCountries.where(rankCountries.Total>=50.).dropna()

mains
others = pd.DataFrame(rankCountries.where(rankCountries.Total < 50).sum())

others = others.rename(columns={0:'Others'}).T

others
import matplotlib



matplotlib.style.use('ggplot')

%matplotlib inline  



by_country = pd.concat([mains,others])

pd.Series(by_country['Total']).plot.pie(figsize=(10, 10), autopct='%0.1f')
df = df.rename(columns={'UserName':'User Name',

                        'DisplayName':'Display Name', 

                        'RegisterDate':'Register Date', 

                        'PerformanceTier':'Performance Tier',

                        'AchievementType':'Achievement Type',

                        'CurrentRanking':'Current Ranking', 

                        'HighestRanking':'Highest Ranking', 

                        'TotalGold':'Gold', 

                        'TotalSilver':'Silver',

                        'TotalBronze':'Bronze'})
df = pd.concat([df['Country'],df.iloc[:, 1:-1]], axis=1)

#df.to_csv(r'/kaggle/working/kaggler_ranking_by_countries.csv')

df.to_html("kaggler_ranking_by_countries.htm")
pd.set_option('display.max_rows', 40)

df
from IPython.display import IFrame

display(IFrame(src='https://quasar.kz/kaggler_ranking_by_countries.htm', width="100%", height=600))