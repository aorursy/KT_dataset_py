# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import requests

import re

import pandas as pd

from bs4 import BeautifulSoup

from time import sleep

from selenium import webdriver

import numpy as np 
driver = webdriver.Chrome('chromedriver/chromedriver')

#request url 

driver.get('https://www.goalzz.com/?region=-7&team=144')

# give it some time

sleep(5)

# retrive , download html page 

html = driver.page_source

# close 

driver.close()
len(html) 
soup = BeautifulSoup(html,'lxml')
date = soup.find_all('div',attrs={'class':'matchDate'})
match_dates = []
for i in date:

    match_dates.append(i.text)
match_dates
df = pd.DataFrame()
df['date'] = match_dates
df
teams = soup.find_all('a',attrs={'class':'tl'})
27 * 2
teamss = []
for i in teams:

    teamss.append(i.text)
teamss.insert(2719,'Y20')
home_teams = []

away_teams = []
for i in range(0,len(teamss)):

    if i%2:

         away_teams.append(teamss[i])

    else:

         home_teams.append(teamss[i])

     
len(home_teams) , len(away_teams)
score = soup.find_all('td',attrs={'class':'sc'})
scores = []
for i in score: 

    scores.append(i.text)
scores
sc = pd.DataFrame()
sc['scores'] = scores
sc
sc.index
sc.loc[0][0]
w = []

u = []
len(sc.loc[5][0])
for i in range(0,1426):

    if len(sc.loc[i][0]) == 10 :

        u.append(sc.loc[i][0])

    else:

        w.append(sc.loc[i][0])

        

    
w.insert(1387,'0 : 0')
df['home_team'] = home_teams
df['away_team'] = away_teams
df
df['scores'] = w
df
df.loc[0][1]
win_team = []
for i in range(0,1386):

    if df.loc[i][3][0] > df.loc[i][3][4]:

        win_team.append(df.loc[i][1])

    elif df.loc[i][3][0] < df.loc[i][3][4]:

        win_team.append(df.loc[i][2])

    else :

        win_team.append('draw')
win_team.insert(1387,0)
df['waining_team'] = win_team
df
driver = webdriver.Chrome('chromedriver/chromedriver')

#request url 

driver.get('https://www.goalzz.com/?region=-7&team=144')

xx = []

for i in range(0,1386): 

       xx.append(driver.find_elements_by_xpath(f'//*[@id="matchesTable"]/tr[{i}]/td[1]/font/a'))
comp = []
for i in xx[1:] : 

       comp.append(i[0].text)

    
comp
df = df[df['date'] != 'N/A']
df['competiton'] = comp
df
df = df.rename(columns={'waining_team' : 'wining_team'})
df = df[['home_team' , 'scores' , 'away_team' , 'date' , 'wining_team' , 'competiton']]
df
stage = soup.find_all('td',attrs={'id':'jm10x5'})
st = []
for i in range (1,1387) : 

    stage = soup.find_all('td',attrs={'id':'jm{}x5'.format(i)})

    st.append(stage[0].text)
st
df['stage'] = st[:1363]
df
dff = df 
dff.stage[dff['stage'] == ''] = np.nan
dff.isnull().sum()
dff
year = [] 
dff.loc[0][3][:4]
for i in range (0,1362) : 

      if len(dff.loc[i][3]) <= 4 : 

            dff.loc[i][3] = '{}/1/1'.format(dff.loc[i][3])
dff
dff['date'] = pd.to_datetime(dff['date'])
dff


dff['home_team'] = dff['home_team'].str.replace(' ','_')
dff['home_team']
dff['away_team'] = dff['away_team'].str.replace('-','')
dff['away_team'] = dff['away_team'].str.replace(' ','_')
dff
dff.to_csv('Al_Hilal Scores Archive.csv')
dff.info()