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
import sqlite3
# creating the connection

conn = sqlite3.connect('../input/soccer/database.sqlite')
# have a look at the table types.I can see various items that i could join.

pd.read_sql("""
                SELECT * 
                FROM sqlite_master
                WHERE type='table'
                ;""",
                conn)
# attempting a join

all_players = pd.read_sql("""
                SELECT *
                FROM Player
                JOIN Player_Attributes
                ON Player.player_api_id = Player_attributes.player_api_id
                ;""",
                conn)
pd.set_option('display.max_columns', None)
all_players.head()
all_players = all_players.drop(['id', 'date', 'player_api_id','player_name','player_fifa_api_id','birthday','attacking_work_rate','defensive_work_rate'], axis=1)
all_players.head()
all_players.tail()
all_players = pd.get_dummies(all_players, drop_first=True)
all_players.head()
all_players.isnull().sum()
all_players.dropna(inplace=True)
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(all_players.drop('overall_rating', axis=1), all_players['overall_rating'])
y_train.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_normed = scaler.fit_transform(x_train)
x_test_normed = scaler.transform(x_test) 
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train, y_train)
reg.score(x_train,y_train)
reg.score(x_test, y_test)
reg_summary = pd.DataFrame(x_train.columns.values, columns=['Features'])
reg_summary['coefs'] = reg.coef_
reg_summary.sort_values('coefs')
from sklearn.feature_selection import f_regression
p_val = f_regression(x_train,y_train)[1]
reg_summary['p_score'] =p_val.round(3)
reg_summary.sort_values('coefs')
x_train.shape
r2 = reg.score(x_train,y_train)

n = 135948

p = 38
ar2 = (1-(1-r2)*(n-1)) / (n-p-1)
ar2
