# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/shot_logs.csv')
df.columns
df.head(5)
defense_df = df.groupby('player_name').agg({'FGM': pd.Series.count,

                                            'CLOSEST_DEFENDER_PLAYER_ID': pd.Series.nunique})
defense_df.sort('FGM', ascending = False)
df['player_name'].unique()
defense_df.ix['stephen curry']
df[df['PERIOD'] == 7]
def seconds(clock):

    x = time.strptime(clock, '%M:%S')

    return(datetime.timedelta(minutes=x.tm_min,seconds=x.tm_sec).total_seconds())
import datetime

import time

df['GAME_CLOCK_SECONDS'] = df['GAME_CLOCK'].apply(seconds)
df.GAME_CLOCK_SECONDS.describe()
df.ix[df['TOUCH_TIME'] < 0, 'TOUCH_TIME'] = np.nan

df = df[df['FINAL_MARGIN'] < 5]
clutch_df = df[df.GAME_CLOCK_SECONDS < 300]
clutch_df.shape
plt.style.use('fivethirtyeight')

plt.scatter(clutch_df['GAME_CLOCK_SECONDS'], clutch_df['TOUCH_TIME'])