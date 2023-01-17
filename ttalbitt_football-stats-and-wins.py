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
df = pd.read_csv('/kaggle/input/american-football-team-stats-1998-2019/AmericanFootball98.csv')

df.head(2)
df.select_dtypes(object).head(2)
df['avg start'] = df['avg start'].apply(lambda x : float(x[4:]))

df['opp avg start'] = df['opp avg start'].apply(lambda x : float(x[4:]))
df['avg time per drive'] = df['avg time per drive'].apply(lambda x : int(x[0])*60 + int(x[2:]))

df['opp avg time per drive'] = df['opp avg time per drive'].apply(lambda x : int(x[0])*60 + int(x[2:]))
df.dtypes.value_counts()
win_corr = df.drop('team_code', axis=1).corr().loc['wins']
win_corr_abs = abs(win_corr)
win_corr_abs.sort_values().head(60)