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
df = pd.read_csv('/kaggle/input/ipl-data-set/matches.csv')

df
df["running_total"] = df["win_by_runs"].cumsum()

df["running_total_by_person"] = df.groupby("win_by_runs")["win_by_wickets"].cumsum()

df
df["win_by_runs_diff"] = df["win_by_runs"].diff()

df["win_by_runs_diff"]
df["A_diff_pct"] = df["win_by_runs"].pct_change()*100

df["A_diff_pct"]
df.style.format({"A_diff_pct":'{:.2f}%'})
df["city"].str.cat(df["winner"], sep = ", ").head()
# using + sign

df["city"] + ", " + df["winner"].head()
d = {"A": [100, 200, 300, 400, 100], "W":[10, 5, 0, 3, 8]}

df = pd.DataFrame(d)

df

# with replacement

df.sample(n = 5, replace = True, random_state = 2)



# adding weights

df.sample(n = 5, replace = True, random_state = 2, weights = "W")
df.style.hide_index().set_caption("Styled df with no index and a caption")
s = pd.Series(range(1552194000, 1552212001, 3600))

s = pd.to_datetime(s, unit = "s")

s



# set timezome to current time zone (UTC)

s = s.dt.tz_localize("UTC")

s



# set timezome to another time zone (Chicago)

s = s.dt.tz_convert("America/Chicago")

s