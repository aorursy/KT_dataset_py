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
from tqdm import tqdm_notebook as tqdm

from collections import defaultdict

from ipywidgets import interactive
# Standard plotly imports

import chart_studio.plotly as py

#import chart_studio.graph_objs as go

from plotly.offline import iplot, init_notebook_mode# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)
source = pd.read_csv("/kaggle/input/clouds-lb/understanding_cloud_organization-publicleaderboard.csv", parse_dates=["SubmissionDate"])
source.head()
print("First submission: {fs}, last submission: {ls}".format(fs = source.SubmissionDate.min(), ls = source.SubmissionDate.max()))
best_subs_data = defaultdict(list)
for team, data in tqdm(source.groupby('TeamName')):

    for day in range(97):

        date = pd.Timestamp('2019-08-16') + pd.Timedelta(days=day)

        data_upto = data[data.SubmissionDate <= date]

        if data_upto.empty:

            bestscore = 0

        else:

            bestscore = data_upto.Score.max()

        best_subs_data[team].append(bestscore)

    
df_lb_daily = pd.DataFrame(best_subs_data).transpose()

df_lb_daily.head()
print("We have {t} teams competing for {d} days".format(t = df_lb_daily.shape[0], d = df_lb_daily.shape[1]))
df_lb_daily.loc['Victor Zaguskin'].iplot()
df_lb_daily.max().iplot()
df_lb_daily.max().diff().iloc[-20:].iplot()
df_lb_daily.apply(lambda s: s.nlargest(13).min()).iplot()
df_lb_daily.apply(lambda s: s.nlargest(13).min()).diff().iloc[-20:].iplot()
df_lb_daily.quantile(q=.95).iplot()
df_lb_daily.quantile(q=.95).diff().iloc[-20:].iplot()
df_lb_daily.quantile(q=.90).iplot()
df_lb_daily.quantile(q=.90).diff().iloc[-20:].iplot()