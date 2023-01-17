import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import altair as alt
alt.renderers.enable('notebook')
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/restaurant-scores-lives-standard.csv')
data.head()
most_recent_daily_data = data[data['inspection_date'] == data['inspection_date'].max()]
daily_risk_category = most_recent_daily_data.groupby('risk_category').size().reset_index()\
.rename(columns={0:'count'})
daily_risk_category
alt.Chart(daily_risk_category).mark_bar().encode(
x = 'risk_category:N',
y = 'count:Q'
)
inspection_score_daily = data.groupby('inspection_date').inspection_score.mean().reset_index()
inspection_score_daily['date'] = pd.to_datetime(inspection_score_daily['inspection_date'])
inspection_score_daily['year'] = inspection_score_daily['date'].dt.year
alt.Chart(inspection_score_daily).mark_line().encode(
x = 'date',
y = 'inspection_score:Q',
color = 'year:N'
)
