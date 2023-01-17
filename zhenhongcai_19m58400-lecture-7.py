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
import pandas as pd
import plotly.express as px
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale="Rainbow",range_color=(0.1,200000.))
fig.update_layout(title_text='Confirmed Cumulative Cases per Country',title_x=0.5)
fig.show()



df1 = pd.read_csv('../input/march-11-2011-earthquakeoldest-first/query (1).csv',header=0)
df1.index = pd.to_datetime(df1['time'])
df1['time']=df1.index.strftime("%y-%m-%d %H:00:00")
fig1 = px.scatter_geo(df1,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='Rainbow',range_color=(5.,7.))
fig1.show()
