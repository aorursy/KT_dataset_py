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
df = pd.read_csv('../input/2019-covid19-ncov19-data-set-in-korean/COVID-19_Korean.csv')

df.head()
import plotly.express as px

max_val = df["확진자"].max()

fig = px.bar(df, x="국가/지역", y="확진자", title='전세계 확진자', color='국가/지역'

  ,animation_frame="날짜", animation_group="국가/지역", range_y=[0,max_val*1.1])

fig.show()

df = df[df['국가/지역']!='중국']

max_val = df["확진자"].max()

fig = px.bar(df, x="국가/지역", y="확진자", title='중국 이외 나라별 확진자', color='국가/지역'

            ,animation_frame="날짜", animation_group="국가/지역", range_y=[0,max_val*1.1])

fig.show()
