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

        fullpath = os.path.join(dirname, filename)

        print(fullpath)

        df = pd.read_csv(fullpath)

        print(df.count())

        df.head()



# Any results you write to the current directory are saved as output.
import pandas as pd

filename = "/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv"

df = pd.read_csv(filename)

#df.head()

x = df

#x = x[(x['Country/Region']=='US')]

#x = x[(x['Province/State']=='Washington')]

#x = x[(x['Country/Region']=='US') & ((x['Province/State'].str.endswith(', WA')) | (x['Province/State']=='Washington'))]

x = x[(x['Country/Region']=='US') & ((x['Province/State'].str.endswith(', NY')) | (x['Province/State']=='New York'))]

if False:

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        print(x)

print(x['3/13/20'].sum())

y = x.sum(axis=0)

if True:

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):

        print(y)