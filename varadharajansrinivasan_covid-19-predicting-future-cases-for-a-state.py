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
!pip install fbprophet
from fbprophet import Prophet
import pandas as pd
covid_ds = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
covid_ds.head(10)
covid_ds['Date'] =  pd.to_datetime(covid_ds.Date, format='%d/%m/%y')
covid_ds.head(10)
def predit_future_state(ds, state, days):
    ds_state = ds[ds['State/UnionTerritory']==state]
    ds_model = ds_state[['Date','Confirmed']].rename(columns={'Date': 'ds', 'Confirmed': 'y'})
    
    m = Prophet()
    m.fit(ds_model)
    
    future =  m.make_future_dataframe(periods=days)
    
    
    forecast=m.predict(future)
    fig1 = m.plot(forecast)
    fig2 = m.plot_components(forecast)
predit_future_state(covid_ds,'Tamil Nadu',36)