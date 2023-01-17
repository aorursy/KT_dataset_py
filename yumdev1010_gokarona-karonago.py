import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
import requests



response = requests.get('https://api.covid19india.org/states_daily.json')

response = response.json()

response = response['states_daily']

df = pd.DataFrame(response)

df = df.set_index('date')

# df = pd.to_datetime(df['date'])



df
df_confirmed = df.loc[df['status'] == 'Confirmed']

df_confirmed = df_confirmed.drop(['status'], axis=1)

cols = list(df_confirmed)





# df_confirmed[cols].apply(pd.to_numeric, errors='coerce', axis=1)

df_confirmed = pd.to_numeric(df_confirmed[cols].stack(), errors='coerce').unstack()

df_confirmed
df_state = df_confirmed[['mh', 'kl', 'up', 'dl']]

df_state.info()
df_state.plot(figsize=(20,5), grid=True, kind='bar')

plt.show()