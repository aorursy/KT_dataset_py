



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from fbprophet import Prophet



import os

for dirname, _, filenames in os.walk('/kaggle/input/bitcoin-data/BTC-USD.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



path='/kaggle/input/bitcoin-data/BTC-USD.csv'

df=pd.read_csv(path)
df.drop('Open',axis=1,inplace=True)

df.drop('High',axis=1,inplace=True)

df.drop('Low',axis=1,inplace=True)

df.drop('Adj Close',axis=1,inplace=True)

df.drop('Volume',axis=1,inplace=True)

df
df.rename(columns={'Date': 'ds', 'Close': 'y'},inplace=True)

# df

df['y']=np.log(df['y'])

m=Prophet(daily_seasonality=True,interval_width=0.95)



m.fit(df)



future=m.make_future_dataframe(periods=60,freq='D')

future.tail()



pred=m.predict(future)

pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()



# df.plot(figsize=(15,7))

fig=m.plot(pred)

fig=m.plot_components(pred)

fig.show()

# fig.savefig('/home/bigpenguin/01_fbprophet_getting_started-02.png')