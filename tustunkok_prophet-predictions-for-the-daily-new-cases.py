import numpy as np

import pandas as pd

from fbprophet import Prophet
df = pd.read_csv("../input/covid19-turkey-as-published/covid19-turkey-data.csv")

df.head()
df.info()
nonnull_df = df.dropna(subset=["Daily Recovered"]).drop("Pneumonia Cases/Active Cases", axis=1)

nonnull_df.info()
nonnull_df["Active Cases"] = nonnull_df.apply(lambda x: x["Total Cases"] - x["Total Recovered"] - x["Total Deaths"], axis=1)
prophet_train_df = nonnull_df[["Date", "Daily Cases"]].rename(columns={"Date": "ds", "Daily Cases": "y"})

prophet_train_df.head()
prophet = Prophet().fit(prophet_train_df)

future = prophet.make_future_dataframe(periods=10)

forecast = prophet.predict(future)
fig1 = prophet.plot(forecast.tail(10))
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10)