import pandas as pd

import matplotlib.pylab as plt

import seaborn as sns

from pathlib import Path

from pandas.plotting import register_matplotlib_converters

%matplotlib inline
sns.set(font_scale=1.5)
BASE_FOLDER_PATH = Path("../input/m5-forecasting-accuracy/")

SALES_TRAIN_VALIDATION_PATH = BASE_FOLDER_PATH / "sales_train_validation.csv"

TRAIN_START_DATE = pd.to_datetime("2011-01-29")

TRAIN_END_DATE = pd.to_datetime("2016-04-24")
df = pd.read_csv(SALES_TRAIN_VALIDATION_PATH)
store_mean_df = pd.concat([df.filter(like="d_"), df["store_id"]], axis=1).groupby("store_id").mean().reset_index()

# Need to melt the data to get better data format

store_mean_df = store_mean_df.melt(id_vars="store_id", value_name="qty", var_name="date")

dates_s = pd.date_range(TRAIN_START_DATE, TRAIN_END_DATE, freq="1D")

date_labels = store_mean_df["date"].unique()

date_labels_to_date_d = dict(zip(date_labels, dates_s))
# Map date labels (d_1, d_2 and so on ) to actual dates

store_mean_df["date"] = store_mean_df["date"].map(date_labels_to_date_d)
store_mean_df.sample(10)
def compute_rolling_mean_per_store_df(df, period=30):

    return (df.set_index("date").groupby("store_id")

                                .rolling(period)

                                .mean()

                                .reset_index())
# Approximate month with 30 days and year with 365 days

monthly_rolling_mean_store_df = compute_rolling_mean_per_store_df(store_mean_df, period=30)

yearly_rolling_mean_store_df = compute_rolling_mean_per_store_df(store_mean_df, period=365)
register_matplotlib_converters()
g = sns.FacetGrid(monthly_rolling_mean_store_df.dropna(), col="store_id", col_wrap=3)

g = g.map(plt.plot, "date", "qty")
g = sns.FacetGrid(yearly_rolling_mean_store_df.dropna(), col="store_id",  col_wrap=3)

g = g.map(plt.plot, "date", "qty")
from fbprophet import Prophet

from fbprophet.plot import add_changepoints_to_plot



dfs = []





def fit_and_plot_trend(store_id):

    # Need to rename date and qty columns so that Prophet is happy. :)

    ts_df = (yearly_rolling_mean_store_df.loc[lambda df: df["store_id"] == store_id].dropna()

                                         .drop("store_id", axis=1)

                                         .rename(columns={"date": "ds", "qty": "y"}))

    m = Prophet(daily_seasonality=True)

    m.fit(ts_df)

    future = m.make_future_dataframe(periods=28)

    forecast = m.predict(future)

    fig = m.plot(forecast)

    dfs.append(forecast[["ds", "trend"]].assign(store_id=store_id))

    return add_changepoints_to_plot(fig.gca(), m, forecast)
# 10 stores

STORE_IDS = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
for store_id in STORE_IDS:

    print(f"Fitting a trend for {store_id}")

    fit_and_plot_trend(store_id)
store_to_trend_dict = (pd.concat(dfs).loc[lambda df: df["ds"] == TRAIN_END_DATE, ["store_id", "trend"]]

                                     .set_index("store_id")["trend"]

                                     .to_dict())
store_to_trend_dict