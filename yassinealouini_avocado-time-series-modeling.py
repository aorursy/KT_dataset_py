# The usual suspects
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.cross_validation import cross_val_score, time
from sklearn.model_selection import TimeSeriesSplit
# Ignore warnings (this isn't a good practice usually)
import warnings
warnings.filterwarnings("ignore")
# Avocados are green. :) 
GREEN_COLORMAP = sns.color_palette("Greens")
DATA_PATH  = "../input/avocado.csv"
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
df.set_index('Date').plot(y='AveragePrice', ax=ax, color=GREEN_COLORMAP[2])
# Get the number of years and create one plot per year.
# Notice that 2018 has less samples than the previous ones.
years = df.year.unique()
number_years = len(years)
fig, axes = plt.subplots(number_years, 1, figsize=(12, 8))
for i, year in enumerate(years):
    # One green shade per year :)
    # Also, no line connecting the points and marker set to a dot
    # for enhanced readability.
    (df.set_index('Date')
       .loc[lambda df: df.year == year]
       .plot(y='AveragePrice', ax=axes[i], color=GREEN_COLORMAP[i],
             marker="o", linestyle=""))
    axes[i].legend_.remove()

fig.set_tight_layout("tight")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
df.groupby('Date').size().plot(ax=ax)
df.Date.diff().value_counts()
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
df.groupby('Date')['region'].nunique().plot(ax=axes[0])
df.groupby('Date')['type'].nunique().plot(ax=axes[1])
("Bingo, that's it: there are {} unique regions and {} unique" 
" types of Avocado ({})").format(df['region'].nunique(),
                                 df['type'].nunique(),
                                ' and '.join(df['type'].unique())) 
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

df['region'].value_counts().plot(kind='bar', ax=axes[0], 
                                 color=GREEN_COLORMAP)
df['type'].value_counts().plot(kind='bar', ax=axes[1], color=GREEN_COLORMAP)
fig.set_tight_layout("tight")
ts = df.groupby('Date')['AveragePrice'].mean().reset_index()
ts.sample(5)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ts.set_index('Date').plot(ax=ax, marker="o", linestyle="-", color=GREEN_COLORMAP[2])
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
(ts.set_index('Date')
   .resample('1M')
   .mean()
   .plot(ax=ax, marker="o", linestyle="-", color=GREEN_COLORMAP[2]))
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
(ts.set_index('Date')
   .assign(month=lambda df: df.index.month)
   .groupby('month')['AveragePrice'].agg(["mean", "std", "median", "min", "max"])
   .plot(ax=ax, marker="o"))
ax.set_xlabel('Month')
# Renaming the ts DataFrame's columns (you will see why soon) before temporal split
renamed_ts = ts.rename(columns={"Date": "ds", "AveragePrice": "y"})
train_ts = renamed_ts.loc[lambda df: df['ds'].dt.year < 2018, :]
test_ts = renamed_ts.loc[lambda df: df['ds'].dt.year == 2018, :]
train_ts.head()
train_ts.tail()
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics


# TODO: Add some comments
HORIZON = "90days"
PERIOD = "7days"

prophet_model = Prophet()
prophet_model.fit(train_ts)
prophet_cv_df = cross_validation(prophet_model, horizon=HORIZON, 
                                 period=PERIOD)
prophet_cv_df.head()
prophet_perf_df = performance_metrics(prophet_cv_df)
prophet_perf_df.sample(5)
from fbprophet.plot import plot_cross_validation_metric
plot_cross_validation_metric(prophet_cv_df, metric='mae');
fig ,ax = plt.subplots(1, 1, figsize=(12, 8))
(prophet_perf_df.groupby('horizon')['mae']
                .mean()
                .plot(ax=ax, marker="o", colors=GREEN_COLORMAP[2]))
ax.set_ylabel('MAE')
future_prophet_df = prophet_model.make_future_dataframe(periods=365)
predicted_prophet_df = prophet_model.predict(future_prophet_df)
prophet_model.plot(predicted_prophet_df);
prophet_model.plot_components(predicted_prophet_df);
def add_calendar_features(df):
    # TODO: Add some comments
    return (df.assign(month=lambda df: df['ds'].dt.month, 
                                     week=lambda df: df['ds'].dt.week,
                                     year=lambda df: df['ds'].dt.year,
                                     past_month_mean_y=lambda df: 
                                      (df['y'].rolling(window=4)
                                              .mean()
                                              .fillna(method='bfill')),
                                     past_year_mean_y=lambda df: 
                                      (df['y'].rolling(window=52)
                                              .mean())
                                              .fillna(method='bfill'))
                              )



augmented_ts = add_calendar_features(renamed_ts)
augmented_train_ts = augmented_ts.loc[lambda df: df['ds'].dt.year < 2018, :].drop('ds', axis=1)
augmented_test_ts = augmented_ts.loc[lambda df: df['ds'].dt.year == 2018, :].drop('ds', axis=1)
augmented_train_ts.head()
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
augmented_train_ts.plot(y='past_month_mean_y', ax=ax)
augmented_train_ts.plot(y='past_year_mean_y', ax=ax)
augmented_train_ts.plot(y='y', ax=ax)
tscv = TimeSeriesSplit(n_splits=3)
from tpot import TPOTRegressor

# TODO: Try more generations and a bigger population size. 
# Be careful not to run out of time!

tpot_model = TPOTRegressor(generations=20, population_size=100, cv=tscv, 
                           scoring="neg_mean_absolute_error", 
                           n_jobs=2, verbosity=2)
tpot_model.fit(augmented_train_ts.drop('y', axis=1), 
               augmented_train_ts['y'])
test_timestamps = test_ts.ds.values
predicted_prophet_s = predicted_prophet_df.loc[lambda df: df['ds']
                                               .isin(test_timestamps), "yhat"]
predicted_tpot_s = tpot_model.predict(augmented_test_ts.drop("y", axis=1))
assert predicted_tpot_s.shape == predicted_prophet_s.shape
assert predicted_tpot_s.shape == test_ts["y"].shape
predictions_df = pd.DataFrame({'tpot': predicted_tpot_s, 
                              'prophet': predicted_prophet_s,
                              'true': test_ts['y'].values,
                              'Date': test_ts['ds'].values})
print("MAE for tpot on the test dataset is: {}".format(
    (predictions_df['tpot'] - predictions_df['true']).abs().mean(axis=0)))
print("MAE for prophet on the test dataset is: {}".format(
    (predictions_df['prophet'] - predictions_df['true']).abs().mean(axis=0)))
fig, ax = plt.subplots(1, 1, figsize=(12, 10))
predictions_df.set_index('Date').plot(marker='o', ax=ax)