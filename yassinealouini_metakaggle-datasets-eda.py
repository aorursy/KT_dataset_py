# Data loading and processing
import pandas as pd 

# Plotting
import matplotlib.pylab as plt
import seaborn as sns
# Some setup
sns.set(font_scale=1.5)
META_DATASETS_PATH = "../input/all_kaggle_datasets.csv"
meta_datasets_df = pd.read_csv(META_DATASETS_PATH)
_dfs = []
for datasetId, category_row in meta_datasets_df.set_index("datasetId")["categories"].iteritems():
    # Each category_dict has two keys: "categories" and "type". Notice that the "type" column is always equal to "dataset"
    # so will ignore it (doesn't add anything).
    category_dict = eval(category_row)
    # A list of categories
    categories = category_dict["categories"]
    _df = pd.DataFrame(categories)
    _df["datasetId"] = datasetId 
    if category_dict["type"] != "dataset":
        print(category_dict["type"])
    _dfs.append(_df)
categories_df = pd.concat(_dfs, sort=False)
categories_df.head(1).T
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
categories_df.groupby("name").size().nlargest(10).plot(kind='bar', ax=ax)
ax.set_ylabel("Number of datasets")
ax.set_xlabel("Categories")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
categories_df.groupby("name")["totalCount"].sum().nlargest(10).plot(kind='bar', ax=ax)
ax.set_ylabel("Total count")
ax.set_xlabel("Categories")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
(meta_datasets_df.assign(date=lambda df: pd.to_datetime(df["dateUpdated"]))
                 .set_index('date')
                 .resample('1M')
                 .size()
                 .plot(ax=ax)
)

ax.set_ylabel("Number of datasets per month")


fig, ax = plt.subplots(1, 1, figsize=(12, 8))
(meta_datasets_df.assign(date=lambda df: pd.to_datetime(df["dateUpdated"]))
                 .set_index('date')
                 .resample('1M')
                 .size()
                 .cumsum()
                 .plot(ax=ax)
)
"The number of unique datasets as of {} is {}".format(meta_datasets_df.dateUpdated.max(), meta_datasets_df.datasetId.nunique())
from scipy.optimize import curve_fit
import numpy as np


def exp_f(x, a, b, c):
    return a*np.exp(b*x)+c

to_fit_s = (meta_datasets_df.assign(date=lambda df: pd.to_datetime(df["dateUpdated"]))
                            .set_index('date')
                            .resample('1M')
                            .size()
                            .cumsum())

x = range(0, len(to_fit_s))
y = to_fit_s.values

fitted_params, covariance = curve_fit(exp_f, x, y, p0=(1, 1, 0))
fitted_s = pd.Series(exp_f(x, *fitted_params), index=to_fit_s.index)
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
fitted_s.plot(ax=ax, label='Fitted', style='o--')
to_fit_s.plot(ax=ax, label='Original', style='o-')
ax.legend()
ax.set_ylabel("Cumulative number of datasets")
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
day_of_week_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
(meta_datasets_df.assign(date=lambda df: pd.to_datetime(df["dateUpdated"]))
                 .assign(day_of_week=lambda df: df.date.dt.day_name())
                 .groupby("day_of_week")
                 .size()
                 .loc[day_of_week_order]
                 .plot(kind='bar', ax=ax))
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
(meta_datasets_df.assign(date=lambda df: pd.to_datetime(df["dateUpdated"]))
                 .assign(month=lambda df: df.date.dt.month)
                 .groupby("month")
                 .size()
                 .plot(kind='bar', ax=ax))
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
(meta_datasets_df.groupby("creatorName")["datasetId"]
                 .count()
                 .nlargest(10)
                 .plot(kind='bar', ax=ax)
)
ax.set_xlabel("Creator's name")
ax.set_ylabel("Number of datasets")