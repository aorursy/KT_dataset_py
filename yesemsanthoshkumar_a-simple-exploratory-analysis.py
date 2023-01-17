import numpy as np

import pandas as pd

from scipy import stats

import matplotlib.pyplot as mplt

import seaborn as sns

% matplotlib inline
train = pd.read_csv("../input/train.csv",

                    parse_dates=["timestamp"],

                   date_parser=lambda x: pd.datetime.strptime(x, "%Y-%m-%d"))
train.shape
train.id.min(), train.id.max(), train.id.nunique()
train.timestamp.min(), train.timestamp.max(), train.timestamp.nunique()
train["year_month"] = train.timestamp.map(lambda x: x.year * 100 + x.month)

train["month"] = train.timestamp.dt.month

train["year"] = train.timestamp.dt.year

train["weekday"] = train.timestamp.dt.weekday
ts_df = train[["id", "year_month", "year", "month", "weekday"]]
y_df = ts_df.groupby("year").count().reset_index()

m_df = ts_df.groupby("month").count().reset_index()

wd_df = ts_df.groupby("weekday").count().reset_index()



fig, ax = mplt.subplots(ncols=2)

fig.set_size_inches(13, 3)



sns.barplot(data=y_df, x="year", y="id", ax=ax[0])

ax[0].set_title("Transactions over the years")



sns.barplot(data=wd_df, x="weekday", y="id", ax=ax[1])

ax[1].set_title("Transactions over weekdays")



fig, axs = mplt.subplots()

fig.set_size_inches(13, 3)



sns.barplot(data=m_df, x="month", y="id", ax=axs)

axs.set_title("Transaction over months")
yr_grp = train.groupby("year").mean().reset_index()

fig, ax = mplt.subplots(ncols=2)

fig.set_size_inches(10, 3)



sns.barplot(data=yr_grp, x="year", y="full_sq", orient="v", ax=ax[0])

ax[0].set_title("full_sq over the years")



sns.barplot(data=yr_grp, x="year", y="life_sq", orient="v", ax=ax[1])

ax[1].set_title("life_sq over the years")
sns.heatmap(train[["full_sq", "life_sq", "num_room", "price_doc"]].corr(), annot=True)
mode_by_own = train.loc[train.product_type == "OwnerOccupier", "build_year"].mode()[0]

mode_by_invest = train.loc[train.product_type == "Investment", "build_year"].mode()[0]

(mode_by_own, mode_by_invest)
train.loc[(train.product_type == "OwnerOccupier") & (train.build_year.isnull()), "build_year"] = mode_by_own

train.loc[(train.product_type == "Investment") & (train.build_year.isnull()), "build_year"] = mode_by_invest
train["year_difference"] = train.year - train.build_year
inv_val = train.loc[train.product_type == "Investment", "year_difference"].values

own_val = train.loc[train.product_type == "OwnerOccupier", "year_difference"].values
fig, ax = mplt.subplots(nrows=2)

fig.set_size_inches(15, 10)

ax[0].hist(inv_val)

ax[0].set_title("Year difference for investment buildings")

sns.countplot(own_val, ax=ax[1])

ax[1].set_title("Year difference for owner occupied buildings")
train.loc[train.full_sq < 10, :].shape
train.loc[train.full_sq < train.life_sq,:].shape
train.loc[train.full_sq < train.life_sq, "full_sq"] = train.life_sq
train.loc[train.floor > train.max_floor, :].shape
train.loc[train.kitch_sq > train.full_sq, :].shape
rooms = train[["num_room", "price_doc"]].groupby("num_room").aggregate(np.mean).reset_index()

mplt.scatter(x=rooms.num_room, y=rooms.price_doc)

mplt.xlabel("Num rooms")

mplt.ylabel('Mean Price')
population_errors = train.full_all - (train.male_f + train.female_f)

sns.countplot(population_errors)
train.loc[(train.state < 1) ^ (train.state > 4), "state"] = np.nan

inv_counts = train[train.product_type == "Investment"]["state"].value_counts()

own_counts = train[train.product_type == "OwnerOccupier"]["state"].value_counts()

product_category = pd.DataFrame([inv_counts, own_counts])

product_category.index = ["Investment", "OwnerOccupier"]

product_category.plot(kind="bar", stacked=True)