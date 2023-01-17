import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as mdates



import matplotlib

matplotlib.style.use('ggplot')





# Read in CSV to dataframe

df = pd.read_csv('../input/hacker_news_sample.csv', parse_dates=[13], infer_datetime_format=True)



# Examine the fields

df.columns
df.shape
for column in df.columns:

    print(column, df[column].dtype)
df["ranking"].unique()
# "ranking" is useless, drop it

del df["ranking"]

df.columns
# don't count deleted/invalid/ no user posts

df["deleted"].unique()
df["dead"].unique()
df = df[pd.isnull(df["dead"])]

df = df[pd.isnull(df["deleted"])]

del df["dead"]

del df["deleted"]

df.columns
df = df[pd.notnull(df["by"])]
df.shape
by_user = df.groupby(["by"]).size()

by_user
by_user.nlargest(30)
by_user.describe()
plt.figure(figsize=(10,5))

by_user.plot.box(vert=False, logx=True)
one_users = by_user[by_user <= 1].count()

total_users = by_user.count()

one_users, total_users, 100 * one_users / total_users
100 * by_user[by_user <= 1].sum() / by_user.sum()
two_users = by_user[by_user <= 2].count()

two_users, total_users, 100 * two_users / total_users
100 * by_user[by_user <= 2].sum() / by_user.sum()
100 * by_user[by_user <= 5].count() / total_users
100 * by_user[by_user <= 100].count() / total_users
plt.figure(figsize=(8,6))

ax = by_user.plot.hist()

ax.set(xlabel="number of posts", ylabel="number of users")

ax
plt.figure(figsize=(8,6))

ax = by_user.plot.hist(logx=True, logy=True, bins=2**np.arange(0, 15))

ax.set(xlabel="number of posts", ylabel="number of users")

ax
by_user["tptacek"], by_user["tptacek"] / by_user.mean()
100 * by_user.nlargest(2).sum() / by_user.sum()
by_user.nlargest(20)
100 * by_user.nlargest(20).sum() / by_user.sum()
100 * by_user.nlargest(1000).sum() / by_user.sum()
100 * by_user.nlargest(20000).sum() / by_user.sum()
df["year"] = df["timestamp"].dt.year
by_year = df.groupby("year").size()

by_year.plot.bar(figsize=(8,6))
# https://stackoverflow.com/a/10374456

user_year = pd.DataFrame({'count' : df.groupby( [ "year", "by"] ).size()}).reset_index()

user_year
user_year["year"] = user_year["year"].astype("str")
user_year[["year", "count"]].boxplot(by="year", figsize=(10,6))
# https://stackoverflow.com/a/27844045

group_useryear_sizes = df.groupby( [ "year", "by"] ).size()

group_useryear = group_useryear_sizes.groupby(level="year", group_keys=False)
group_useryear.nlargest(1)
group_useryear.nlargest(5)
pareto = group_useryear.apply(lambda x: 100 * x.nlargest(int(x.count() / 5)).sum() / x.sum())

pareto
pareto.plot(kind="bar", figsize=(8,6))
df["month"] = pd.to_datetime(df["timestamp"].dt.strftime("%Y-%m-01"))
group_usermonth_sizes = df.groupby( [ "month", "by"] ).size()

group_usermonth = group_usermonth_sizes.groupby(level="month", group_keys=False)
paretomonth = group_usermonth.apply(lambda x: 100 * x.nlargest(int(x.count() / 5)).sum() / x.sum())

paretomonth
paretomonth.describe()
paretomonth.plot.box(vert=False, figsize=(8,6))
paretomonth.plot(figsize=(12,10))
top_users = group_useryear.nlargest(1).index.levels[1]

list(top_users)
user_month = pd.DataFrame({'count' : df.groupby( [ "month", "by"] ).size()}).reset_index()
fig, ax = plt.subplots(figsize=(12,10))

for user in top_users:

    user_month_partial = user_month[user_month["by"] == user]

    ax.plot(user_month_partial["month"], user_month_partial["count"], label=user)

ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

ax.legend()

ax