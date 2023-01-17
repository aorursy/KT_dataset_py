import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('darkgrid')

data = pd.read_csv("/kaggle/input/indian-food-101/indian_food.csv", index_col="name")

data.head()
fig, ax = plt.subplots(2,2, figsize=(18,12))

ax = ax.ravel()

sns.swarmplot(ax=ax[0], x=data.diet, y=data.prep_time)

sns.swarmplot(ax=ax[2], x=data.diet, y=data.cook_time)



sns.barplot(ax=ax[1], x=data.diet, y=data.prep_time)

sns.barplot(ax=ax[3], x=data.diet, y=data.cook_time)



ax[0].set_title("Prep Time")

ax[2].set_title("Cook Time")

ax[1].set_title("Prep Time")

ax[3].set_title("Cook Time")
total_time = data.prep_time + data.cook_time

print(f"Making {total_time.index[total_time.argmax()]} takes {total_time.iloc[total_time.argmax()]} minutes.")
fig, ax = plt.subplots(2,2, figsize=(18,12))



ax = ax.ravel()

sns.swarmplot(ax=ax[0], x=data.region, y=data.prep_time)

sns.swarmplot(ax=ax[2], x=data.region, y=data.cook_time)



sns.barplot(ax=ax[1], x=data.region, y=data.prep_time)

sns.barplot(ax=ax[3], x=data.region, y=data.cook_time)





ax[0].set_title("Prep Time, by region")

ax[1].set_title("Prep Time, by region")

ax[2].set_title("Cook Time, by region")

ax[3].set_title("Cook Time, by region")
fig, ax = plt.subplots(2,2, figsize=(18,12))



ax = ax.ravel()

sns.swarmplot(ax=ax[0], x=data.flavor_profile, y=data.prep_time)

sns.swarmplot(ax=ax[2], x=data.flavor_profile, y=data.cook_time)



sns.barplot(ax=ax[1], x=data.flavor_profile, y=data.prep_time)

sns.barplot(ax=ax[3], x=data.flavor_profile, y=data.cook_time)





ax[0].set_title("Prep Time, by flavor")

ax[1].set_title("Prep Time, by flavor")

ax[2].set_title("Cook Time, by flavor")

ax[3].set_title("Cook Time, by flavor")
fig, ax = plt.subplots(2,2, figsize=(18,12))



ax = ax.ravel()

sns.swarmplot(ax=ax[0], x=data.course, y=data.prep_time)

sns.swarmplot(ax=ax[2], x=data.course, y=data.cook_time)



sns.barplot(ax=ax[1], x=data.course, y=data.prep_time)

sns.barplot(ax=ax[3], x=data.course, y=data.cook_time)





ax[0].set_title("Prep Time, by course type")

ax[1].set_title("Prep Time, by course type")

ax[2].set_title("Cook Time, by course type")

ax[3].set_title("Cook Time, by course type")
fig, ax = plt.subplots(1,2, figsize=(18,12))



ax = ax.ravel()

sns.barplot(ax=ax[0], y=data.state, x=data.prep_time)

sns.barplot(ax=ax[1], y=data.state, x=data.cook_time)





ax[0].set_title("Prep Time, by state")

ax[1].set_title("Cook Time, by state")

fig.tight_layout()
plt.figure(figsize=(8,8))

sns.barplot(y=data.state.value_counts().index, x=data.state.value_counts())

plt.title("Value Count by state")