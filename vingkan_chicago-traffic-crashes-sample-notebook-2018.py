CRASH_DATA_URL = "https://data.cityofchicago.org/resource/85ca-t3if.json"



f_target = ["hit_and_run"]



f_dummy = ["year", "month", "dow", "hour", "beat", "street"]



f_binary = ["intersection_related", "dooring", "work_zone"]



f_cont = [

    "posted_speed_limit",

    "lane_count",

    "num_units",

    "injuries_fatal",

    "injuries_incapacitating",

    "injuries_no_indication",

    "injuries_non_incapacitating",

    "injuries_reported_not_evident",

]



f_categor = [

    "alignment",

    "contributory_cause",

    "device_condition",

    "first_crash_type",

    "most_severe_injury",

    "lighting_condition",

    "road_defect",

    "roadway_surface_cond",

    "traffic_control_device",

    "trafficway_type",

    "weather_condition",

]



f_other = ["longitude", "latitude"]



feature_names = ["i"] + f_target + f_dummy + f_binary + f_cont + f_categor + f_other



weekday_order = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import Image
df = pd.read_csv("../input/crashes_data.csv")

df[feature_names].head().T
df_year = df.query("year == 2018")

crash_count = df_year.groupby("beat")["i"].count()

cmap = sns.cubehelix_palette(as_cmap=True)

fig, ax = plt.subplots()

hue = df_year["beat"].apply(lambda beat: crash_count[beat])

points = ax.scatter(df_year["longitude"], df_year["latitude"], c=hue, s=20, cmap="Reds")

fig.colorbar(points)

fig.set_size_inches(8, 8)

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Count of Traffic Crashes by Police Beat (2018)")

plt.show()
df_year = pd.DataFrame(df.query("year == 2018"))

df_year["has_injuries_fatal"] = (df_year["injuries_fatal"] > 0).astype(int)

fatal_count = df_year.groupby("beat")["has_injuries_fatal"].sum()

cmap = sns.cubehelix_palette(as_cmap=True)

fig, ax = plt.subplots()

hue = df_year["beat"].apply(lambda beat: fatal_count[beat])

points = ax.scatter(df_year["longitude"], df_year["latitude"], c=hue, s=20, cmap="Reds")

fig.colorbar(points)

fig.set_size_inches(8, 8)

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Count of Fatal Traffic Crashes by Police Beat (2018)")

plt.show()
df_year = pd.DataFrame(df.query("year == 2018"))

df_year["has_injuries_fatal"] = (df_year["injuries_fatal"] > 0).astype(int)

fatal_count = df_year.groupby("beat")["has_injuries_fatal"].mean()

cmap = sns.cubehelix_palette(as_cmap=True)

fig, ax = plt.subplots()

hue = df_year["beat"].apply(lambda beat: fatal_count[beat])

points = ax.scatter(df_year["longitude"], df_year["latitude"], c=hue, s=20, cmap="Reds")

fig.colorbar(points)

fig.set_size_inches(8, 8)

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Proportion of Fatal Traffic Crashes by Police Beat (2018)")

plt.show()
df_year = df.query("year == 2018")

crash_count = df_year.groupby("street")["i"].count()

cmap = sns.cubehelix_palette(as_cmap=True)

fig, ax = plt.subplots()

hue = df_year["street"].apply(lambda street: crash_count[street])

points = ax.scatter(df_year["longitude"], df_year["latitude"], c=hue, s=20, cmap=cmap)

fig.colorbar(points)

fig.set_size_inches(8, 8)

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Count of Traffic Crashes by Street (2018)")

plt.show()
df_year = pd.DataFrame(df.query("year == 2018"))

df_year["has_injuries_fatal"] = (df_year["injuries_fatal"] > 0).astype(int)

fatal_count = df_year.groupby("street")["has_injuries_fatal"].sum()

cmap = sns.cubehelix_palette(as_cmap=True)

fig, ax = plt.subplots()

hue = df_year["street"].apply(lambda street: fatal_count[street])

points = ax.scatter(df_year["longitude"], df_year["latitude"], c=hue, s=20, cmap=cmap)

fig.colorbar(points)

fig.set_size_inches(8, 8)

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Count of Fatal Traffic Crashes by Street (2018)")

plt.show()
df_year = pd.DataFrame(df.query("year == 2018"))

df_year["has_injuries_fatal"] = (df_year["injuries_fatal"] > 0).astype(int)

fatal_count = df_year.groupby("street")["has_injuries_fatal"].mean()

cmap = sns.cubehelix_palette(as_cmap=True)

fig, ax = plt.subplots()

hue = df_year["street"].apply(lambda street: fatal_count[street])

points = ax.scatter(df_year["longitude"], df_year["latitude"], c=hue, s=20, cmap=cmap)

fig.colorbar(points)

fig.set_size_inches(8, 8)

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("Proportion of Fatal Traffic Crashes by Street (2018)")

plt.show()
df_month = df.query("year == 2018 and month == 1")

fig, axes = plt.subplots(4, 6, sharex=True, sharey=True)

for h, ax in enumerate(list(axes.reshape(-1, 1))):

    df_hour = df.query("hour == {}".format(h))

    crash_count = df_hour.groupby("beat")["i"].count()

    sns.scatterplot(

        x="longitude", y="latitude",

        hue=df_hour["beat"].apply(lambda beat: crash_count[beat] / len(df_month)),

        s=20, palette="Reds", edgecolor=None,

        data=df_hour, ax=ax[0]

    )

    ax[0].legend().set_visible(False)

    ax[0].set_title("{}:00 {}".format(12 if h % 12 == 0 else h % 12, "AM" if h < 12 else "PM"))

fig.set_size_inches((15, 10))

fig.suptitle("Relative Frequency of Traffic Crashes by Police Beat (January 2018)")

plt.show()
fig, ax = plt.subplots(1, 2)

group_dow_sum = df.groupby("dow")["hit_and_run"].sum()

sns.barplot(

    x=group_dow_sum.index, y=df.groupby("dow")["i"].count().values,

    color="lightgray", ax=ax[0], label="All Crashes"

)

sns.barplot(

    x=group_dow_sum.index, y=group_dow_sum.values,

    palette="Reds", ax=ax[0], label="Hit and Runs"

)

ax[0].set_xticklabels(weekday_order)

ax[0].set_xlabel("Day of Week")

ax[0].set_ylabel("Count")

ax[0].set_ylim((0, df.groupby("dow")["i"].count().max()))

group_dow_mean = df.groupby("dow")["hit_and_run"].mean()

sns.barplot(

    x=group_dow_mean.index, y=np.ones(len(group_dow_mean.index)),

    color="lightgray", ax=ax[1]

)

sns.barplot(

    x=group_dow_mean.index, y=group_dow_mean.values,

    palette="Reds", ax=ax[1]

)

ax[1].set_xticklabels(weekday_order)

ax[1].set_xlabel("Day of Week")

ax[1].set_ylabel("Proportion")

ax[1].set_ylim((0, 1))

fig.suptitle("Hit and Runs by Day of Week", y=0.95)

fig.subplots_adjust(wspace=0.2)

fig.set_size_inches((15, 5))

fig.legend(bbox_to_anchor=(0.5, 0.11))

plt.show()
fig, ax = plt.subplots(1, 2)

group_hour_sum = df.groupby("hour")["hit_and_run"].sum()

sns.barplot(

    x=group_hour_sum.index, y=df.groupby("hour")["i"].count().values,

    color="lightgray", ax=ax[0], label="All Crashes"

)

sns.barplot(

    x=group_hour_sum.index, y=group_hour_sum.values,

    palette="Reds", ax=ax[0], label="Hit and Runs"

)

ax[0].set_xlabel("Hour")

ax[0].set_ylabel("Count")

ax[0].set_ylim((0, df.groupby("hour")["i"].count().max()))

group_hour_mean = df.groupby("hour")["hit_and_run"].mean()

sns.barplot(

    x=group_hour_mean.index, y=np.ones(len(group_hour_mean.index)),

    color="lightgray", ax=ax[1]

)

sns.barplot(

    x=group_hour_mean.index, y=group_hour_mean.values,

    palette="Reds", ax=ax[1]

)

ax[1].set_xlabel("Hour")

ax[1].set_ylabel("Proportion")

ax[1].set_ylim((0, 1))

fig.suptitle("Hit and Runs by Hour of Day", y=0.95)

fig.subplots_adjust(wspace=0.2)

fig.set_size_inches((15, 5))

fig.legend(bbox_to_anchor=(0.5, 0.11))

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
predictors = ["hour", "dow"]

data = pd.DataFrame(df[predictors])

for p in predictors:

    data[p] = data[p].astype(str)

one_hot = pd.get_dummies(data)

one_hot.head(15)
predictors_cont = f_cont

# predictors_cont = ["posted_speed_limit", "injuries_non_incapacitating"]

scaled = pd.DataFrame(StandardScaler().fit_transform(df[predictors_cont].astype(float)), columns=predictors_cont)

scaled.head()
fig, axes = plt.subplots(1, 2, sharey=True)

sns.distplot(df["posted_speed_limit"], kde=False, ax=axes[0], color="red")

axes[0].set_xlabel("Posted Speed Limit (mph)")

axes[0].set_ylabel("Count of Crashes")

sns.distplot(scaled["posted_speed_limit"], kde=False, ax=axes[1], color="blue")

axes[1].set_xlabel("Scaled Posted Speed Limit")

fig.set_size_inches(10, 4)

fig.suptitle("Scaling Posted Speed Limit")

plt.show()
SEED = 0

X = pd.concat([scaled, df[f_binary], one_hot], axis=1)

y = df["hit_and_run"]

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, shuffle=True, random_state=SEED, stratify=df["hit_and_run"]

)

print("Train: N = {0}, P(hit_and_run) = {1:.3f}".format(len(X_train), y_train.mean()))

print("Test:  N = {0}, P(hit_and_run) = {1:.3f}".format(len(X_test), y_test.mean()))
reg = LogisticRegression(solver="saga", class_weight="balanced", max_iter=1000)

reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

print("F1 Score = {0:.3f}".format(f1_score(y_test, y_pred)))

print("Precision = {0:.3f}".format(precision_score(y_test, y_pred)))

print("Recall = {0:.3f}".format(recall_score(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred) / len(y_test)

ax = sns.heatmap(cm, annot=True, cmap="YlGnBu")

plt.xlabel("Predicted")

plt.ylabel("Actual")

ax.invert_xaxis()

ax.invert_yaxis()

plt.show()
tree = DecisionTreeClassifier(class_weight="balanced")

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print("F1 Score = {0:.3f}".format(f1_score(y_test, y_pred)))

print("Precision = {0:.3f}".format(precision_score(y_test, y_pred)))

print("Recall = {0:.3f}".format(recall_score(y_test, y_pred)))

cm = confusion_matrix(y_test, y_pred) / len(y_test)

ax = sns.heatmap(cm, annot=True, cmap="YlGnBu")

plt.xlabel("Predicted")

plt.ylabel("Actual")

ax.invert_xaxis()

ax.invert_yaxis()

plt.show()