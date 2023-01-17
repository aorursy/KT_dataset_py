import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.metrics import roc_auc_score # official evaluation score of challenge

from sklearn.metrics import roc_curve

import xgboost as xgb

import lightgbm as lgbm



import warnings

import gc

warnings.filterwarnings("ignore")
def load_data(which="sample", skiprows=160000000):

    dtypes = {

        "ip" : "uint64",

        "app": "uint64",

        "device": "uint64",

        "os": "uint64",

        "channel": "uint64",

        "is_attributed": "uint64"

    }

    if which == "sample":

        data = pd.read_csv("../input/train_sample.csv", dtype=dtypes)

    elif which == "whole":

        data = pd.read_csv("../input/train.csv", names = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed'], skiprows=skiprows, dtype=dtypes)

    return data
train_df = load_data(which="sample") # which = "sample" or "whole"

train_df.head()
train_df.info()
train_df.isnull().sum()/len(train_df.index)*100
train_df = train_df.drop("attributed_time", axis=1)
train_df["click_month"] = pd.to_datetime(train_df["click_time"]).dt.month

train_df["click_day_of_week"] = pd.to_datetime(train_df["click_time"]).dt.dayofweek

train_df["click_hour"] = pd.to_datetime(train_df["click_time"]).dt.hour

train_df["click_year"] = pd.to_datetime(train_df["click_time"]).dt.year

train_df = train_df.drop("click_time", axis=1)
train_df.head()
train_df.click_year.value_counts()
train_df = train_df.drop(["click_year", "click_month"], axis=1)
train_df.head()
cols = ["click_day_of_week", "click_hour"]

train_df[cols] = train_df[cols].astype("uint64")

train_df.info()
train_df_int = train_df.select_dtypes(include=["uint64"])

train_df_int = train_df_int.apply(pd.to_numeric, downcast="unsigned")

train_df_int.info()
train_df = train_df.drop(train_df.dtypes[train_df.dtypes=="uint64"].index, axis=1)

train_df = pd.concat([train_df, train_df_int], axis=1)

train_df.head()
train_df.isnull().sum()
variable_value_counts = train_df["app"].value_counts()

variable_value_quantile = variable_value_counts[variable_value_counts>variable_value_counts.quantile(0.8)]

variable_value_quantile = pd.Series(variable_value_quantile).reset_index(name="count").rename(index=str, columns={"index": "app", "count": "count"})

variable_value_quantile
shortened_data = pd.merge(train_df, variable_value_quantile, on="app", how="inner").drop("count", axis=1)

# Plot app count distribution - for 80 % larger counts

plt.figure(figsize=(18, 8))

sns.countplot(x="app", data=shortened_data)
variable_value_counts = train_df["device"].value_counts()

variable_value_quantile = variable_value_counts[variable_value_counts>variable_value_counts.quantile(0.8)]

variable_value_quantile = pd.Series(variable_value_quantile).reset_index(name="count").rename(index=str, columns={"index": "device", "count": "count"})

variable_value_quantile
shortened_data = pd.merge(train_df, variable_value_quantile, on="device", how="inner").drop("count", axis=1)

# Plot device count distribution - for 80 % larger counts

plt.figure(figsize=(18, 8))

sns.countplot(x="device", data=shortened_data)
variable_value_counts = train_df["os"].value_counts()

variable_value_quantile = variable_value_counts[variable_value_counts>variable_value_counts.quantile(0.8)]

variable_value_quantile = pd.Series(variable_value_quantile).reset_index(name="count").rename(index=str, columns={"index": "os", "count": "count"})

variable_value_quantile
shortened_data = pd.merge(train_df, variable_value_quantile, on="os", how="inner").drop("count", axis=1)

# Plot os count distribution - for 80 % larger counts

plt.figure(figsize=(18, 8))

sns.countplot(x="os", data=shortened_data)
variable_value_counts = train_df["channel"].value_counts()

variable_value_quantile = variable_value_counts[variable_value_counts>variable_value_counts.quantile(0.5)]

variable_value_quantile = pd.Series(variable_value_quantile).reset_index(name="count").rename(index=str, columns={"index": "channel", "count": "count"})

variable_value_quantile
shortened_data = pd.merge(train_df, variable_value_quantile, on="channel", how="inner").drop("count", axis=1)

# Plot os count distribution - for 80 % larger counts

plt.figure(figsize=(18, 8))

sns.countplot(x="channel", data=shortened_data)
train_df.channel.value_counts()[train_df.channel.value_counts() == train_df.channel.value_counts().max()]
perc_attributed = pd.DataFrame((train_df.is_attributed.value_counts()/train_df.is_attributed.value_counts().sum()*100).values, columns=["perc_of_occur[%]"]).reset_index().rename(columns={"index": "is_attributed"})

perc_attributed
print("{:.1f}% of the IPs are unique and {:.1f}% are repetitions.".format(len(train_df.ip.unique())/len(train_df.index)*100, 100-len(train_df.ip.unique())/len(train_df.index)*100))
ip_counts = train_df["ip"].value_counts()

suspicious_ips = pd.DataFrame(ip_counts[ip_counts>50].reset_index(name="count")).rename(columns={"index": "ip"}) # 50 clicks or more

suspicious_ips
suspicious_ips_shortened = pd.merge(train_df, suspicious_ips, on="ip", how="inner").drop("count", axis=1)

# Plot IP count distribution

plt.figure(figsize=(16, 8))

sns.countplot(x="ip", data=suspicious_ips_shortened)
train_df.head()
suspicious_df = train_df.set_index("ip").loc[suspicious_ips.ip.values]

suspicious_df.head()
def mode(x):

    return x.mode()



suspicious_df.groupby(["ip"]).apply(mode)
ip_fraud_count = suspicious_df[suspicious_df["is_attributed"]==0].groupby("ip").size()

ip_fraud_perc = pd.DataFrame(ip_fraud_count/suspicious_df.groupby("ip").size()*100, columns=["Fraud_Percentage[%]"], dtype="float16")

del(ip_fraud_count)

ip_fraud_perc.head()
ip_is_attributed = train_df.groupby(["ip"]).is_attributed.sum()

ip_is_attributed = ip_is_attributed[ip_is_attributed > 0].sort_values(ascending=False).reset_index(name="is_attributed_count")

ip_is_attributed = ip_is_attributed.iloc[:int(0.1*ip_is_attributed.shape[0])]

ip_is_attributed.head()
sns.set(font_scale=1.0)

plt.figure(figsize=(18, 6))

sns.barplot(x="ip", y="is_attributed_count", data=ip_is_attributed)
app_is_attributed = train_df.groupby(["app"]).is_attributed.sum()

app_is_attributed = app_is_attributed[app_is_attributed > 0].sort_values(ascending=False).reset_index(name="is_attributed_count")

app_is_attributed = app_is_attributed.iloc[:int(0.5*app_is_attributed.shape[0])]

app_is_attributed.head()
sns.set(font_scale=1.0)

plt.figure(figsize=(18, 6))

sns.barplot(x="app", y="is_attributed_count", data=app_is_attributed)
os_is_attributed = train_df.groupby(["os"]).is_attributed.sum()

os_is_attributed = os_is_attributed[os_is_attributed > 0].sort_values(ascending=False).reset_index(name="is_attributed_count")

os_is_attributed = os_is_attributed.iloc[:int(0.5*os_is_attributed.shape[0])]

os_is_attributed.head()
sns.set(font_scale=1.0)

plt.figure(figsize=(18, 6))

sns.barplot(x="os", y="is_attributed_count", data=os_is_attributed)
device_is_attributed = train_df.groupby(["device"]).is_attributed.sum()

device_is_attributed = device_is_attributed[device_is_attributed > 0].sort_values(ascending=False).reset_index(name="is_attributed_count")

device_is_attributed = device_is_attributed.iloc[:int(0.5*device_is_attributed.shape[0])]

device_is_attributed.head()
sns.set(font_scale=1.0)

plt.figure(figsize=(18, 6))

sns.barplot(x="device", y="is_attributed_count", data=device_is_attributed)
channel_is_attributed = train_df.groupby(["channel"]).is_attributed.sum()

channel_is_attributed = channel_is_attributed[channel_is_attributed > 0].sort_values(ascending=False).reset_index(name="is_attributed_count")

channel_is_attributed = channel_is_attributed.iloc[:int(0.5*channel_is_attributed.shape[0])]

channel_is_attributed.head()
sns.set(font_scale=1.0)

plt.figure(figsize=(18, 6))

sns.barplot(x="channel", y="is_attributed_count", data=channel_is_attributed)
try:

    

    del channel_is_attributed

    del device_is_attributed

    del os_is_attributed

    del app_is_attributed

    del ip_is_attributed

    

finally:

    

    gc.collect()
train_df.groupby("click_day_of_week").is_attributed.size().plot()

plt.ylabel("Click Counts")

plt.xlabel("Day of week")

plt.xticks(ticks=[0, 1, 2, 3], labels=["Monday", "Tuesday", "Wednesday", "Thursday"])

_ = plt.title("Clicks per Weekday", {"fontsize": 15})
train_df.groupby("click_day_of_week").is_attributed.sum().plot()

plt.ylabel("Download Counts")

plt.xlabel("Day of week")

plt.xticks(ticks=[0, 1, 2, 3], labels=["Monday", "Tuesday", "Wednesday", "Thursday"])

_ = plt.title("Downloads per Weekday", {"fontsize": 15})
train_df.groupby("click_day_of_week").is_attributed.mean().plot()

plt.ylabel("Download Ratio")

plt.xlabel("Day of week")

plt.xticks(ticks=[0, 1, 2, 3], labels=["Monday", "Tuesday", "Wednesday", "Thursday"])

_ = plt.title("Download Ratio per Weekday", {"fontsize": 15})
plt.figure(figsize=(10, 6))

click_hour_attributed = train_df.groupby("click_hour").is_attributed.size()

click_hour_attributed[24] = click_hour_attributed[0]

click_hour_attributed =  click_hour_attributed.drop(0)

click_hour_attributed.plot()

plt.ylabel("Click Counts")

plt.xlabel("Hour")

plt.xticks(ticks=range(1, 25), labels=range(1, 25))

_ = plt.title("Clicks per Hour", {"fontsize": 15})
plt.figure(figsize=(10, 6))

click_hour_attributed = train_df.groupby("click_hour").is_attributed.sum()

click_hour_attributed[24] = click_hour_attributed[0]

click_hour_attributed =  click_hour_attributed.drop(0)

click_hour_attributed.plot()

plt.ylabel("Download Counts")

plt.xlabel("Hour")

plt.xticks(ticks=range(1, 25), labels=range(1, 25))

_ = plt.title("Downloads per Hour", {"fontsize": 15})
plt.figure(figsize=(10, 6))

click_hour_attributed = train_df.groupby("click_hour").is_attributed.mean()

click_hour_attributed[24] = click_hour_attributed[0]

click_hour_attributed =  click_hour_attributed.drop(0)

click_hour_attributed.plot()

plt.ylabel("Download Ratio")

plt.xlabel("Hour")

plt.xticks(ticks=range(1, 25), labels=range(1, 25))

_ = plt.title("Downloads Ratio per Hour", {"fontsize": 15})
day_week_hour_count = train_df.groupby(["click_day_of_week", "click_hour"]).is_attributed.count().reset_index(name="click_count")

day_week_hour_count["index"] = day_week_hour_count["click_day_of_week"].astype(str) + "_" + day_week_hour_count["click_hour"].astype(str)

day_week_hour_count.head()
plt.figure(figsize=(50, 20))

sns.lineplot(x="index", y="click_count", data=day_week_hour_count.loc[:, "click_count":"index"])

plt.xticks(ticks=range(len(day_week_hour_count["index"])), labels=day_week_hour_count["index"])

plt.tick_params(labelsize=25)

plt.title("Clicks per Day of Week per Hour")

sns.set(font_scale=3.0)
plt.figure(figsize=(50, 20))

sns.lineplot(x="index", y="click_count", data=day_week_hour_count.loc[:30, "click_count":"index"])

plt.xticks(ticks=range(len(day_week_hour_count.loc[:30, "index"])), labels=day_week_hour_count.loc[:30, "index"])

plt.tick_params(labelsize=25)

plt.title("Downloaded per Day of Week per Hour [:30]")

sns.set(font_scale=3.0)
plt.figure(figsize=(50, 20))

sns.lineplot(x="index", y="click_count", data=day_week_hour_count.loc[28:, "click_count":"index"])

plt.xticks(ticks=range(len(day_week_hour_count.loc[28:, "index"])), labels=day_week_hour_count.loc[28:, "index"])

plt.tick_params(labelsize=22)

plt.title("Downloaded per Day of Week per Hour [28:]")

sns.set(font_scale=3.0)
day_week_hour_ratio = train_df.groupby(["click_day_of_week", "click_hour"]).is_attributed.mean().reset_index(name="is_attributed_ratio")

day_week_hour_ratio["index"] = day_week_hour_ratio["click_day_of_week"].astype(str) + "_" + day_week_hour_ratio["click_hour"].astype(str)

plt.figure(figsize=(50, 20))

sns.lineplot(x="index", y="is_attributed_ratio", data=day_week_hour_ratio.loc[:, "is_attributed_ratio":"index"])

plt.xticks(ticks=range(len(day_week_hour_ratio["index"])), labels=day_week_hour_ratio["index"])

plt.tick_params(labelsize=25)

plt.title("Downloaded Ratio per Day of Week per Hour")

sns.set(font_scale=1.5)
train_df.head()
new_features = [

    {"op": "mode", "groupby": ["ip"], "select": "os", "agg": lambda x: x.mode() if x.mode() is int else x.mode().max()},

    {"op": "mode", "groupby": ["ip"], "select": "channel", "agg": lambda x: x.mode() if x.mode() is int else x.mode().max()},

    {"op": "mode", "groupby": ["ip"], "select": "device", "agg": lambda x: x.mode() if x.mode() is int else x.mode().max()},

    

    {"op": "count", "groupby": ["ip"], "select": "app", "agg": lambda x: x.count()},

    

    {"op": "mode", "groupby": ["ip", "app", "device"], "select": "os", "agg": lambda x: x.mode() if x.mode() is int else x.mode().max()},

    {"op": "mode", "groupby": ["ip", "app", "device"], "select": "channel", "agg": lambda x: x.mode() if x.mode() is int else x.mode().max()},

    {"op": "mode", "groupby": ["ip", "app", "device", "os"], "select": "channel", "agg": lambda x: x.mode() if x.mode() is int else x.mode().max()}

]

for new_feature in new_features:

    new_feature_name = str(new_feature["op"]) + "_" + str(new_feature["select"]) + "_per_" + '_'.join(new_feature["groupby"])

    new_feature_df = train_df.groupby(new_feature["groupby"])[new_feature["select"]].agg(new_feature["agg"]).reset_index(name=new_feature_name)

    train_df = pd.merge(train_df, new_feature_df, how="inner", on=new_feature["groupby"])

                                                          
train_df.head()
test_df = pd.read_csv("../input/test.csv")
ip_occur_train = len(test_df.ip.value_counts()[train_df.ip.unique()].index)*100/len(test_df.ip.value_counts().index)

print(round(ip_occur_train, 2), "% of the test set IPs have appeared in the train set", round(100 - ip_occur_train, 2), "% are new occurences.")
train_df = train_df.drop("click_day_of_week", axis=1)
def perc_of_train_fea_cat_in_test_data(test_df, features):

    for feature in features:

        test_df_unique = test_df[feature].unique()

        train_df_unique = train_df[feature].unique()

        train_df_unique_len = len(train_df_unique)

        count = 0

        for unique_feature in test_df_unique:

            if unique_feature in train_df_unique:

                count += 1

        perc = round(count / train_df_unique_len * 100, 2)

        print(perc, "% of feature named: " + feature.upper() + "'s categories of the training set have appeared in the test set.")

        

def perc_of_new_fea_in_test(test_df, features):

    for feature in features:

        test_df_unique = test_df[feature].unique()

        test_df_unique_len = len(test_df_unique)

        train_df_unique = train_df[feature].unique()

        count = 0

        for unique_feature in test_df_unique:

            if unique_feature not in train_df_unique:

                count += 1

        perc = round((count / test_df_unique_len) * 100, 2)

        print(perc, "% of categories in feature: " + feature.upper() + " are new occurences in the test set.")
perc_of_train_fea_cat_in_test_data(test_df, test_df.columns[(test_df.columns!="click_id") & (test_df.columns!="click_time")])
perc_of_new_fea_in_test(test_df, test_df.columns[(test_df.columns!="click_id") & (test_df.columns!="click_time")])
# train_df["attributed_time"] = pd.to_datetime(train_df["attributed_time"])

# train_df.attributed_time[~train_df.attributed_time.isnull()]
# train_df["attributed_time_day"] = train_df["attributed_time"].dt.day

# train_df["attributed_time_hour"] = train_df["attributed_time"].dt.hour

# train_df["attributed_time_weekday"] = train_df["attributed_time"].dt.dayofweek

# train_df = train_df.drop("attributed_time", axis=1)

# train_df.head()
train_df.is_attributed.mean()
train_df = train_df.drop("ip", axis=1)
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
KFold = StratifiedKFold(n_splits=int(train_df.shape[0]/10000), shuffle=True)
scale_pos_weight = round(train_df.is_attributed.value_counts()[0]/train_df.is_attributed.value_counts()[1], 2)

scale_pos_weight
param_grid = {"max_depth": [2, 4, 5, 10],

             "learning_rate": [0.0001, 0.001, 0.01],

             "n_estimators": [10, 100, 200],

             }
bst = xgb.XGBModel(objective="binary:logistic", booster="dart",

                  scale_pos_weight=scale_pos_weight, n_jobs=-1)
grid_search = GridSearchCV(estimator=bst,

                        param_grid=param_grid,

                        scoring=make_scorer(roc_auc_score),

                        cv=KFold,

                        verbose=1,

                        return_train_score=True)
grid_search.fit(X=train_df.drop("is_attributed", axis=1), y=train_df["is_attributed"])
xgb_df = pd.DataFrame(grid_search.cv_results_)

xgb_df
print("The best auc score is:", grid_search.best_score_)

print("The best params are:", grid_search.best_params_)
plt.plot(list(range(1, 37)), xgb_df["mean_train_score"], label="Train Score")

plt.plot(list(range(1, 37)), xgb_df["mean_test_score"], label="Test Score")

plt.grid()

plt.xlabel("Param Index")

plt.ylabel("AUC Score")

plt.show()
param_grid = {"max_depth": [2, 4, 5, 10],

             "learning_rate": [0.0001, 0.001, 0.01],

             "n_estimators": [10, 100, 200],

             }
lg = lgbm.LGBMClassifier(objective="binary", scale_pos_weight=scale_pos_weight, n_jobs=-1)
grid_search_lg = GridSearchCV(estimator=lg,

                        param_grid=param_grid,

                        scoring=make_scorer(roc_auc_score),

                        cv=KFold,

                        verbose=1,

                        return_train_score=True)
grid_search_lg.fit(X=train_df.drop("is_attributed", axis=1), y=train_df["is_attributed"])
lg_df = pd.DataFrame(grid_search_lg.cv_results_)

lg_df
print("The best auc score is:", grid_search_lg.best_score_)

print("The best params are:", grid_search_lg.best_params_)
plt.plot(list(range(1, 37)), lg_df["mean_train_score"], label="Train Score")

plt.plot(list(range(1, 37)), lg_df["mean_test_score"], label="Test Score")

plt.grid()

plt.xlabel("Param Index")

plt.ylabel("AUC Score")

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_df.drop("is_attributed", axis=1), train_df["is_attributed"], test_size=.3, shuffle=True)
print("Ratio of 'is_attributed (1)' in train data:", y_train.mean())

print("Ratio of 'is_attributed (1)' in test data:", y_test.mean())
xgb_final = xgb.XGBClassifier(objective="binary:logistic", booster="dart",

                  scale_pos_weight=scale_pos_weight, n_jobs=-1, **grid_search.best_params_)
xgb_final.fit(X_train, y_train)
xgb_predict = xgb_final.predict(X_test)

xgb_proba = xgb_final.predict_proba(X_test)
xgb_predict
xgb_proba
roc_auc_score(y_test, xgb_proba[:, 1])
count = 0

for i, j in zip(y_test.values, xgb_predict):

    if i == j:

        count += 1

print("Accuracy:", count/xgb_predict.shape[0])
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test, xgb_proba[:, 1])
plt.plot(fpr_xgb, tpr_xgb)

plt.grid()

plt.title("XGBoost ROC Curve")

plt.xlabel("False Positive Ratio (FPR)")

plt.ylabel("True Positive Ratio (TPR)")

plt.show()
ROC_df_xgb = pd.DataFrame(data=np.concatenate([thresholds_xgb.reshape(-1, 1), tpr_xgb.reshape(-1, 1), fpr_xgb.reshape(-1, 1)], axis=1), columns=["Threshold", "True Positive Ratio (TPR)", "False Positive Ratio (FPR)"])

ROC_df_xgb.head()
lg_final = lgbm.LGBMClassifier(objective="binary", scale_pos_weight=scale_pos_weight, n_jobs=-1, **grid_search_lg.best_params_)
lg_final.fit(X_train, y_train)
lg_predict = lg_final.predict(X_test)

lg_proba = lg_final.predict_proba(X_test)
lg_predict
lg_proba
roc_auc_score(y_test, lg_proba[:, 1])
count = 0

for i, j in zip(y_test.values, lg_predict):

    if i == j:

        count += 1

print("Accuracy:", count/lg_predict.shape[0])
fpr_lg, tpr_lg, thresholds_lg = roc_curve(y_test, xgb_proba[:, 1])
plt.plot(fpr_lg, tpr_lg)

plt.grid()

plt.title("LightGBM ROC Curve")

plt.xlabel("False Positive Ratio (FPR)")

plt.ylabel("True Positive Ratio (TPR)")

plt.show()
thresholds_lg
ROC_df_lg = pd.DataFrame(data=np.concatenate([thresholds_lg.reshape(-1, 1), tpr_lg.reshape(-1, 1), fpr_lg.reshape(-1, 1)], axis=1), columns=["Threshold", "True Positive Ratio (TPR)", "False Positive Ratio (FPR)"])

ROC_df_lg
try:

    

    del train_df

    del X_train

    del X_test

    del y_train

    del y_test

    del test_df # Delete it because of RAM shortage when working on the whole data; It's gonna be loaded later again after the final model is trained and tested on the whole data.

    del ROC_df_xgb

    del ROC_df_lg

    

except:

    pass

    

finally:

    _ = gc.collect()
train_whole =  load_data(which="whole")

train_whole = train_whole.drop("attributed_time", axis=1)
train_whole.head()
train_whole.info()
int_columns = ["ip", "app", "device", "os", "channel", "is_attributed"]

train_whole[int_columns] = train_whole[int_columns].apply(pd.to_numeric, downcast="unsigned")

train_whole.info()
gc.collect()
import sys

var, obj = None, None

total_size = 0

for var, obj in locals().items():

    print(str(var) + " : " + str(sys.getsizeof(obj)))

    total_size += sys.getsizeof(obj)

print("Total memory usage:", total_size)

del total_size
try:

    del train_df_int

    del variable_value_counts

    del variable_value_quantile

    del shortened_data

    del ip_counts

    del suspicious_ips

    del suspicious_ips_shortened

    del suspicious_df

    del click_hour_attributed

    del day_week_hour_count

    del day_week_hour_ratio

    del new_feature_df

    del xgb_df

    del lg_df

    del xgb_predict

    del lg_predict

    del StratifiedKFold    

finally:

    _ = gc.collect()
import sys

var, obj = None, None

total_size = 0

for var, obj in locals().items():

    print(str(var) + " : " + str(sys.getsizeof(obj)))

    total_size += sys.getsizeof(obj)

print("Total memory usage:", total_size)

del total_size
train_whole["click_month"] = pd.to_datetime(train_whole["click_time"]).dt.month

train_whole["click_day_of_week"] = pd.to_datetime(train_whole["click_time"]).dt.dayofweek

train_whole["click_hour"] = pd.to_datetime(train_whole["click_time"]).dt.hour

train_whole["click_year"] = pd.to_datetime(train_whole["click_time"]).dt.year

train_whole = train_whole.drop("click_time", axis=1)
train_whole.info()
train_whole = train_whole.drop(["click_year", "click_month"], axis=1)
int_columns = ["click_day_of_week", "click_hour"]

train_whole[int_columns] = train_whole[int_columns].apply(pd.to_numeric, downcast="unsigned")

train_whole.info()
for new_feature in new_features:

    new_feature_name = str(new_feature["op"]) + "_" + str(new_feature["select"]) + "_per_" + '_'.join(new_feature["groupby"])

    new_feature_df = train_whole.groupby(new_feature["groupby"])[new_feature["select"]].agg(new_feature["agg"]).reset_index(name=new_feature_name)

    train_whole = pd.merge(train_whole, new_feature_df, how="inner", on=new_feature["groupby"])
train_whole.info()
# float_columns = ["attributed_time_day", "attributed_time_hour", "attributed_time_weekday"]

# train_whole[float_columns] = train_whole[float_columns].apply(pd.to_numeric, downcast="float")

int_columns = ["count_app_per_ip"]

train_whole[int_columns] = train_whole[int_columns].apply(pd.to_numeric, downcast="unsigned")

train_whole.info()
train_whole = train_whole.drop("ip", axis=1)
vars_ = dir()

var_list = []

for var in vars_:

    if not var.startswith("_"):

        var_list.append(var)

        

var_list
var, obj = None, None

total_size = 0

for var, obj in locals().items():

    print(str(var) + " : " + str(sys.getsizeof(obj)) + " Bytes")

    total_size += sys.getsizeof(obj)

print("Total memory usage:", total_size/1000000000, "GB")

del total_size
try:

    del new_feature_df

finally:

    _ = gc.collect()
var, obj = None, None

total_size = 0

for var, obj in locals().items():

    print(str(var) + " : " + str(sys.getsizeof(obj)) + " Bytes")

    total_size += sys.getsizeof(obj)

print("Total memory usage:", total_size/1000000000, "GB")

del total_size
# Not enough memory, maybe some more can be freed in order to perform this operation and test the accuracy of the model trained on almost all the data

# X_train, X_test, y_train, y_test = train_test_split(train_whole.drop("is_attributed", axis=1), train_whole["is_attributed"], test_size=.3, shuffle=True)

# try:

#     del train_whole

# except:

#     pass
# print("Ratio of 'is_attributed' in y_train:", y_train.mean())

# print("Ratio of 'is_attributed' in y_test:", y_test.mean())
# lg_final = lgbm.LGBMClassifier(objective="binary", is_unbalance=True, n_jobs=-1, **grid_search_lg.best_params_).fit(X_train, y_train)

lg_final = lgbm.LGBMClassifier(objective="binary", scale_pos_weight=scale_pos_weight, n_jobs=-1, **grid_search_lg.best_params_).fit(train_whole.drop("is_attributed", axis=1), train_whole["is_attributed"])
try:

    del train_whole

#     del X_train

#     del X_test

#     del y_train

#     del y_test

finally:

    _ = gc.collect()
test_df = pd.read_csv("../input/test.csv")

test_df.head()
test_df.head()
test_df.info()
test_df["click_month"] = pd.to_datetime(test_df["click_time"]).dt.month

test_df["click_day_of_week"] = pd.to_datetime(test_df["click_time"]).dt.dayofweek

test_df["click_hour"] = pd.to_datetime(test_df["click_time"]).dt.hour

test_df["click_year"] = pd.to_datetime(test_df["click_time"]).dt.year

test_df = test_df.drop("click_time", axis=1)

test_df.info()
test_df = test_df.drop(["click_year", "click_month"], axis=1)
test_df = test_df.astype("uint64")

test_df.info()
test_df_int = test_df.select_dtypes(include=["uint64"])

test_df_int = test_df_int.apply(pd.to_numeric, downcast="unsigned")

test_df = test_df.drop(test_df.dtypes[test_df.dtypes=="uint64"].index, axis=1)

test_df = pd.concat([test_df, test_df_int], axis=1)

test_df.info()
for new_feature in new_features:

    new_feature_name = str(new_feature["op"]) + "_" + str(new_feature["select"]) + "_per_" + '_'.join(new_feature["groupby"])

    new_feature_df = test_df.groupby(new_feature["groupby"])[new_feature["select"]].agg(new_feature["agg"]).reset_index(name=new_feature_name)

    test_df = pd.merge(test_df, new_feature_df, how="inner", on=new_feature["groupby"])
try:

    del new_feature_df

    del test_df_int

finally:

    _ = gc.collect()
test_df.info()
int_columns = ["count_app_per_ip"]

test_df[int_columns] = test_df[int_columns].apply(pd.to_numeric, downcast="unsigned")

test_df.info()
test_df.head()
click_id = test_df.click_id

X_test = test_df.drop(["click_id", "ip"], axis=1)
lg_predict = lg_final.predict_proba(X_test)
lg_predict
try:

    del test_df

    del X_test

except:

    pass

finally:

    _ = gc.collect()
results = pd.concat([click_id, pd.Series(lg_predict[:, 1], name="is_attributed")], axis=1)

results.head()
results = results.sort_values(by="click_id", axis=0).reset_index().drop("index", axis=1)

results.head()
results.to_csv("submission_file.csv", sep=',', index=False)