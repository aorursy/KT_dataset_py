# Basic Libraries
import numpy as np
import pandas as pd
import time
import warnings
warnings.simplefilter("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Visualization
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import seaborn as sns

# statiscics
import scipy

# Data preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Grid search
from sklearn.model_selection import GridSearchCV

# StratifiedKFold
from sklearn.model_selection import StratifiedKFold

# Learning curve
from sklearn.model_selection import learning_curve

# Validation curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from scipy import interp

# Dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Classification method
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

# Imbalanced data preprocessing
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

# Validation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# train data
train = pd.read_csv("/kaggle/input/aps-failure-at-scania-trucks-data-set/aps_failure_training_set.csv")
# test data
test = pd.read_csv("/kaggle/input/aps-failure-at-scania-trucks-data-set/aps_failure_test_set.csv")
# datacheck
train.head()
# data size
train.shape
# null value
train.isnull().sum().sum()
# data info
train.dtypes
# train data, Reload 'na'as Null value
train = pd.read_csv("/kaggle/input/aps-failure-at-scania-trucks-data-set/aps_failure_training_set.csv", na_values="na")
# test data
test = pd.read_csv("/kaggle/input/aps-failure-at-scania-trucks-data-set/aps_failure_test_set.csv", na_values="na")

# or train.replace('nan', np.nan) + type change
# null chack
col = train.iloc[:,1:].columns
null_ratio = train.iloc[:,1:].isnull().sum().values / train.shape[0]*100

# Check by visualization
plt.figure(figsize=(20,8))
plt.plot(col, null_ratio)
plt.xlabel("variables")
plt.ylabel("ratio(%)")
plt.xticks(rotation=90, fontsize=10)
plt.title("Null ratio")
plt.legend()
# data type check
train.dtypes
plt.figure(figsize=(10,6))
sns.countplot(train["class"])
plt.title("neg:{0} / pos:{1}".format(train["class"].value_counts()[0], train["class"].value_counts()[1]))
# define function
def class_flg(x):
    if x["class"] == 'pos':
        res = 1
    else:
        res = 0
    return res

train["class"] = train.apply(class_flg, axis=1)
# test data
test["class"] = test.apply(class_flg, axis=1)
# min and max and mean and std
col = train.iloc[:,1:].columns
mean = train.iloc[:,1:].mean()
min_ = train.iloc[:,1:].min()
max_ = train.iloc[:,1:].max()
std = train.iloc[:,1:].std()
# Visualization by plot
plt.figure(figsize=(25,8))

plt.plot(col, mean, linewidth=5, color="blue", label='mean') #mean
plt.fill_between(col, mean+std, mean-std, alpha=0.15, color='green', label='±1σ') # ±1σ
plt.plot(col, min_, linewidth=5, color='blue', linestyle='--', label='max-min') # min
plt.plot(col, max_, linewidth=5, color='blue', linestyle='--') # max
plt.xlabel("variables")
plt.ylabel("values")
plt.yscale("log")
plt.xticks(rotation=90, fontsize=10)
plt.title("variables range")
plt.legend()
train_mean = train.groupby("class").mean().T
train_std = train.groupby("class").std().T

# Visualization by plot
plt.figure(figsize=(25,8))

plt.plot(train_mean.index, train_mean[1], linewidth=5, color="red", label='pos') #mean
plt.fill_between(train_mean.index, train_mean[1]+train_std[1], train_mean[1]-train_std[1], alpha=0.15, color='orange', label='±1σ') # ±1σ

plt.plot(train_mean.index, train_mean[0], linewidth=5, color="blue", label='nag') #mean
plt.fill_between(train_mean.index, train_mean[0]+train_std[0], train_mean[0]-train_std[0], alpha=0.15, color='green', label='±1σ') # ±1σ

plt.xlabel("variables")
plt.ylabel("values")
plt.yscale("log")
plt.xticks(rotation=90, fontsize=10)
plt.title("variables range")
plt.legend()
# Null data are tempolary filled mean value.
matrix = train.iloc[:,1:].iloc[1:].corr()
plt.figure(figsize=(15,15))
sns.heatmap(matrix, vmax=1, vmin=-1, cmap='bwr', square=True, annot=False, center=0, yticklabels=False, xticklabels=False)
# skew
col = train.iloc[:,1:].columns
# Roop, calculate with drop na values.
skew = []
for i in col:
    sk = scipy.stats.skew(train[i].dropna())
    skew.append(sk)
# kurtosis
# Roop, calculate with drop na values.
kurt = []
for i in col:
    ku = scipy.stats.kurtosis(train[i].dropna())
    kurt.append(ku)
# check with graph
fig, ax = plt.subplots(1, 2, figsize=(20,6))
sns.distplot(skew, ax=ax[0], kde=False, bins=100)
ax[0].set_xlabel("Skewness")
ax[0].set_ylabel("Frequency")
ax[0].set_title("Skewness")
sns.distplot(kurt, ax=ax[1], kde=False, bins=100)
ax[1].set_xlabel("Kurtosis")
ax[1].set_ylabel("Frequency")
ax[1].set_title("Kurtosis")
# Data
label = train["class"]
X = train.iloc[:,1:]
col = train.iloc[:,1:].columns

# Scaling
# Create instance
sc = StandardScaler()
# Fitting
sc.fit(X)
# Transform
X_std = sc.fit_transform(X)
## test data
X_test_std = sc.fit_transform(test.iloc[1:])
Y_test = test["class"]
# Create data frame
train_std = pd.DataFrame(X_std, columns=col)
train_std["class"] = label
train_std.head()
# Create test data frame
test_std = pd.DataFrame(X_test_std, columns=test.iloc[1:].columns)
test_std["class"] = Y_test
test_std.head()
# Null value
null_df = pd.DataFrame({"variables":train.iloc[:,1:].columns,
                        "null_ratio":null_ratio})
null_over15_col = null_df[null_df["null_ratio"]>15]["variables"]
# Result columns
null_over15_col
# null ratio df
null_ratio_df = pd.DataFrame({"variables":train_std[null_over15_col].isnull().sum().index, 
                             "null_ratio":train_std[null_over15_col].isnull().sum()/len(train_std)*100}).sort_values(by="null_ratio", ascending=False)

# over 15% null value df
null_df = train_std[null_ratio_df["variables"]]

# distribution check
col = null_df.columns
fig, ax = plt.subplots(4, 7, figsize=(25, 20))
plt.subplots_adjust(hspace=0.5)
for i in range(len(col)):
    if i <= 6:
        sns.distplot(null_df[col[i]], ax=ax[0,i], kde=False)
        ax[0,i].set_title("Null ratio(%):\n{}".format(null_ratio_df[null_ratio_df["variables"]==col[i]]["null_ratio"].values))
    if i > 6 and i <= 13:
        sns.distplot(null_df[col[i]], ax=ax[1,i-7], kde=False)
        ax[1,i-7].set_title("Null ratio(%):\n{}".format(null_ratio_df[null_ratio_df["variables"]==col[i]]["null_ratio"].values))
    if i > 13 and i <= 20:
        sns.distplot(null_df[col[i]], ax=ax[2,i-14], kde=False)
        ax[2,i-14].set_title("Null ratio(%):\n{}".format(null_ratio_df[null_ratio_df["variables"]==col[i]]["null_ratio"].values))
    if i > 20 and i <= 27:
        sns.distplot(null_df[col[i]], ax=ax[3,i-21], kde=False)
        ax[3,i-21].set_title("Null ratio(%):\n{}".format(null_ratio_df[null_ratio_df["variables"]==col[i]]["null_ratio"].values))
# Columns of drop
drop_col = null_over15_col.values
drop_col = np.delete(drop_col, np.where((drop_col == 'bl_000') & (drop_col == 'bk_000')))
# Checking
drop_col
# Drop over15% null columns, and fill mean values
train_std.drop(drop_col, axis=1, inplace=True)

# Roop fill mean
for i in train_std.columns:
    mean = train_std[i].mean()
    train_std[i].fillna(mean, inplace=True)
    
train_std.head()
# Drop over15% null columns, and fill mean values
test_std.drop(drop_col, axis=1, inplace=True)

# Roop fill mean
for i in test_std.columns:
    mean = test_std[i].mean()
    test_std[i].fillna(mean, inplace=True)
    
test_std.head()
train_std.corr().isnull().sum()
# Source of error that occurred during analysis
train_std["cd_000"].var()
train_std.drop("cd_000", axis=1, inplace=True)
# test data
test_std.drop("cd_000", axis=1, inplace=True)
# Difine variables
X = train_std.iloc[:,:-1]
Y = train_std["class"]
# Calculation of distiortions
distortions = []
for i in range(1,20):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=10)
    km.fit(X)
    distortions.append(km.inertia_)
    
# Plotting 
plt.figure(figsize=(10,6))
plt.plot(range(1,20), distortions, marker='o')
plt.xlabel("Number of clusters")
plt.xticks(range(1,20))
plt.ylabel("Distortion")
# Clustering n=8
kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=100, random_state=10)
# Fitting
kmeans.fit(X)
# output
cluster = kmeans.labels_
# test data fit transform and labels
cluster_test = kmeans.fit_predict(test_std.iloc[:,:-1])
eigen_vals = sorted(np.linalg.eigvals(X.corr()), reverse=True)

# plot
fig, ax = plt.subplots(2, 1, figsize=(20,10))
plt.subplots_adjust(hspace=0.4)
ax[0].plot(eigen_vals, 's-')
ax[0].set_xlabel("factor")
ax[0].set_ylabel("eigenvalue")

ax[1].plot(eigen_vals, 's-')
ax[1].set_xlabel("factor")
ax[1].set_ylabel("eigenvalue")
ax[1].set_ylim([0,10])
ax[1].set_title("Scale up")
# Create instance, n=10
pca = PCA(n_components=10)

# Fitting
pca_result = pca.fit_transform(X)
pca_result = pd.DataFrame(pca_result, columns=["pca1","pca2","pca3","pca4","pca5","pca6","pca7","pca8","pca9","pca10"])

pca_result.head()
# Visualization by heatmap
plt.figure(figsize=(20,20))
sns.heatmap(pca.components_.T, vmax=1, vmin=-1, cmap='bwr', square=False, annot=False, center=0, yticklabels=X.columns, xticklabels=pca_result.columns)
plt.xlabel("pca")
plt.ylabel("variables")
# Visualization by plot
x = pca_result["pca1"]
y = pca_result["pca2"]
color = Y

plt.figure(figsize=(10,6))
plt.scatter(x, y, c=color, alpha=0.5)
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar()
# create dataframe
pca_result["class"] = Y

fig, ax = plt.subplots(1,2,figsize=(20,6))
sns.distplot(pca_result[pca_result["class"]==1]["pca1"], label="pos", ax=ax[0])
sns.distplot(pca_result[pca_result["class"]==0]["pca1"], label="neg", ax=ax[0])
ax[0].legend()

sns.distplot(pca_result[pca_result["class"]==1]["pca2"], label="pos", ax=ax[1])
sns.distplot(pca_result[pca_result["class"]==0]["pca2"], label="neg", ax=ax[1])
ax[1].legend()
# create dataframe
pca_result["cluster"] = cluster

# Visualization by plot
x = pca_result["pca1"]
y = pca_result["pca2"]

plt.figure(figsize=(10,6))
plt.scatter(x, y, c=cluster, alpha=0.5, cmap="Set1")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar()
# pivot count 
pivot = pd.pivot_table(data=pca_result, index="class", columns="cluster", values="pca1", aggfunc="count", fill_value=0)
pivot.columns = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
pivot.reset_index()
# Characteristics of each cluster
# Create dataframe
cluster_name = ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]
pca_label = ["pca1","pca2","pca3","pca4","pca5","pca6","pca7","pca8","pca9","pca10"]

cluster_stats = pd.DataFrame({"cluster":range(0,7)})
cluster_pos_mean = pd.merge(cluster_stats, pca_result[pca_result["class"]==1].groupby("cluster").mean()[pca_label].reset_index(),
                            left_on="cluster", right_on="cluster", how="left").drop("cluster", axis=1)
cluster_pos_std = pd.merge(cluster_stats, pca_result[pca_result["class"]==1].groupby("cluster").std()[pca_label].reset_index(),
                            left_on="cluster", right_on="cluster", how="left").drop("cluster", axis=1)
cluster_neg_mean = pd.merge(cluster_stats, pca_result[pca_result["class"]==0].groupby("cluster").mean()[pca_label].reset_index(),
                            left_on="cluster", right_on="cluster", how="left").drop("cluster", axis=1)
cluster_neg_std = pd.merge(cluster_stats, pca_result[pca_result["class"]==0].groupby("cluster").std()[pca_label].reset_index(),
                            left_on="cluster", right_on="cluster", how="left").drop("cluster", axis=1)

# Change name of columns
cluster_pos_mean.index = cluster_name
cluster_pos_std.index = cluster_name
cluster_neg_mean.index = cluster_name
cluster_neg_std.index = cluster_name
# Visualization
fig, ax = plt.subplots(2,5,figsize=(20,10))
plt.subplots_adjust(hspace=0.5, wspace=0.4)

for i in range(len(cluster_pos_mean.columns)):
    if i <5:
        ax[0,i].plot(cluster_pos_mean.index, cluster_pos_mean[pca_label[i]], color="red", label="pos")
        ax[0,i].fill_between(cluster_pos_mean.index, cluster_pos_mean[pca_label[i]]+cluster_pos_std[pca_label[i]],
                             cluster_pos_mean[pca_label[i]]-cluster_pos_std[pca_label[i]], color="orange", alpha=0.3, label="±1σ")
        ax[0,i].plot(cluster_neg_mean.index, cluster_neg_mean[pca_label[i]], color="blue", label="neg")
        ax[0,i].fill_between(cluster_neg_mean.index, cluster_neg_mean[pca_label[i]]+cluster_neg_std[pca_label[i]],
                             cluster_neg_mean[pca_label[i]]-cluster_neg_std[pca_label[i]], color="green", alpha=0.3, label="±1σ")
        ax[0,i].set_title(pca_label[i])
        ax[0,i].set_xlabel("Cluster")
        ax[0,i].set_ylabel("Standarlized values")
        ax[0,i].tick_params(axis='x', labelrotation=90)
        ax[0,i].legend(ncol=2)
    else:
        ax[1,i-5].plot(cluster_pos_mean.index, cluster_pos_mean[pca_label[i]], color="red", label="pos")
        ax[1,i-5].fill_between(cluster_pos_mean.index, cluster_pos_mean[pca_label[i]]+cluster_pos_std[pca_label[i]],
                               cluster_pos_mean[pca_label[i]]-cluster_pos_std[pca_label[i]], color="orange", alpha=0.3, label="±1σ")
        ax[1,i-5].plot(cluster_neg_mean.index, cluster_neg_mean[pca_label[i]], color="blue", label="neg")
        ax[1,i-5].fill_between(cluster_neg_mean.index, cluster_neg_mean[pca_label[i]]+cluster_neg_std[pca_label[i]],
                               cluster_neg_mean[pca_label[i]]-cluster_neg_std[pca_label[i]], color="green", alpha=0.3, label="±1σ")
        ax[1,i-5].set_title(pca_label[i])
        ax[1,i-5].set_xlabel("Cluster")
        ax[1,i-5].set_ylabel("Standarlized values")
        ax[1,i-5].tick_params(axis='x', labelrotation=90)
        ax[1,i-5].legend(ncol=2)
# Difine variables
X = train_std.iloc[:,:-1]
Y = train_std["class"]

# Data preprocessing, oversampling method
# create instance
ros = RandomOverSampler(sampling_strategy="auto", random_state=10)

# train test data split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Apply to data
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
# Create instance
forest  = RandomForestClassifier(n_estimators=10, random_state=10)
# parameters
param_range = [10,15,20]
leaf = [70, 75, 80, 85]
criterion = ["entropy", "gini", "error"]
param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]

# Optimization by Grid search, scoring is f1 score
gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
gs = gs.fit(X_resampled, y_resampled)
print("gs best:%.3f" % gs.best_score_)
print("gs params:{}".format(gs.best_params_))
# Prediction
gs_best = gs.best_estimator_

y_pred = gs_best.predict(X_test)
# Scores
print("Confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))
print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))
print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
# Create best model of random forest classifier
forest  = RandomForestClassifier(n_estimators=10, random_state=10, criterion="entropy", max_depth=15, max_leaf_nodes=60)
forest.fit(X_resampled, y_resampled)

importance = forest.feature_importances_

indices = np.argsort(importance)[::-1]

# Due to the large number of items, only the top 30 were written.
for i in range(30):
    print("%2d) %-*s %f" %(i+1, 10, X.columns[indices[i]], importance[indices[i]]))
# Visualization with paret0 graph
forest_importance1 = pd.DataFrame({})
variables = []
feature_importance1 = []
for i in range(len(indices)):
    col = X.columns[indices[i]]
    impor = importance[indices[i]]
    variables.append(col)
    feature_importance1.append(impor)
forest_importance1["variables"] = variables
forest_importance1["feature_importance1"] = feature_importance1
forest_importance1["feature_importance1"] = forest_importance1["feature_importance1"]*100
forest_importance1["cumsum"] = forest_importance1["feature_importance1"].cumsum()


# Graph
fig, ax1 = plt.subplots(figsize=(20,8))
ax1.bar(forest_importance1["variables"], forest_importance1["feature_importance1"], label="importance")
ax1.grid()
ax1.set_xlabel("variables")
ax1.tick_params(axis='x', rotation=90, labelsize=10)
ax1.set_ylabel("importance(%)")
plt.legend(loc='lower left')
ax2 = ax1.twinx()
ax2.plot(forest_importance1["variables"], forest_importance1["cumsum"], color="red", label="ratio")
ax2.set_ylim([0,110])
ax2.set_ylabel("Ratio(%)")
plt.legend(loc="upper right")
# combine cluster label
train_std["cluster"] = cluster
# Create variable data and label data by removing the cluster data of c1 and c5 (0th and 4th in the label).

# test data
test_std["cluster"] = cluster_test
# Difine variables
X = train_std.query("cluster!=0 & cluster!=4").iloc[:,:-2]
Y = train_std.query("cluster!=0 & cluster!=4")["class"]

# train test data split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Create instance
forest  = RandomForestClassifier(n_estimators=10, random_state=10)

# parameters
param_range = [5, 10,15]
leaf = [60, 65, 70, 75]
criterion = ["entropy", "gini", "error"]
param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]

# Optimization by Grid search, scoring is f1
gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
gs = gs.fit(X_train, y_train)

print("gs best:%.3f" % gs.best_score_)
print("gs params:{}".format(gs.best_params_))
# Prediction
gs_best = gs.best_estimator_

y_pred = gs_best.predict(X_test)

# Scores
print("Confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))
print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))
print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
# Create best model of random forest classifier
forest  = RandomForestClassifier(n_estimators=10, random_state=10, criterion="entropy", max_depth=15, max_leaf_nodes=60)
forest.fit(X_train, y_train)

importance = forest.feature_importances_

indices = np.argsort(importance)[::-1]

# Due to the large number of items, only the top 30 were written.
for i in range(30):
    print("%2d) %-*s %f" %(i+1, 10, X.columns[indices[i]], importance[indices[i]]))
# Visualization with paret0 graph
forest_importance2 = pd.DataFrame({})
variables = []
feature_importance2 = []
for i in range(len(indices)):
    col = X.columns[indices[i]]
    impor = importance[indices[i]]
    variables.append(col)
    feature_importance2.append(impor)
forest_importance2["variables"] = variables
forest_importance2["feature_importance2"] = feature_importance2
forest_importance2["feature_importance2"] = forest_importance2["feature_importance2"]*100
forest_importance2["cumsum"] = forest_importance2["feature_importance2"].cumsum()


# Graph
fig, ax1 = plt.subplots(figsize=(20,8))
ax1.bar(forest_importance2["variables"], forest_importance2["feature_importance2"], label="importance")
ax1.grid()
ax1.set_xlabel("variables")
ax1.tick_params(axis='x', rotation=90, labelsize=10)
ax1.set_ylabel("importance(%)")
plt.legend(loc='lower left')
ax2 = ax1.twinx()
ax2.plot(forest_importance2["variables"], forest_importance2["cumsum"], color="red", label="ratio")
ax2.set_ylim([0,110])
ax2.set_ylabel("Ratio(%)")
plt.legend(loc="upper right")
forest_importance = pd.merge(forest_importance1.drop("cumsum", axis=1), forest_importance2.drop("cumsum", axis=1),
                             left_on="variables", right_on="variables", how='left')
forest_importance
count = []
ratio = []
for i in range(0,11):
    r = i*0.1
    c = forest_importance[(forest_importance["feature_importance1"]<=i*0.1) & (forest_importance["feature_importance2"]<=i*0.1)]["variables"].count()
    count.append(c)
    ratio.append(r)
    
pd.DataFrame({"Both ratio(%)<":ratio,
              "count":count})
# Variables
X = train_std.iloc[:,:-2]
Y = train_std["class"]

# Roop
Threshold = []
col_count = []
accuracy = []
precision = []
recall = []
f1_ = []

for i in range(0,10):
    # Create instance, parameters are default.
    forest  = RandomForestClassifier(n_estimators=10, random_state=10)
    
    # threshold
    thre = "> 0." + str(i) +"%"

    col = forest_importance[(forest_importance["feature_importance1"] >= i*0.1) | (forest_importance["feature_importance2"] >= i*0.1)]["variables"].values
    col_c = len(col)
    # Select data
    X = train_std[col]
    y = train_std["class"]
    # With over sampling
    # Data preprocessing, oversampling method
    # create instance
    ros = RandomOverSampler(sampling_strategy="auto", random_state=10)

    # train test data split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

    # Apply to data
    X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
    
    # Fitting
    forest.fit(X_resampled, y_resampled)
    
    # Prediction
    y_pred = forest.predict(X_test)

    # Scores
    acc = accuracy_score(y_true=y_test, y_pred=y_pred).round(3)
    pre = precision_score(y_true=y_test, y_pred=y_pred).round(3)
    rec = recall_score(y_true=y_test, y_pred=y_pred).round(3)
    f1 = f1_score(y_true=y_test, y_pred=y_pred).round(3)
    
    # list append
    Threshold.append(thre)
    col_count.append(col_c)
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    f1_.append(f1)

# create dataframe
pd.DataFrame({"Threshold":Threshold,
              "Col_count":col_count,
             "accuracy":accuracy,
             "precision":precision,
             "recall":recall,
             "f1_score":f1_})
# Variables
X = train_std.iloc[:,:-2]
Y = train_std["class"]

# Roop
Threshold = []
col_count = []
accuracy = []
precision = []
recall = []
f1_ = []

for i in range(0,10):
    # Create instance, parameters are default.
    forest  = RandomForestClassifier(n_estimators=10, random_state=10)
    
    # threshold
    thre = "> 0." + str(i) +"%"

    col = forest_importance[(forest_importance["feature_importance1"] >= i*0.1) | (forest_importance["feature_importance2"] >= i*0.1)]["variables"].values
    col_c = len(col)
    # Select data
    X = train_std.query("cluster!=0 & cluster!=4")[col]
    y = train_std.query("cluster!=0 & cluster!=4")["class"]

    # train test data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    # Fitting
    forest.fit(X_train, y_train)
    
    # Prediction
    y_pred = forest.predict(X_test)

    # Scores
    acc = accuracy_score(y_true=y_test, y_pred=y_pred).round(3)
    pre = precision_score(y_true=y_test, y_pred=y_pred).round(3)
    rec = recall_score(y_true=y_test, y_pred=y_pred).round(3)
    f1 = f1_score(y_true=y_test, y_pred=y_pred).round(3)
    
    # list append
    Threshold.append(thre)
    col_count.append(col_c)
    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    f1_.append(f1)

# create dataframe
pd.DataFrame({"Threshold":Threshold,
              "Col_count":col_count,
             "accuracy":accuracy,
             "precision":precision,
             "recall":recall,
             "f1_score":f1_})
best_col = forest_importance[(forest_importance["feature_importance1"] >= 0.6) | (forest_importance["feature_importance2"] >= 0.6)]["variables"].values
# Difine variables
X = pca_result.iloc[:,:-2]
Y = pca_result["class"]

# train test data split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Data preprocessing, oversampling method
# create instance
ros = RandomOverSampler(sampling_strategy="auto", random_state=10)

# Apply to data
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
# Create instance
forest  = RandomForestClassifier(n_estimators=10, random_state=10)

# parameters
param_range = [20, 25, 30]
leaf = [85, 90, 95, 100]
criterion = ["entropy", "gini", "error"]
param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]

# Optimization by Grid search, scoring is f1
gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
gs = gs.fit(X_resampled, y_resampled)

print("gs best:%.3f" % gs.best_score_)
print("gs params:{}".format(gs.best_params_))
# Prediction
gs_best = gs.best_estimator_

y_pred = gs_best.predict(X_test)

# Scores
print("Confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))
print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))
print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
# Create best model of random forest classifier
forest  = RandomForestClassifier(n_estimators=10, random_state=10, criterion="entropy", max_depth=15, max_leaf_nodes=60)
forest.fit(X_resampled, y_resampled)

importance = forest.feature_importances_

indices = np.argsort(importance)[::-1]

# Due to the large number of items, only the top 30 were written.
for i in range(10):
    print("%2d) %-*s %f" %(i+1, 10, X.columns[indices[i]], importance[indices[i]]))
# Visualization with paret0 graph
forest_importance = pd.DataFrame({})
variables = []
feature_importance = []
for i in range(len(indices)):
    col = X.columns[indices[i]]
    impor = importance[indices[i]]
    variables.append(col)
    feature_importance.append(impor)
forest_importance["variables"] = variables
forest_importance["feature_importance"] = feature_importance
forest_importance["feature_importance"] = forest_importance["feature_importance"]*100
forest_importance["cumsum"] = forest_importance["feature_importance"].cumsum()


# Graph
fig, ax1 = plt.subplots(figsize=(20,8))
ax1.bar(forest_importance["variables"], forest_importance["feature_importance"], label="importance")
ax1.grid()
ax1.set_xlabel("variables")
ax1.tick_params(axis='x', rotation=90, labelsize=10)
ax1.set_ylabel("importance(%)")
plt.legend(loc='lower left')
ax2 = ax1.twinx()
ax2.plot(forest_importance["variables"], forest_importance["cumsum"], color="red", label="ratio")
ax2.set_ylim([0,110])
ax2.set_ylabel("Ratio(%)")
plt.legend(loc="upper right")
# Difine variables
X = train_std.iloc[:,:-2]
Y = train_std["class"]

# Data preprocessing, oversampling method
# create instance
rus = RandomUnderSampler(sampling_strategy="auto", random_state=10)

# train test data split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Apply to data
X_resampled, y_resampled = rus.fit_sample(X_train, y_train)
# Create instance
forest  = RandomForestClassifier(n_estimators=10, random_state=10)

# parameters
param_range = [5, 10,15,20]
leaf = [60, 65, 70, 75]
criterion = ["entropy", "gini", "error"]
param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]

# Optimization by Grid search, scoring is f1 score
gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
gs = gs.fit(X_resampled, y_resampled)

print("gs best:%.3f" % gs.best_score_)
print("gs params:{}".format(gs.best_params_))
# Prediction
gs_best = gs.best_estimator_

y_pred = gs_best.predict(X_test)

# Scores
print("Confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))
print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))
print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
# Difine variables
X = train_std.iloc[:,:-2]
Y = train_std["class"]

# Data preprocessing, oversampling method
# create instance
smote = SMOTE(sampling_strategy="auto", random_state=10)

# train test data split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Apply to data
X_resampled, y_resampled = smote.fit_sample(X_train, y_train)
# Create instance
forest  = RandomForestClassifier(n_estimators=10, random_state=10)

# parameters
param_range = [10,15,20]
leaf = [80, 85, 90, 95]
criterion = ["entropy", "gini", "error"]
param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]

# Optimization by Grid search, scoring is f1 score
gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
gs = gs.fit(X_resampled, y_resampled)

print("gs best:%.3f" % gs.best_score_)
print("gs params:{}".format(gs.best_params_))
# Prediction
gs_best = gs.best_estimator_

y_pred = gs_best.predict(X_test)

# Scores
print("Confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))
print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))
print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
# Difine variables
X = train_std.iloc[:,:-2]
Y = train_std["class"]

# Data preprocessing, oversampling method
# create instance
nem = NearMiss(sampling_strategy="auto")

# train test data split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Apply to data
X_resampled, y_resampled = nem.fit_sample(X_train, y_train)
# Create instance
forest  = RandomForestClassifier(n_estimators=10, random_state=10)

# parameters
param_range = [5, 10,15,20]
leaf = [60, 65, 70, 75]
criterion = ["entropy", "gini", "error"]
param_grid = [{"max_depth":param_range, "criterion":criterion, "max_leaf_nodes":leaf}]

# Optimization by Grid search, scoring is f1 score
gs = GridSearchCV(estimator=forest, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
gs = gs.fit(X_resampled, y_resampled)

print("gs best:%.3f" % gs.best_score_)
print("gs params:{}".format(gs.best_params_))
# Prediction
gs_best = gs.best_estimator_

y_pred = gs_best.predict(X_test)

# Scores
print("Confusion_matrix = \n", confusion_matrix(y_true=y_test, y_pred=y_pred))
print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))
print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
class k_fold_cross_val:
    def __init__(self, X_train, y_train, estimator, cv):
        self.X_train = X_train
        self.y_train = y_train
        self.estimator = estimator
        self.cv = cv
        
    def cross_val_kfold(self):
        kfold = StratifiedKFold(n_splits=self.cv, random_state=10)
        self.kfold = kfold
        
        scores = []
        for train_idx, test_idx in self.kfold.split(self.X_train, self.y_train):
            self.estimator.fit(self.X_train[train_idx], self.y_train.values[train_idx])
            score = self.estimator.score(self.X_train[test_idx], self.y_train.values[test_idx])
            scores.append(score)
            print("Class: %s, Acc: %.3f" % (np.bincount(self.y_train.values[train_idx]), score))
            self.scores = scores
            
    def score(self):
        scores = cross_val_score(estimator=self.estimator, X=self.X_train, y=self.y_train, cv=self.cv, n_jobs=1)
        print("CV accuracy scores: %s" % self.scores)
        print("CV accuracy: %.3f +/- %.3f" % (np.mean(self.scores), np.std(self.scores)))
        
    def draw_roc_curve(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        
        mean_tpr=0
        mean_fpr=np.linspace(0,1,100)
        plt.figure(figsize=(10,6))
        for train_idx, test_idx in self.kfold.split(self.X_train, self.y_train):
            proba = self.estimator.fit(self.X_train[train_idx], self.y_train.values[train_idx]).predict_proba(self.X_train[test_idx])
            fpr, tpr, thresholds = roc_curve(y_true=self.y_train.values[test_idx], y_score=proba[:,1], pos_label=1)
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=1, label="ROC fold (area=%.2f)" %(roc_auc))
        
        # Line
        plt.plot([0,1], [0,1], linestyle='--', color=(0.6,0.6,0.6), label="random guessing")
        # plot mean of fpr, tpr roc_auc
        mean_tpr /= self.cv
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k--', label="mean ROC (area = %.2f)" % mean_auc, color="blue")
        # Line
        plt.plot([0,0,1], [0,1,1], lw=2, linestyle=':', color="black", label='perfect performance')
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.title("Receiver Operator Characteristic")
        plt.legend()
def draw_learning_curve(estimator, X_train, y_train):
    # learning curve
    train_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=X_train, y=y_train, train_sizes=np.linspace(0.1,1,10), cv=10, n_jobs=1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # plot
    plt.figure(figsize=(10,6))
    # train data
    plt.plot(train_sizes, train_mean, color="blue", marker='o', markersize=5, label='training accuracy')
    plt.fill_between(train_sizes, train_mean+train_std, train_mean-train_std, color="blue", alpha=0.15)
    # val data
    plt.plot(train_sizes, test_mean, color="green", marker='s', linestyle='--', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean+test_std, test_mean-test_std, color="green", alpha=0.15)

    plt.grid()
    plt.xlabel("Number of trainig samples")
    plt.ylabel("Accuracy")
    plt.ylim([0.8,1.0])
    plt.title("Learning curve")
    plt.legend()
def draw_validation_curve(estimator, X_train, y_train, param_name, param_range, xscale):
    # validation curve
    train_scores, test_scores = validation_curve(estimator=estimator, X=X_train, y=y_train, param_name=param_name, param_range=param_range, cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # plot
    plt.figure(figsize=(10,6))
    # train data
    plt.plot(param_range, train_mean, color="blue", marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean+train_std, train_mean-train_std, color="blue", alpha=0.15)
    # val data
    plt.plot(param_range, test_mean, color="green", marker='s', linestyle='--', markersize=5, label='validation accuracy')
    plt.fill_between(param_range, test_mean+test_std, test_mean-test_std, color="green", alpha=0.15)

    plt.grid()
    plt.xlabel("{}".format(param_name))
    if xscale=="log":
        plt.xscale("log")
    else:
        pass
    plt.ylabel("Accuracy")
    plt.ylim([0.8,1.0])
    plt.title("Validation curve")
    plt.legend()
def confmat_roccurve(X_test, y_test, y_pred, estimator):
    # create confusion matrix
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    # visualiazation confusion matrix
    fig, ax = plt.subplots(1,2,figsize=(18,6))
    
    ax[0].matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax[0].text(x=j, y=i, s=confmat[i,j], va="center", ha="center")
            
    ax[0].set_xlabel("predicted label")
    ax[0].set_ylabel("true label")
    ax[0].set_title("confusion matrix")
    # Score
    print("accuracy = %.3f" % accuracy_score(y_true=y_test, y_pred=y_pred))
    print("precision = %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
    print("recall = %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
    print("f1_score = %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
    
    # visualization roc curve
    y_score = estimator.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_score)
    ax[1].plot(fpr, tpr, label="roc curve (area = %.3f)" % auc(fpr, tpr), color="blue")
    ax[1].plot([0,1], [0,1], linestyle='--', color=(0.6,0.6,0.6), label='random')
    ax[1].plot([0,0,1], [0,1,1], linestyle=':', color="black", label='perfect performance')
    ax[1].set_xlabel("false positive rate")
    ax[1].set_ylabel("true positive rate")
    ax[1].set_title("Receiver Operator Characteristic")
    ax[1].legend()
best_col
# Selected parameters, feature importance threshold >=0.6
col = best_col
# Variables
X = train_std[col]
y = train_std["class"]

# train test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Apply over sampling method
ros = RandomOverSampler(sampling_strategy="auto", random_state=10)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)

# Study best parameter by Cross validation
# Instance
lgb_ = lgb.LGBMClassifier()

# prameters
max_depth = [5, 10, 15]
min_samples_leaf = [1,3,5,7]
min_samples_split = [4,6, 8, 10]

param_grid = [{"max_depth":max_depth,
               "min_samples_leaf":min_samples_leaf, "min_samples_split":min_samples_split}]

# Optimization by Grid search
gs = GridSearchCV(estimator=lgb_, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
gs = gs.fit(X_resampled, y_resampled)

print("gs best:%.3f" % gs.best_score_)
print("gs params:{}".format(gs.best_params_))
# best params
gs_best_all = gs.best_estimator_

# Cross validation
cv = k_fold_cross_val(X_resampled.values, y_resampled, gs_best_all, 5)
cv.cross_val_kfold()
# cross val score
cv.score()
# learning curve
draw_learning_curve(gs_best_all, X_resampled, y_resampled)
# validation curve
draw_validation_curve(gs_best_all, X_resampled, y_resampled, "max_depth", param_range, "")
# cv training roc curve
cv.draw_roc_curve(X_resampled, y_resampled)
# test data prediction
y_pred_all = gs_best_all.predict(X_test)

# Confusion matrix and ROC curve
confmat_roccurve(X_test, y_test, y_pred_all, gs_best_all)
# Selected parameters, feature importance threshold >=0.6
col = best_col
# Variables
X = train_std.query("cluster!=0 & cluster!=4")[col]
y = train_std.query("cluster!=0 & cluster!=4")["class"]

# train test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# Apply over sampling method
ros = RandomOverSampler(sampling_strategy="auto", random_state=10)
X_resampled, y_resampled = ros.fit_sample(X_train, y_train)

# Study best parameter by Cross validation
# Instance
lgb_ = lgb.LGBMClassifier()

# prameters
max_depth = [5, 10, 15]
min_samples_leaf = [1,3,5,7]
min_samples_split = [4,6, 8, 10]

param_grid = [{"max_depth":max_depth,
               "min_samples_leaf":min_samples_leaf, "min_samples_split":min_samples_split}]

# Optimization by Grid search
gs = GridSearchCV(estimator=lgb_, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1)
gs = gs.fit(X_resampled, y_resampled)

print("gs best:%.3f" % gs.best_score_)
print("gs params:{}".format(gs.best_params_))
# best params
gs_best_cluster = gs.best_estimator_

# Cross validation
cv = k_fold_cross_val(X_resampled.values, y_resampled, gs_best_cluster, 5)
cv.cross_val_kfold()
# cross val score
cv.score()
# learning curve
draw_learning_curve(gs_best_cluster, X_resampled, y_resampled)
# validation curve
draw_validation_curve(gs_best_cluster, X_resampled, y_resampled, "max_depth", param_range, "")
# cv training roc curve
cv.draw_roc_curve(X_resampled, y_resampled)
# test data prediction
y_pred_cluster = gs_best_cluster.predict(X_test)

# Confusion matrix and ROC curve
confmat_roccurve(X_test, y_test, y_pred_cluster, gs_best_cluster)
# Selected parameters, feature importance threshold >=0.6
col = best_col
# Variables
X_Test = test_std[col]
y_Test = test_std["class"]

# test data prediction
y_Test_pred_all = gs_best_all.predict(X_Test)
# Variables
X_Test_cluster = test_std.query("cluster!=0 & cluster!=4")[col]
y_Test_cluster = test_std.query("cluster!=0 & cluster!=4")["class"]

# test data prediction
y_Test_pred_cluster = gs_best_cluster.predict(X_Test_cluster)
# Scores
# All data prediction
print("-"*30, "all data", "-"*30)
print("Confusion_matrix = \n", confusion_matrix(y_true=y_Test, y_pred=y_Test_pred_all))
print("accuracy = %.3f" % accuracy_score(y_true=y_Test, y_pred=y_Test_pred_all))
print("precision = %.3f" % precision_score(y_true=y_Test, y_pred=y_Test_pred_all))
print("recall = %.3f" % recall_score(y_true=y_Test, y_pred=y_Test_pred_all))
print("f1_score = %.3f" % f1_score(y_true=y_Test, y_pred=y_Test_pred_all))

# with cluster data prediction
print("-"*30, "cluster data", "-"*30)
print("Confusion_matrix = \n", confusion_matrix(y_true=y_Test_cluster, y_pred=y_Test_pred_cluster))
print("accuracy = %.3f" % accuracy_score(y_true=y_Test_cluster, y_pred=y_Test_pred_cluster))
print("precision = %.3f" % precision_score(y_true=y_Test_cluster, y_pred=y_Test_pred_cluster))
print("recall = %.3f" % recall_score(y_true=y_Test_cluster, y_pred=y_Test_pred_cluster))
print("f1_score = %.3f" % f1_score(y_true=y_Test_cluster, y_pred=y_Test_pred_cluster))
