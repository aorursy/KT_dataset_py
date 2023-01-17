import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
from time import time
from tqdm import tqdm
%matplotlib inline

sns.set(style="darkgrid")
train = pd.read_csv('../input/hse-practical-ml-1/car_loan_train.csv')
test = pd.read_csv('../input/hse-practical-ml-1/car_loan_test.csv')
data = pd.concat([train, test])
data["Date.of.Birth"] = pd.to_datetime(data["Date.of.Birth"])
data["Date.of.Birth"].loc[data["Date.of.Birth"].dt.year > 2019] = data["Date.of.Birth"].loc[data["Date.of.Birth"].dt.year > 2019] - (pd.to_datetime("2019-01-01") - pd.to_datetime("1919-01-01")) 

data["DisbursalDate"] = pd.to_datetime(data["DisbursalDate"])
data["AVERAGE.ACCT.AGE"] = pd.to_timedelta(data["AVERAGE.ACCT.AGE"].str[0], unit="Y") + pd.to_timedelta(data["AVERAGE.ACCT.AGE"].str[5], unit="M")
data["CREDIT.HISTORY.LENGTH"] = pd.to_timedelta(data["CREDIT.HISTORY.LENGTH"].str[0], unit="Y") + pd.to_timedelta(data["CREDIT.HISTORY.LENGTH"].str[5], unit="M")
data["Employment.Type"] = data["Employment.Type"].fillna("XNA")
categorical = ["branch_id", "supplier_id", "manufacturer_id", "Current_pincode_ID", "State_ID", "Employee_code_ID", "PERFORM_CNS.SCORE.DESCRIPTION", "Employment.Type"]
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
for col in categorical:
    print(col)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
data["Date.of.Birth"] = data["Date.of.Birth"].astype("int") / 10 ** 9
data["DisbursalDate"] = data["DisbursalDate"].astype("int") / 10 ** 9
data["AVERAGE.ACCT.AGE"] = data["AVERAGE.ACCT.AGE"].astype("int") / 10 ** 9
data["CREDIT.HISTORY.LENGTH"] = data["CREDIT.HISTORY.LENGTH"].astype("int") / 10 ** 9
data["age"] = data["DisbursalDate"]  - data["Date.of.Birth"]
br = data["branch_id"].value_counts()
data["branch_id_pr"] = data["branch_id"].replace({i : 999 for i in br[br < 300].index}) #объединить малые группы

s_id = data["supplier_id"].value_counts()
data["supplier_id_pr"] = data["supplier_id"].replace({i : i % 499 for i in s_id.index}) #хэш


data["manufacturer_id_pr"] = data["manufacturer_id"].replace({9 : 7, 8: 7}) #объединить малые группы

pin_id = data["Current_pincode_ID"].value_counts()
data["Current_pincode_ID_pr"] = data["Current_pincode_ID"].replace({i : 10000 for i in pin_id[499:].index}) #объединить малые группы, если код редкий то это хорошо

data["State_ID_pr"] = data["State_ID"].replace({19 : 15, 20: 15, 21:15})

em_id = data["Employee_code_ID"].value_counts()
data["Employee_code_ID_pr"] = data["Employee_code_ID"].replace({i : i % 1103 for i in em_id.index}) #хэш
y_data = data["target"]
data = data.drop(["target","UniqueID", 
                                "branch_id", "supplier_id", "manufacturer_id", "Current_pincode_ID", "State_ID", "Employee_code_ID"
                                ], axis=1)
categorical_pr = ["branch_id_pr", "supplier_id_pr", "manufacturer_id_pr", "Current_pincode_ID_pr", "State_ID_pr", "Employee_code_ID_pr", 'PERFORM_CNS.SCORE.DESCRIPTION',  'Employment.Type']
numeric = list(set(data.columns) - set(categorical_pr))
numeric1 = numeric
for col1 in numeric:
    numeric1 = set(numeric1) - set([col1])
    for col2 in numeric1:
        data[col1 + "+" + col2] = data[col1] + data[col2]
        data[col1 + "-" + col2] = data[col1] - data[col2]
        data[col1 + "*" + col2] = data[col1] * data[col2]
        data[col1 + "/" + col2] = data[col1] / data[col2]
categorical_pr1 = set(categorical_pr)
new_cat = []
for col1 in categorical_pr1:
    categorical_pr1 = categorical_pr1 - set([col1])
    for col2 in set(categorical_pr) - set([col1]):
        data[col1 + "_" + col2] = data[col1].astype("str") + data[col2].astype("str")
        new_cat.append(col1 + "_" + col2)
for col in new_cat:
    print(col)
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
train, test = data[data["target"].notnull()], data[data["target"].isna()]
kf = KFold(n_splits=5, shuffle=True)
auc_new_feature= []
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
best_score = 0
for train_index, test_index in tqdm(kf.split(train)):

    X_train = train.drop("target", axis=1).iloc[train_index]
    X_test = train.drop("target", axis=1).iloc[test_index]
    y_train = train["target"].iloc[train_index]
    y_test = train["target"].iloc[test_index]

    cat = CatBoostClassifier(iterations=10000, learning_rate=0.1, depth=4, eval_metric="AUC", early_stopping_rounds=100, verbose=False, rsm=0.1)
    cat.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], cat_features=set(categorical_pr) | set(new_cat))
    y_pred = cat.predict_proba(X_test)

    plot_train = cat.eval_metrics(Pool(X_train, y_train, cat_features=set(categorical_pr) | set(new_cat)), metrics="AUC")["AUC"]
    plt.plot(plot_train)
    plot_test = cat.eval_metrics(Pool(X_test, y_test, cat_features=set(categorical_pr) | set(new_cat)), metrics="AUC")["AUC"]
    plt.plot(plot_test)
    
    if plot_test[-100] > best_score:
        cat_best = cat

    print("test: ", roc_auc_score(y_test, y_pred[:, 1]), "train: ", plot_train[-100])
    auc_new_feature.append(roc_auc_score(y_test, y_pred[:, 1]))
to_plot = pd.DataFrame(sorted(list(zip(cat_best.feature_importances_, X_train.columns.values)), key=lambda x: x[0], reverse=True), columns=["value", "feature"])
fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (30, 5))
sns.barplot(x="feature", y="value", data=to_plot[:50]);
plt.xticks(rotation=90)
plt.title("Feature importance", fontsize=18)
kf = KFold(n_splits=3, shuffle=True)
auc_new_feature= []
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

cat_now =((set(categorical_pr) | set(new_cat)) & set(to_plot[to_plot["value"] > 0]["feature"]))
best_score = 0
for train_index, test_index in tqdm(kf.split(train)):

    X_train = train.drop("target", axis=1).iloc[train_index][to_plot[to_plot["value"] > 0]["feature"]]
    X_test = train.drop("target", axis=1).iloc[test_index][to_plot[to_plot["value"] > 0]["feature"]]
    y_train = train["target"].iloc[train_index]
    y_test = train["target"].iloc[test_index]

    cat = CatBoostClassifier(iterations=10000, learning_rate=0.1, depth=4, eval_metric="AUC", early_stopping_rounds=100, verbose=False, rsm=0.1)
    cat.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], cat_features=cat_now)
    y_pred = cat.predict_proba(X_test)

    plot_train = cat.eval_metrics(Pool(X_train, y_train, cat_features=cat_now), metrics="AUC")["AUC"]
    plt.plot(plot_train)
    plot_test = cat.eval_metrics(Pool(X_test, y_test, cat_features=cat_now), metrics="AUC")["AUC"]
    plt.plot(plot_test)
    
    if plot_test[-100] > best_score:
        cat_best2 = cat

    print("test: ", roc_auc_score(y_test, y_pred[:, 1]), "train: ", plot_train[-100])
    auc_new_feature.append(roc_auc_score(y_test, y_pred[:, 1]))
to_plot2 = pd.DataFrame(sorted(list(zip(cat_best2.feature_importances_, X_train.columns.values)), key=lambda x: x[0], reverse=True), columns=["value", "feature"])
fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (30, 5))
sns.barplot(x="feature", y="value", data=to_plot[:50]);
plt.xticks(rotation=90)
plt.title("Feature importance", fontsize=18)
def permutation_feature(df, y_true, model, metric, iterations=1, cols=None):
    if list(cols):
        imp = {col : [] for col in cols}
    else:
        imp = {col : [] for col in df.columns}
    base_pred = model.predict_proba(df)
    base_score = metric(y_true, base_pred[:, 1])
    for col in tqdm(imp.keys()):
        imp[col] = 0
        temp = df[col].values.copy()
        for i in range(iterations):
            df[col] = np.random.permutation(df[col].values)
            y_pred = model.predict_proba(df)
            score = metric(y_true, y_pred[:, 1])
            diff = base_score - score
            imp[col] += diff
            df[col] = temp
        imp[col] = imp[col] / iterations
        print(col, imp[col])
    return imp
res_agg = permutation_feature(X_test, y_test, cat_best2, metric=roc_auc_score, cols=to_plot2[to_plot2["value"] > 0]["feature"].values)
to_plot_agg = pd.DataFrame(sorted(list(res_agg.items()), key=lambda x: x[1], reverse=True), columns=["feature", "value"])
fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (30, 10))
sns.barplot(x="feature", y="value", data=to_plot_agg[:50]);
plt.xticks(rotation=90)
plt.title("Feature importance", fontsize=18)
X_train, X_test, y_train, y_test = train_test_split(train[to_plot_agg[to_plot_agg["value"] > 1e-5]["feature"].values],  train["target"], train_size=0.9)


cat = CatBoostClassifier(iterations=10000, learning_rate=0.1, depth=4, eval_metric="AUC", 
                        l2_leaf_reg=3, early_stopping_rounds=100, verbose=True, rsm=0.1)
cat.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], cat_features=set(cat_now) & set(to_plot_agg[to_plot_agg["value"] > 1e-5]["feature"].values))
y_pred = cat.predict_proba(X_test)


print("test: ", roc_auc_score(y_test, y_pred[:, 1]), "train: ", plot_train[-100])
cat2 = CatBoostClassifier(iterations=10000, learning_rate=0.01, depth=4, eval_metric="AUC", 
                        l2_leaf_reg=3, early_stopping_rounds=100, verbose=True)
cat2.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], cat_features=set(cat_now) & set(to_plot_agg[to_plot_agg["value"] > 1e-5]["feature"].values), init_model=cat)
y_pred = cat.predict_proba(X_test)


print("test: ", roc_auc_score(y_test, y_pred[:, 1]), "train: ", plot_train[-100])
y_pred = cat2.predict_proba(test[to_plot_agg[to_plot_agg["value"] > 1e-5]["feature"].values])
test_new = test[y_pred[:, 1] < 0.05]
test_new["target"] = 0
train_new = pd.concat([train, test_new])
X_train, X_test, y_train, y_test = train_test_split(train_new[to_plot_agg[to_plot_agg["value"] > 1e-5]["feature"].values],  train_new["target"], train_size=0.9)


cat = CatBoostClassifier(iterations=10000, learning_rate=0.1, depth=4, eval_metric="AUC", 
                        l2_leaf_reg=3, early_stopping_rounds=100, verbose=True, rsm=0.1)
cat.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], cat_features=set(cat_now) & set(to_plot_agg[to_plot_agg["value"] > 1e-5]["feature"].values))
y_pred = cat.predict_proba(X_test)


print("test: ", roc_auc_score(y_test, y_pred[:, 1]), "train: ", plot_train[-100])
y_pred = cat.predict_proba(test[to_plot_agg[to_plot_agg["value"] > 1e-5]["feature"].values])
res = pd.DataFrame(range(test.shape[0]), columns=["ID"])
res["Predicted"] = y_pred[:, 1]
res.to_csv("res_catic.csv", index=False)
res = pd.DataFrame(range(test.shape[0]), columns=["ID"])

feature_all = to_plot_agg[to_plot_agg["value"] > 1e-5]["feature"].values
feature_cat = set(cat_now) & set(to_plot_agg[to_plot_agg["value"] > 1e-5]["feature"].values)






kf = KFold(n_splits=10, shuffle=True)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))

i = 0
auc = []
for train_index, test_index in tqdm(kf.split(train)):
    i += 1
    X_train = train[feature_all].iloc[train_index]
    X_test = train[feature_all].iloc[test_index]
    y_train = train["target"].iloc[train_index]
    y_test = train["target"].iloc[test_index]

    cat = CatBoostClassifier(iterations=10000, learning_rate=0.1, depth=4, eval_metric="AUC", early_stopping_rounds=100, verbose=False, rsm=0.1)
    cat.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], cat_features=feature_cat)



    cat2 = CatBoostClassifier(iterations=10000, learning_rate=0.01, depth=4, eval_metric="AUC", 
                        l2_leaf_reg=3, early_stopping_rounds=100, verbose=False)
    cat2.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], cat_features=feature_cat, init_model=cat)

    res[str(i)] = cat2.predict_proba(test[feature_all])[:, 1]

    plot_train = cat2.eval_metrics(Pool(X_train, y_train, cat_features=feature_cat), metrics="AUC")["AUC"]
    plt.plot(plot_train, label="train_" + str(i))
    plot_test = cat2.eval_metrics(Pool(X_test, y_test, cat_features=feature_cat), metrics="AUC")["AUC"]
    plt.plot(plot_test, label="test_" + str(i))

    y_pred = cat2.predict_proba(X_test)
    
    print("test: ", roc_auc_score(y_test, y_pred[:, 1]), "train: ", plot_train[-100])
    auc.append(roc_auc_score(y_test, y_pred[:, 1]))
res["Predicted"] = 0
for i in range(1, 11):
    res["Predicted"] += softmax(auc)[i - 1] * res[str(i)]

res[["ID", "Predicted"]].to_csv("resotto23.csv", index=False)
from scipy.special import softmax
res["Predicted"] = 0
for i in range(1, 11):
    res["Predicted"] += softmax(auc)[i - 1] * res[str(i)]
res[["ID", "Predicted"]].to_csv("resotto.csv", index=False)

