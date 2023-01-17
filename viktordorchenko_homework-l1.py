# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import xgboost

import lightgbm

import catboost

import time

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score



df = pd.concat([pd.read_csv(f"../input/telecom-churn-datasets/churn-bigml-{i}.csv") for i in [20, 80]], axis=0, ignore_index=True)

df["Churn"] = df["Churn"].astype(np.int64)



for col in [col for col in df.columns.tolist() if col.endswith("plan")]:

    df[col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)



df.drop(["State", "Account length"], axis=1, inplace=True)

df = pd.get_dummies(df, columns=["Area code"])





print(df.shape, "\n")

df.head()
estimators = ["XGBoost", "LightGBM", "CatBoost"]



X_train, X_test, y_train, y_test = train_test_split(df.drop(["Churn"], axis=1), df["Churn"], test_size=.2, random_state=42)

params = [list(np.arange(100, 1100, 100)), list(np.arange(.05, 0, -.005))]



results = pd.DataFrame(data={})



for idx, name in enumerate(estimators):

    for param in list(zip(*params)):

        t_start = time.time()



        if idx == 0:

            estimator = xgboost.XGBClassifier(n_estimators=param[0], learning_rate=param[1])

        elif idx == 1:

            estimator = lightgbm.LGBMClassifier(n_estimators=param[0], learning_rate=param[1])

        else:

            estimator = catboost.CatBoostClassifier(verbose=False, n_estimators=param[0], learning_rate=param[1])



        estimator.fit(X_train, y_train)

        probas_ = estimator.predict_proba(X_test)[:, 1]

        spent_time = np.around(time.time() - t_start, 4)

        roc_auc = np.around(roc_auc_score(y_test, probas_), 4)



        dct = {"estimator": name,

               "n_estimators": param[0],

               "learning_rate": param[1],

               "spent_time": spent_time,

               "roc_auc": roc_auc}



        results = pd.concat([results, pd.DataFrame(data=dct, index=[0])], axis=0, ignore_index=True)



results.to_csv("/kaggle/working/cpu_results.csv", index=False)

results
results.groupby(["estimator"])["spent_time"].mean().reset_index().sort_values(by="spent_time").reset_index(drop=True).round(4).rename(columns={"spent_time": "mean_spent_time"})