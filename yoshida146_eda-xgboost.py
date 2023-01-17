import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
import catboost
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
input_path = os.path.join(os.getcwd(), "..", "input")
target_col = "isBankrupted"
df_train = pd.read_csv(os.path.join(input_path, "train.csv"), index_col="id")
df_test = pd.read_csv(os.path.join(input_path, "test.csv"), index_col="id")
df_train.head()
df_test.head()
df_train.describe()
print(df_train[target_col].value_counts())
sns.set_style("whitegrid")
sns.countplot(df_train[target_col])
os.makedirs('../files/', exist_ok=True)
plt.savefig('../files/unbalnced.jpg')
df_train.info()
df_train.isnull().any()
df_test.isnull().any()
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median", missing_values=np.nan)),
    ("scaler", StandardScaler()),
])
X = pipe.fit_transform(df_train.drop(target_col, axis=1))
X_test = pipe.transform(df_test)
y = df_train[target_col].values
clfs = [
    xgb.XGBClassifier(),
    lgb.LGBMClassifier(),
    catboost.CatBoostClassifier(verbose=0, task_type="GPU"),
    SVC(gamma="auto"),
    MLPClassifier(),
    GaussianNB(),
#     GaussianProcessClassifier()
]
clf_names = [
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "SVC",
    "NeuralNetwork",
    "NaiveBayes",
#     "Gaussian"
]
scores = np.zeros(len(clfs))
for i in np.arange(len(clfs)):
    print(clf_names[i])
    scores[i] = np.mean(cross_validate(clfs[i], X, y, scoring="roc_auc", cv=5, return_train_score=False)["test_score"])
pd.DataFrame(scores, index=clf_names, columns=["Score"])
clf = xgb.XGBClassifier()
clf.fit(X, y)
predict = clf.predict_proba(X_test)[:, 1]
submit = pd.read_csv("../input/sampleSubmission.csv")
submit["isBankrupted"] = predict
os.makedirs("../output/", exist_ok=True)
submit.to_csv("../output/submit_xgb.csv", index=False)