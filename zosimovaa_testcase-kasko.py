# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import *
from scipy.special import boxcox1p

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.preprocessing import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.feature_selection import SelectFromModel

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, recall_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

#Options
random_state = 100
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
warnings.filterwarnings("ignore")




from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve

from xgboost import XGBClassifier
#Добавим необходимые функции

#Расчет показателя Gini  и AUC
def model_score(y_test, y_pred_prob, return_str=False):
    gini = 2*roc_auc_score(y_test, y_pred_prob[:,1:])-1
    auc = roc_auc_score(y_test, y_pred_prob[:,1:])
    if return_str:
        return "Gini: {0:.4} | AUC: {1:.4}".format(gini, auc)
    else:
        return gini, auc

#Функция генерирует отрезки для задания интервалов. Работает в паре с последующей функцией.
def make_groups(ser, num):
    desc = ser.describe()
    return(np.linspace(desc["min"], desc["max"]+1, num+1))
    
#Функция присваивает номер группы (отрезка) по значению.
def set_groups(x, groups):
    for i in range(len(groups)-1):
        if (x >= groups[i])&(x < groups[i+1]):
            return i
    #return -1
    raise ValueError("Value not in group")


def feat_types(df):
    data_describe = df.describe(include="all")
    num_cols = [c for c in df.columns if df[c].dtype.name != 'object']
    cat_cols = [c for c in df.columns if df[c].dtype.name == 'object']
    bin_cols = [c for c in cat_cols if data_describe[c]['unique'] == 2]
    nonbin_cols = [c for c in cat_cols if data_describe[c]['unique'] > 2]
    
    return num_cols, cat_cols, bin_cols, nonbin_cols


def validate_model(models):
    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.008])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    
    for model in models:
        score = cross_val_predict(model, X, y, cv=5, method='predict_proba')
        gini, roc_auc = model_score(y, score)
        fpr, tpr, _ = roc_curve(y,  score[:,1:])
        ax.plot(fpr,tpr, label="{0} \n{1}".format(type(model).__name__, model_score(y, score, return_str=True)))
    
    ax.legend(loc="lower right")
    ax.set_title("ROC")
#data = pd.read_excel("Test case - Regression.xlsx", header=1)
data = pd.read_excel("/kaggle/input/testcatse-kasko-sber/Test case - Regression.xlsx", header=1)
data.drop("Unnamed: 0", axis=1, inplace=True)

#Посмотрим на данные и пропущенные значения
display(data.head())
display(data.info())

#Бинаризуем признак пола
map_Sex = {'Male': 1, 'Female':0}
data.replace({"Gender": map_Sex}, inplace=True)
#Разобьем выборку на группы по признаках возраста и дохода
n_groups = 6
gr = make_groups(data["Age"], n_groups)
data["Age_gr"] = data["Age"].apply(set_groups, args=[gr])

#Add Income groups
gr = make_groups(data["Income"], n_groups)
data["Income_gr"] = data["Income"].apply(set_groups, args=[gr])

data.head(3)
#Оценим распределение мужчин и женщин
ct = pd.crosstab(data["Gender"], data["KASKO_flg"])
display(ct)
#Оценка распределения дохода по возрастным группам в разрезе мужчин и женщин.
groups = np.unique(data["Age_gr"])
fig, ax = plt.subplots(figsize=(15,4), ncols=2)
plt.suptitle("Income distribution per Age groups")
for g in [0,1]:
    ax[g].set_title("Gender: {}".format(g))
    for gr in groups:
        sns.distplot(data.loc[(data["Age_gr"]==gr)&(data["Gender"]==g), "Income"], hist=False, ax=ax[g], label="Age_gr_{}".format(gr))
    ax[g].grid()
fig, ax = plt.subplots(figsize=(24,5), ncols=2)

grouped = data.groupby(["Age_gr", "Gender"]).agg({"KASKO_flg":"mean"})
grouped = grouped.unstack(level=-1)
grouped.columns = [col[0]+"_"+str(col[1]) for col in grouped.columns]

ax[0].set_title("Age groups vs KASKO prob")
ax[0].grid()
ax[0] = sns.lineplot(grouped.index, grouped["KASKO_flg_0"], ax=ax[0], label="Female")
ax[0] = sns.lineplot(grouped.index, grouped["KASKO_flg_1"], ax=ax[0], label="Male")

grouped = data.groupby(["Income_gr", "Gender"]).agg({"KASKO_flg":"mean"})
grouped = grouped.unstack(level=-1)
grouped.columns = [col[0]+"_"+str(col[1]) for col in grouped.columns]

ax[1].set_title("Income groups vs KASKO prob")
ax[1].grid()
ax[1] = sns.lineplot(grouped.index, grouped["KASKO_flg_0"], ax=ax[1], label="Female")
ax[1] = sns.lineplot(grouped.index, grouped["KASKO_flg_1"], ax=ax[1], label="Male")
fig, ax = plt.subplots(figsize=(24,5), ncols=2)
plt.suptitle("Age groups")
for g in [0,1]:
    grouped = data[data["Gender"]==g].groupby(["Age_gr", "Income_gr"]).agg({"KASKO_flg":"mean"})
    grouped = grouped.unstack()
    grouped.columns = ["Income_gr_"+str(col[1]) for col in grouped.columns]
    sns.lineplot(data=grouped, ax=ax[g])
    ax[g].set_title("Gender: {}".format(g))
    ax[g].grid()
fig, ax = plt.subplots(figsize=(24,5), ncols=2)
plt.suptitle("Income groups")
for g in [0,1]:
    grouped = data[data["Gender"]==g].groupby(["Income_gr", "Age_gr"]).agg({"KASKO_flg":"mean"})
    grouped = grouped.unstack()
    grouped.columns = ["Age_group_"+str(col[1]) for col in grouped.columns]
    sns.lineplot(data=grouped, ax=ax[g])
    ax[g].set_title("Gender: {}".format(g))
    ax[g].grid()
    
data2 = data.copy()
data = data2.copy()
data["Gender"] = data["Gender"].astype(object)
data.drop(["Age_gr", "Income_gr"], axis=1, inplace=True)
display(data.head(3))
X = data.drop(["KASKO_flg"], axis=1)
y = data['KASKO_flg']

num_cols, cat_cols, bin_cols, nonbin_cols = feat_types(X)

X_cat=None
if len(nonbin_cols):
    X_cat = pd.get_dummies(X[nonbin_cols])
#X = pd.get_dummies(X, columns=nonbin_cols)

poly = PolynomialFeatures(interaction_only=True, include_bias=True)
X_poly = poly.fit_transform(X[bin_cols + num_cols])
#X = pd.DataFrame(poly.fit_transform(X), index=X.index)

scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = RobustScaler()

X_poly = pd.DataFrame(scaler.fit_transform(X_poly), index=X.index)
#X = pd.DataFrame(scaler.fit_transform(X), index=X.index)

X = pd.concat([X_poly, X_cat], axis=1)
X.head(5)
clf = LogisticRegression(random_state=random_state)
validate_model([clf])
clf = XGBClassifier(seed=random_state)
validate_model([clf])
clf = RandomForestClassifier(random_state=random_state)
validate_model([clf])
def hyperopt_gini(X_, y_, params):
    try:
        Model = globals()[params.pop("model")]
        model = Model(**params)
        score = cross_val_predict(model, X_, y_, cv=5, method='predict_proba')
        return -model_score(y_, score)[0]
    
    except Exception as ex :
        #print(ex)
        return np.inf

def f_model(params):
    global best
    global best_params
    global best_ext_params
    acc = hyperopt_gini(X, y, params.copy())
    if (acc < best):
        best = acc
        best_params = params
        print("new best: {0:.4} {1}".format(best, params))
    return {'loss': acc, 'status': STATUS_OK}


def model_tune(space, X, y, random_state=random_state, iters=10):
    print(space["model"])
    global best
    global best_params
    best, best_params = np.inf, None 
    res = fmin(f_model, space, algo=tpe.suggest, max_evals=iters, rstate=np.random.RandomState(random_state))
    
    Model = globals()[best_params.pop("model")]
    print("\nBest_params: \n", best_params)
    
    model = Model(random_state=random_state, **best_params)
        
    print("--------------------------------------------------")
    return model
space_lr = {
    'model': 'LogisticRegression',
    'penalty': hp.choice('penalty',["l1", "l2"]),
    'C': hp.uniform('C', 0.00001,5),
    'tol': hp.uniform('tol', 0.0001, 0.1),
    'solver': hp.choice('solver',["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
}

space_xgbc = {
        'model': 'XGBClassifier',
        'lambda' : hp.uniform('lambda', 0.0001, 1),

        'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.31, 0.05)),
        'max_depth':        hp.choice('max_depth',        np.arange(2, 5, 1, dtype=int)),
        'min_child_weight': hp.choice('min_child_weight', np.arange(1, 8, 1, dtype=int)),
        'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
        'subsample':        hp.uniform('subsample', 0.6, 1),
        'n_estimators':     130,
        'eval_metric': 'auc',
        'objective': 'binary:logistic',
        'nthread': 4,
        'early_stopping_rounds': 10,
}

space_rf = {
        'model': 'RandomForestClassifier',
    
        'n_estimators':     hp.choice('n_estimators',    np.arange(20, 200, 10)),
        'max_depth':        hp.choice('max_depth',        np.arange(2, 5, 1, dtype=int)),
        'min_samples_split':hp.choice('min_samples_split',  np.arange(2, 5, 1, dtype=int)),
        'min_samples_leaf':hp.choice('min_samples_leaf',  np.arange(1, 5, 1, dtype=int)),
        'n_jobs': 4

}
model_rf = model_tune(space_rf, X, y, iters=20)

validate_model([model_rf])
fitted = []
fitted.append(model_tune(space_lr, X, y, iters=20))
fitted.append(model_tune(space_xgbc, X, y, iters=20))
fitted.append(model_tune(space_rf, X, y, iters=20))

validate_model(fitted)