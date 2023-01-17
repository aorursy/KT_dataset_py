import numpy as np

import pandas as pd

from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import model_selection

import matplotlib.pyplot as plt

from sklearn.linear_model import RidgeCV

from sklearn.preprocessing import StandardScaler

import seaborn as sns

from sklearn.impute import KNNImputer

import missingno as msno

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.linear_model import LinearRegression
hitters = pd.read_csv("../input/hitterss/Hitters.csv")
df = hitters.copy()
df.head()
df.info()
df.isnull().sum()
df[df["Salary"].isnull()].head(10)
df[df["Hits"] < 70]
df[df["Hits"] > 70]
sns.lineplot(x = "Salary",y = "Years",data= df,hue = "League",style = "Division");
df.describe([0.01,0.25,0.75,0.99]).T
f, ax = plt.subplots(figsize= [20,15])

sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap = "coolwarm" )

ax.set_title("Correlation Matrix", fontsize=20)

plt.show()
df.groupby(["League"])["Salary"].mean()
msno.matrix(df);
for i in ["Hits","HmRun","Runs","RBI","Walks","Years","CAtBat","CHits","CHmRun","CRuns","CRBI","CWalks","PutOuts","Assists","Errors","Salary","AtBat"]:



    Q1 = df[i].quantile(0.25)

    Q3 = df[i].quantile(0.75)

    IQR = Q3-Q1

    upper = Q3 + 1.5*IQR

    lower = Q1 - 1.5*IQR



    if df[(df[i] > upper) | (df[i] < lower)].any(axis=None):

        print(i,"yes")

        print(df[(df[i] > upper) | (df[i] < lower)].shape[0])

    else:

        print(i, "no")
df = pd.get_dummies(df, columns =["League","Division","NewLeague"], drop_first = True)

cols = df.columns
cols
imputer = KNNImputer(n_neighbors=6)

df_filled = imputer.fit_transform(df)
df.shape
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors= 20,contamination= 0.1)
clf.fit_predict(df_filled)
df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:30]
th = np.sort(df_scores)[8]

th
outlier = df_scores > th
dff = df_filled[df_scores > th]
dff = pd.DataFrame(dff,columns = cols)
dff.shape
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

scaler = StandardScaler()

y = dff["Salary"]

X = dff.drop('Salary', axis=1)
df_ = pd.DataFrame(dff, columns = cols)
dummies = dff[["League_N","Division_W","NewLeague_N"]]

dummies
X = X.drop(["League_N","Division_W","NewLeague_N"],axis = 1)
X.head()
cols = X.columns
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns = cols)
X.head()
X.shape
dummies.shape
X_ = pd.concat([X,dummies],axis = 1)
X_.head()
print("X shape :", X_.shape,"Y shape",y.shape)
X_train, X_test, y_train, y_test = train_test_split(X_, 

                                                    y, 

                                                    test_size = 0.20, random_state = 46)
reg_model = LinearRegression()

reg_model.fit(X_train, y_train)
#train

y_pred = reg_model.predict(X_train)

np.sqrt(mean_squared_error(y_train, y_pred))
#test

y_pred = reg_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
linear_tuned = np.sqrt(np.mean(-cross_val_score(reg_model, X_train, y_train, cv = 10, scoring = "neg_mean_squared_error")))
y_pred = reg_model.predict(X_test)
linear_sc = np.sqrt(mean_squared_error(y_test, y_pred))

linear_sc
from sklearn.linear_model import RidgeCV, LassoCV

from sklearn.linear_model import Ridge, Lasso
ridge_model = Ridge().fit(X_train,y_train)
## train

y_pred = ridge_model.predict(X_train)

np.sqrt(mean_squared_error(y_train, y_pred))

#test

y_pred = ridge_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
ridge_cv = RidgeCV(alphas = alphas, scoring = "neg_mean_squared_error", cv = 10,normalize = True).fit(X_train,y_train)
ridge_cv.alpha_
ridge_tuned = Ridge(alpha =ridge_cv.alpha_).fit(X_train,y_train)
#test

y_pred = ridge_tuned.predict(X_test)

ridge_sc = np.sqrt(mean_squared_error(y_test, y_pred))

ridge_sc
from sklearn.linear_model import Lasso
alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
lasso_model = Lasso().fit(X_train, y_train)
#train

y_pred  = lasso_model.predict(X_train)

np.sqrt(mean_squared_error(y_train,y_pred))
#test

y_pred  = lasso_model.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))
lasso_cv_model = LassoCV(alphas = alphas, cv = 10,max_iter = 10000).fit(X_train, y_train)
lasso_cv_model.alpha_
lasso_tuned = Lasso(alpha =lasso_cv_model.alpha_).fit(X_train,y_train)
y_pred = lasso_tuned.predict(X_test)
lasso_sc =np.sqrt(mean_squared_error(y_test,y_pred))

lasso_sc
from sklearn.linear_model import ElasticNet
enet_model = ElasticNet().fit(X_train, y_train)

y_pred = enet_model.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
enet_params = {"l1_ratio": [0.001,0.01,0.1,0.4,0.5,0.6,0.8,1],

              "alpha":[0.1,0.01,0.001,0.2,0.3,0.5,0.8,0.9,1],

              "max_iter" :[1000,5000,10000]}
from sklearn.model_selection import GridSearchCV
gs_cv_enet = GridSearchCV(enet_model, enet_params, cv = 10,n_jobs = -1).fit(X_train, y_train)
gs_cv_enet.best_params_
enet_tuned = ElasticNet(**gs_cv_enet.best_params_,normalize = True)
enet_tuned = ElasticNet().fit(X_train, y_train)

y_pred = enet_tuned.predict(X_test)

enet_sc = np.sqrt(mean_squared_error(y_test, y_pred))

enet_sc
models = pd.DataFrame({"Model" : ["Linear","Rigde","Lasso","ElasticNET"],

                     "Score" : [linear_sc,ridge_sc,lasso_sc,enet_sc]})
models.sort_values("Score")