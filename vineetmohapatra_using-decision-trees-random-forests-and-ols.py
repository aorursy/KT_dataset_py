import warnings
warnings.filterwarnings("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'): 
    for filename in filenames:
        print(os.path.join(dirname,filename))


import numpy as np
import pandas as pd

## reading the insurance dataset

insurance=pd.read_csv("/kaggle/input/insurance/insurance.csv")
insurance.info()
insurance.head()
import matplotlib.pyplot as plt
import seaborn as sns
## checking the null percentage in the columns of the dataset

(insurance.isnull().sum()/len(insurance.index))*100
insurance["smoker"]=insurance["smoker"].apply(lambda x: 1 if x=="yes" else 0)
## Creating males and females dataset for predicting charges as per the gender

male_cost=insurance[insurance["sex"]=="male"]
female_cost=insurance[insurance["sex"]=="female"]
male_cost.head()
female_cost.head()
males=male_cost.drop(["sex"], axis=1)
females=female_cost.drop(["sex"], axis=1)
male_cost.head()
males.head()
females.head()
males.age.describe()
males["age"]=males["age"].apply(lambda x: "young_adulthood" if x>=18 and x<=35 else "middle_aged" if x>35 and x<=45 else "older_adulthood")
males["age"].value_counts(normalize=True)
females["age"]=females["age"].apply(lambda x: "young_adulthood" if x>=18 and x<=35 else "middle_aged" if x>35 and x<=45 else "older_adulthood")
females["age"].value_counts(normalize=True)
males.head()
males["region"].value_counts(normalize=True)
### Creating dummies for categorical variables
dummy_males = pd.get_dummies(males[["age","region"]], drop_first=True)

# Adding the results to the master dataframe
males = pd.concat([males, dummy_males], axis=1)
males.head()
males.drop(columns=["age","region"], inplace=True)
males.head()
dummy_females = pd.get_dummies(females[["age","region"]], drop_first=True)

# Adding the results to the master dataframe
females = pd.concat([females, dummy_females], axis=1)

females.head()
females.drop(columns=["age","region"], inplace=True)
females.head()
from sklearn.model_selection import train_test_split
np.random.seed(0)
df_train,df_test= train_test_split(males, train_size=0.7, random_state=100)

print(df_train.shape)
print(df_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df_train[["charges","bmi"]]=scaler.fit_transform(df_train[["charges","bmi"]])
y_train=df_train[["charges"]]
X_train=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]

df_test[["charges","bmi"]]=scaler.transform(df_test[["charges","bmi"]])
y_test=df_train[["charges"]]
X_test=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]
from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_leaf=20)
X_train.head()
dt.fit(X_train, y_train)
y_test_pred=dt.predict(X_test)
y_test_pred[:10]
from sklearn.metrics import r2_score
r2_score(y_test,y_test_pred)
import statsmodels.api as sm
y_train=df_train[["charges"]]
X_train=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]
y_test=df_test[["charges"]]
X_test=df_test[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]
X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
lr.params
print(lr.summary()) ## We will have to drop "region_northwest" as the p-value lies in critical zone and needs to be rejected
X_train=X_train.drop(columns=["region_northwest"])
X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
print(lr.summary())                 ## We will have to drop "region_southeast" as the p-value lies in critical zone and needs to be rejected
X_train=X_train.drop(columns=["region_southeast"])
X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
print(lr.summary())      ## We will have to drop "region_southwest" as the p-value lies in critical zone and needs to be rejected
X_train=X_train.drop(columns=["region_southwest"])
X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
print(lr.summary())   ## We will have to drop "children" as the p-value lies in critical zone and needs to be rejected
X_train=X_train.drop(columns=["children"])
X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
print(lr.summary())    
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_test=X_test[X_train.columns]
X_test_lm=sm.add_constant(X_test)

y_test_pred=lr.predict(X_test_lm)
y_test_pred[:5]
r2_score(y_test,y_test_pred)
y_test[:5]
from sklearn.model_selection import GridSearchCV
dt=DecisionTreeRegressor()
params={"max_depth":[2,3,4],"min_samples_leaf":[5,10,15,20,25]}
grid_search=GridSearchCV(estimator=dt, param_grid=params,cv=4,n_jobs=-1,verbose=1, scoring="r2")

y_train=df_train[["charges"]]
X_train=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]
%%time
grid_search.fit(X_train,y_train)
grid_search.best_score_
dt_best=grid_search.best_estimator_
dt_best
y_test=df_train[["charges"]]
X_test=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]
y_test_pred=dt_best.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_test_pred)
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(random_state=42, n_jobs=-1,max_depth=5)
rf.fit(X_train,y_train)
rf.feature_importances_
imp_df=pd.DataFrame({"var_name":X_train.columns,"Importance":rf.feature_importances_})
imp_df.sort_values(by="Importance", ascending=False)
y_test_pred=rf.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_test_pred)
scores=[78.64, 87.18, 88.35, 90.86]
df_males=pd.DataFrame({"model":["Linear Regression","Decision Tree","cross validation using decision trees","random forests"],"r2_scores":scores})
df_males
np.random.seed(0)
df_train,df_test= train_test_split(females, train_size=0.7, random_state=100)
print(df_train.shape)
print(df_test.shape)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df_train[["charges","bmi"]]=scaler.fit_transform(df_train[["charges","bmi"]])
y_train=df_train[["charges"]]
X_train=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]
df_test[["charges","bmi"]]=scaler.transform(df_test[["charges","bmi"]])
y_test=df_train[["charges"]]
X_test=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]
dt=DecisionTreeRegressor(random_state=42, max_depth=3, min_samples_leaf=15)
dt.fit(X_train, y_train)
y_test_pred=dt.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_test_pred)
X_train=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]
y_train=df_train[["charges"]]
                 
    

X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
print(lr.summary())   ## We will have to drop "region_northwest" as the p-value lies in critical zone and needs to be rejected
X_train=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_southeast","region_southwest"]]
X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
print(lr.summary())    ## We will have to drop "region_southwest" as the p-value lies in critical zone and needs to be rejected
X_train=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_southeast"]]
X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
print(lr.summary())     ## We will have to drop "region_southeast" as the p-value lies in critical zone and needs to be rejected
X_train=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood"]]
X_train_lm=sm.add_constant(X_train)
lr=sm.OLS(y_train,X_train_lm).fit()
print(lr.summary())
X_test=df_test[X_train.columns]
X_test_lm=sm.add_constant(X_test)
y_test_pred=lr.predict(X_test_lm)
y_test=df_test[["charges"]]
from sklearn.metrics import r2_score
r2_score(y_test,y_test_pred)
X_train=df_train[["bmi","children","smoker","age_older_adulthood","age_young_adulthood","region_northwest","region_southeast","region_southwest"]]
y_train=df_train[["charges"]]
dt=DecisionTreeRegressor()
grid_search=GridSearchCV(estimator=dt,param_grid=params,cv=4,n_jobs=-1,verbose=1,scoring="r2")

grid_search.fit(X_train,y_train)


grid_search.best_score_
dt_best=grid_search.best_estimator_
X_test=df_test[X_train.columns]
y_test=df_test[["charges"]]
y_test_pred=dt_best.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_test_pred)
rf=RandomForestRegressor(random_state=42, n_jobs=-1,max_depth=5,n_estimators=100)
rf.fit(X_train,y_train)
rf.feature_importances_
imp_df=pd.DataFrame({"var_name":X_train.columns,"Importance":rf.feature_importances_})
imp_df.sort_values(by="Importance", ascending=False)
y_test_pred=rf.predict(X_test)
r2_score(y_test, y_test_pred)
scores=[62.89, 83.41, 75.35, 75.04]
df_females=pd.DataFrame({"model":["Linear Regression","Decision Tree","cross validation using decision trees","random forests"],"r2_scores":scores})
df_females
prediction_scores=pd.merge(df_males,df_females, on="model")
prediction_scores
prediction_scores=prediction_scores.rename(columns={"r2_scores_x":"r2_scores_males","r2_scores_y":"r2_scores_females"})
prediction_scores.set_index('model')
