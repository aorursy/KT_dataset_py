import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import xgboost as xgb # import xgboost KEY PART

import os

print(os.listdir("../input"))
seed = 888 # keep in constant to ouput same result
df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

df.head()
print(df.shape)
print(df.info())
print("Number of columns: ",len(df.columns))

print(df.columns)
# dropping Serial No. as the index will be not needed here

df.drop(['Serial No.'], axis = 1, inplace = True)

# rename columns to make things easier

df.rename(columns = {'LOR ':'LOR', 'Chance of Admit ':'Chance of Admit'}, inplace = True)

print(df.columns)
df.describe()
# plotting heatmap for easier visuallization of correlation

fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df.corr(), ax=ax, annot=True)

plt.show()
df[df.columns[:-2]].hist(bins = 100,figsize = (12,10))

plt.tight_layout()

plt.show()
# remove spaces in features: it is to prevent error later for plotting XGBoost tree

df.rename(columns = {'GRE Score':'GRE', 'TOEFL Score':'TOEFL', 'University Rating':'UniRating', 'LOR ':'LOR', 'Chance of Admit ':'Chance_of_Admit'}, inplace = True)

print(df.columns)



# seperate data into features and output

X, y = df.iloc[:, :-1], df.iloc[:, -1]
# seperate data into train and validation data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)

# prepare model to train

xg_reg = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.4, learning_rate = 0.1,

                alpha = 0, n_estimators = 10)
# fit & predict

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)
# compute RMSE root mean square error

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE (test): %f" % (rmse))

preds_train = xg_reg.predict(X_train)

# compute RMSE for trained data

rmse = np.sqrt(mean_squared_error(y_train, preds_train))

print("RMSE (train): %f" % (rmse))



# import function for computing r2 score

from sklearn.metrics import r2_score



# compute r2 score

print("r_square score (test): ", r2_score(y_test, preds))

print("r_square score (train): ", r2_score(y_train, preds_train))
print("Some predictions vs real data:")

for i in range(0,100,20):

    print(preds[i], "\t",y_test.iloc[i])
# convert dataset into optimized data structure 'DMatrix'

data_dmatrix = xgb.DMatrix(data = X, label = y)
def gini(actual, pred, cmpcol = 0, sortcol = 1):

    assert( len(actual) == len(pred) )

    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)

    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]

    totalLosses = all[:,0].sum()

    giniSum = all[:,0].cumsum().sum() / totalLosses

    

    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)



def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return [('gini', gini_score)]
params = {'objective':'binary:logistic','colsample_bytree': 0.4, 'subsample': 0.5, 'learning_rate': 0.005,

                'max_depth': 6, 'alpha':0}



cv_results = xgb.cv(dtrain = data_dmatrix, params = params, nfold = 10,

                    num_boost_round = 3000, early_stopping_rounds = 10,

                    metrics = ["rmse"], feval = gini_xgb, as_pandas=True, seed = seed)
cv_results.head()
cv_results.tail()
# prepare model to train

xg_reg_v2 = xgb.XGBRegressor(objective ='binary:logistic', colsample_bytree = 0.4, subsample = 0.5, learning_rate = 0.005,

                alpha = 0, n_estimators = 1087, eval_metric= ["rmse", "auc"])

# fit & predict

xg_reg_v2.fit(X_train,y_train)

# predict results for later comparisons

preds_v2 = xg_reg_v2.predict(X_test)
for i in range(4):

    xgb.plot_tree(xg_reg_v2,num_trees=i, rankdir='LR')

    plt.rcParams['figure.figsize'] = [30, 30]

plt.show()
xgb.plot_importance(xg_reg_v2)

plt.rcParams['figure.figsize'] = [8, 8]

plt.show()
# compute RMSE root mean square error

rmse = np.sqrt(mean_squared_error(y_test, preds_v2))

print("RMSE(test): %f" % (rmse))

preds_train_v2 = xg_reg_v2.predict(X_train)

# compute RMSE for trained data

rmse = np.sqrt(mean_squared_error(y_train, preds_train_v2))

print("RMSE(train): %f" % (rmse))



from sklearn.metrics import r2_score



print("r_square score (test): ", r2_score(y_test, preds_v2))

print("r_square score (train): ", r2_score(y_train, preds_train_v2))
print("Some predictions vs real data:")

for i in range(0,100,20):

    print(preds_v2[i], "\t",y_test.iloc[i])
blue = plt.scatter(np.arange(len(preds)//2), preds[::2], color = "blue")

green = plt.scatter(np.arange(len(y_test)//2), y_test[::2], color = "green")

red = plt.scatter(np.arange(len(preds_v2)//2), preds_v2[::2], color = "red")

plt.title("Comparison of XGBoost")

plt.xlabel("Index of Candidate")

plt.ylabel("Chance of Admit")

plt.legend((blue, red, green),('XGBv1', 'XGBv2', 'Actual'))

plt.show()
df["Chance of Admit"].plot(kind = 'hist',bins = 300,figsize = (6,6))

plt.title("Chance of Admit")

plt.xlabel("Chance of Admit")

plt.ylabel("Frequency")

plt.show()