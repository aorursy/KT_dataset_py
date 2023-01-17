import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr





%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
import os

print(os.listdir("../input"))

local = 0

if (local):

    train = pd.read_csv("input/train.csv")

    test = pd.read_csv("input/test.csv")

    # os.chdir('C:\\Users\\wjhol\\Documents\\GitHub\\DataScienceCurriculum\\DataScienceCurriculum\\DiamondRegression')

    # print(os.getcwd())

else:

    train = pd.read_csv("../input/train.csv")

    test = pd.read_csv("../input/test.csv")

train.head()
test.head()
# remove the ID and price

train_clean = train.drop(['ID','price'],1)

test_clean = test.drop('ID',1)

test_clean.head()

train_clean.shape
all_data = pd.concat((train_clean[:],test_clean[:]))

all_data["caret_sqroot"] = np.sqrt(all_data["carat"])

all_data["caret_cubtroot"] = all_data.carat ** (1/3)

all_data.shape

matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":train["price"], "log(price + 1)":np.log1p(train["price"])})

prices.hist(bins=40, grid=True)
#log transform the target:

train["logprice"] = np.log1p(train["price"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

print(numeric_feats)


#skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

#skewed_feats = skewed_feats[skewed_feats > 0.75]

#skewed_feats = skewed_feats.index

#print(skewed_feats)



#all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)

all_data.head()
#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())
#creating matrices for sklearn:



print (train.shape[0])

print (all_data.shape[0])

# select the all_data values with the number of rows in the train dataset; test is everything else

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

train_price = train["logprice"]

print(train_price.head())

y = train_price
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression

from sklearn.model_selection import cross_val_score

# http://localhost:8888/notebooks/notebook-Copy3.ipynb#

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)



def rmse (ypred, yval):

    rmseval = np.sqrt((yval - ypred)^2)

    return (rmseval)
lm = LinearRegression()

lm_fit = lm.fit(X_train,y)

lm_pred = np.array(lm_fit.predict (X_train))

#val = rmse(lm_pred,y)

#print (val)

#print(lm_pred.head())

#print (y.head())



print (lm_fit)

print(round(rmse_cv(lm_fit).mean(),4))



# 
predp = pd.DataFrame(lm_pred)
predp.columns = ['x']

predp.set_index ('x')



from scipy.interpolate import interp1d

cdf = predp.sort_values('x').reset_index()

cdf['p'] = cdf.index / float(len(cdf) - 1)

# setup the interpolator using the value as the index

interp = interp1d(cdf['x'], cdf['p'])



# a is the value, b is the percentile

print (cdf.head())

#Now we can see that the two functions are inverses of each other.



print (predp['x'].quantile(0.57))

print (interp(8.0403611))

#interp(0.61167933268395969)

#array(0.57)

print (interp(predp['x'].quantile(0.43)))

#array(0.43)
cdfx
len(x_values)
sz = len(lm_pred)

sz
from sklearn.metrics import r2_score

lm_predict = lm.predict (X_train)

#print (lm_predict)

#print (y)

print(r2_score(lm_predict,y))
plt.scatter(lm_predict,lm_predict - y,c = 'b',s=40,alpha= 1.0)

plt.hlines (y = 0, xmin=5, xmax = 11)

plt.title ('Residuals using training data')

plt.ylabel ('Residuals')
model_ridge = Ridge()
alphas = [0.00001,0.0001, 0.001, 0.005, 0.01, 0.03, 0.05, 0.08, 0.1, .25, .5, 1]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - RMSE vs Alpha")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.min()
model_lasso = LassoCV(alphas = [10, 5,1, 0.1, 0.001, 0.0005],tol = 0.001).fit(X_train, y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

print (coef.sort_values(ascending=False))
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = pd.concat([coef.sort_values().head(12),

                     coef.sort_values().tail(11)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")
#let's look at the residuals as well:

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)



preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")
import xgboost as xgb


dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(X_test)



params = {"max_depth":2, "eta":0.1}

model = xgb.cv(params, dtrain,  num_boost_round=1000, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(n_estimators=1000, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv

model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))

lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})

predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = 0.7*lasso_preds + 0.3*xgb_preds
solution = pd.DataFrame({"ID":test.ID, "price":preds})

solution.to_csv("ridge_sol.csv", index = False)
from keras.layers import Dense

from keras.models import Sequential

from keras.regularizers import l1

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
X_train = StandardScaler().fit_transform(X_train)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state = 3)
X_tr.shape
X_tr
model = Sequential()

#model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))

model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))



model.compile(loss = "mse", optimizer = "adam")
model.summary()
hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val))
pd.Series(model.predict(X_val)[:,0]).hist()