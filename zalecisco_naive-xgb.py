import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn import model_selection, preprocessing

import xgboost as xgb

import datetime

#now = datetime.datetime.now()



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

macro = pd.read_csv('../input/macro.csv')

id_test = test.id

train.sample(3)

# Any results you write to the current directory are saved as output.
# refdate = pd.to_datetime("2009-01-01").toordinal()  # Just to make the numbers small



# train["time"] = pd.to_datetime(train["timestamp"]).apply(lambda x: x.toordinal()) - refdate



# test["time"] = pd.to_datetime(train["timestamp"]).apply(lambda x: x.toordinal()) - refdate


y_train = train["price_doc"]

x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)

x_test = test.drop(["id", "timestamp"], axis=1)



for c in x_train.columns:

    if x_train[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_train[c].values)) 

        x_train[c] = lbl.transform(list(x_train[c].values))

        #x_train.drop(c,axis=1,inplace=True)

        

for c in x_test.columns:

    if x_test[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(x_test[c].values)) 

        x_test[c] = lbl.transform(list(x_test[c].values))

        #x_test.drop(c,axis=1,inplace=True)        
x_train.sub_area.value_counts()

x_train.product_type.value_counts()
# Adding sample weights

# OK, this didn't help



# Note that this means my RMSEs on any part of the training set 

#   are not comparable with Reynaldo's

# I'm deliberately downweighting points that are hard to fit, 

#   so RMSE will be lower, but this might not mean better performance 



# million1 = (y_train==1e6)

# million2 = (y_train==2e6)

# million3 = (y_train==3e6)

# owner_occ = (x_train.product_type==1)

# nek = (x_train.sub_area==72) # Nekrasovka

# weights = 5 - 3*million1 - 2*million2 - 1*million3 + owner_occ + nek
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.75,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



#dtrain = xgb.DMatrix(x_train, y_train, weight=weights)

dtrain = xgb.DMatrix(x_train, y_train)

dtest = xgb.DMatrix(x_test)
# OK, put this back.  Maybe we needed it to get a random seed right?



# Why waste time with this?  We know it came up with a lucky guess,

#   so just remember the guess



cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)

cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()
num_boost_rounds = len(cv_output)

#num_boost_rounds = 384

# but it used to be 455. Not sure what's going on here.

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

                
fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
print( num_boost_rounds )
y_predict = model.predict(dtest)

y_predict[(y_predict<=1.5e6)&(x_test.product_type==0)] = 1000000

y_predict[(y_predict>1.5e6)&(y_predict<2.5e6)&(x_test.product_type==0)] = 2000000

y_predict[(y_predict>=2.5e6)&(y_predict<3.1e6)&(x_test.product_type==0)] = 3000000
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

output.head()
output.to_csv('xgbSub.csv', index=False)