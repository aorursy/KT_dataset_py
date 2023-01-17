import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
train=pd.read_csv("../input/train.csv")


train.head()
train_y=train.SalePrice

predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_x=train[predictor_cols]
my_model=RandomForestRegressor()
my_model.fit(train_x,train_y)
test=pd.read_csv("../input/test.csv")
test_x=test[predictor_cols]
predicted_price=my_model.predict(test_x)
my_submission=pd.DataFrame({'Id':test.Id,'SalePrice':predicted_price})
my_submission.to_csv('submission.csv', index=False)