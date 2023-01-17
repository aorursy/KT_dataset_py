import numpy as np
import pandas as pd

data = pd.read_csv('../input/train.tsv', delimiter='\t')
data.head()
data.dropna(axis=0, subset=['price'], inplace=True)
y = data.price
df = data.category_name.str.split('/', expand=True)
df = pd.get_dummies(df)
X = data.drop(['price'], axis=1).select_dtypes(exclude=['object'])
X = pd.concat([X, df], axis=1)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor


my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=10, learning_rate=0.05))
my_pipeline.set_params(xgbregressor__n_jobs=4).fit(X, y)

# make predictions
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(- scores.mean())
# Read the data
train = pd.read_csv('../input/train.tsv', delimiter='\t')
test = pd.read_csv('../input/test.tsv', delimiter='\t')

# Create training predictors data and training target data
train.dropna(axis=0, subset=['price'], inplace=True)
train_y = train.price
train_X = train.drop(['price'], axis=1).select_dtypes(exclude=['object'])
test_X = test.select_dtypes(exclude=['object'])
train_X = pd.get_dummies(train_X)
test_X = pd.get_dummies(test_X)

# Fit
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=30, learning_rate=0.05))
my_pipeline.set_params(xgbregressor__n_jobs=4).fit(train_X, train_y)

# Use the model to make predictions
predicted_prices = my_pipeline.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
my_submission = pd.DataFrame({'test_id': test.test_id, 'price': predicted_prices})
my_submission.to_csv('samplesubmission.csv', index=False)
