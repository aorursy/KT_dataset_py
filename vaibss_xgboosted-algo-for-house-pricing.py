import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor


# Load data
data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
# X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
X = data.drop(['SalePrice'], axis=1)
# print(X.dtypes.sample(10))
X = pd.get_dummies(X) ##Applying one-hot encoding
# print(data.describe())
# train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)
my_pipeline = make_pipeline(Imputer(), XGBRegressor(n_estimators=1000, learning_rate = 0.1))
# my_pipeline.fit(train_X, train_y).early_stopping_rounds=5
# predictions = my_pipeline.predict(test_X)

scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error', cv=5)
print(scores)

print('Mean Absolute Error %2f' %(-1 * scores.mean()))
# from sklearn.metrics import mean_absolute_error
# print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


"""Rest of the code commented out due to implemenation of 
pipeline in the above lines"""
# my_imputer = Imputer()
# train_X = my_imputer.fit_transform(train_X)
# test_X = my_imputer.transform(test_X)
# # from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

# my_model = XGBRegressor(n_estimators=1000, learning_rate=0.1)
# # my_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1)

# my_model.fit(train_X, train_y, early_stopping_rounds=5, 
#              eval_set=[(test_X, test_y)], verbose = False)
# # my_model.fit(train_X, train_y)
# predictions = my_model.predict(test_X)
# from sklearn.metrics import mean_absolute_error
# print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
