import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor



data = pd.read_csv('../input/train.csv')

data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = data.SalePrice

X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)



my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)



my_model = XGBRegressor(n_estimators=1000, 

                        max_depth=5, 

                        learning_rate=0.1, 

                        subsample=0.7, 

                        colsample_bytree=0.8, 

                        colsample_bylevel=0.8, 

                        base_score=train_y.mean(), 

                        random_state=42, seed=42)

hist = my_model.fit(train_X, train_y, 

                    early_stopping_rounds=5, 

                    eval_set=[(test_X, test_y)], eval_metric='rmse', 

                    verbose=100)
test_pred = my_model.predict(test_X)

errors = test_pred - test_y

sorted_errors = np.argsort(abs(errors))

worse_5 = sorted_errors[-5:]

best_5 = sorted_errors[:5]



print(pd.DataFrame({'worse':errors[worse_5]}))

print()

print(pd.DataFrame({'best':errors[best_5]}))
import lime

import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(train_X, feature_names=X.columns, class_names=['SalePrice'], verbose=True, mode='regression')
i = worse_5[0]

print('Error =', errors[i])

exp = explainer.explain_instance(test_X[i], my_model.predict, num_features=10)

exp.show_in_notebook(show_table=True)
i = worse_5[1]

print('Error =', errors[i])

exp = explainer.explain_instance(test_X[i], my_model.predict, num_features=10)

exp.show_in_notebook(show_table=True)
i = best_5[0]

print('Error =', errors[i])

exp = explainer.explain_instance(test_X[i], my_model.predict, num_features=10)

exp.show_in_notebook(show_table=True)
i = best_5[1]

print('Error =', errors[i])

exp = explainer.explain_instance(test_X[i], my_model.predict, num_features=10)

exp.show_in_notebook(show_table=True)