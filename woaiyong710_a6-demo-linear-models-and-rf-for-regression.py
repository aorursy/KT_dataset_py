import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

from sklearn.metrics.regression import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.linear_model import LinearRegression, LassoCV, Lasso

from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv('../input/winequality-white.csv')
data.head()
data.info()
# y = None # you code here

X=data.drop(['quality'],axis=1)

y=data.quality





X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3,random_state=17) # you code here

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_holdout_scaled = scaler.transform(X_holdout)
linreg = LinearRegression()

linreg.fit(X_train_scaled, y_train)
predict_y_train=linreg.predict(X_train_scaled)

predict_y_test=linreg.predict(X_holdout_scaled)

mes1=mean_squared_error(predict_y_train, y_train)

mes2=mean_squared_error(predict_y_test, y_holdout)

print("Mean squared error (train): %.3f" % mes1)# you code here

print("Mean squared error (test): %.3f" % mes2)# you code here
linreg_coef = pd.DataFrame({'Features':X_train.columns, 'Influence':linreg.coef_}) # you code here

linreg_coef.sort_values('Influence')
lasso1 = Lasso(alpha=0.01, random_state=17) # you code here

lasso1.fit(X_train_scaled, y_train) # you code here
lasso1_coef = pd.DataFrame({'Features':X_train.columns, 'Information':lasso1.coef_}) # you code here

lasso1_coef.sort_values('Information') # you code here
alphas = np.logspace(-6, 2, 200)

lasso_cv = LassoCV(random_state=17, alphas=alphas, cv=5) # you code here

lasso_cv.fit(X_train_scaled, y_train) # you code here
lasso_cv.alpha_
lasso_cv_coef = pd.DataFrame({'Features':X_train.columns, 'Information':lasso_cv.coef_}) # you code here

lasso_cv_coef.sort_values('Information') # you code here
lasso_y_predict=lasso_cv.predict(X_train_scaled)

lasso_mse_train=mean_squared_error(lasso_y_predict,y_train)

print("Mean squared error (train): %.3f" %lasso_mse_train)

lasso_y_predict_2=lasso_cv.predict(X_holdout_scaled)

lasso_mse_test=mean_squared_error(lasso_y_predict_2, y_holdout)

print("Mean squared error (test): %.3f" %lasso_mse_test)
forest = RandomForestRegressor(random_state=17) # you code here

forest.fit(X_train_scaled, y_train) # you code here
rf_y_predict=forest.predict(X_train_scaled)

rf_mse1=mean_squared_error(rf_y_predict, y_train)

print("Mean squared error (train): %.3f" %rf_mse1)

cvs = cross_val_score(forest,X=X_train_scaled, y=y_train, scoring='neg_mean_squared_error')

print("Mean squared error (cv): %.3f" %(-np.mean(cvs)))

rf_y_predict_test=forest.predict(X_holdout_scaled)

rf_mse2=mean_squared_error(rf_y_predict_test, y_holdout)

print("Mean squared error (test): %.3f" %rf_mse2) # you code here
forest_params = {'max_depth': list(range(10, 25)), 

                 'min_samples_leaf': list(range(1, 8)),

                 'max_features': list(range(6,12))}



locally_best_forest = GridSearchCV(forest, param_grid=forest_params,scoring='neg_mean_squared_error')



locally_best_forest.fit(X_train_scaled, y_train) # you code here
locally_best_forest.best_params_, locally_best_forest.best_score_
cvs1=cross_val_score(locally_best_forest.best_estimator_, X=X_train_scaled, y=y_train, cv=5, scoring='neg_mean_squared_error')

print("Mean squared error (cv): %.3f" %(-np.mean(cvs1)))

y_test_predict=locally_best_forest.best_estimator_.predict(X_holdout_scaled)

best_rf_mse=mean_squared_error(y_test_predict, y_holdout)

print("Mean squared error (test): %.3f" %best_rf_mse) # you code here
best_forest=locally_best_forest.best_estimator_

rf_importance = pd.DataFrame({'Feature':X_train.columns, 'Importance':best_forest.feature_importances_}) # you code here

rf_importance.sort_values('Importance') # you code here