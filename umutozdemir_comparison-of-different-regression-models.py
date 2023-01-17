import pandas as pd

import numpy as np

import os

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt
file = ("../input/winequality-red.csv")



wine = pd.read_csv(file, delimiter = ";")

wine.head()
wine.columns
# Split into Features and Labels



X = wine.drop(['alcohol'], axis = 1)



y = wine[['alcohol']]
# Split into train and test data



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, shuffle = True, random_state = 42)

X_train.head()
# Definition of categorical and numerical attributes



cat_attribs = ['quality']

num_attribs = list(X_train.drop(cat_attribs, axis=1))
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import FeatureUnion

from sklearn.linear_model import LinearRegression



# Since Scikit-Learn doesn't hanldes DataFrame, we build a class for it



class DFSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values



num_pipe = Pipeline([

    ('DFSelector', DFSelector(num_attribs)),

    ('scaler', StandardScaler()) # Feature Scaling



])



cat_pipe = Pipeline([

    ('DFSelector', DFSelector(cat_attribs)),

    ('OneHot', OneHotEncoder(sparse = False)) #OneHotEncoding of Categorical Attributes

])





full_pipeline = FeatureUnion(transformer_list =[

    ("num_pipeline", num_pipe),

    ("cat_pipeline", cat_pipe)

])
# Preprocessing of the training set



X_train_prepared = full_pipeline.fit_transform(X_train)
# Proof that Feature Scaling and OneHotEncoding worked



pd.DataFrame(X_train_prepared).head()
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(X_train_prepared, y_train)
lin_reg.score(X_train_prepared, y_train)



# Thats a promising score, best score is 1
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict = lin_reg.predict(X_train_prepared)



lin_mse = mean_squared_error(y_train, y_predict)

lin_rmse = np.sqrt(lin_mse)

lin_rmse



# Thats a promising RMSE
# Cross Validation



from sklearn.model_selection import cross_val_score





scores = cross_val_score(lin_reg, X_train_prepared, y_train, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



lin_rmse_scores = np.sqrt(-scores)



lin_rmse_scores.mean()



# Thats still a promising RMSE, especially for Validation
# Plotting each feature except quality against y_predict to see if there is any obvious linearity.



f, axarr = plt.subplots(3, 3, sharex='col', sharey='row', figsize = [15,8])



axarr[0, 0].scatter(X_train_prepared[:,0], y_predict, alpha = 0.05)

axarr[0, 0].set_title('fixed acidity')



axarr[0, 1].scatter(X_train_prepared[:,1], y_predict, alpha = 0.05)

axarr[0, 1].set_title('volatile acidity')

axarr[0, 2].scatter(X_train_prepared[:,2], y_predict, alpha = 0.05)

axarr[0, 2].set_title('citric acid')



axarr[1, 0].scatter(X_train_prepared[:,3], y_predict, alpha = 0.05)

axarr[1, 0].set_title('residual sugar')

axarr[1, 1].scatter(X_train_prepared[:,4], y_predict, alpha = 0.05)

axarr[1, 1].set_title('chlorides')

axarr[1, 2].scatter(X_train_prepared[:,5], y_predict, alpha = 0.05)

axarr[1, 2].set_title('free sulfur dioxide')



axarr[2, 0].scatter(X_train_prepared[:,6], y_predict, alpha = 0.05)

axarr[2, 0].set_title('total sulfur dioxide')

axarr[2, 1].scatter(X_train_prepared[:,7], y_predict, alpha = 0.05)

axarr[2, 1].set_title('density')

axarr[2, 2].scatter(X_train_prepared[:,8], y_predict, alpha = 0.05)

axarr[2, 2].set_title('pH')





plt.show()
# Just to proof the graphs



corr_matrix = wine.corr()



corr_matrix["alcohol"].sort_values(ascending=False)
from sklearn.linear_model import Ridge



ridge_reg = Ridge(alpha=0.05, solver="cholesky")

ridge_reg.fit(X_train_prepared, y_train)



ridge_reg.score(X_train_prepared, y_train)
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_ridge = ridge_reg.predict(X_train_prepared)



ridge_mse = mean_squared_error(y_train, y_predict_ridge)

ridge_rmse = np.sqrt(ridge_mse)

ridge_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score





scores_ridge = cross_val_score(ridge_reg, X_train_prepared, y_train, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



ridge_rmse_scores = np.sqrt(-scores_ridge)

ridge_rmse_scores.mean()
from sklearn.linear_model import Lasso



lasso_reg = Lasso(alpha=0.05, random_state = 42)

lasso_reg.fit(X_train_prepared, y_train)



lasso_reg.score(X_train_prepared, y_train)
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_lasso = lasso_reg.predict(X_train_prepared)



lasso_mse = mean_squared_error(y_train, y_predict_lasso)

lasso_rmse = np.sqrt(lasso_mse)

lasso_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score





scores_lasso = cross_val_score(lasso_reg, X_train_prepared, y_train, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



lasso_rmse_scores = np.sqrt(-scores_lasso)

lasso_rmse_scores.mean()
from sklearn.linear_model import ElasticNet



# l1_ratio = 0 = penalty = l2 (Ridge)

# l1_ratio = 1 = panalty = l1 (Lasso)



elastic_reg = ElasticNet(alpha=0.005, l1_ratio = 0.5, random_state = 42)

elastic_reg.fit(X_train_prepared, y_train)



elastic_reg.score(X_train_prepared, y_train)
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_elastic = elastic_reg.predict(X_train_prepared)



elastic_mse = mean_squared_error(y_train, y_predict_elastic)

elastic_rmse = np.sqrt(elastic_mse)

elastic_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score





scores_elastic = cross_val_score(elastic_reg, X_train_prepared, y_train, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



elastic_rmse_scores = np.sqrt(-scores_elastic)

elastic_rmse_scores.mean()
# Transform y_train for cross val score. It works witout, but an error occurs



y_train_rs = y_train.as_matrix()
from sklearn.linear_model import SGDRegressor



sgd_reg = SGDRegressor(penalty=None, max_iter = 1000, random_state = 42)

sgd_reg.fit(X_train_prepared, y_train_rs.ravel())



sgd_reg.score(X_train_prepared, y_train_rs.ravel())
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_sgd = sgd_reg.predict(X_train_prepared)



sgd_mse = mean_squared_error(y_train, y_predict_sgd)

sgd_rmse = np.sqrt(sgd_mse)

sgd_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score





scores_sgd = cross_val_score(sgd_reg, X_train_prepared, y_train_rs.ravel(), cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



sgd_rmse_scores = np.sqrt(-scores_sgd)

sgd_rmse_scores.mean()
from sklearn.linear_model import SGDRegressor



sgd_reg_ridge = SGDRegressor(penalty="l2", max_iter = 1000, random_state = 42)

sgd_reg_ridge.fit(X_train_prepared, y_train_rs.ravel())



sgd_reg_ridge.score(X_train_prepared, y_train_rs.ravel())
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_sgd_ridge = sgd_reg_ridge.predict(X_train_prepared)



sgd_ridge_mse = mean_squared_error(y_train, y_predict_sgd_ridge)

sgd_ridge_rmse = np.sqrt(sgd_mse)

sgd_ridge_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score





scores_sgd_ridge = cross_val_score(sgd_reg_ridge, X_train_prepared, y_train_rs.ravel(), cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



sgd_ridge_rmse_scores = np.sqrt(-scores_sgd_ridge)

sgd_ridge_rmse_scores.mean()
from sklearn.linear_model import SGDRegressor



sgd_reg_lasso = SGDRegressor(penalty="l1", max_iter = 1000, random_state = 42)

sgd_reg_lasso.fit(X_train_prepared, y_train_rs.ravel())



sgd_reg_lasso.score(X_train_prepared, y_train_rs.ravel())
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_sgd_lasso = sgd_reg_lasso.predict(X_train_prepared)



sgd_lasso_mse = mean_squared_error(y_train, y_predict_sgd_lasso)

sgd_lasso_rmse = np.sqrt(sgd_mse)

sgd_lasso_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score





scores_sgd_lasso = cross_val_score(sgd_reg_lasso, X_train_prepared, y_train_rs.ravel(), cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



sgd_ridge_lasso_scores = np.sqrt(-scores_sgd_ridge)

sgd_ridge_lasso_scores.mean()
from sklearn.linear_model import SGDRegressor



sgd_reg_elastic = SGDRegressor(penalty="elasticnet", alpha = 0.005, l1_ratio = 0.5, max_iter = 1000, random_state = 42)

sgd_reg_elastic.fit(X_train_prepared, y_train_rs.ravel())



sgd_reg_elastic.score(X_train_prepared, y_train_rs.ravel())
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_sgd_lasso = sgd_reg_lasso.predict(X_train_prepared)



sgd_lasso_mse = mean_squared_error(y_train, y_predict_sgd_lasso)

sgd_lasso_rmse = np.sqrt(sgd_mse)

sgd_lasso_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score





scores_sgd_lasso = cross_val_score(sgd_reg_lasso, X_train_prepared, y_train_rs.ravel(), cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



sgd_ridge_lasso_scores = np.sqrt(-scores_sgd_ridge)

sgd_ridge_lasso_scores.mean()
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=4, include_bias=False)



X_poly = poly_features.fit_transform(X_train_prepared)
poly_reg = LinearRegression()

poly_reg.fit(X_poly, y_train)
poly_reg.score(X_poly, y_train)
# Before we had 16 Features (n) and 1279 Samples (m)



X_train_prepared.shape
# After adding the Polynomial-Features (degree = 4) we end up with 4844 Features!



X_poly.shape
y_poly_predicted = poly_reg.predict(X_poly)
from sklearn.metrics import mean_squared_error



poly_mse = mean_squared_error(y_train, y_poly_predicted)

poly_rmse = np.sqrt(poly_mse)

poly_rmse
# Try reducing the number of features, increasing the number samples, and decreasing the number of folds 

# (if you are using cross_validation).



# m/f > n^2: 255 > 23464336, which is clearly not given



from sklearn.model_selection import cross_val_score



scores_poly = cross_val_score(poly_reg, X_poly, y_train, cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



poly_rmse_scores = np.sqrt(-scores_poly)
poly_rmse_scores.mean()
poly_features = PolynomialFeatures(degree=2, include_bias=False)



X_poly_2 = poly_features.fit_transform(X_train_prepared)
poly_2_reg = LinearRegression()

poly_2_reg.fit(X_poly_2, y_train)
poly_2_reg.score(X_poly_2, y_train)
y_poly_2_predicted = poly_2_reg.predict(X_poly_2)
from sklearn.metrics import mean_squared_error



poly_2_mse = mean_squared_error(y_train, y_poly_2_predicted)

poly_2_rmse = np.sqrt(poly_2_mse)

poly_2_rmse
from sklearn.model_selection import cross_val_score



scores_poly_2 = cross_val_score(poly_2_reg, X_poly_2, y_train, cv=2, n_jobs=-1, scoring = "neg_mean_squared_error")



poly_2_rmse_scores = np.sqrt(-scores_poly_2)

poly_2_rmse_scores.mean()
from sklearn.svm import LinearSVR # I could have also used SVR with kernel="linear", but LinearSVR is faster



svm_reg_linear = LinearSVR(epsilon = 1, C = 1)



svm_reg_linear.fit(X_train_prepared, y_train_rs.ravel())



svm_reg_linear.score(X_train_prepared, y_train_rs.ravel())
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_svm_linear = svm_reg_linear.predict(X_train_prepared)



svm_linear_mse = mean_squared_error(y_train_rs.ravel(), y_predict_svm_linear)

svm_linear_rmse = np.sqrt(svm_linear_mse)

svm_linear_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score



scores_svm_linear = cross_val_score(svm_reg_linear, X_train_prepared, y_train_rs.ravel(), cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



svm_reg_linear_scores = np.sqrt(-scores_svm_linear)

svm_reg_linear_scores.mean()
from sklearn.svm import SVR



svm_reg_poly = SVR(kernel = "poly", degree = 2, C=1, epsilon = 0)



svm_reg_poly.fit(X_train_prepared, y_train_rs.ravel())



svm_reg_poly.score(X_train_prepared, y_train_rs.ravel())
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_svm_poly = svm_reg_poly.predict(X_train_prepared)



svm_poly_mse = mean_squared_error(y_train_rs.ravel(), y_predict_svm_poly)

svm_poly_rmse = np.sqrt(svm_poly_mse)

svm_poly_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score



scores_svm_poly = cross_val_score(svm_reg_poly, X_train_prepared, y_train_rs.ravel(), cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



svm_reg_poly_scores = np.sqrt(-scores_svm_poly)

svm_reg_poly_scores.mean()
from sklearn.svm import SVR



svm_reg_rbf = SVR(kernel = "rbf", C=1)



svm_reg_rbf.fit(X_train_prepared, y_train_rs.ravel())



svm_reg_rbf.score(X_train_prepared, y_train_rs.ravel())
# Calculating the MSE



from sklearn.metrics import mean_squared_error



y_predict_svm_rbf = svm_reg_rbf.predict(X_train_prepared)



svm_rbf_mse = mean_squared_error(y_train_rs.ravel(), y_predict_svm_rbf)

svm_rbf_rmse = np.sqrt(svm_rbf_mse)

svm_rbf_rmse
# Cross Validation



from sklearn.model_selection import cross_val_score



scores_svm_rbf = cross_val_score(svm_reg_rbf, X_train_prepared, y_train_rs.ravel(), cv=5, n_jobs=-1, scoring = "neg_mean_squared_error")



svm_reg_rbf_scores = np.sqrt(-scores_svm_rbf)

svm_reg_rbf_scores.mean()
# Due to the promising scores we will do a randomizes search on the RBF Kernel



from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import expon, reciprocal



params = {

        'kernel': ['rbf'],

        'C': reciprocal(1, 200000),

        'gamma': expon(scale=1.0),

    }



svm_reg = SVR()

rnd_search = RandomizedSearchCV(svm_reg, param_distributions= params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)

rnd_search.fit(X_train_prepared, y_train_rs.ravel())

rnd_search.best_params_
rnd_search.best_score_
final_model = rnd_search





X_test_prepared = full_pipeline.transform(X_test) ## call transform NOT fit_transform



final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)



final_rmse = np.sqrt(final_mse)
final_rmse
Results = pd.DataFrame(y_test)

Results["Final_Predictions"] = final_predictions

Results.head(10)