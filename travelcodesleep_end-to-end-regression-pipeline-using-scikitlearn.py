# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#support python 2&3
from __future__ import division, print_function, unicode_literals

#common imports
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#make notebook's output stable across runs
np.random.seed(42)

#Pretty Figures
%matplotlib inline
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#Ignore useless warnings
import warnings
warnings.filterwarnings(action='ignore',message='^internal gelsd')
pd.options.mode.chained_assignment = None
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# function to load data
dataset_path = os.path.join("../input",'forestfires.csv')
def load_data(path):
        return pd.read_csv(dataset_path)
    
fire = load_data(dataset_path)
fire.head()
fire.FFMC[np.random.choice(fire.index,15)] = np.nan
fire.info() #now we have 15 missing values for attribute FFMC
fire.describe() #some attributes skewed to the right(mean>median) some to the left(mean<median)
#let's name the categorical and numeical attributes 
categorical_attributes = list(fire.select_dtypes(include=['object']).columns)
numerical_attributes = list(fire.select_dtypes(include=['float64', 'int64']).columns)
print('categorical_attributes:', categorical_attributes)
print('numerical_attributes:', numerical_attributes)
#Here we face an uniqe issue where a few months have only 1 or 2 data points. I chose to get rid of these so that the glm models
#can handle the tests properly. Let me know how you would have hanlded.
print('months', fire.month.value_counts(), sep='\n')
print('\n')
print('days', fire.day.value_counts(), sep='\n')
months_to_remove = ['nov','jan','may']
forest_fire = fire.drop(fire[fire.month.isin(months_to_remove)].index ,axis=0)
forest_fire.month.value_counts()
#visualizing distributions 
forest_fire.hist(bins=50, figsize=(15,10), ec='w')
plt.show()
#target-area-is heavily skewed, we have extreme outliers.
plt.hist(forest_fire.area, ec='w', bins=100, color='red')
plt.text(800,100, 'max: '+str(forest_fire.area.max()), color='black', fontsize=14)
#Burnt area attribute ranges from 0 to 1091.
#Grouping the the burnt area to get a better understanding
forest_fire['area_cat'] = pd.cut(forest_fire['area'], bins=[0,5, 10, 50, 100, 1100], include_lowest=True, 
                                 labels=['0-5', '5-10', '10-50', '50-100', '>100'])
forest_fire.area_cat.value_counts()
#Interquartile range
Q1 = forest_fire.area.quantile(.25)
Q3 = forest_fire.area.quantile(.75)
IQR = 1.5*(Q3-Q1)
IQR
#we are loosing quite a number of data points in already a small data set if we remove all outliers
forest_fire.query('(@Q1 - 1.5 * @IQR) <= area <= (@Q3 + 1.5 * @IQR)').area_cat.value_counts(sort=False)
#remove outliers
forest_fire.drop(forest_fire[forest_fire.area>100].index,axis=0, inplace=True)
forest_fire.area_cat.value_counts()
#Let's understand what temp ranges we have here.
plt.hist(forest_fire.temp, ec='w', bins=50, color='red')
plt.show()
forest_fire['temp_bins'] = pd.cut(forest_fire.temp, bins=[0, 15, 20, 25, 40], include_lowest=True, 
                                 labels=['0-15', '15-20', '20-25', '>25'])
forest_fire.temp_bins.value_counts(sort=False)
#so we have from very cold 0-15 degrees to hot >25 degree.
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(forest_fire.values, forest_fire.temp_bins.values):
    st_train_set = forest_fire.iloc[train_index]
    st_test_set = forest_fire.iloc[test_index]
#this works like magic.
print(st_test_set.temp_bins.value_counts(sort=False)/len(st_test_set), sep='\n')
print(forest_fire.temp_bins.value_counts(sort=False)/len(forest_fire), sep='\n')
#now lets drop the area and temp categories 
for _ in (st_train_set, st_test_set):
    _.drop(['area_cat','temp_bins'], axis=1, inplace=True)
    
forest_fire = st_train_set.copy()
forest_fire.head()
#December had a few incidents but all on the higher burnt area side. Is it becuase dry weather or because more tourists? 
ax = plt.figure(figsize=(12,8))
ax = sns.boxplot(x='month', y='area', data=forest_fire, color='lightgrey', )
ax = sns.stripplot(x='month', y='area', data=forest_fire, color='red', jitter=0.4, size=4)
#There are more incidents on weekends - Friday/Sat/Sun, it might mean that campers vactioning might have caused/spotted fires.
ax = plt.figure(figsize=(12,8))
ax = sns.boxplot(x='day', y='area', data=forest_fire, color='lightgrey', )
ax = sns.stripplot(x='day', y='area', data=forest_fire, color='red', jitter=0.4, size=4)
#I am checking the temparature distribution as per the forest cordinates during the month of december
#it looks like the temps are low, so it could be a combinaton of dry weather and human made fire. But as data scientists,
#we must make important discoveries to arrive at any conclusive evidence. Else the above statments are just my mind made 
#fantsies
forest_fire[forest_fire.month=='dec'].plot(kind='scatter', x='X', y='Y', c='temp', cmap=plt.get_cmap('coolwarm'), colorbar=True)
plt.show()
corr_matrix = forest_fire.corr(method='spearman')
corr_matrix
ax = plt.figure(figsize=(12,8))
ax = sns.heatmap(corr_matrix, cmap='PiYG')
#corrleation with area
corr_matrix.area.sort_values(ascending=False)
#visualizing relations of most related attributes
attributes = ['area', 'wind', 'temp', 'rain', 'RH']
sns.pairplot(forest_fire[attributes])
plt.show()
#create a fresh copy of train to preprocess
forest_fire = st_train_set.drop('area', axis=1)
forest_fire_labels = st_train_set.area.copy()
from sklearn.base import BaseEstimator, TransformerMixin

class AttributeDeleter(BaseEstimator, TransformerMixin):
    def __init__(self, delete=True):
        self.delete = delete
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return np.delete(X,[fire.columns.get_loc(i) for i in['X','Y','area']],axis=1)
            
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

categorical_attributes = list(forest_fire.select_dtypes(include=['object']).columns)
numerical_attributes = list(forest_fire.select_dtypes(include=['float64', 'int64']).columns)

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('drop_attributes', AttributeDeleter()),
                         ('std_scaler', StandardScaler()),
                        ])
full_pipeline = ColumnTransformer([('num', num_pipeline, numerical_attributes),
                                   ('cat', OneHotEncoder(), categorical_attributes),
                                  ])

train = full_pipeline.fit_transform(forest_fire)
train_labels = forest_fire_labels
train.shape ,forest_fire.shape
#check the train data
train_df = pd.DataFrame(train, columns= numerical_attributes[2:] + list(full_pipeline.named_transformers_.cat.categories_[0]) +
             list(full_pipeline.named_transformers_.cat.categories_[1]))
train_df.head()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()
lin_reg.fit(train, train_labels)

area_predictions = lin_reg.predict(train)
lin_mse = mean_squared_error(train_labels, area_predictions)
lin_rmse = np.sqrt(lin_mse)
print('linear_train_rmse', lin_rmse) #model might be underfitting

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, train, train_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-scores)

def explain_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
 
explain_scores(lin_rmse_scores)
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(train, train_labels)

area_predictions = tree_reg.predict(train)
tree_mse = mean_squared_error(train_labels, area_predictions)
tree_rmse = np.sqrt(tree_mse)
print('tree_train_rmse', tree_rmse) #model obviously overfitting

scores = cross_val_score(tree_reg, train, train_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)
explain_scores(tree_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(train, train_labels)

area_predictions = rf_reg.predict(train)
rf_mse = mean_squared_error(train_labels, area_predictions)
rf_rmse = np.sqrt(rf_mse)
print('rf_train_rmse', rf_rmse) #model is overfitting 

scores = cross_val_score(rf_reg, train, train_labels, scoring='neg_mean_squared_error', cv=10)
rf_rmse_scores = np.sqrt(-scores)
explain_scores(rf_rmse_scores)
from sklearn.svm import SVR

svm_reg = SVR(kernel='linear')
svm_reg.fit(train, train_labels)

area_predictions = svm_reg.predict(train)
svm_mse = mean_squared_error(train_labels, area_predictions)
svm_rmse = np.sqrt(svm_mse)
print('svm_train_rmse', svm_rmse) #svm is generalizing well to crossvalidation set

scores = cross_val_score(svm_reg, train, train_labels, scoring='neg_mean_squared_error', cv=10)
svm_rmse_scores = np.sqrt(-scores)
explain_scores(svm_rmse_scores)
from sklearn.neighbors import KNeighborsRegressor

knn_reg = KNeighborsRegressor()
knn_reg.fit(train, train_labels)

area_predictions = knn_reg.predict(train)
knn_mse = mean_squared_error(train_labels, area_predictions)
knn_rmse = np.sqrt(knn_mse)
print('knn_train_rmse', knn_rmse) #overfiiting

scores = cross_val_score(knn_reg, train, train_labels, scoring='neg_mean_squared_error', cv=10)
knn_rmse_scores = np.sqrt(-scores)
explain_scores(knn_rmse_scores)
#lets improve the models with hyperparameter tuning

from sklearn.model_selection import GridSearchCV

param_grid = [{'bootstrap':[False,True],'n_estimators':[75,100,125,150,200], 'max_features':[1,2,4,6]}]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(train, train_labels)
grid_search.best_params_ 
grid_search.best_estimator_
feature_importances = grid_search.best_estimator_.feature_importances_
attributes = numerical_attributes + list(full_pipeline.named_transformers_.cat.categories_[0]) +\
             list(full_pipeline.named_transformers_.cat.categories_[1])
    
sorted(zip(feature_importances, attributes), reverse=True)
indices = np.argsort(feature_importances)
plt.figure(figsize=(12,8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), feature_importances[indices], color='lightblue', align='center')
plt.yticks(range(len(indices)), [attributes[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
#lets try with RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {'n_estimators': randint(low=10, high=250),
              'max_features': randint(low=1, high=24),
             }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_dist, n_iter=40, cv=5, scoring='neg_mean_squared_error', random_state=42)

rnd_search.fit(train, train_labels)

rnd_search.best_params_
np.sqrt(-rnd_search.best_score_), np.sqrt(-grid_search.best_score_) #both have very similar scores even though they came up with 
#different best parameters. which one to use, I leave to you explain
#tuning svr

param_grid = [{'kernel': ['linear'], 'C':[0.5,1,5,10,30]},
              {'kernel':['rbf'], 'C':[5,10,15,20], 'gamma':[0.5,1.0,1.5,2.0]},
             ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=4)
grid_search.fit(train,train_labels)

negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
grid_search.best_params_
from scipy.stats import expon, reciprocal

param_dist = {'kernel':['linear','rbf'],
                  'C':reciprocal(1,100),
                  'gamma':expon(scale=1.0)}
svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_dist, n_iter=100, cv=5,
                               scoring='neg_mean_squared_error', verbose=2, n_jobs=4, random_state=42)
rnd_search.fit(train, train_labels)

negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
rnd_search.best_params_ #randomized search is able to find better parameters for rbf kernel in same number of iterations
def indices_top_features(impotance, top):
    return np.sort(np.argpartition(np.array(impotance), -top)[-top:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importnaces, top):
        self.feature_importnaces = feature_importances
        self.top = top
    def fit(self, X, y=None):
        self.feature_indcies_ = indices_top_features(self.feature_importnaces, self.top)
        return self
    def transform(self,X):
        return X[:, self.feature_indcies_]
data_prep_feature_seletion_pipe = Pipeline([('prep', full_pipeline),
                                            ('fe_select', TopFeatureSelector(feature_importances,5)) #here am choosing top 5 features you can choose others depending on 
                                           ])                                                        #on what you want to keep  
train_fe_selected = data_prep_feature_seletion_pipe.fit_transform(forest_fire)
train_fe_selected.shape
#now let's try knn with these reduced dimensions 

knn_reg = KNeighborsRegressor()
knn_reg.fit(train_fe_selected, train_labels)

area_predictions = knn_reg.predict(train_fe_selected)
knn_mse = mean_squared_error(train_labels, area_predictions)
knn_rmse = np.sqrt(knn_mse)
print('knn_train_rmse', knn_rmse) #knn is generalizing well to crossvalidation set

scores = cross_val_score(knn_reg, train_fe_selected, train_labels, scoring='neg_mean_squared_error', cv=10)
knn_rmse_scores = np.sqrt(-scores)
explain_scores(knn_rmse_scores)
#lets tune KNN 
param_grid = {'weights': ['uniform', 'distance'], 'n_neighbors': list(range(1,36,5))}

knn_reg= KNeighborsRegressor()
knn_grid_search = GridSearchCV(knn_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=4)
knn_grid_search.fit(train_fe_selected,train_labels)

knn_negative_mse = knn_grid_search.best_score_
knn_rmse = np.sqrt(-knn_negative_mse)
knn_rmse
#and for final model I want to try GBM
from xgboost import XGBRegressor

xgb_reg = XGBRegressor()
xgb_reg.fit(train_fe_selected, train_labels)

area_predictions = xgb_reg.predict(train_fe_selected)
xgb_mse = mean_squared_error(train_labels, area_predictions)
xgb_rmse = np.sqrt(xgb_mse)
print('xgb_train_rmse', xgb_rmse) #overfitting

scores = cross_val_score(xgb_reg, train_fe_selected, train_labels, scoring='neg_mean_squared_error', cv=10)
xgb_rmse_scores = np.sqrt(-scores)
explain_scores(xgb_rmse_scores)
param_grid = {'objective':['reg:linear'],
              'learning_rate': [0.02,0.03,0.04], 
              'max_depth': [1,2],
              'min_child_weight': [2,3,4],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.2,0.3,0.4],
              'n_estimators': [50,60,70,100]}

xgb_reg = XGBRegressor()

xgb_grid_search = GridSearchCV(xgb_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=4)
xgb_grid_search.fit(train_fe_selected,train_labels)

xgb_negative_mse = grid_search.best_score_
xgb_rmse = np.sqrt(-xgb_negative_mse)
xgb_rmse
xgb_grid_search.best_params_
final_model = knn_grid_search.best_estimator_

X_test = st_test_set.drop(['area'], axis=1)
y_test = st_test_set['area'].copy()

X_test_prepared = data_prep_feature_seletion_pipe.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, 5)),
    ('final_model', XGBRegressor(**xgb_grid_search.best_params_))
])
prepare_select_and_predict_pipeline.fit(forest_fire,forest_fire_labels)
final_predictions = prepare_select_and_predict_pipeline.predict(X_test)
some_data = forest_fire[:4]
some_labels = forest_fire_labels[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))
#Confidence Interval of our Predictions will help us better understand the ouput of our model
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions-y_test)**2
np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))
