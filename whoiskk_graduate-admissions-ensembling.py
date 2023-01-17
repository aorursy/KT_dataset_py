import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.simplefilter('ignore')
%matplotlib inline
# data = pd.read_csv("../input/Admission_Predict.csv")
data = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
data.head()
data1.shape
data.set_index('Serial No.', inplace=True)
data.head()
num_vars = list(set(data.columns) - set(data.select_dtypes("object").columns))
num_vars
for i in num_vars:
    print(i)
    plt.figure()
    sns.distplot(data[i])
    plt.show()
# We can see that Research column is a categorical variable i.e :: 0(Not experience) and 1(Having Experience)
num_vars.remove('Research')
num_vars.remove('Chance of Admit ')
target_col = 'Chance of Admit '
cat_vars = []
cat_vars.append('Research')
# lambda_dict = {}
# rc_bc, bc_params = stats.boxcox(data['University Rating'])
# bc_params
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

# data['University Rating'].hist(ax=ax1, bins=100)
# ax1.set_yscale('log')
# ax1.tick_params(labelsize=1)
# ax1.set_title('Normal')
# ax1.set_xlabel('')
# ax1.set_ylabel('Occurence :: ', fontsize=1)

# data['University Rating 1'] = np.log(data['University Rating'])
# data['University Rating'].hist(ax=ax2, bins=100)
# ax2.set_yscale('log')
# ax2.tick_params(labelsize=1)
# ax2.set_title('Log')
# ax2.set_xlabel('')
# ax2.set_ylabel('Occurence :: ', fontsize=1)

# data['University Rating 2'] = (data['University Rating']**bc_params - 1)/bc_params
# data['University Rating 2'].hist(ax=ax3, bins=100)
# ax3.set_yscale('log')
# ax3.tick_params(labelsize=1)
# ax3.set_title('BoxCox')
# ax3.set_xlabel('')
# ax3.set_ylabel('Occurence :: ', fontsize=1)
def ohe(data, features):
    for feature in features:
        oh = pd.get_dummies(data[feature], prefix=feature, prefix_sep='_')
        data.drop([feature], inplace=True, axis=1)
        data = data.join(oh)
    return data
data = ohe(data, cat_vars)
data.head()
mm = MinMaxScaler()
for i in num_vars:
    data[i] = mm.fit_transform(data[[i]])
    print(i)
data.head()
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
# from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import cross_val_score
models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), AdaBoostRegressor(), BaggingRegressor(),
          ExtraTreesRegressor(), GradientBoostingRegressor(), lgb.LGBMRegressor(), xgb.XGBRegressor()]
def cross_val(model, X, y, n_splits):
    print(cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_squared_error').mean())
f1 = num_vars + ['Research_1']
for i in models:
    print(str(i).split("(")[0])
    cross_val(i, data[f1], data[target_col].values, 3)
xgbM = xgb.XGBClassifier()
xgbM.fit(data[f1], data[target_col])
def plot_imporatances(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()
plot_imporatances(xgbM, f1)
# f1.remove('Research_1')
def plot_correlations(data, features, target):
    corr = data[features + [target]].corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
plot_correlations(data, f1, target_col)
# Research_0 and Research_1 are highly correlated. As expected because they're exactly inverse of each other.
# Removing Research_0 instead of Research_1 because Research_0 is highly correlated with other features too (all blue line.)
f1.remove("Research_0")
xgbM = xgb.XGBClassifier()
xgbM.fit(data[f1], data[target_col])

plot_imporatances(xgbM, f1)
f1 = num_vars + ['Research_0', 'Research_1']
for i in models:
    print(str(i).split("(")[0])
    cross_val(i, data[f1], data[target_col].values, 3)
model_scores = {
    'LinearRegression': 0.7730041243734104,
    'KNeighborsRegressor': 0.6782033337353565,
    'DecisionTreeRegressor': 0.5727816065309351,
    'RandomForestRegressor': 0.732617047489113,
    'AdaBoostRegressor': 0.6848478260769751,
    'BaggingRegressor': 0.7365759033202579,
    'ExtraTreesRegressor': 0.7149594005565629,
    'GradientBoostingRegressor': 0.7433033713983969,
    'LGBMRegressor': 0.723100524729666,
    'XGBRegressor': 0.7485287892412207
}
def getEnsembleWeights(model_scores):
    total_score = 0
    for key, value in model_scores.items():
        total_score = total_score + model_scores[key]
    for key, value in model_scores.items():
        model_scores[key] = model_scores[key]/total_score
    return model_scores
weights = getEnsembleWeights(model_scores)
weights
def createEnsemble(X, y, X_test, weights):
    
    regM = LinearRegression()
    knnM = KNeighborsRegressor()
    dtreeM = DecisionTreeRegressor()
    randM = RandomForestRegressor()
    adaM = AdaBoostRegressor()
    bagM = BaggingRegressor(n_jobs=-1)
    extM = ExtraTreesRegressor()
    gbtM = GradientBoostingRegressor()
    lgbM = lgb.LGBMRegressor(n_jobs=-1)
    xgbM = xgb.XGBRegressor(n_jobs=-1)
    
    regM.fit(X, y)
    knnM.fit(X, y)
    dtreeM.fit(X, y)
    randM.fit(X, y)
    adaM.fit(X, y)
    bagM.fit(X, y)
    extM.fit(X, y)
    gbtM.fit(X, y)
    lgbM.fit(X, y)
    xgbM.fit(X, y)
    
    regP = regM.predict(X_test)
    knnP = knnM.predict(X_test)
    dtreeP = dtreeM.predict(X_test)
    randP = randM.predict(X_test)
    adaP = adaM.predict(X_test)
    bagP = bagM.predict(X_test)
    extP = extM.predict(X_test)
    gbtP = gbtM.predict(X_test)
    lgbP = lgbM.predict(X_test)
    xgbP = xgbM.predict(X_test)
    
    final_pred = regP*weights['LinearRegression'] + knnP*weights['KNeighborsRegressor'] + dtreeP*weights['DecisionTreeRegressor'] + randP*weights['RandomForestRegressor'] + adaP*weights['AdaBoostRegressor'] + bagP*weights['BaggingRegressor'] + extP*weights['ExtraTreesRegressor'] + gbtP*weights['GradientBoostingRegressor'] + lgbP*weights['LGBMRegressor'] + xgbP*weights['XGBRegressor']
    
    return final_pred
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data[f1], data[target_col], test_size=0.2, random_state=13)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
def rmse(y_true, y_pred):
    return(np.sqrt(mean_squared_error(y_true, y_pred)))

y_pred = createEnsemble(X_train, y_train, X_test, weights)
mean_squared_error(y_test, y_pred)
for i in models:
    print(str(i).split("(")[0])
    cross_val(i, data[f1], data[target_col].values, 3)
# The ensembled score is quite good.
