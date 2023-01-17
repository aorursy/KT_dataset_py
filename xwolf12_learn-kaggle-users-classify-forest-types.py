from datetime import datetime



print("last update: {}".format(datetime.now())) 
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction import FeatureHasher

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import SelectFromModel

from sklearn.svm import LinearSVC

from sklearn.preprocessing import PowerTransformer
# seed

np.random.seed(1231)
df_train = pd.read_csv("/kaggle/input/learn-together/train.csv" , index_col=['Id'])

df_test = pd.read_csv("/kaggle/input/learn-together/test.csv" , index_col=['Id'])
df_train.head()
df_test.head()
print("shape training csv: %s" % str(df_train.shape)) 

print("shape test csv: %s" % str(df_test.shape)) 
df_train.dtypes.value_counts()
df_test.dtypes.value_counts()
df_train.iloc[:,10:].columns
df_test.iloc[:,10:].columns
df_train.iloc[:,10:] = df_train.iloc[:,10:].astype("category")

df_test.iloc[:,10:] = df_test.iloc[:,10:].astype("category")
df_train.isna().sum().sum()
df_test.isna().sum().sum()
df_train[df_train.duplicated()].shape
df_train.describe()
df_test.describe()
#Is it possible to have a negative value in *vertical distance to hydrology* ? how many are?



print("percent of negative values (training): " + '%.3f' % ((df_train.loc[df_train.Vertical_Distance_To_Hydrology < 0].shape[0] / df_train.shape[0])*100))

print("percent of negative values (testing): " + '%.3f' % ((df_test.loc[df_test.Vertical_Distance_To_Hydrology < 0].shape[0]/ df_test.shape[0])*100))
sns.boxplot(df_train.Vertical_Distance_To_Hydrology)
sns.boxplot(df_test.Vertical_Distance_To_Hydrology)
columns_t_analyze = df_train.select_dtypes(["float64", "int64"]).columns.tolist()

columns_t_analyze.append("Cover_Type")

plot = sns.pairplot(df_train.loc[:,columns_t_analyze], hue="Cover_Type")

plot.savefig("pairplot.png")
columns_t_analyze = df_train.select_dtypes(["float64", "int64"])

transformer =  PowerTransformer(method='yeo-johnson').fit(columns_t_analyze)
columns_t_analyze = df_train.select_dtypes(["float64", "int64"])

#columns_transformed =  RobustScaler(quantile_range=(25, 75)).fit_transform(columns_t_analyze)

columns_transformed =  PowerTransformer(method='yeo-johnson').fit_transform(columns_t_analyze)
columns_transformed = pd.DataFrame(columns_transformed)

columns_transformed.columns = columns_t_analyze.columns

columns_transformed = pd.concat([columns_transformed, df_train.loc[:,"Cover_Type"]], axis=1, join='inner')
X_train, X_test, y_train, y_test = train_test_split(columns_transformed.loc[:,columns_transformed.columns].drop("Cover_Type", axis=1), columns_transformed.loc[:,'Cover_Type'], test_size=0.33, random_state=42)
X_train.shape
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False)

lsvc.fit(X_train, y_train)

pred = lsvc.predict(X_test)

print("LinearSVC")

print(classification_report(y_test,pred, labels=None))
from sklearn.linear_model import SGDClassifier



sgdc= SGDClassifier()

sgdc.fit(X_train, y_train)

pred = sgdc.predict(X_test)

print("SGDC")

print(classification_report(y_test,pred, labels=None))
from sklearn.ensemble import RandomForestClassifier

randomfr= RandomForestClassifier()

randomfr.fit(X_train, y_train)

pred = randomfr.predict(X_test)

print("randomfr")

print(classification_report(y_test,pred, labels=None))
model = SelectFromModel(randomfr, prefit=True)

X_new = model.transform(X_train)

X_new.shape
# elevation, horizontal_distance_to_roadways, horizontal_distance_to_fire_points



pd.DataFrame(X_new).describe()
sns.pairplot(pd.concat([pd.DataFrame(X_new), df_train.loc[:,'Cover_Type']], axis=1, join='inner'), hue="Cover_Type")
sns.pairplot(columns_transformed.drop(columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points'], axis=1), hue="Cover_Type")
df_train.columns
fig, axs = plt.subplots(nrows=2)

sns.boxplot(df_train.Hillshade_3pm, ax=axs[0])

sns.boxplot(df_test.Hillshade_3pm, ax=axs[1], color="green")
#training 

quan = df_train.select_dtypes(["int", "float64"])

Q1 = quan.quantile(0.25)

Q3 = quan.quantile(0.75)

IQR =  Q3 - Q1



(((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).sum() / quan.shape[0]) * 100
#testing 

quan = df_test.select_dtypes(["int", "float"])

Q1 = quan.quantile(0.25)

Q3 = quan.quantile(0.75)

IQR =  Q3 - Q1



(((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).sum() / quan.shape[0]) * 100
quan = df_train.select_dtypes(["int", "float"]).copy()

Q1 = quan.quantile(0.25)

Q3 = quan.quantile(0.75)

IQR =  Q3 - Q1

sns.boxplot(quan.loc[~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_3pm].Hillshade_3pm)
for i in list(range(1,8)):

    sns.distplot(df_train.loc[(~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_3pm) & (df_train.Cover_Type == i), 'Hillshade_3pm'])

print("Normal shape {}".format(quan.shape[0]))

print("Without outliers {}".format(quan.loc[~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_3pm].shape[0]))
sns.boxplot(quan.loc[~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Slope].Slope)
sns.boxplot(data=df_train, y="Slope", x="Cover_Type")
for i in list(range(1,8)):

    sns.distplot(df_train.loc[(~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Slope) & (df_train.Cover_Type == i), 'Slope'])

print("Normal shape {}".format(quan.shape[0]))

print("Without outliers {}".format(quan.loc[~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Slope].shape[0]))
sns.boxplot(quan.loc[~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_Noon].Hillshade_Noon)
sns.boxplot(data=df_train, y="Hillshade_Noon", x="Cover_Type")
for i in list(range(1,8)):

    sns.distplot(df_train.loc[(~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_Noon) & (df_train.Cover_Type == i), 'Hillshade_Noon'])

print("Normal shape {}".format(quan.shape[0]))

print("Without outliers {}".format(quan.loc[~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_Noon].shape[0]))
sns.boxplot(data=df_train, y="Hillshade_9am", x="Cover_Type")
sns.boxplot(data=df_train, y="Hillshade_3pm", x="Cover_Type")
sns.boxplot(data=df_train, y="Aspect", x="Cover_Type")
quan.shape [0] - quan.loc[(~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_Noon) & (~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Slope)

        & (~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_3pm)].shape[0]
df_train_copy = df_train[(~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_Noon) & (~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Slope)

        & (~((quan < (Q1 - 1.5 * IQR)) | (quan > (Q3 + 1.5 * IQR))).Hillshade_3pm)].copy()
df_train_copy.shape
columns_t_analyze = df_train_copy.select_dtypes(["float64", "int64"]).copy()

X_train, X_test, y_train, y_test = train_test_split(df_train_copy.loc[:,columns_t_analyze.columns], df_train_copy.loc[:,'Cover_Type'], test_size=0.33, random_state=42)



from sklearn.ensemble import RandomForestClassifier

randomfr= RandomForestClassifier()

randomfr.fit(X_train, y_train)

pred = randomfr.predict(X_test)

print("randomfr")

print(classification_report(y_test,pred, labels=None))
model = SelectFromModel(randomfr, prefit=True)

X_new = model.transform(X_train)

X_new.shape
pd.DataFrame(X_new).head()
X_train.head()
sns.jointplot(data=df_train_copy, x="Horizontal_Distance_To_Hydrology", y="Vertical_Distance_To_Hydrology", kind="reg",)
df_train_copy['Horizontal_Distance_To_Hydrology'].corr(df_train_copy['Vertical_Distance_To_Hydrology'])
plt.figure(figsize=(15, 15))

sns.heatmap(df_train_copy.corr(), annot=True, cmap="YlGnBu")
from scipy.stats import norm

sns.distplot(df_train_copy.Horizontal_Distance_To_Hydrology, fit=norm)
from scipy import stats



res = stats.probplot(df_train_copy.Horizontal_Distance_To_Hydrology, plot=plt)
res = stats.probplot(pd.np.log(df_train_copy.loc[df_train_copy.Horizontal_Distance_To_Hydrology>0].Horizontal_Distance_To_Hydrology), plot=plt)
res = stats.probplot(pd.np.log(df_train_copy.loc[df_train_copy.Vertical_Distance_To_Hydrology>0].Vertical_Distance_To_Hydrology), plot=plt)
sns.boxplot(df_train_copy.Horizontal_Distance_To_Hydrology)
# np.where()

df_train_copy['horizontal_distance_to_hidrology_modified'] = np.where(df_train_copy.Horizontal_Distance_To_Hydrology <=0, df_train_copy.Horizontal_Distance_To_Hydrology.mean(), df_train_copy.Horizontal_Distance_To_Hydrology)

df_train_copy['horizontal_distance_to_hidrology_modified'] = np.log(df_train_copy.horizontal_distance_to_hidrology_modified)
df_train_copy['vertical_distance_to_hidrology_modified'] = np.where(df_train_copy.Vertical_Distance_To_Hydrology <=0, df_train_copy.Vertical_Distance_To_Hydrology.mean(), df_train_copy.Vertical_Distance_To_Hydrology)

df_train_copy['vertical_distance_to_hidrology_modified'] = np.log(df_train_copy.vertical_distance_to_hidrology_modified)
sns.jointplot(data=df_train_copy, x="vertical_distance_to_hidrology_modified", y="horizontal_distance_to_hidrology_modified", kind="reg")
df_train_copy["total_distance"] = (df_train_copy.vertical_distance_to_hidrology_modified**2) + (df_train_copy.horizontal_distance_to_hidrology_modified**2)
sns.boxplot(data=df_train_copy, y="total_distance", x="Cover_Type")
#columns_t_analyze = df_train_copy.select_dtypes(["int64", "float64"]).copy()

X_train, X_test, y_train, y_test = train_test_split(df_train_copy.loc[:,['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Roadways',

       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points',

       'horizontal_distance_to_hidrology_modified',

       'vertical_distance_to_hidrology_modified', 'total_distance']], df_train_copy.loc[:,'Cover_Type'], test_size=0.33, random_state=42)



from sklearn.ensemble import RandomForestClassifier

randomfr= RandomForestClassifier()

randomfr.fit(X_train, y_train)

pred = randomfr.predict(X_test)

print("randomfr")

print(classification_report(y_test,pred, labels=None))



model = SelectFromModel(randomfr, prefit=True)

X_new = model.transform(X_train)

print(X_new.shape)
pd.DataFrame(X_new).head()
X_train.head()
list(range(2,14,2))
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(df_train_copy[['Slope', 'Cover_Type']],df_train_copy.Cover_Type , test_size = 0.3)



score_ls = []     

score_std_ls = [] 

for tree_depth in range(2,14,2):

    tree_model = DecisionTreeClassifier(max_depth=tree_depth)

    

    scores = cross_val_score(tree_model, X_train.Slope.to_frame(), y_train, cv=3, scoring='balanced_accuracy')   

    

    score_ls.append(np.mean(scores))

    

    score_std_ls.append(np.std(scores))

    

temp = pd.concat([pd.Series(list(range(2,14,2))), pd.Series(score_ls), pd.Series(score_std_ls)], axis=1)

temp.columns = ['depth', 'accuracy', 'roc_acc_std']

print(temp)


tree_model = DecisionTreeClassifier(max_depth=2)



tree_model.fit(X_train.Slope.to_frame(), y_train)



X_train['Slope_tree'] = tree_model.predict_proba(X_train.Slope.to_frame())[:,1] 



X_train.Slope_tree.unique()
fig = plt.figure()

fig = pd.DataFrame(X_train.groupby(['Slope_tree'])["Cover_Type"])[0].plot()

fig.set_title('Monotonic relationship between discretised Slope and target')

fig.set_ylabel('Cover_Type')
pd.concat( [X_train.groupby(['Slope_tree'])['Slope'].min(),

            X_train.groupby(['Slope_tree'])['Slope'].max()], axis=1)
df_train['Slope_tree'] = tree_model.predict_proba(df_train.Slope.to_frame())[:,1]

df_test['Slope_tree'] = tree_model.predict_proba(df_test.Slope.to_frame())[:,1]



df_train.Slope_tree = df_train.Slope_tree.astype("category")

df_test.Slope_tree = df_test.Slope_tree.astype("category")







df_train_copy['Slope_tree'] = tree_model.predict_proba(df_train_copy.Slope.to_frame())[:,1]



df_train_copy.Slope_tree = df_train_copy.Slope_tree.astype("category")
df_train.Cover_Type.value_counts()
X = df_train.select_dtypes("category").drop(columns=["Cover_Type"])
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder

d = defaultdict(LabelEncoder)

fit = X.apply(lambda x: d[x.name].fit_transform(x))
fit.columns
Y_train = df_train.loc[:,'Cover_Type']
X_train, X_test, y_train, y_test = train_test_split(fit, Y_train, test_size=0.33, random_state=42)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy = accuracy_score(pred, y_test)

print(clf)

print(classification_report(pred, y_test, labels=None))
feature_importances = pd.DataFrame(clf.feature_importances_,

                                   index = X_train.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)

feature_importances
qualitative = df_train.select_dtypes("category").drop(columns=["Cover_Type"])

columns_t_analyze = df_train.select_dtypes(["float64", "int64"])

#columns_transformed =  RobustScaler(quantile_range=(25, 75)).fit_transform(columns_t_analyze)



columns_transformed =  PowerTransformer(method='yeo-johnson').fit_transform(columns_t_analyze)

columns_transformed = pd.DataFrame(columns_transformed)

columns_transformed.columns = columns_t_analyze.columns

columns_transformed = pd.concat([columns_transformed, df_train.loc[:,"Cover_Type"]], axis=1, join='inner')

d = defaultdict(LabelEncoder)

fit = X.apply(lambda x: d[x.name].fit_transform(x))

fit.reset_index(drop=True, inplace=True)

columns_transformed.reset_index(drop=True, inplace=True)

features_preprocessing = pd.concat([fit, columns_transformed], axis=1, join='inner')
features_preprocessing.columns
selected_columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type10', 

                  'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type4', 'Soil_Type3', 'Soil_Type17', 'Soil_Type2']



X_train, X_test, y_train, y_test = train_test_split(features_preprocessing.loc[:,selected_columns], features_preprocessing.loc[:,'Cover_Type'], test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsClassifier



for i in range(3, 21, 3):

    neigh = KNeighborsClassifier(n_neighbors=i)

    neigh.fit(X_train, y_train)

    pred = neigh.predict(X_test)

    print("KNeighborsClassifier {}".format(i))

    print(classification_report(pred, y_test, labels=None))
from sklearn.naive_bayes import GaussianNB, BernoulliNB

gnb = GaussianNB()

gnb.fit(X_train, y_train)

pred = gnb.predict(X_test)

## accuracy

accuracy = accuracy_score(y_test,pred)

print("naive_bayes")

print(classification_report(y_test,pred, labels=None))
from sklearn import svm

Sv=svm.SVC(gamma='scale',kernel='rbf')

Sv.fit(X_train, y_train)



pred = Sv.predict(X_test)

# accuracy

accuracy = accuracy_score(y_test,pred)

print(classification_report(y_test,pred, labels=None))
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

accuracy = accuracy_score(pred, y_test)

print(clf)

print(classification_report(pred, y_test, labels=None))
from xgboost import XGBClassifier



xgb = XGBClassifier(max_depth=10, subsample=0.8, colsample_bytree=0.7,missing=-999)



xgb.fit(X_train, y_train)

pred = xgb.predict(X_test)

accuracy = accuracy_score(pred, y_test)

print(xgb)

print(classification_report(pred, y_test, labels=None))
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_auc_score

params = {

        'min_child_weight': [1, 5, 10, 13, 15],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5, 10, 20]

        }



xgb = XGBClassifier(silent=True, nthread=1)

folds = 3

param_comb = 5



skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=4, cv=skf.split(X_train, y_train), verbose=3, random_state=1001 )



random_search.fit(X_train, y_train)
print('\n All results:')

print(random_search.cv_results_)

print('\n Best estimator:')

print(random_search.best_estimator_)

print('\n Best hyperparameters:')

print(random_search.best_params_)

results = pd.DataFrame(random_search.cv_results_)

results.to_csv('xgb-random-grid-search-results-01.csv', index=False)
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.8, gamma=5,

              learning_rate=0.1, max_delta_step=0, max_depth=10,

              min_child_weight=10, missing=None, n_estimators=100, n_jobs=1,

              nthread=1, objective='multi:softprob', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=True, subsample=0.8, verbosity=1)



xgb.fit(X_train, y_train)

pred = xgb.predict(X_test)

print(classification_report(pred, y_test, labels=None))
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svc = svm.SVC(gamma="scale")

clf = GridSearchCV(svc, parameters, cv=5)

clf.fit(X_train, y_train)

print('\n All results:')

print(clf.cv_results_)

print('\n Best estimator:')

print(clf.best_estimator_)

print('\n Best hyperparameters:')

print(clf.best_params_)
clf = svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,

    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',

    max_iter=-1, probability=False, random_state=None, shrinking=True,

    tol=0.001, verbose=False)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print(classification_report(pred, y_test, labels=None))
# selected_columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type10', 

#                   'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type4', 'Soil_Type3', 'Soil_Type17', 'Soil_Type2', 'Soil_Type30', 'Soil_Type13',

#                  'Soil_Type22', 'Soil_Type12', 'Soil_Type35', 'Soil_Type11', 'Wilderness_Area1', 'Soil_Type14']







selected_columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type10', 

                  'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type4', 'Soil_Type3', 'Soil_Type17', 'Soil_Type2', 'Soil_Type30', 'Soil_Type13',

                 'Soil_Type22', 'Soil_Type12', 'Soil_Type35', 'Soil_Type11', 'Wilderness_Area1', 'Soil_Type14',

                 'Wilderness_Area3', 'Soil_Type37', 'Soil_Type23', 'Soil_Type16', 'Soil_Type20', 'Soil_Type24', 'Soil_Type18', 'Wilderness_Area2', 'Slope_tree']
df_train_copy.columns
# without outliers

X_train, X_test, y_train, y_test = train_test_split(df_train_copy.loc[:,selected_columns], df_train_copy.loc[:,'Cover_Type'], test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

randomfr= RandomForestClassifier()

randomfr.fit(X_train, y_train)

pred = randomfr.predict(X_test)

print("randomfr")

print(classification_report(y_test,pred, labels=None))
from sklearn.neighbors import KNeighborsClassifier



for i in range(3, 21, 3):

    neigh = KNeighborsClassifier(n_neighbors=i)

    neigh.fit(X_train, y_train)

    pred = neigh.predict(X_test)

    print("KNeighborsClassifier {}".format(i))

    print(classification_report(pred, y_test, labels=None))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

# from sklearn.metrics import roc_auc_score

random_grid = {'bootstrap': [True, False],

               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],

               'max_features': ['auto', 'sqrt'],

               'min_samples_leaf': [1, 2, 4],

               'min_samples_split': [2, 5, 10],

               'n_estimators': [130, 180, 230]}



# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

rf = RandomForestClassifier()

random_search = RandomizedSearchCV(rf, param_distributions=random_grid, n_iter=param_comb, scoring='accuracy', n_jobs=4, cv=5, verbose=3, random_state=1001 )



random_search.fit(X_train, y_train)
random_search.best_estimator_.fit(X_train, y_train)

pred = random_search.best_estimator_.predict(X_test)

print("randomfr")

print(classification_report(y_test,pred, labels=None))
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_auc_score





qualitative = df_train_copy.select_dtypes("category").drop(columns=["Cover_Type"])

columns_t_analyze = df_train_copy.select_dtypes(["float64", "int64"])

#columns_transformed =  RobustScaler(quantile_range=(25, 75)).fit_transform(columns_t_analyze)



columns_transformed =  PowerTransformer(method='yeo-johnson').fit_transform(columns_t_analyze)

columns_transformed = pd.DataFrame(columns_transformed)

columns_transformed.columns = columns_t_analyze.columns

columns_transformed = pd.concat([columns_transformed, df_train_copy.loc[:,"Cover_Type"]], axis=1, join='inner')

d = defaultdict(LabelEncoder)

fit = X.apply(lambda x: d[x.name].fit_transform(x))

fit.reset_index(drop=True, inplace=True)

columns_transformed.reset_index(drop=True, inplace=True)

features_preprocessing = pd.concat([fit, columns_transformed], axis=1, join='inner')







selected_columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type10', 

                  'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type4', 'Soil_Type3', 'Soil_Type17', 'Soil_Type2', 'Soil_Type30', 'Soil_Type13',

                 'Soil_Type22', 'Soil_Type12', 'Soil_Type35', 'Soil_Type11', 'Wilderness_Area1', 'Soil_Type14',

                 'Wilderness_Area3', 'Soil_Type37', 'Soil_Type23', 'Soil_Type16', 'Soil_Type20', 'Soil_Type24', 'Soil_Type18', 'Wilderness_Area2']



X_train, X_test, y_train, y_test = train_test_split(features_preprocessing.loc[:,selected_columns], features_preprocessing.loc[:,'Cover_Type'], test_size=0.33, random_state=42)



params = {

        'min_child_weight': [1, 5, 10, 13, 15],

        'gamma': [0.5, 1, 1.5, 2, 5],

        'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],

        'colsample_bytree': [0.6, 0.8, 1.0],

        'max_depth': [3, 4, 5, 10, 20]

        }



xgb = XGBClassifier(silent=True, nthread=1)

#folds = 3

#param_comb = 5



#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)



random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='accuracy', n_jobs=4, cv=7, verbose=3, random_state=1001 )



random_search.fit(X_train, y_train)
xgb = random_search.best_estimator_



xgb.fit(X_train, y_train)

pred = xgb.predict(X_test)

print(classification_report(pred, y_test, labels=None))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

# from sklearn.metrics import roc_auc_score

random_grid = {'bootstrap': [True, False],

               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],

               'max_features': ['auto', 'sqrt'],

               'min_samples_leaf': [1, 2, 4],

               'min_samples_split': [2, 5, 10],

               'n_estimators': [130, 180, 230]}



# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

rf = RandomForestClassifier()

random_search = RandomizedSearchCV(rf, param_distributions=random_grid, n_iter=param_comb, scoring='accuracy', n_jobs=4, cv=5, verbose=3, random_state=1001 )



random_search.fit(X_train, y_train)



random_search.best_estimator_.fit(X_train, y_train)

pred = random_search.best_estimator_.predict(X_test)

print("randomfr")

print(classification_report(y_test,pred, labels=None))
# 1 experiment

#neigh = KNeighborsClassifier(n_neighbors=3)

#neigh.fit(features_preprocessing.loc[:,selected_columns], features_preprocessing.loc[:,'Cover_Type'])

# 2 experiment

# xgb = XGBClassifier(max_depth=10, subsample=0.8, colsample_bytree=0.7,missing=-999)

# xgb.fit(features_preprocessing.loc[:,selected_columns], features_preprocessing.loc[:,'Cover_Type'])

# 3 experiment 



selected_columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type10', 

                  'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type4', 'Soil_Type3', 'Soil_Type17', 'Soil_Type2', 'Soil_Type30', 'Soil_Type13',

                 'Soil_Type22', 'Soil_Type12', 'Soil_Type35', 'Soil_Type11', 'Wilderness_Area1', 'Soil_Type14',

                 'Wilderness_Area3', 'Soil_Type37', 'Soil_Type23', 'Soil_Type16', 'Soil_Type20', 'Soil_Type24', 'Soil_Type18', 'Wilderness_Area2', 'Slope_tree']





# 4 experiment



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

# from sklearn.metrics import roc_auc_score

random_grid = {'bootstrap': [True, False],

               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],

               'max_features': ['auto', 'sqrt'],

               'min_samples_leaf': [1, 2, 4],

               'min_samples_split': [2, 5, 10],

               'n_estimators': [130, 180, 230]}



# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

rf = RandomForestClassifier()

random_search = RandomizedSearchCV(rf, param_distributions=random_grid, n_iter=param_comb, scoring='accuracy', n_jobs=4, cv=5, verbose=3, random_state=1001 )



random_search.fit(df_train_copy.loc[:,selected_columns], df_train_copy.loc[:,'Cover_Type'])



random_search.best_estimator_.fit(df_train_copy.loc[:,selected_columns], df_train_copy.loc[:,'Cover_Type'])



# from sklearn.ensemble import RandomForestClassifier

# randomfr= RandomForestClassifier()

# randomfr.fit(df_train_copy.loc[:,selected_columns], df_train_copy.loc[:,'Cover_Type'])

columns_t_analyze = df_test.select_dtypes(["float64", "int64"])

columns_transformed =  transformer.transform(columns_t_analyze)

columns_transformed = pd.DataFrame(columns_transformed)

columns_transformed.columns = columns_t_analyze.columns
columns_transformed.shape
columns_transformed.head()
columns_transformed.columns
from collections import defaultdict

from sklearn.preprocessing import LabelEncoder

d = defaultdict(LabelEncoder)

X = df_test.select_dtypes("category")

fit = X.apply(lambda x: d[x.name].fit_transform(x))
fit.columns
fit.reset_index(drop=True, inplace=True)

columns_transformed.reset_index(drop=True, inplace=True)

features_test_preprocessing = pd.concat([columns_transformed, fit], axis=1, join='inner')

features_test_preprocessing.shape
features_test_preprocessing.isna().sum().sum()
#selected_columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type10', 

#                  'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type4']



results = xgb.predict(features_test_preprocessing.loc[:,selected_columns])
selected_columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type10', 

                  'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type4', 'Soil_Type3', 'Soil_Type17', 'Soil_Type2']



results = randomfr.predict(df_test.loc[:,selected_columns])
selected_columns=["Elevation", 'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area4', 'Soil_Type10', 

                  'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type4', 'Soil_Type3', 'Soil_Type17', 'Soil_Type2', 'Soil_Type30', 'Soil_Type13',

                 'Soil_Type22', 'Soil_Type12', 'Soil_Type35', 'Soil_Type11', 'Wilderness_Area1', 'Soil_Type14',

                 'Wilderness_Area3', 'Soil_Type37', 'Soil_Type23', 'Soil_Type16', 'Soil_Type20', 'Soil_Type24', 'Soil_Type18', 'Wilderness_Area2', 'Slope_tree']



results = random_search.best_estimator_.predict(df_test.loc[:,selected_columns])

output = pd.DataFrame({'Id': df_test.index,

                       'Cover_Type': results})

output.to_csv('submission_all.csv', index=False)