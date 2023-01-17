import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



# Model libraries

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC

import xgboost as xgb

from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.tree import DecisionTreeClassifier





#Other Libraries

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split,RandomizedSearchCV

from sklearn.metrics import recall_score,precision_score,confusion_matrix

import scipy.stats as stats
data = pd.read_csv('/kaggle/input/novartis-data/Train.csv')

data_test = pd.read_csv('/kaggle/input/novartis-data/Test.csv')





print('Shape of the data:',data.shape)

data.head()
data.info() #getting more information on the dtype
data.describe()  # To check on the statistics
data.isnull().sum() #checking for missing values in data
data.columns #columns of the data
print(data['MULTIPLE_OFFENSE'].value_counts())

plt.figure(figsize=(5,3))

sns.countplot(data['MULTIPLE_OFFENSE'])

plt.show()
X = data.drop('MULTIPLE_OFFENSE', axis=1)

y = data['MULTIPLE_OFFENSE']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.25, random_state = 42, stratify=y)
print('y_train:\n',y_train.value_counts(normalize = True))

print('y_test:\n',y_test.value_counts(normalize = True))
#function to create histogram, Q-Q Plot and boxplot



def diagnostic_plots(df,variable):

    

    #define figure size

    plt.figure(figsize =(16,4))

    

    #histogram

    plt.subplot(1,3,1)

    sns.distplot(df[variable], bins = 30, kde = False)

    plt.title('Histogram')

    

    #Q-Q plot

    plt.subplot(1,3,2)

    stats.probplot(df[variable], dist = "norm", plot = plt)

    plt.ylabel('RM quantiles')

    

    # box plot

    plt.subplot(1,3,3)

    sns.boxplot(y=df[variable])

    plt.title('Boxplot')

    

    plt.show()
diagnostic_plots(data,'X_2')
diagnostic_plots(data,'X_3')
diagnostic_plots(data,'X_6')
diagnostic_plots(data,'X_7')
diagnostic_plots(data,'X_8')
diagnostic_plots(data,'X_10')
diagnostic_plots(data,'X_11')
diagnostic_plots(data.dropna(),'X_12')
diagnostic_plots(data,'X_13')
diagnostic_plots(data,'X_14')
diagnostic_plots(data,'X_15')
f, axes = plt.subplots(ncols=4, figsize=(20,4))

sns.countplot(data['X_1'],ax=axes[0])

sns.countplot(data['X_4'],ax=axes[1])

sns.countplot(data['X_5'],ax=axes[2])

sns.countplot(data['X_9'],ax=axes[3])

plt.show()
### We need to install the libraries required for the preprocesing steps

!pip install -U imbalanced-learn

!pip install feature_engine
# Pipelines

from sklearn.pipeline import Pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as pl1



#preprocessing methods

from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler



#preprocessing methods using feature engine

from feature_engine import categorical_encoders as ce

from feature_engine.outlier_removers import Winsorizer
# Segregating the data into numerical, categorical and features with outliers

numerical_features = ['X_2', 'X_3', 'X_6','X_7', 'X_8', 'X_10', 'X_11','X_12', 'X_13', 'X_14','X_15']

categorical_features = ['X_1', 'X_4', 'X_5','X_9']

outliers_data = ['X_6', 'X_7','X_8','X_10','X_11','X_12','X_13','X_15']
X_train['X_12'] = X_train['X_12'].fillna(X_train['X_12'].median())

X_test['X_12'] = X_test['X_12'].fillna(X_test['X_12'].median())
categorical_features = ['X_1', 'X_4', 'X_5','X_9']

X_train[categorical_features] = X_train[categorical_features].astype('object')

X_test[categorical_features] = X_test[categorical_features].astype('object')
outlier_treat =Pipeline(steps = [

              ('outlier1', Winsorizer(distribution = 'gaussian', tail = 'right',fold = 3, variables = ['X_6', 'X_7','X_8','X_10','X_12'])),

              ('outlier2', Winsorizer(distribution = 'gaussian', tail = 'left',fold = 3, variables = ['X_11', 'X_13'])),

              ('outlier3', Winsorizer(distribution = 'gaussian', tail = 'both',fold = 3, variables = ['X_15']))

                                      ])
outlier_treat.fit(X_train)
X_train = outlier_treat.transform(X_train)
# Converting the categorical variables to 'object' for doing the one hot encoding operation

X_train[categorical_features] = X_train[categorical_features].astype('object')

X_test[categorical_features] = X_test[categorical_features].astype('object')
numeric_transformer = Pipeline(steps = [

              ('scaler', StandardScaler())

                     ])

categorical_transformer = Pipeline(steps=[

    ('onehot3',ce.OneHotCategoricalEncoder(top_categories = 3, variables = ['X_9','X_1'] )),

    ('onehot4',ce.OneHotCategoricalEncoder(top_categories = 4, variables = ['X_5'] )),

    ('onehot10',ce.OneHotCategoricalEncoder(top_categories = 9, variables = ['X_4'] ))

])
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(

    transformers=[

                 ('drop_columns', 'drop', ['INCIDENT_ID','DATE']), #dropping the columns 

                 ('num', numeric_transformer, numerical_features),

                 ('cat', categorical_transformer,categorical_features)

    ])
preprocessor.fit(X_train)
X_train = preprocessor.transform(X_train)

X_test = preprocessor.transform(X_test)
# from sklearn.model_selection import RandomizedSearchCV

# rs = RandomizedSearchCV(xgb_model, {

#         'scale_pos_weight': [1,2],

#         'learning_rate'   : [0.05,0.10,0.15,0.20,0.25,0.30],

#         'min_child_weight': [1,3,5,7],

#         'gamma'           : [0.0,0.1,0.2,0.3,0.4],

#         'colsample_bytree': [0.3,0.4,0.5,0.7]

#     }, 

#     cv=5, 

#     scoring = 'f1',

#     return_train_score=False, 

#     n_iter=50

# )

# rs.fit(X_train, y_train)

# pd.DataFrame(rs.cv_results_)[['param_scale_pos_weight','param_learning_rate','param_min_child_weight','param_gamma','param_colsample_bytree','mean_test_score']]
xgb_model = xgb.XGBClassifier(scale_pos_weight= 1,min_child_weight=1,learning_rate= 0.35,gamma= 0.3,colsample_bytree= 0.3 )

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)



score = recall_score(y_test,y_pred_xgb)

print('Recall score :',score)

confusion_matrix(y_test,y_pred_xgb, labels = [1,0])
#Create an object of the classifier.

bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),sampling_strategy='auto',

replacement=False,random_state=0)

from sklearn.metrics import precision_score

bbc.fit(X_train, y_train)

y_pred_bbc = bbc.predict(X_test)



score = recall_score(y_test,y_pred_bbc)

precision = precision_score(y_test,y_pred_bbc)

print('recall score :',score)

confusion_matrix(y_test,y_pred_bbc, labels = [1,0])

# from sklearn.model_selection import RandomizedSearchCV

# fold = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# svm = SVC()



# rs = RandomizedSearchCV(SVC(class_weight = 'balanced'), {

#         'C': [0.1, 1, 10],

#         'kernel': ['linear', 'poly', 'rbf'],

#         'tol' :[0.1,0.001,0.001]

#     }, 

#     cv=fold, 

#     scoring="recall", 

#     n_iter=5

# )

# rs.fit(X_train, y_train)

# pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','param_tol','mean_test_score']]
model = SVC(class_weight = 'balanced', C = 0.1 , kernel = 'poly', tol = 0.001)

model.fit(X_train, y_train)

y_pred_svm = model.predict(X_test)



score = recall_score(y_pred_svm,y_test)

print('recall score',score)

confusion_matrix(y_pred_svm,y_test, labels = [1,0])

model_rf = RandomForestClassifier(class_weight='balanced_subsample')

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)



score = recall_score(y_pred_rf,y_test)

print(score)

confusion_matrix(y_pred_rf,y_test, labels = [1,0])

ada_model = AdaBoostClassifier()

ada_model.fit(X_train, y_train)

y_pred_ada = ada_model.predict(X_test)



score = recall_score(y_test,y_pred_ada)

print('Recall score :', score)

confusion_matrix(y_test,y_pred_ada, labels = [1,0])
data_test.head()
data_test.isnull().sum()
# Replacing the missing values with the median values

data_test['X_12'] = data_test['X_12'].fillna(data_test['X_12'].median())
#Converting the datatype to 'object' for all the categorical features for transformations

data_test[categorical_features] = data_test[categorical_features].astype('object')
# Performing scaling , one hot encoding on the data using sklearn pipelines

data_test1 = preprocessor.transform(data_test)
#Missing value imputation

X['X_12'] = X['X_12'].fillna(X['X_12'].median())
# Outlier treatment

X = outlier_treat.transform(X)
# Preprocessing 

X = preprocessor.transform(X)
xgb_model.fit(X, y)

prediction_xgb = xgb_model.predict(data_test1)
output_xgb=pd.DataFrame({"INCIDENT_ID":data_test["INCIDENT_ID"],"MULTIPLE_OFFENSE":prediction_xgb}) 

output_xgb.head()
print(output_xgb['MULTIPLE_OFFENSE'].value_counts())

sns.countplot(output_xgb['MULTIPLE_OFFENSE'])