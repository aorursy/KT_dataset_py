import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
import statsmodels.api as sm 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


dataset = pd.read_excel(io='/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx' ,sheet_name='Data')
dataset.head()
dataset.info()
colormap = plt.cm.viridis # Color range to be used in heatmap
plt.figure(figsize=(15,15))
plt.title('Dataset Correlation of attributes', y=1.05, size=19)
sns.heatmap(dataset.corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
dataset.isnull().sum()
dataset.isnull().values.any()
dataset.describe().transpose()
dataset.apply(lambda x: len(x.unique()))
for col in dataset.columns:
    print(col + ' Col Unique values: ', dataset[col].unique(), '\n\n')
zero_mortgage = 0
for zero in dataset['Mortgage']:
    if zero == 0:
        zero_mortgage += 1
print('Number of people with zero mortgage ', zero_mortgage)
cc_avg = 0
for avg in dataset['CCAvg']:
    if avg == 0:
        cc_avg += 1
print('Number of people with zero credit card spending per month: ', cc_avg)
categorical_col = ['Personal Loan', 'Securities Account', 'CD Account', 'Online Col', 'CreditCard']

for col in categorical_col:
    val = 0
    for value in col:
        val += 1
    print('Value count of ' + col + ':', val)
plt.figure(figsize=(16,4))
sns.set_color_codes()
sns.countplot(dataset["Age"])
plt.figure(figsize=(18,5))
sns.set_color_codes()
sns.distplot(dataset["Age"])
plt.figure(figsize=(12,4))
sns.set_color_codes()
sns.barplot(dataset["Age"],dataset["Personal Loan"])
plt.figure(figsize=(12,4))
sns.set_color_codes()
sns.boxplot(y=dataset["Age"],x=dataset["Personal Loan"])
X = dataset.drop(columns = ['ID', 'Personal Loan'])
y = dataset['Personal Loan']
kbest = SelectKBest(k=5)
k_best_features = kbest.fit_transform(X, y)
list(dataset.columns[kbest.get_support (indices=True)])
dataset.corrwith(dataset["Personal Loan"]).abs().nlargest(5)
X = dataset.drop(columns = ['ID','Age','Experience','ZIP Code', 'Family', 'Education','Personal Loan','Securities Account', 'Online','CreditCard']).values
y = dataset['Personal Loan'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
sc = MinMaxScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
def model_evaluate(model, test):
    y_pred = model.predict(test)
    print('Metrics: \n', classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, cmap = 'Blues', fmt = '', annot = True)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
model = LogisticRegression(random_state = 0)
model.fit(x_train, y_train)

model_evaluate(model, x_test)
model = SVC(kernel = 'rbf')
model.fit(X_train, y_train)

model_evaluate(model, X_test)
model = KNeighborsClassifier(n_neighbors = 7, metric = 'euclidean')
model.fit(X_train, y_train)

model_evaluate(model, X_test)
model = GaussianNB()
model.fit(X_train, y_train)

model_evaluate(model, X_test)
xgb = XGBClassifier()
xgb.fit(X_train, y_train)

model_evaluate(xgb, X_test)
crossVal= cross_val_score(estimator = xgb, X = X_train, y = y_train, cv = 10)
print('XGBoost Accuracy: ', crossVal.mean())
print('XGBoost Std: ', crossVal.std())
skf = StratifiedKFold(n_splits=10)
scores = cross_val_score(xgb, X_train, y_train, cv=skf)
print("scores:\n{}".format(scores))
print("average score:\n{}".format(scores.mean()))
params = [{'learning_rate':[0.1,0.01],
           'colsample_bytree':[1,3],
           'gamma':[0,1],
           'reg_alpha':[2,3],
           'reg_lambda':[1,2,4,16],
           'n_estimators':[50,100,150],
           'colsample_bylevel':[1,2],
           'missing':[False, True],
           'subsample':[1,2],
           'base_score':[0.2,0.5]
           }
    ]
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator = xgb,
                  param_grid = params,
                  scoring = 'accuracy',
                  cv = 10,
                  n_jobs = -1)
grid_search = gs.fit(x_train, y_train)
best_result = grid_search.best_score_
best_params = grid_search.best_params_
print('Best_Result', best_result)
print('Best_Params', best_params)

xgb = XGBClassifier(base_score = 0.2, colsample_bylevel = 1, colsample_bytree = 1, gamma = 0, learning_rate = 0.1, missing = True, n_estimators = 150, reg_alpha = 3, reg_lambda = 1, subsample = 1)
xgb.fit(X_train, y_train)

model_evaluate(xgb, X_test)