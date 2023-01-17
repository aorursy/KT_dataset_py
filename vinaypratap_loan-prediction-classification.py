import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from imblearn.over_sampling import SMOTE

df_train = pd.read_csv("../input/train.csv")

df_train.tail()
df_train[df_train['CoapplicantIncome']==0]
df_train['Education'].unique()
df_train.shape
df_train['Loan_Status'].value_counts()
df_train.groupby('Gender')['Loan_Status'].count()
df_train['Loan_Amount_Term'].unique()
df_train[df_train['Loan_Amount_Term'].isnull()]
df_test = pd.read_csv("../input/test.csv")

df_test.head()
df_test.shape
# lets check for null values but lest concate both dataset first

#df = pd.concat([df_train,df_test])

#df.head()
#df.shape
df_train.isnull().sum()
df_train.describe()
df_train.info()
#df.drop(['CoapplicantIncome'],axis=1,inplace=True)
#lets work on categorical variables and identify all of them

cat_vars1 = [feature for feature in df_train.columns if df_train[feature].dtypes == 'O' and feature !='Loan_Status']

cat_vars1

cat_vars2 = [feature for feature in df_test.columns if df_test[feature].dtypes == 'O' and feature !='Loan_Status']

cat_vars2
df_train['Gender'].fillna('Female',inplace=True)

df_test['Gender'].fillna('Female',inplace=True)
# lets impute all those except loan status with mode values for null values

#df[cat_vars] = df[cat_vars].fillna(df[cat_vars].mode()[0])

for feat in cat_vars1:

    df_train[feat]= df_train[feat].fillna(df_train[feat].mode()[0])

for feat in cat_vars2:

    df_test[feat]= df_test[feat].fillna(df_test[feat].mode()[0])
df_train.isnull().sum()
df_test.isnull().sum()
df_train.info()
df_train.head()
# lets find numerical variables

num_vars1 = [feature for feature in df_train.columns if df_train[feature].dtypes != 'O' and feature !='Loan_Status']

num_vars1

num_vars2 = [feature for feature in df_test.columns if df_test[feature].dtypes != 'O' and feature !='Loan_Status']

num_vars2
df_train['Credit_History'] = df_train['Credit_History'].fillna(2)

df_test['Credit_History'] = df_test['Credit_History'].fillna(2)
df_train[num_vars1].isnull().sum()
df_train.info()
for feat in num_vars1:

    sns.boxplot(df_train[feat])

    plt.show()
# data is not much and and it is not giving any clear indication

# lets impute with median as there seems to be outliers

df_train[num_vars1] = df_train[num_vars1].fillna(df_train[num_vars1].median())

df_test[num_vars2] = df_test[num_vars2].fillna(df_test[num_vars2].median())
df_test.head()
df_train.isnull().sum()
df_train['Dependents'].unique()
df_train.groupby(['Dependents'])['Loan_Status'].count()
df_train['Dependents']= df_train['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})

#df['Dependents']= df['Dependents'].map({'0':'No_Dep','1':'OneR1+_Dep','2':'OneR1+_Dep','3+':'OneR1+_Dep'})

df_train['Loan_Status']= df_train['Loan_Status'].map({'Y':1,'N':0})

df_train['Gender']= df_train['Gender'].map({'Male':1,'Female':0})

df_train['Married']= df_train['Married'].map({'Yes':1,'No':0})

df_train['Self_Employed']= df_train['Self_Employed'].map({'Yes':1,'No':0})

df_train['Education']= df_train['Education'].map({'Graduate':1,'Not Graduate':0})

df_train['Property_Area']= df_train['Property_Area'].map({'Urban':0,'Semiurban':1,'Rural':2})
df_test['Dependents']= df_test['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})

#df['Dependents']= df['Dependents'].map({'0':'No_Dep','1':'OneR1+_Dep','2':'OneR1+_Dep','3+':'OneR1+_Dep'})

#df_test['Loan_Status']= df_test['Loan_Status'].map({'Y':1,'N':0})

df_test['Gender']= df_test['Gender'].map({'Male':1,'Female':0})

df_test['Married']= df_test['Married'].map({'Yes':1,'No':0})

df_test['Self_Employed']= df_test['Self_Employed'].map({'Yes':1,'No':0})

df_test['Education']= df_test['Education'].map({'Graduate':1,'Not Graduate':0})

df_test['Property_Area']= df_test['Property_Area'].map({'Urban':0,'Semiurban':1,'Rural':2})
#df.groupby('Credit_History')['Loan_Status'].sum()
#df.groupby('Credit_History')['Loan_Status'].count()
#df.groupby('Self_Employed')['Loan_Status'].sum()
#lets get the dummy vars for categorical variables

#df = pd.concat([df, pd.get_dummies(df['Married'],drop_first=True)],axis=1).drop('Married',axis=1)

#df = pd.concat([df, pd.get_dummies(df['Gender'],drop_first=True)],axis=1).drop('Gender',axis=1)

#df = pd.concat([df, pd.get_dummies(df['Dependents'],drop_first=True)],axis=1).drop('Dependents',axis=1)

#df = pd.concat([df, pd.get_dummies(df['Education'],drop_first=True)],axis=1).drop('Education',axis=1)

#df = pd.concat([df, pd.get_dummies(df['Self_Employed'],drop_first=True)],axis=1).drop('Self_Employed',axis=1)

#df = pd.concat([df, pd.get_dummies(df['Property_Area'],drop_first=True)],axis=1).drop('Property_Area',axis=1)
df_train.info()
df_test.head()
df_train['Total_Inc'] = df_train['ApplicantIncome'] + df_train['CoapplicantIncome']

df_train['Total_Inc'] = np.log(df_train['Total_Inc'])

#df_train['Debt_Income_Ratio'] = df_train['Total_Inc'] / df_train['LoanAmount']

#df_train['EMI'] = (df_train['LoanAmount']*0.09*(1.09**df_train['Loan_Amount_Term']))/(1.09**df_train['Loan_Amount_Term']-1)

df_train.drop(['ApplicantIncome','CoapplicantIncome'],axis=1,inplace=True)

#df_train.drop(['Total_Inc','LoanAmount','Loan_Amount_Term'],axis=1,inplace=True)
df_test['Total_Inc'] = df_test['ApplicantIncome'] + df_test['CoapplicantIncome']

df_test['Total_Inc'] = np.log(df_test['Total_Inc'])

#df_test['Debt_Income_Ratio'] = df_test['Total_Inc'] / df_test['LoanAmount']

#df_test['EMI'] = (df_test['LoanAmount']*0.09*(1.09**df_test['Loan_Amount_Term']))/(1.09**df_test['Loan_Amount_Term']-1)

#df_test['Total_Inc'] = np.log(df_test['Total_Inc'])
df_test.drop(['ApplicantIncome','CoapplicantIncome'],axis=1,inplace=True)

#df_test.drop(['Total_Inc','LoanAmount','Loan_Amount_Term'],axis=1,inplace=True)
#df_train = df[:614]

#df_test = df[614:]
df_1 = df_train.copy()
df_train.drop(['Loan_ID'],inplace=True,axis=1)
df_train.head()
X = df_train.drop('Loan_Status',axis=1)

y = df_train['Loan_Status']
# Rerunning above with resampled data



#sm = SMOTE(random_state=1, ratio = 1)

#X_train_res, y_train_res = sm.fit_sample(X, y)
X.shape
# Normalize using MinMaxScaler to constrain values to between 0 and 1.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#scaler.fit(X_train_res)

scaler.fit(X)

X_train = scaler.transform(X)
#X_train = pd.DataFrame(X_train,columns=X.columns)

#X_train.head()
# Create first pipeline for base without reducing features.



pipe = Pipeline([('classifier' , RandomForestClassifier())])



# Create param grid.



param_grid = [

    {'classifier' : [LogisticRegression()],

     'classifier__penalty' : ['l1', 'l2'],

    'classifier__C' : np.logspace(-4, 4, 20),

    'classifier__solver' : ['liblinear']},

    {'classifier' : [RandomForestClassifier()],

    'classifier__n_estimators' : list(range(10,101,10)),

    'classifier__max_features' : list(range(5,15,5))}

]



# Create grid search object



clf = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)



# Fit on data



best_clf = clf.fit(X_train, y)
best_clf.best_estimator_.get_params()['classifier']
print('Model accuracy is',best_clf.score(X_train, y))
test = df_test.copy()
df_test.drop(['Loan_ID'],axis=1,inplace=True)
df_test.head()
X_test = scaler.transform(df_test)
clf = LogisticRegression()

clf.fit(X_train, y)

y_pred_log_reg = clf.predict(X_test)

acc_log_reg = round( clf.score(X_train,y) * 100, 2)

print ("Train Accuracy: " + str(acc_log_reg) + '%')
clf = SVC()

clf.fit(X_train, y)

y_pred_SVC = clf.predict(X_test)

acc_svc = round( clf.score(X_train, y) * 100, 2)

print ("Train Accuracy: " + str(acc_svc) + '%')
clf = LinearSVC()

clf.fit(X_train, y)

y_pred_linearsvc = clf.predict(X_test)

acc_linear_svc = round( clf.score(X_train, y) * 100, 2)

print ("Train Accuracy: " + str(acc_linear_svc) + '%')
clf = KNeighborsClassifier(n_neighbors = 4)

clf.fit(X_train, y)

y_pred_knn = clf.predict(X_test)

acc_knn = round( clf.score(X_train, y) * 100, 2) 

print ("Train Accuracy: " + str(acc_knn) + '%')
clf = DecisionTreeClassifier()

clf.fit(X_train, y)

y_pred_DT = clf.predict(X_test)

acc_decision_tree = round( clf.score(X_train, y) * 100, 2)

print ("Train Accuracy: " + str(acc_decision_tree) + '%')
clf = RandomForestClassifier(n_estimators=50)

clf.fit(X_train, y)

y_pred_random_forest = clf.predict(X_test)

acc_random_forest = round( clf.score(X_train, y) * 100, 2)

print ("Train Accuracy: " + str(acc_random_forest) + '%')
clf = GaussianNB()

clf.fit(X_train, y)

y_pred_GB = clf.predict(X_test)

acc_gnb = round( clf.score(X_train, y) * 100, 2)

print ("Train Accuracy: " + str(acc_gnb) + '%')
clf = Perceptron(max_iter=6, tol=None)

clf.fit(X_train, y)

y_pred_perceptron = clf.predict(X_test)

acc_perceptron = round( clf.score(X_train, y) * 100, 2)

print ("Train Accuracy: " + str(acc_perceptron) + '%')
clf = SGDClassifier(max_iter=5, tol=None)

clf.fit(X_train, y)

y_pred_SGD = clf.predict(X_test)

acc_sgd = round( clf.score(X_train, y) * 100, 2)

print ("Train Accuracy: " + str(acc_sgd) + '%')
#X_train.get_values
!pip install xgboost

import xgboost

classifier = xgboost.XGBClassifier()

classifier.fit(X_train, y)

# Predicting the Test set results

y_pred_XGB = classifier.predict(X_test)

acc_XGB = round( clf.score(X_train, y) * 100, 2)

print ("Train Accuracy: " + str(acc_XGB) + '%')
models = pd.DataFrame({

    'Model': ['LR', 'SVM', 'L-SVC', 

              'KNN', 'DTree', 'RF', 'NB', 

              'Perceptron', 'SGD','XGB'],

    

    'Score': [acc_log_reg, acc_svc, acc_linear_svc, 

              acc_knn,  acc_decision_tree, acc_random_forest, acc_gnb, 

              acc_perceptron, acc_sgd, acc_XGB]

    })



models = models.sort_values(by='Score', ascending=False)

models
submission = pd.DataFrame({

        "Loan_ID": test["Loan_ID"],

        "Loan_Status":y_pred_log_reg

    })

submission['Loan_Status'] = submission['Loan_Status'].map({1:'Y',0:'N'})

submission.to_csv('Loan_submission.csv', index=False)
submission.Loan_Status.value_counts()