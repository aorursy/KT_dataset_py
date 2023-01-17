import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, learning_curve

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score, roc_auc_score

from sklearn.decomposition import PCA

from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier


from sklearn.feature_selection import SelectFromModel

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, learning_curve
human = pd.read_csv('human.csv',encoding='cp949')
human_new = pd.read_csv('human_new.csv',encoding='cp949')
na_col = human.columns[human.isnull().sum()>0]
human[na_col] = human[na_col].fillna('A')

na_col1 = human_new.columns[human_new.isnull().sum()>0]
human_new[na_col1] = human_new[na_col1].fillna('A')

df = human.copy()
df['Widowed'] = df['혼인 상태'].apply(lambda x: 1 if str(x)==' Widowed' else 0)
df['w'] = df.관계.apply(lambda x: 1 if str(x)==' Wife' else 0)
df['h'] = df.관계.apply(lambda x: 1 if str(x)==' Husband' else 0)
df.관계 = df.관계.apply(lambda x: 0 if str(x) in [' Husband',' Wife'] else x)

df['P'] = df.직업.apply(lambda x: 1 if str(x)==' Priv-house-serv' else 0)
df.직업 = df.직업.apply(lambda x: 0 if str(x) in [' Priv-house-serv'] else x)

categorical_feature_mask = df.dtypes==object
categorical_cols = df.columns[categorical_feature_mask].tolist()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for i in categorical_cols[1:]:
    df[i] = df[i].astype('str')
    encoder.fit(df[i])
    df[i] = encoder.transform(df[i])

col = ['fnlwgt','자본 이득','자본 손실','주당 시간']
for i in col:
    df[i] = np.log(df[i]+1)

X=df.drop(['아이디','성별','모국'],axis=1)
y=df.성별

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res,y_res,test_size=0.3, random_state=1)

random_state=2

predictor = X_train.columns
clf = XGBClassifier(n_estimators=320,random_state = random_state)
clf = clf.fit(X_train, y_train)
fti = clf.feature_importances_
for i, feat in enumerate(predictor):
    print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

model = SelectFromModel(clf, prefit=True)
train_new = model.transform(X_train)


xgb_clf = XGBClassifier(n_estimators=100,random_state=random_state,n_jobs=4)

params = {'max_depth':[3,5,7,9],'min_child_weight':[3,5,7,9],
         'colsample_bytree':[0.5,0.75,1.0]}

gridcv = GridSearchCV(xgb_clf,param_grid=params)
gridcv.fit(X_train, y_train, early_stopping_rounds = 30,
          eval_metric='error',
           eval_set = [(X_train,y_train),(X_test,y_test)],verbose=10)


pred = gridcv.predict(X_test)
print('XGB')
print('accuracy: {:.3f}'.format(accuracy_score(y_test,pred)))
print('precision: {:.3f}'.format(precision_score(y_test,pred)))
print('recall: {:.3f}'.format(recall_score(y_test,pred)))
print('f1: {:.3f}'.format(f1_score(y_test,pred)))
print('roc_auc: {:.3f}'.format(roc_auc_score(y_test,pred)))
confusion_matrix(y_test,pred)
test = human_new.copy()

test['Widowed'] = test['혼인 상태'].apply(lambda x: 1 if str(x)==' Widowed' else 0)
test['w'] = test.관계.apply(lambda x: 1 if str(x)==' Wife' else 0)
test['h'] = test.관계.apply(lambda x: 1 if str(x)==' Husband' else 0)
test.관계 = test.관계.apply(lambda x: 0 if str(x) in [' Husband',' Wife'] else x)


test['P'] = test.직업.apply(lambda x: 1 if str(x)==' Priv-house-serv' else 0)
test.직업 = test.직업.apply(lambda x: 0 if str(x) in [' Priv-house-serv'] else x)

categorical_feature_mask = test.dtypes==object
categorical_cols = test.columns[categorical_feature_mask].tolist()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

for i in categorical_cols[1:]:
    test[i] = test[i].astype('str')
    encoder.fit(test[i])
    test[i] = encoder.transform(test[i])

col = ['fnlwgt','자본 이득','자본 손실','주당 시간']
for i in col:
    df[i] = np.log(df[i]+1)

    
X=test.drop(['아이디'],axis=1)
target = pd.DataFrame()
target['ID'] = test['아이디']

target['SEX'] = gridcv.predict_proba(X)[:,1]
target.to_csv('proba.csv',index=False)