import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
mem = pd.read_csv('../input/bigdata-ui-class-competition-1th/mem_data.csv')
transaction = pd.read_csv('../input/bigdata-ui-class-competition-1th/mem_transaction.csv')
df = pd.merge(mem,transaction,on='MEM_ID',how='left')

df.info()
sns.countplot(x='GENDER',data=df)
plt.show()
print('생일 NA수 : %d'%df.BIRTH_DT.isnull().sum())

print('남성 생일 NA수 : %d'%df[df.GENDER=='M'].BIRTH_DT.isnull().sum())

print('여성 생일 NA수 : %d'%df[df.GENDER=='F'].BIRTH_DT.isnull().sum())
# 생일을 입력하면 0 입력 안하면 1
# 양력 음력 0,1 변환
df["BIRTH"] = df.BIRTH_DT.apply(lambda x: 0 if type(x)==str else 1)
df["BIRTH_SL"] = df.BIRTH_SL.apply(lambda x: 0 if x=='S' else 1)
sns.countplot(x='GENDER',hue='BIRTH',data=df)
plt.show()
#각 ID 별 개별 구매 가격을 총합해서 SELL 변수에 추가
SELL = df.groupby('MEM_ID')['SELL_AMT'].sum()
SELL.name = "SELL"
df = pd.merge(df,SELL,on='MEM_ID',how='left')
print(df.MEMP_STY.value_counts())
#MEMM_STY에서 웹입력(W)값이 없어서 0,1로 변환 추후 적립이면 1
df.MEMP_STY = df.MEMP_STY.apply(lambda x: 0 if x=='O' else 1)
#각 아이디 별 추후적립비율 변수 추가
추후적립비율 = df.groupby(['MEM_ID'])["MEMP_STY"].sum() / df.groupby(['MEM_ID'])["MEMP_STY"].count()
추후적립비율.name = "STY"
df = pd.merge(df,추후적립비율,on='MEM_ID',how='left')
#SMS 수신 동의 0,1 변환 Y = 0
df.SMS = df.SMS.apply(lambda x: 0 if x=='Y' else 1)

#우편 번호 입력 하면 1
df.ZIP_CD = df.ZIP_CD.apply(lambda x: 0 if x=='-' else 1)
sns.countplot(x='GENDER',hue='ZIP_CD',data=df)
plt.show()
df = df.drop(['RGST_DT','LAST_VST_DT','SELL_DT',"MEMP_DT",'BIRTH_DT','MEMP_TP','SELL_AMT','MEMP_STY','STORE_ID','SALES_AMT'],axis=1)
#ID별 첫행만 불러와서 중복 아이디 제거
df = df.groupby('MEM_ID').head(1)
#평균 지출 금액 추가
df['S/V'] = df['SELL']/df['VISIT_CNT']
df = df.drop('VISIT_CNT',axis=1)
#차이가 많이 나는 변수들 log  (0값에 로그 취하면 -inf 이므로 +1 후 로그)
col = ['USABLE_PNT','USED_PNT','ACC_PNT','USABLE_INIT',"SELL",'S/V']
df[col] = np.log(df[col]+1)
#정규화
from sklearn.preprocessing import StandardScaler

col = ['USABLE_PNT','USED_PNT','ACC_PNT','USABLE_INIT',"SELL",'S/V']
standard = StandardScaler()
df[col] = standard.fit(df[col]).transform(df[col])
# UNKNOWN 값 분리
test = df[df.GENDER == 'UNKNOWN']
train = df[df.GENDER != 'UNKNOWN']
#GENER 0,1 남성일경우 1
train.GENDER = train.GENDER.apply(lambda x: 0 if x=='F' else 1)
X=train.drop(['MEM_ID','GENDER'],axis=1)
y=train.GENDER
X
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, learning_curve

from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score, roc_auc_score
import scikitplot as skplt
#테스트 사이즈 0.3
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=1)

kfold = StratifiedKFold(n_splits=10)
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(XGBClassifier())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","LogisticRegression",'XGB']})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
test_target = test[['MEM_ID','GENDER']]
test = test.drop(['MEM_ID','GENDER'],axis=1)
xgb_clf = XGBClassifier(n_estimators=100,random_state=random_state,n_jobs=4)

params = {'max_depth':[3,5,7,9],'min_child_weight':[3,5,7,9],
         'colsample_bytree':[0.5,0.75,1.0]}

# XGBoost의 최적의 파라미터를 찾고
# 최적의 파라미터를 포함하는 XGBoost 객체를 생성
# GridSearchCV 객체 생성
gridcv = GridSearchCV(xgb_clf,param_grid=params)
gridcv.fit(X_train, y_train, early_stopping_rounds = 30,
          eval_metric='error',#error select BEST
           eval_set = [(X_train,y_train),(X_test,y_test)])


pred = gridcv.predict(X_test)

print('XGB')
print('accuracy: {:.3f}'.format(accuracy_score(y_test,pred)))
print('precision: {:.3f}'.format(precision_score(y_test,pred)))
print('recall: {:.3f}'.format(recall_score(y_test,pred)))
print('f1: {:.3f}'.format(f1_score(y_test,pred)))
print('roc_auc: {:.3f}'.format(roc_auc_score(y_test,pred)))
confusion_matrix(y_test,pred)
print(pd.Series(gridcv.predict(test)).value_counts())

y_pred = gridcv.predict(X_test)

#예측 확률 저장
#test_target.GENDER = gridcv.predict_proba(test)[:,1]

#test_target.sort_values('MEM_ID').to_csv('gridcv.csv',index=False)
