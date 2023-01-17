import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
!pip install missingno
import missingno as mn
# 경고창이 뜨지 않게
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_train.info()
df_train.describe()
df_train['age'].value_counts()
df_train['fnlwgt'].value_counts()
df_train['education-num'].value_counts()
df_train['capital-gain'].value_counts()
df_train['capital-loss'].value_counts()
df_train['hours-per-week'].value_counts()
df_train['income'].value_counts()
df_train['workclass'].value_counts()
df_train['occupation'].value_counts()     # ???
df_train['native-country'].value_counts() # 미국에다 물음표
df_train['race'].value_counts()
df_train['education'].value_counts()
# workclass 결측치( ?)를 private로 채우기
df_train["workclass"]=df_train["workclass"].replace(" ?","Private")
df_test["workclass"]=df_train["workclass"].replace(" ?","Private")
# native-country 결측치( ?)를 U-S 로 채우기
df_train["native-country"]=df_train["native-country"].replace(" ?","United-States")
df_test["native-country"]=df_test["native-country"].replace(" ?","United-States")
# occupation의 결측치를 Machine-op-inspct
df_train['occupation'] = df_train['occupation'].replace(' ?',' Prof-specialty')
df_test['occupation'] = df_test['occupation'].replace(' ?',' Prof-specialty')
df_train['occupation'].value_counts() # occupation 등 결측치 제거되었는지 확인하기
plt.figure(figsize=(10,10))
plt.xticks(rotation=45)
sb.countplot("occupation", hue='income', data=df_train)
plt.figure(figsize=(10,10))
sb.countplot("native-country", hue='income', data=df_train)
plt.figure(figsize=(8,8))
sb.countplot("country", hue='income', data=df_train)
plt.figure(figsize=(8,8))
sb.countplot("race", hue='income', data=df_train)
plt.figure(figsize=(8,8))
sb.countplot("race", hue='income', data=df_train)
plt.figure(figsize=(8,8))
sb.countplot("workclass", hue='income', data=df_train)
plt.figure(figsize=(8,8))
sb.countplot("sex", hue='income', data=df_train)
# 히스토그램 : 클래스별 계수 그래프 (kdeplot()) 사용해서 확인
plt.figure(figsize=(8,8))
sb.kdeplot(df_train[df_train['income']==1]["hours-per-week"])
sb.kdeplot(df_train[df_train['income']==0]["hours-per-week"])
plt.legend(['income(1)','income(0)'])
df_train['hours-per-week'].groupby(df_train['income'], as_index=1).count()
#### 2개 이상의 관계를 보여주는 "그래프" ####

# factorplot : 2개 컬럼과 Survived컬럼과의 관계를 한번에 보여주는 것
# 세로선의 길이는 임의 표본을 뽑았을 때 최소비율과 최대비율을 표시한 것
sb.factorplot("Pclass", "Survived", hue="Sex", data=df_train)

# 히스토그램 : 클래스별 계수 그래프 (kdeplot())
# 나이를 기준으로 생존률을 표시 (나이가 연속적인 값의 특징을 가지고 있기때문에 사용)
sb.kdeplot(df_train[df_train['Survived']==1]["Age"])
sb.kdeplot(df_train[df_train['Survived']==0]["Age"])
plt.legend(['Survived(1)','Survived(0)'])

# violinplot() : 2개 칼럼과 Survived 컬럼과의 관계를 표시하는 것
# hue = 기준점!
sb.violinplot('Pclass','Age',hue='Survived', data=df_train,scale="count",split=1)
plt.show()
####  2개 이상의 관계를 보여주는 "표:groupby / crosstab" ####

# Groupby : 해당 칼럼을 기존의 상대칼럼으로 묶음
# as_index=True : 클래스를 출력 (False : 인덱스 값으로 출력)
# count() : 개수를 세는 함수
df_train[['Pclass','Survived']].groupby(df_train['Pclass'], as_index=1).count()
# 생존자 수만 계산
df_train[["Pclass", "Survived"]].groupby(df_train['Pclass'], as_index=True).sum()
# 생존률 계산
df_train[["Pclass", "Survived"]].groupby(df_train['Pclass'], as_index=True).mean()
# 형제,자매와 생존률
pd.crosstab(df_train['SibSp'],df_train['Survived'],margins=True).style.background_gradient(cmap="summer_r")
df_train['marital']=df_train['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse',' Divorced',' Never-married',' Widowed',' Married-spouse-absent',' Separated'],
                        ['Family','Family','N_Family','N_Family','N_Family','N_Family','N_Family'])
df_test['marital']=df_test['marital-status'].replace([' Married-AF-spouse',' Married-civ-spouse',' Divorced',' Never-married',' Widowed',' Married-spouse-absent',' Separated'],
                        ['Family','Family','N_Family','N_Family','N_Family','N_Family','N_Family'])
# Asia로 바꾸기 (12개국가)
df_train['country']=df_train['native-country'].replace([' Philippines',' India',' China',' Japan',' Vietnam',' Taiwan',
                                                        ' Cambodia',' Hong',' Thailand',' Laos', ' South',' Iran'],
                                                       ['Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia'])
df_test['country']=df_test['native-country'].replace([' Philippines',' India',' China',' Japan',' Vietnam',' Taiwan',
                                                        ' Cambodia',' Hong',' Thailand',' Laos', ' South',' Iran'],
                                                       ['Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia','Asia'])                                                       
# N_America로 바꾸기 (4개국가(미국이 2개! = 5개)
df_train['country']=df_train['country'].replace([' United-States','United-States',' Mexico',' Canada',' Outlying-US(Guam-USVI-etc)'],
                                                       ['N_America','N_America','N_America','N_America','N_America'])
df_test['country']=df_test['country'].replace([' United-States','United-States',' Mexico',' Canada',' Outlying-US(Guam-USVI-etc)'],
                                                       ['N_America','N_America','N_America','N_America','N_America'])                                                       
# S_America로 바꾸기 (13개국가)','
df_train['country']=df_train['country'].replace([' Puerto-Rico',' El-Salvador',' Cuba',' Jamaica',' Columbia',' Dominican-Republic',
                                                       ' Guatemala',' Haiti',' Nicaragua',' Peru',' Ecuador',' Trinadad&Tobago',' Honduras'],
                                                       ['S_America','S_America','S_America','S_America','S_America','S_America','S_America',
                                                       'S_America','S_America','S_America','S_America','S_America','S_America'])
df_test['country']=df_test['country'].replace([' Puerto-Rico',' El-Salvador',' Cuba',' Jamaica',' Columbia',' Dominican-Republic',
                                                       ' Guatemala',' Haiti',' Nicaragua',' Peru',' Ecuador',' Trinadad&Tobago',' Honduras'],
                                                       ['S_America','S_America','S_America','S_America','S_America','S_America','S_America',
                                                       'S_America','S_America','S_America','S_America','S_America','S_America'])                                                       
# Europe로 바꾸기 (12개국가)
df_train['country']=df_train['country'].replace([' Germany',' England',' Italy',' Poland',' Portugal',' Greece',
                                                        ' Ireland',' France',' Yugoslavia',' Hungary', ' Scotland',' Holand-Netherlands'],
                                                       ['Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe',
                                                       'Europe','Europe','Europe','Europe'])
df_test['country']=df_test['country'].replace([' Germany',' England',' Italy',' Poland',' Portugal',' Greece',
                                                        ' Ireland',' France',' Yugoslavia',' Hungary', ' Scotland',' Holand-Netherlands'],
                                                       ['Europe','Europe','Europe','Europe','Europe','Europe','Europe','Europe',
                                                       'Europe','Europe','Europe','Europe'])                                                       
# df_train["Age_cat"] = 0

# df_train.loc[df_train['age'] < 20, "Age_cat"] = 1
# df_train.loc[(20 <= df_train['age'])&(df_train['age']< 40), "Age_cat"] = 2
# df_train.loc[(40 <= df_train['age'])&(df_train['age']< 60), "Age_cat"] = 3
# df_train.loc[60 <= df_train['age'], "Age_cat"] = 4

# df_test["Age_cat"] = 0

# df_test.loc[df_test['age'] < 20, "Age_cat"] = 1
# df_test.loc[(20 <= df_test['age'])&(df_test['age']< 40), "Age_cat"] = 2
# df_test.loc[(40 <= df_test['age'])&(df_test['age']< 60), "Age_cat"] = 3
# df_test.loc[60 <= df_test['age'], "Age_cat"] = 4
df_train['education'] = df_train['education'].replace([' Preschool', ' 1st-4th', ' 5th-6th',' 7th-8th', ' 9th', ' 10th', ' 11th', 
                                                       ' 12th', ' Assoc-acdm', ' Assoc-voc', ' HS-grad', ' Some-college',
                                                      ' Prof-school',' Bachelors', ' Masters', ' Doctorate' ],
                                                      ['minor','minor','minor','minor','minor','minor','minor','minor',
                                                       'Associates','Associates','HS-graduate','HS-graduate','Prof-school', 
                                                       'Bachelors', 'Masters', 'Doctorate' ])
df_test['education'] = df_test['education'].replace([' Preschool', ' 1st-4th', ' 5th-6th',' 7th-8th', ' 9th', ' 10th', ' 11th', 
                                                       ' 12th', ' Assoc-acdm', ' Assoc-voc', ' HS-grad', ' Some-college',
                                                      ' Prof-school',' Bachelors', ' Masters', ' Doctorate' ],
                                                      ['minor','minor','minor','minor','minor','minor','minor','minor',
                                                       'Associates','Associates','HS-graduate','HS-graduate','Prof-school', 
                                                       'Bachelors', 'Masters', 'Doctorate' ])
df_train['occupation'] = df_train['occupation'].replace([' Prof-specialty',' Exec-managerial',' Tech-support',' Protective-serv',
                                                         " Craft-repair"," Adm-clerical",' Sales', ' Other-service', ' Machine-op-inspct',
                                                          ' Transport-moving', ' Handlers-cleaners', ' Farming-fishing',
                                                          ' Protective-serv', ' Priv-house-serv', ' Armed-Forces'],
                                                         ['a','a','a','a','b','b','b','b','b','b','b','b','b','b','b'])
df_test['occupation'] = df_test['occupation'].replace([' Prof-specialty',' Exec-managerial',' Tech-support',' Protective-serv',
                                                         " Craft-repair"," Adm-clerical",' Sales', ' Other-service', ' Machine-op-inspct',
                                                          ' Transport-moving', ' Handlers-cleaners', ' Farming-fishing',
                                                          ' Protective-serv', ' Priv-house-serv', ' Armed-Forces'],
                                                       ['a','a','a','a','b','b','b','b','b','b','b','b','b','b','b'])
df_train['marital'] = df_train['marital'].map({'Family':0, 'N_Family':1})
df_test['marital'] = df_test['marital'].map({'Family':0, 'N_Family':1})
# race을 숫자로
df_train.race = df_train.race.map({' Amer-Indian-Eskimo':0,' Asian-Pac-Islander':1,' Black':2,' Other':3,' White':4})
df_test.race = df_test.race.map({' Amer-Indian-Eskimo':0,' Asian-Pac-Islander':1,' Black':2,' Other':3,' White':4})
# country을 숫자로
df_train.country = df_train.country.map({'N_America':0,'S_America':1,'Asia':2,'Europe':3})
df_test.country = df_test.country.map({'N_America':0,'S_America':1,'Asia':2,'Europe':3})
# 성별 문자를 숫자로 바꾸기 
df_train['sex'] = df_train['sex'].map({' Male':0, ' Female':1})
df_test['sex'] = df_test['sex'].map({' Male':0, ' Female':1})
# # hours-per-week 분류 및 숫자로 바꾸기

# df_train['New_hour'] = 0

# df_train.loc[df_train['hours-per-week'] < 25, "New_hour"] = 1
# df_train.loc[(25 <= df_train['hours-per-week'])&(df_train['hours-per-week']< 40), "New_hour"] = 2
# df_train.loc[(40 <= df_train['hours-per-week'])&(df_train['hours-per-week']< 60), "New_hour"] = 3
# df_train.loc[60 <= df_train['hours-per-week'], "New_hour"] = 4

# df_test["New_hour"] = 0

# df_test.loc[df_test['hours-per-week'] < 25, "New_hour"] = 1
# df_test.loc[(25 <= df_test['hours-per-week'])&(df_test['hours-per-week']< 40), "New_hour"] = 2
# df_test.loc[(40 <= df_test['hours-per-week'])&(df_test['hours-per-week']< 60), "New_hour"] = 3
# df_test.loc[60 <= df_test['hours-per-week'], "New_hour"] = 4
# education을 숫자로
df_train.education = df_train.education.map({'minor':0,'Associates':1,'HS-graduate':2,'Prof-school':3,'Bachelors':4,'Masters':5,'Doctorate':6})
df_test.education = df_test.education.map({'minor':0,'Associates':1,'HS-graduate':2,'Prof-school':3,'Bachelors':4,'Masters':5,'Doctorate':6})
# education 컬럼 one-hot encoding
df_train = pd.get_dummies(df_train, columns=["education"], prefix="edu")
df_test = pd.get_dummies(df_test, columns=["education"], prefix="edu")
# occupation 컬럼 one-hot encoding
df_train = pd.get_dummies(df_train, columns=["occupation"], prefix="occ")
df_test = pd.get_dummies(df_test, columns=["occupation"], prefix="occ")
# race 컬럼 one-hot encoding
df_train = pd.get_dummies(df_train, columns=["race"], prefix="ra")
df_test = pd.get_dummies(df_test, columns=["race"], prefix="ra")
# sex 컬럼 one-hot encoding
df_train = pd.get_dummies(df_train, columns=["sex"], prefix="sex")
df_test = pd.get_dummies(df_test, columns=["sex"], prefix="sex")
# marital 컬럼 one-hot encoding
df_train = pd.get_dummies(df_train, columns=["marital"], prefix="mar")
df_test = pd.get_dummies(df_test, columns=["marital"], prefix="mar")
# country 컬럼 one-hot encoding
df_train = pd.get_dummies(df_train, columns=["country"], prefix="c")
df_test = pd.get_dummies(df_test, columns=["country"], prefix="c")
df_train #확인
# del df_train['age']
# del df_test['age']

del df_train['workclass']
del df_test['workclass']

del df_train['education-num']
del df_test['education-num']

del df_train['marital-status']
del df_test['marital-status']

# del df_train['hours-per-week']
# del df_test['hours-per-week']

del df_train['native-country']
del df_test['native-country']

del df_train['relationship']
del df_test['relationship']

del df_train['no']
del df_test['no']
df_test #확인
heatmapCo = df_train[['income', 'capital-gain', 'capital-loss', 'marital', 'country', 'Age_cat', 'New_hour']]
plt.figure(figsize=(12,12))
sb.heatmap(heatmapCo.astype(float).corr(),annot=True)
df_train
Y = df_train.iloc[:,5]
X = df_train.drop(labels=['income'],axis=1)
Y

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
trainX, testX, trainY, testY = train_test_split(X,Y,random_state=0)
!pip install xgboost
import xgboost
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler_m = StandardScaler()
scaler_m.fit(trainX)

trainX_scaler = scaler_m.transform(trainX)
testX_scaler = scaler_m.transform(testX)
df_test_scaler = scaler_m.transform(df_test)
# LinearSVC (scaler O
model_svc = LinearSVC(C=0.1,random_state=3)
model_svc.fit(trainX_scaler, trainY)
model_svc.score(testX_scaler,testY)
# xgboost (scaler X
model_xgb = XGBClassifier(learning_rate=0.1,n_estimators=1000,max_depth=4,gamma=2)
model_xgb.fit(trainX, trainY)
model_xgb.score(testX,testY)
# xgboost (scaler O
model_xgb = XGBClassifier(learning_rate=0.1,n_estimators=500,max_depth=2,gamma=0)
model_xgb.fit(trainX_scaler, trainY)
model_xgb.score(testX_scaler,testY)
# RandomForest
model_rf = RandomForestClassifier(n_estimators=500,max_depth=2,random_state=0)
model_rf.fit(trainX, trainY)
model_rf.score(testX,testY)
# GradientBoosting (scaler X
model_gbm = GradientBoostingClassifier(random_state=0)
model_gbm.fit(trainX, trainY)
model_gbm.score(testX,testY)
# GradientBoosting (scaler O
model_gbm = GradientBoostingClassifier(random_state=3)
model_gbm.fit(trainX_scaler, trainY)
model_gbm.score(testX_scaler,testY)
# 다른 방법으로 / 파라미터값 조정 / 확률은 잘?

learning_rate = [0.0001, 0.001, 0.1, 1, 10, 100, 1000]
n_estimators = [100,200,300,400,500,600,700]
max_depth = [1,2,3,4,5,6,7]
gamma = [0.001,0.01,0.1,1,10,100,1000]


for i in range(len(learning_rate)):
    model_xgb = XGBClassifier(learning_rate=learning_rate[i],n_estimators=n_estimators[i],max_depth=max_depth[i])
    model_xgb.fit(trainX, trainY)
    print("learning_rate 값 : ", learning_rate[i])
    print("n_estimators 값 : ", n_estimators[i])
    print("max_depth 값 : ",max_depth[i])
    print("Score : ", model_xgb.score(testX, testY))
# GridSearchCV
from sklearn.model_selection import GridSearchCV

# 파라미터 값을 설정
param = {"C":[0.001,0.01,0.1,1,10,100,1000],'n_estimators':[500],'max_depth':[2], 'gamma':[1]}

# (사용할 모델, 검색할 파라미터 목록,Kfold값, 교차 검증값을 반환할지 )
grid_search = GridSearchCV(XGBClassifier(), param, cv=5, return_train_score=True)
grid_search.fit(trainX, trainY)
grid_search.score(testX, testY)
answer_xgb = model_xgb.predict(df_test_scaler)
answer_df = pd.DataFrame(columns=['income'])
for i in range(len(df_test)):
    answer_df.loc[i,['income']] = answer_xgb[i]
answer_df
answer_df.to_csv('answer_df_xgb66.csv', encoding='euc-kr', index=False) # submisstion