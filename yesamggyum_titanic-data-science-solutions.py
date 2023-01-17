# data analysis and wrangling
# 왜 랜덤 패키지를 임포트 한거지?
# 그리고 데이터 랭글링은 왜 필요한거지?
## 데이터 랭글링은 데이터 전처리와 같은 의미로 보여진다.
import pandas as pd
import numpy as np
import random as rnd

# visualization
# 시각화를 위해 seaborn 패키지와 matplotlib.pyplot 패키지를 임포트했다.
# %matplotlib inline는 노트북 내에서 그래프가 보이게끔 한다.
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
# 머신러닝을 위해 다양한 분류기, 모델 패키지를 임포트했다.
# 대부분 scikit learn인 sklearn에서 불러왔다.
# 로지스틱 회귀분석, 서포트 벡터 머신, 선형 서포트 벡터 머신
# 랜덤포레스트 분류기, K근접 분류기, 가우시안 나이브베이즈
# 퍼셉트론, 확률 그라디언트 하강 분류기, 의사결정나무
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# pandas 패키지의 read_csv 함수를 실행해 트레이닝, 테스트 데이터를 불러오고 이름을 지정한다.
# 특정 상황에서 두 데이터를 모두 사용하기 위해 결합한다. (combine에 저장됨)
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]
# 트레이닝 데이터셋의 속성 이름을 출력한다.
# 데이터셋의 속성으로 무엇이 있는지 확인하는 것
print(train_df.columns.values)
# preview the data
# 본격적인 분석에 앞서 트레이닝 데이터셋의 상위 5개를 출력한다.
# 각 속성과 속성에 해당하는 값으로 어떤 것이 있는지 확인할 수 있다.
# 값의 유형도 확인할 수 있다.
train_df.head()
# 트레이닝 데이터셋의 마지막 5개 데이터를 확인하다.
# 속성은 다를게 없는데 head와 마찬가지로 많은 결측값을 확인할 수 있다.
# 확인해둔 결측값은 추후 전처리 대상이 될 예정이다.
# 문자형 속성, 값도 숫자형으로 변환될 것이다.
train_df.tail()
# 트레이닝, 테스트 데이터셋의 속성이름, 데이터 유형, 개수, null 허용여부, 용량을 알 수 있다.
# 트레이닝 데이터셋의 수는 테스트 데이터셋의 것보다 2배 많고 속성도 1개 많다. (Survived)
train_df.info()
print('_'*40)
test_df.info()
train_df.describe()
# Review survived rate using `percentiles=[.61, .62]` knowing our problem description mentions 38% survival rate.
# Review Parch distribution using `percentiles=[.75, .8]`
# SibSp distribution `[.68, .69]`
# Age and Fare `[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]`

# describe를 통해 전체 속성의 통계를 확인할 수 있다.
# 코드 아래의 문구는 describe 함수의 파라미터인 percentiles를 변경하며 리뷰를 진행하라는 의미이다.
# Survived는 61%와 62%를 기준으로 1과 0이 나뉜다. 그래서 percentiles=[.61,.62]를 입력하면  
# 두 %에 따라 값이 다른 것을 확인할 수 있다.
# 이것으로 Survived 즉, 생존률이 얼마나 되는지 알 수 있다. (38%)
# 승선요금은 정말 다양하다.
train_df.describe(include=['O'])

#pandas의 describe 함수를 실행한다.
# 이 때 include의 값으로 ['O']를 입력한다.
# 문자형 데이터를 값으로 가지는 속성의 통계 결과가 나온다.
# 숫자형 데이터와 다르게 count, unique, top, freq에 대해 정리된다.
# Sex, Embarked의 고유값이 적은 걸로 보아 (1개, 3개) 모델 학습 속성으로 사용할 수 있을 것 같다.
# 트레이닝 데이터셋의 Pclass와 Survived를 선택했다.
# Pclass로 groupby한다. as_index는 False로 두어 첫번째 속성이 Pclass가 인덱스되지 않게 한다.
# Pclass의 고유값을 기준으로 그룹으로 묶으며 그에 따라 그룹화되는 Survived에 대해서는 평균이 계산된다. (mean)
# 마지막에 sort_values 함수를 통해 Survived 기준으로 정렬하고 ascending을 False로 두어 내림차순으로 확인한다.
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# seaborn 패키지를 사용해 히스토그램을 그렸다.
# column, 속성에는 Survived를 입력해 0과 1일 때의 그래프를 비교할 수 있게 했다.
# map은 FacetGrid를 통해 생성한 subplot에 실제 그래프를 그리는 작업이다.
# x축은 Age 속성을 20개로 나눈 것과 같다.
# 생존률이 가장 높은 연령대는 20~40세 사이로 보인다.
# 사망자보다 생존자가 많은 연령대는 흔치 않은데 유아 0~4세는 생존자가 더 많은 것으로 보여진다.
# 승객 대부분의 연령대를 확인할 수 있다.

g = sns.FacetGrid(train_df, col='Survived')
#g.map(plt.hist, 'Age', bins=20)
g.map(plt.hist, 'Age', bins=20)
# Survived와 Pclass의 상관관계를 확인할 수 있는 히스토그램을 그린다.
# 엑셀의 표처럼 행은 Pclass, 열은 Survived로 한다.
# 그러면 2*3의 그래프가 만들어진다.
# add_legend()는 뭐지? legend는 꼭 전설뿐만 아니라 범례라는 뜻도 갖고있다.
# 정말 Pclass=1 에서 생존률이 가장 높아보인다.
# Pclass=3 에서 생존률이 가장 낮아보인다.

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20, color='r')
grid.add_legend();
# 출발지, 생존, 성별, 좌석등급을 pointplot으로 표현했다.
# 출발지는 행으로 구분지었고 생존율은 그래프 내의 y축으로
# 성별은 색상으로 좌석등급은 x축으로 결정했다.
# 총 4개 속성의 상관을 한 그래프에서 확인할 수 있다.

# 아래 그래프에서 Pclass가 높을 수록 생존률이 높다는 게 어느정도 확인된다.
# 하지만 Pclass는 이미 Survived와의 상관을 위에서 확인했다.
# Pclass = 3의 생존률은 각기 다르다.

# S, Q는 여성의 생존률이 높고 C만 남성의 생존률이 높다.
# 이 점을 의심한 것으로 보인다. 
# 하지만 이게 꼭 Embarked - Survived의 상관이 아닐 수도 있다는 내용이다.
# Embarked - Pclass 일 수도 있고 여성이 C에서 적게 탑승했을 수 있다.

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
# 출발지와 생존율에 대한 그래프를 바플롯으로 나타냈다.
# 대부분 여성의 생존율이 높다.
# 출발지는 행, 속성은 생존여부, x축은 성별, y축은 운임요금
# 운임요금이 비쌀수록 생존율이 높은 것처럼 보여진다.

# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
# 속성을 drop하기 이전의 shape을 확인하고 drop한 후의 shape을 확인한다.
# 2개의 행이 줄어듬을 확인할 수 있다.

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
# combinea에는 데이터셋과 테스트셋이 모두 있다.
# 모든 데이터의 타이틀과 성별을 확인한다.
# 왜 테스트셋에 한정짓지 않았을까? 최대한 많은 데이터 타이틀과 성별의 사례를 수집하기 위함인걸까?

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])
# 아~ 타이틀이 성별에 의해 구분되기 때문에 성별 속성을 추가하여 표시했음을 알 수 있다.
# 소수의 타이틀은 Rare로 통일을 한다.
# 나머지 타이틀은 통합을 한다.
# 결과 5개의 타이틀만 남았다.
# 여성의 타이틀이 가장 높은 생존율을 보이고 박사(Master)도 높은 생존율을 보였다.

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
# 모델 학습을 위해 범주형 데이터를 서수형, 숫자형으로 바꾼 것을 확인할 수 있다.

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
# 등급, 성별, 연령에 따른 상관관계를 확인한다.
# 여아보다 남아가 많다.

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
guess_ages = np.zeros((2,3))
guess_ages
# null값을 drop하고 성별과 등급에 맞게 평균+-표준편차에 해당하는 랜덤값을 입력시켜준다.
# 다시 한 번 봐둬야겠다.

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()
# cut() 함수를 사용해 5개의 범위로 구분한다.
# AgeBand라는 신규 속성을 만들었다. 

train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
# 숫자형 데이터였던 Age가 서수형 데이터로 바뀌었다.

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()
# AgeBand 속성을 Age로 대체하고 AgeBand는 drop한다.

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()
# 본인을 포함한 (+1) 형제자매, 부모자식의 수를 더해 FammilySize라는 속성을 만들었다.

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# 혼자일수록 생존율이 높다.

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
# 최빈값을 'S' 로 채운다.

freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# 숫자형인 출발지 속성을 범주로 변경했다.

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
# Fare의 null에 중간값을 채워넣는다.

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()
# FareBand와 생존율과 양의 상관관계를 보인다.

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)
test_df.head(10)
# 모델 학습용 트레이닝 데이터의 Survived를 drop한다.
# PassengerId도 drop한다.

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron
# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc
# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)