import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit, learning_curve
#导入train.csv和test.csv
data_train = pd.read_csv("C:/Users/79359/Desktop/train.csv")
data_test = pd.read_csv("C:/Users/79359/Desktop/test.csv")
data_train["Fname"] = data_train.Name.apply(lambda x: x.split(",")[0])
data_test["Fname"] = data_test.Name.apply(lambda x: x.split(",")[0])
data_train["Family_size"] = data_train["SibSp"] + data_train["Parch"]
data_test["Family_size"] = data_test["SibSp"] + data_test["Parch"]
dead_train = data_train[data_train["Survived"] == 0]
fname_ticket = dead_train[(dead_train["Sex"] == "female") & (dead_train["Family_size"] >= 1)][["Fname", "Ticket"]]
data_train["dead_family"] = np.where(data_train["Fname"].isin(fname_ticket["Fname"]) & data_train["Ticket"].isin(fname_ticket["Ticket"]) & ((data_train["Age"] >=1) | data_train.Age.isnull()), 1, 0)
data_test["dead_family"] = np.where(data_test["Fname"].isin(fname_ticket["Fname"]) & data_test["Ticket"].isin(fname_ticket["Ticket"]) & ((data_test["Age"] >=1) | data_test.Age.isnull()), 1, 0)
live_train = data_train[data_train["Survived"] == 1]
live_fname_ticket = live_train[(live_train["Sex"] == "male") & (live_train["Family_size"] >= 1) & ((live_train["Age"] >= 18) | (live_train["Age"].isnull()))][["Fname", "Ticket"]]
data_train["live_family"] = np.where(data_train["Fname"].isin(live_fname_ticket["Fname"]) & data_train["Ticket"].isin(live_fname_ticket["Ticket"]), 1, 0)
data_test["live_family"] = np.where(data_test["Fname"].isin(live_fname_ticket["Fname"]) & data_test["Ticket"].isin(live_fname_ticket["Ticket"]), 1, 0)
dead_man_fname_ticket = data_train[(data_train["Family_size"] >= 1) & (data_train["Sex"] == "male") & (data_train["Survived"] == 0) & (data_train["dead_family"] == 0)][["Fname", "Ticket"]]
data_train["deadfamily_man"] = np.where(data_train["Fname"].isin(dead_man_fname_ticket["Fname"]) & data_train["Ticket"].isin(dead_man_fname_ticket["Ticket"]) & (data_train.Sex == "male"), 1, 0)
data_train["deadfamily_woman"] = np.where(data_train["Fname"].isin(dead_man_fname_ticket["Fname"]) & data_train["Ticket"].isin(dead_man_fname_ticket["Ticket"]) & (data_train.Sex == "female"), 1, 0)
data_test["deadfamily_man"] = np.where(data_test["Fname"].isin(dead_man_fname_ticket["Fname"]) & data_test["Ticket"].isin(dead_man_fname_ticket["Ticket"]) & (data_test.Sex == "male"), 1, 0)
data_test["deadfamily_woman"] = np.where(data_test["Fname"].isin(dead_man_fname_ticket["Fname"]) & data_test["Ticket"].isin(dead_man_fname_ticket["Ticket"]) & (data_test.Sex == "female"), 1, 0)

data_train.loc[(data_train["dead_family"] == 0) & (data_train["live_family"] == 0) & (data_train["deadfamily_man"] == 0) & (data_train["deadfamily_woman"] == 0) & (data_train["Family_size"] >= 1) & (data_train["Sex"] == "male"), "deadfamily_man"] = 1
data_train.loc[(data_train["dead_family"] == 0) & (data_train["live_family"] == 0) & (data_train["deadfamily_man"] == 0) & (data_train["deadfamily_woman"] == 0) & (data_train["Family_size"] >= 1) & (data_train["Sex"] == "female"), "deadfamily_woman"] = 1
data_test.loc[(data_test["dead_family"] == 0) & (data_test["live_family"] == 0) & (data_test["deadfamily_man"] == 0) & (data_test["deadfamily_woman"] == 0) & (data_test["Family_size"] >= 1) & (data_test["Sex"] == "male"), "deadfamily_man"] = 1
data_test.loc[(data_test["dead_family"] == 0) & (data_test["live_family"] == 0) & (data_test["deadfamily_man"] == 0) & (data_test["deadfamily_woman"] == 0) & (data_test["Family_size"] >= 1) & (data_test["Sex"] == "female"), "deadfamily_woman"] = 1

grp_tk = data_train.drop(["Survived"], axis=1).append(data_test).groupby(["Ticket"])
tickets = []
for grp, grp_train in grp_tk:
    ticket_flag = True
    if len(grp_train) != 1:
        for i in range(len(grp_train) - 1):
            if grp_train.iloc[i]["Fname"] != grp_train.iloc[i+1]["Fname"]:
                ticket_flag = False
    if ticket_flag == False:
        tickets.append(grp)
data_train.loc[(data_train.Ticket.isin(tickets)) & (data_train.Family_size == 0) & (data_train.Sex == "male"), "deadfamily_man"] = 1
data_train.loc[(data_train.Ticket.isin(tickets)) & (data_train.Family_size == 0) & (data_train.Sex == "female"), "deadfamily_woman"] = 1
data_test.loc[(data_test.Ticket.isin(tickets)) & (data_test.Family_size == 0) & (data_test.Sex == "male"), "deadfamily_man"] = 1
data_test.loc[(data_test.Ticket.isin(tickets)) & (data_test.Family_size == 0) & (data_test.Sex == "female"), "deadfamily_woman"] = 1

data_train['Title'] = data_train.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(data_train['Title'], data_train['Sex'])
data_train['Title'] = data_train['Title'].replace(['Lady', 'Countess', 'Capt','Col', 'Don', 'Dr',
                                             'Major','Rev', 'Sir', 'Jonkheer', 'Dona'],
                                            'Rare')
data_train['Title'] = data_train['Title'].replace('Mlle', 'Miss')
data_train['Title'] = data_train['Title'].replace('Ms', 'Miss')
data_train['Title'] = data_train['Title'].replace('Mme', 'Mrs')

data_test['Title'] = data_test.Name.str.extract('([A-Za-z]+)\.', expand=False)
data_test['Title'] = data_test['Title'].replace(['Lady', 'Countess', 'Capt','Col', 'Don', 'Dr',
                                             'Major','Rev', 'Sir', 'Jonkheer', 'Dona'],
                                            'Rare')
data_test['Title'] = data_test['Title'].replace('Mlle', 'Miss')
data_test['Title'] = data_test['Title'].replace('Ms', 'Miss')
data_test['Title'] = data_test['Title'].replace('Mme', 'Mrs')
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train = set_Cabin_type(data_train)
data_test = set_Cabin_type(data_test)
titles=['Rare', 'Miss', 'Mrs', 'Master', 'Mr']
for title in titles:
    data_train.loc[(data_train.Age.isnull()) & (data_train['Title'] == title), 'Age'] = \
        data_train[data_train['Title'] == title].Age.mean()

dummies_Title = pd.get_dummies(data_train['Title'], prefix= 'Title')

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,dummies_Title], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Embarked_C','Embarked_Q','Pclass_2','Title'], axis=1, inplace=True)
df.head()
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].to_numpy().reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].to_numpy().reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].to_numpy().reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].to_numpy().reshape(-1,1), fare_scale_param)
df.head()
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
#对test_data做和train_data中一致的特征变换
tmp_df = data_test[['Age','Fare','Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values

#填补age缺失值 同一title的年龄做平均 王逸轩
for line in data_test:
 xx=data_test['Title']
 data_test.loc[(data_test.Age.isnull()), 'Age'] = data_test[data_test['Title'] == xx].Age.mean()

#特征量化
dummies_Title = pd.get_dummies(data_test['Title'], prefix= 'Title')

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

#设置df_test并去除多余的标签
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,dummies_Title], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Embarked_C','Embarked_Q','Pclass_2'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)
tickets = data_test.Ticket.value_counts().keys().to_numpy()
data_test['Same_Ticket']=range(len(data_test))
for i in range(tickets.size):
    indexList=data_test.loc[data_test.Ticket.astype(str) == tickets[i]].index.to_numpy()
    for j in indexList:
        data_test.loc[j ,'Same_Ticket']=data_test.Ticket.value_counts()[tickets[i]]
df_test['Same_Ticket']=data_test['Same_Ticket']
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df_test['Same_Ticket'].to_numpy().reshape(-1,1))
df_test['Same_Ticket_scaled'] = scaler.fit_transform(df_test['Same_Ticket'].to_numpy().reshape(-1,1), age_scale_param)

tickets = data_train.Ticket.value_counts().keys().to_numpy()
data_train['Same_Ticket']=range(len(data_train))
for i in range(tickets.size):
    indexList=data_train.loc[data_train.Ticket.astype(str) == tickets[i]].index.to_numpy()
    for j in indexList:
        data_train.loc[j ,'Same_Ticket']=data_train.Ticket.value_counts()[tickets[i]]
data_train
df['Same_Ticket']=data_train['Same_Ticket']
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Same_Ticket'].to_numpy().reshape(-1,1))
df['Same_Ticket_scaled'] = scaler.fit_transform(df['Same_Ticket'].to_numpy().reshape(-1,1), age_scale_param)
#简单看看打分情况
clf = LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Same_Ticket_scaled|Title_.*|Family_size|dead_family|live_family|deadfamily_man|deadfamily_woman')
X = all_data.values[:,1:]
y = all_data.values[:,0]
print(cross_val_score(clf, X, y, cv=5))
from sklearn.ensemble import BaggingRegressor
#设置train.csv中用于训练的数据
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Same_Ticket_scaled|Title_.*|Family_size|dead_family|live_family|deadfamily_man|deadfamily_woman')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = LogisticRegression(solver='liblinear',C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

#设置test.csv中将用于预测的数据
test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Same_Ticket_scaled|Title_.*|Family_size|dead_family|live_family|deadfamily_man|deadfamily_woman')
predictions = bagging_clf.predict(test)
#设置预测结果的csv文件内容格式
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
#保存预测结果为csv文件
result.to_csv("C:/Users/79359/Desktop/bagging_predictions1.csv", index=False)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
#设置train.csv中用于训练的数据
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Same_Ticket_scaled|Title_.*|Family_size|dead_family|live_family|deadfamily_man|deadfamily_woman')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingClassifier之中
clf = RandomForestClassifier(n_estimators=100)
bagging_clf = BaggingClassifier(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

#设置test.csv中将用于预测的数据
test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Same_Ticket_scaled|Title_.*|Family_size|dead_family|live_family|deadfamily_man|deadfamily_woman')
predictions = bagging_clf.predict(test)
#设置预测结果的csv文件内容格式
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
#保存预测结果为csv文件
result.to_csv("C:/Users/79359/Desktop/bagging_predictions2.csv", index=False)
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import BaggingClassifier
#设置train.csv中用于训练的数据
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Same_Ticket_scaled|Title_.*|Family_size|dead_family|live_family|deadfamily_man|deadfamily_woman')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingClassifier之中
clf = SVC()
bagging_clf = BaggingClassifier(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

#设置test.csv中将用于预测的数据
test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Same_Ticket_scaled|Title_.*|Family_size|dead_family|live_family|deadfamily_man|deadfamily_woman')
predictions = bagging_clf.predict(test)
#设置预测结果的csv文件内容格式
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
#保存预测结果为csv文件
result.to_csv("C:/Users/79359/Desktop/bagging_predictions3.csv", index=False)
print(data_test)
#把train.csv中的title替换成数字，便于svc分析
data_train["Title"] = data_train["Title"].replace(["Mr"], 1)
data_train["Title"] = data_train["Title"].replace(["Miss"], 2)
data_train["Title"] = data_train["Title"].replace(["Mrs"], 3)
data_train["Title"] = data_train["Title"].replace(["Master"], 4)
data_train["Title"] = data_train["Title"].replace(["Royal"], 5)
data_train["Title"] = data_train["Title"].replace(["Rare"], 6)
#把test.csv中的title替换成数字，便于svc预测
data_test["Title"] = data_test["Title"].replace(["Mr"], 1)
data_test["Title"] = data_test["Title"].replace(["Miss"], 2)
data_test["Title"] = data_test["Title"].replace(["Mrs"], 3)
data_test["Title"] = data_test["Title"].replace(["Master"], 4)
data_test["Title"] = data_test["Title"].replace(["Rare"], 6)

#把年龄Age和票价Fare分层成AgeBin和FareBin，便于进行训练和预测
#年龄分层，不同层年龄替换成数字
data_train.loc[data_train["Age"] <= 15, "AgeBin"] = 0
data_train.loc[(data_train["Age"] > 15) & (data_train["Age"] <= 30), "AgeBin"] = 1
data_train.loc[(data_train["Age"] > 30) & (data_train["Age"] <= 49), "AgeBin"] = 2
data_train.loc[(data_train["Age"] > 49) & (data_train["Age"] < 80), "AgeBin"] = 3
data_train.loc[data_train["Age"] >= 80, "AgeBin"] = 4
data_test.loc[data_test["Age"] <= 15, "AgeBin"] = 0
data_test.loc[(data_test["Age"] > 15) & (data_test["Age"] <= 30), "AgeBin"] = 1
data_test.loc[(data_test["Age"] > 30) & (data_test["Age"] <= 49), "AgeBin"] = 2
data_test.loc[(data_test["Age"] > 49) & (data_test["Age"] < 80), "AgeBin"] = 3
data_test.loc[data_test["Age"] >= 80, "AgeBin"] = 4

#票价分层
data_train.loc[data_train["Fare"] <= 7.854, "FareBin"] = 0
data_train.loc[(data_train["Fare"] > 7.854) & (data_train["Fare"] <= 10.5), "FareBin"] = 1
data_train.loc[(data_train["Fare"] > 10.5) & (data_train["Fare"] <= 21.558), "FareBin"] = 2
data_train.loc[(data_train["Fare"] > 21.558) & (data_train["Fare"] <= 41.579), "FareBin"] = 3
data_train.loc[data_train["Fare"] > 41.579, "FareBin"] = 4
data_test.loc[data_test["Fare"] <= 7.854, "FareBin"] = 0
data_test.loc[(data_test["Fare"] > 7.854) & (data_test["Fare"] <= 10.5), "FareBin"] = 1
data_test.loc[(data_test["Fare"] > 10.5) & (data_test["Fare"] <= 21.558), "FareBin"] = 2
data_test.loc[(data_test["Fare"] > 21.558) & (data_test["Fare"] <= 41.579), "FareBin"] = 3
data_test.loc[data_test["Fare"] > 41.579, "FareBin"] = 4

#由于性别数据为字符串属性，将Sex标签分开为female和male两个标签，分别用1和0确定性别。
train_dummies_sex = pd.get_dummies(data_train["Sex"])
test_dummies_sex = pd.get_dummies(data_test["Sex"])
data_train = pd.concat([data_train, train_dummies_sex], axis=1)
data_test = pd.concat([data_test, test_dummies_sex], axis=1)
data_train = data_train.drop(["Sex"], axis=1)
data_test = data_test.drop(["Sex"], axis=1)
#去除训练集和待预测集中的无关数据、以其他方式替换的数据和不适合用来分析的数据
df_test=data_test
data_train = data_train.drop(["PassengerId", "Ticket", "Cabin", "Embarked", "Fname", "Name", "SibSp", "Parch","Age","Fare"], axis=1)
data_test = data_test.drop(["PassengerId", "Ticket", "Cabin", "Embarked", "Fname", "Name", "SibSp", "Parch","Age","Fare"], axis=1)

#SVC训练过程
y = data_train["Survived"]
train_x, val_x, train_y, val_y = train_test_split(data_train.drop(["Survived"], axis=1), y, test_size=0.2, random_state=0)
#测试发现参数 C 的影响较大
clf = SVC(C=67, probability=True)
#bagging_clf = BaggingClassifier(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
#bagging_clf.fit(train_x, train_y)
clf.fit(train_x, train_y)

predictions = clf.predict(data_test)
#设置预测结果的csv文件内容格式
result = pd.DataFrame({'PassengerId':df_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
#保存预测结果为csv文件
result.to_csv("C:/Users/79359/Desktop/predictions4.csv", index=False)