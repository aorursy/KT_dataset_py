import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_set = pd.read_csv('./train.csv')
test_set = pd.read_csv('./test.csv')

data_set.head()
print(data_set.dtypes)
print(data_set.info())
print(data_set.describe())
'''
1. 里面有空数据
2. 大部分人都挂掉了
3. 头等舱 + 商务舱的人偏少
4. 有婴儿 + 老人
'''
# 观察年龄
data_set.Age[0:10]
data_set.Age.hist()
plt.title('Age')
plt.xlabel('Age')
plt.ylabel('People Count')
plt.show()

plt.scatter(data_set.Survived, data_set.Age)
plt.title('Age Survice')
plt.xlabel('Suivived')
plt.ylabel('Age')
plt.show()
# 看看性别
data_set.Sex[0:10]
m = data_set.Survived[data_set.Sex == 'male'].value_counts()
f = data_set.Survived[data_set.Sex == 'female'].value_counts()

print(m, f)

df = pd.DataFrame({ 'male': m, 'female': f })
df
df.plot(kind = 'bar', stacked=True)
plt.title('Survived by sex')
plt.xlabel('live')
plt.ylabel('count number')
# 看看船票
# data_set.Fare.max()
# data_set.Fare.hist()
# plt.xlabel('Fare')
# plt.ylabel('Count')
# plt.show()

survived_0 = data_set.Fare[data_set.Survived == 0]
survived_0.hist()
plt.title('survived 0')
plt.show()

survived_1 = data_set.Fare[data_set.Survived == 1]
survived_1.hist()
plt.title('survived 1')
plt.show()

plt.scatter(data_set.Survived, data_set.Fare)
plt.xlabel('Survived')
plt.ylabel('Fare Count number')
plt.show()
# 船舱
# data_set.Pclass[0:10]
# data_set.Pclass.hist()
# plt.show()

survived_p1 = data_set.Survived[data_set.Pclass == 1].value_counts()
survived_p2 = data_set.Survived[data_set.Pclass == 2].value_counts()
survived_p3 = data_set.Survived[data_set.Pclass == 3].value_counts()

df = pd.DataFrame({ 'top': survived_p1, 'mid': survived_p2, 'lower': survived_p3 })
df.plot(kind='bar', stacked=True)
plt.xlabel('live')
plt.ylabel('Count')
plt.show()
# 登船地点
data_set.Embarked[0:10]


survived_S = data_set.Survived[data_set.Embarked == 'S'].value_counts()
survived_C = data_set.Survived[data_set.Embarked == 'C'].value_counts()
survived_Q = data_set.Survived[data_set.Embarked == 'Q'].value_counts()

df = pd.DataFrame({ 'S': survived_S, 'C': survived_C, 'Q': survived_Q  })
df.plot(kind='bar', stacked=True)
plt.show()
# 保留有效数据

label = data_set.loc[:, 'Survived']
data = data_set.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdata = test_set.loc[:, ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
def fill_Nan(data):
    data_copy = data.copy(deep=True)
    # 填充值 取0 median，max, mean？ mean or median
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    
    return data_copy
    
data_no_nan = fill_Nan(data)
print(data_no_nan)
def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy.Sex == 'female', 'Sex'] = 0     
    data_copy.loc[data_copy.Sex == 'male', 'Sex'] = 1
    
    return data_copy

data_after_sex = transfer_sex(data_no_nan)

def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy.Embarked == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy.Embarked == 'Q', 'Embarked'] = 1   
    data_copy.loc[data_copy.Embarked == 'C', 'Embarked'] = 2 
    return data_copy

data_after_embarked = transfer_embark(data_after_sex)

data_after_embarked.head()
# 利用KNN训练数据

from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(data_after_embarked, label, random_state=0, train_size=0.8)

print(train_data.shape, test_data.shape, train_label.shape, test_label.shape)
# 训练
from sklearn.neighbors import KNeighborsClassifier
k = 10
classifier = KNeighborsClassifier(n_neighbors = k)
classifier.fit(train_data, train_label)

# 预测
predictions = classifier.predict(test_data)
print(predictions)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


print(accuracy_score(test_label, predictions))
print(classification_report(test_label, predictions))
print(confusion_matrix(test_label, predictions))
# 找到最好的k值
from sklearn.model_selection import cross_val_score

cross_val_score(classifier, data_after_embarked, label, cv=5, scoring='accuracy')