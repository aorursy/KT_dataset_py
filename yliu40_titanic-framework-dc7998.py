import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")
print (train_data.shape,test_data.shape)
train_data.columns
train_data.head(3)
print (train_data.dtypes)
print (train_data.describe())
survived_m = train_data.Survived[train_data.Sex == 'male'].value_counts()
survived_f = train_data.Survived[train_data.Sex == 'female'].value_counts()

df=pd.DataFrame({'male':survived_m,'female':survived_f})
df.plot(kind='bar', stacked=True)
plt.title('survived by sex')
plt.xlabel('survived')
plt.ylabel('count')
plt.show()
train_data[train_data.Survived==0]['Age'].hist()
plt.title('Age distribution of people who did not survive')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()

train_data[train_data.Survived==1]['Age'].hist()
plt.title('Age distribution of people who survive')
plt.xlabel('Age')
plt.ylabel('count')
plt.show()
train_data['Fare'].hist()
plt.title('Fare distribution')
plt.xlabel('Fare')
plt.ylabel('count')
plt.show()

train_data[train_data.Survived==0]['Fare'].hist()
plt.title('Fare distribution of people who did not survive')
plt.xlabel('Fare')
plt.ylabel('count')
plt.show()

train_data[train_data.Survived==1]['Fare'].hist()
plt.title('Fare distribution of people who survive')
plt.xlabel('Fare')
plt.ylabel('count')
plt.show()
train_data['Pclass'].hist()
plt.title('Pclass distribution')
plt.xlabel('Pclass')
plt.ylabel('count')
plt.show()

Survived_p1 = train_data.Survived[train_data['Pclass'] == 1].value_counts()
Survived_p2 = train_data.Survived[train_data['Pclass'] == 2].value_counts()
Survived_p3 = train_data.Survived[train_data['Pclass'] == 3].value_counts()

df = pd.DataFrame({'p1':Survived_p1,'p2':Survived_p2,'p3':Survived_p3})
df.plot(kind='bar', stacked=True)
plt.title('survive by Pclass')
plt.xlabel('Pclass')
plt.ylabel('count')
plt.show()
Survived_S = train_data.Survived[train_data['Embarked'] == 'S'].value_counts()
Survived_C = train_data.Survived[train_data['Embarked'] == 'C'].value_counts()
Survived_Q = train_data.Survived[train_data['Embarked'] == 'Q'].value_counts()

df = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title("Survived by Embarked")
plt.xlabel("Survival") 
plt.ylabel("count")
plt.show()
label = train_data.loc[:,'Survived']
traindata = train_data.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdata=test_data.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
print (traindata.shape, testdata.shape)
def fill_None(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

traindata_filled = fill_None(traindata)
testdata_filled = fill_None(testdata)

print (traindata.isnull().values.any(), traindata_filled.isnull().values.any())
print (testdata.isnull().values.any(), testdata_filled.isnull().values.any())
def convert_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex']=='female','Sex']=0
    data_copy.loc[data_copy['Sex']=='male','Sex']=1
    return data_copy

traindata_convert_sex = convert_sex(traindata_filled)
testdata_convert_sex = convert_sex(testdata_filled)
def convert_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

traindata_convert_embark = convert_embark(traindata_convert_sex)
testdata_convert_embark= convert_embark(testdata_convert_sex)
print(testdata_convert_embark.loc[:3])
from sklearn.model_selection import train_test_split
train_dataset,val_dataset,train_labels,val_labels=train_test_split(traindata_convert_embark,label,random_state=0,test_size=0.2)
print (train_dataset.shape, val_dataset.shape, train_labels.shape, val_labels.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_range = range(1, 51)
k_scores = []
for K in k_range:
    classifer=KNeighborsClassifier(n_neighbors = K)
    classifer.fit(train_dataset,train_labels)
    #print('current K =', K)
    predictions=classifer.predict(val_dataset)
    score = accuracy_score(val_labels,predictions)
    #print(score)
    k_scores.append(score)
plt.plot(k_range,k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print (np.argmax(k_scores)+1)
classifier_final = KNeighborsClassifier(n_neighbors = 33)
classifier_final.fit(traindata_convert_embark, label)
test_pred = classifier_final.predict(testdata_convert_embark)
# 预测
#print (test_pred)

# 检测模型precision， recall 等各项指标
print (test_data.head(3))
pd.DataFrame({"PassengerId":test_data["PassengerId"], "Survived":test_pred}).to_csv('submission.csv', header=True, index=False)
