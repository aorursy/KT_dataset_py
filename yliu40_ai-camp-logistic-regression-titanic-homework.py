import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库
import seaborn as sns
data = pd.read_csv('../input/train.csv')
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
len(data.index)
data.columns
data.head(3)
sns.countplot(data['Survived'])
female = data.Survived[data.Sex == 'female'].value_counts().sort_index()
female.plot(kind='bar',label='female', color='red', alpha=0.6)
plt.show()
male = data.Survived[data.Sex == 'male'].value_counts().sort_index()
male.plot(kind='bar',label='male', color='blue', alpha=0.6)
plt.show()
highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()
lowclass = data.Survived[data.Pclass == 3].value_counts().sort_index()
lowclass.plot(kind='bar',label='lowclass', color='blue', alpha=0.6)
plt.show()
y,X= dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = data, return_type = 'dataframe')
model = LogisticRegression()
print (X.shape,y.shape)
model.fit(X, y)
from sklearn.metrics import accuracy_score
accuracy_score(model.predict(X),y)
1-y.mean()
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))
test_data = pd.read_csv('../input/test.csv')
test_data.head(5)
test_data['Survived'] = 1
test_data.head(5)
ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data = test_data, return_type='dataframe')
Xtest.head(5)
y_pred = model.predict(Xtest)
pd.DataFrame(list(zip(test_data['PassengerId'], y_pred)), columns=['PassengerID', 'Survived']).to_csv('../my_prediction.csv')
y_pred
