# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
print(data.shape)

print(data.head())
data.describe()
data.describe().T
data.dtypes
data.info(verbose= True)
data2=data.copy(deep=True)
data2[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]= data2[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)
data2.isnull().sum()
p=data2.hist(figsize=(20,20))
data2['Pregnancies'].unique()
data2['Glucose'].fillna(data2['Glucose'].mean(), inplace=True)

data2['BloodPressure'].fillna(data2['BloodPressure'].mean(), inplace=True)

data2['SkinThickness'].fillna(data2['SkinThickness'].median(), inplace=True)

data2['Insulin'].fillna(data2['Insulin'].median(), inplace=True)

data2['BMI'].fillna(data2['BMI'].median(), inplace=True)

data2.isnull().sum()
q = data2.hist(figsize=(20,20))
print(data2['Outcome'].value_counts())

print(data2['Outcome'].value_counts().plot(kind="bar"))
plt.figure(figsize=(12,8))

sns.heatmap(data.corr(), annot=True, center=True)
plt.figure(figsize=(12,8))

sns.heatmap(data2.corr(), annot=True, center=True)
#sns.boxplot(data2['BMI'])

for col in data2:

    plt.figure(figsize=(3,2))

    sns.boxplot(data2[col])
data2.Pregnancies.value_counts()
data2.Pregnancies.mean()
data2.loc[data2['Pregnancies']>12, 'Pregnancies']=data2['Pregnancies'].median()
data2.Pregnancies.value_counts()
data2.BloodPressure.value_counts()
data2.loc[(data2.BloodPressure > 100 )]
plt.figure(figsize=(15,12))

sns.pairplot(data2, hue='Outcome')
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc_data = pd.DataFrame(sc.fit_transform(data2.drop('Outcome',axis=1)),columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age'])
sc_data.head()
x = sc_data.copy()

y = data2['Outcome']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42, stratify=y)
# METHOD 1: KNN

from sklearn.neighbors import KNeighborsClassifier



train_score = []

test_score = []



for i in range(1,50):

    knn = KNeighborsClassifier(i)

    knn.fit(x_train,y_train)

    

    train_score.append(knn.score(x_train,y_train))

    test_score.append(knn.score(x_test,y_test))

    

    
max_train_score = max(train_score)

train_index = [i for i, v in enumerate(train_score) if v==max_train_score]



print("Max_train_score = {} % and the value of k = {}".format(max_train_score*100,list(map(lambda x: x+1, train_index))))
max_test_score = max(test_score)

test_index = [i for i, v in enumerate(test_score) if v==max_test_score]



print("Max_test_score = {} % and the value of k = {}".format(max_test_score*100,list(map(lambda x: x+1, test_index))))
#print(train_score)

#print(test_score)
plt.figure(figsize=(15,6))



p = sns.lineplot(range(1,50), train_score, marker='o', label='train_score')

p = sns.lineplot(range(1,50), test_score, marker='o', label='test_score')
# So finally we get the best score at k = 11



knn_final = KNeighborsClassifier(11)

knn_final.fit(x_train,y_train)

knn_final.score(x_test,y_test)
#METHOD 2: RANDOM FORREST



from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train,y_train)
rf.score(x_test,y_test)
pred2= rf.predict(x_test)

pred2=pred2.round().astype('int64')
#print(pred2.shape)

#print(y_pred.shape)
# METHOD 3: SUPPORT VECTOR MACHINE



from sklearn import svm

sup = svm.SVC()
sup.fit(x_train,y_train)

sup.score(x_test,y_test)
# METHOD 4: DECISION TREE



from sklearn.tree import DecisionTreeClassifier

dis = DecisionTreeClassifier(random_state=42)
dis.fit(x_train,y_train)

dis.score(x_test,y_test)
# 1. Confusion matrix  for KNN



from sklearn.metrics import confusion_matrix



y_pred = knn_final.predict(x_test)

confusion_matrix(y_pred,y_test)

pd.crosstab(y_test,y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn import metrics



cnf_matrix = confusion_matrix(y_test,y_pred)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot= True, fmt= 'g')

plt.title('Confusion Matrix', y=1.1)

plt.xlabel('Predicted')

plt.ylabel('Actual')
# 2. Confusion matrix  for Random Forrest



confusion_matrix(pred2,y_test)

pd.crosstab(y_test,pred2, rownames=['True'], colnames=['Predicted'], margins=True)
cnf_matrix2 = confusion_matrix(y_test,pred2)

p = sns.heatmap(pd.DataFrame(cnf_matrix2), annot= True, fmt= 'g')

plt.title('Confusion Matrix', y=1.1)

plt.xlabel('Predicted')

plt.ylabel('Actual')
# 3. Confusion matrix for Support Vector Machine



pred3 = sup.predict(x_test)

confusion_matrix(pred3,y_test)

pd.crosstab(y_test,pred3, rownames=['True'], colnames=['Predicted'], margins=True)
cnf_matrix3 = confusion_matrix(y_test,pred3)

p = sns.heatmap(pd.DataFrame(cnf_matrix3),annot=True, fmt='g')

plt.title('Confusion Matrix', y=1.1)

plt.xlabel('Predicted')

plt.ylabel('Actual')
# 4. Confusion matrix for Decision Tree



pred4 = dis.predict(x_test)

confusion_matrix(pred4,y_test)

pd.crosstab(y_test,pred4, rownames=['True'], colnames=['Predicted'], margins=True)
cnf_matrix4 = confusion_matrix(y_test,pred4)

p = sns.heatmap(pd.DataFrame(cnf_matrix4),annot=True, fmt='g')

plt.title('Confusion Matrix', y=1.1)

plt.xlabel('Predicted')

plt.ylabel('Actual')
# 1. Classification report for KNN



from sklearn.metrics import classification_report

print(classification_report(y_pred,y_test))
# 2. Classification report for Random Forrest



print(classification_report(pred2,y_test))
# 3. Classification report for Support Vector Machine



print(classification_report(pred3,y_test))
# 4. Classification report for Decision Tree Classifier



print(classification_report(pred4,y_test))
#The End