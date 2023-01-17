# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn-dark')
data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.shape
data.describe()
data.info()
data.isnull().values.any()
data.isnull().sum()
import missingno as msno

msno.matrix(data)

plt.show()

missing_percantage=data['salary'].isnull().sum()/len(data)*100



print(round(missing_percantage,2),'%')
column=data.select_dtypes(include=['object'])

for col in column:

    display(data[col].value_counts())
plt.figure(figsize=(10,7))

sns.countplot(x='gender',data=data)

labels = (data['gender'])
plt.figure(figsize=(10,7))

sns.countplot(x='gender',hue='status',data=data)

plt.show()
plt.figure(figsize=(10,7))

sns.boxplot(y='gender',x='salary',data=data)
plt.figure(figsize=(10,7))

sns.countplot(x='ssc_b',data=data)

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x='hsc_b',hue='hsc_s',data=data)

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x='ssc_b',hue='status',data=data)

plt.show()
plt.figure(figsize=(15,8))

sns.catplot(x='hsc_b',hue='hsc_s',col='status',data=data,kind='count')

plt.show()
plt.figure(figsize=(10,7))

sns.countplot(x="degree_t", hue='status',data=data)
plt.figure(figsize=(10,7))

sns.countplot(x="specialisation", hue='status',data=data)
plt.figure(figsize = (15,7))

ax=plt.subplot(121)

sns.violinplot(x='degree_t',y='salary',hue='gender',data=data,split=True,scale="count")

ax.set_title('UG Degree')

ax=plt.subplot(122)

sns.violinplot(x='specialisation',y='salary',hue='gender',data=data,split=True,scale="count")

ax.set_title('MBA ')
plt.figure(figsize = (15, 15))

ax=plt.subplot(221)

sns.boxplot(x='status',y='ssc_p',data=data)

ax.set_title('Secondary school percentage')

ax=plt.subplot(222)

sns.boxplot(x='status',y='hsc_p',data=data)

ax.set_title('Higher Secondary school percentage')

ax=plt.subplot(223)

sns.boxplot(x='status',y='degree_p',data=data)

ax.set_title('UG Degree percentage')

ax=plt.subplot(224)

sns.boxplot(x='status',y='mba_p',data=data)

ax.set_title('MBA percentage')
plt.figure(figsize=(10,7))

sns.violinplot(x=data["gender"], y=data["salary"], hue=data["workex"])

plt.title("Gender vs Salary based on work experience")
plt.figure(figsize=(10,5))

sns.distplot(data['salary'], bins=50, hist=False)

plt.title("Salary Distribution")

plt.show()
plt.figure(figsize=(10,7))

sns.boxplot(x='gender',y='salary',data=data)

plt.show()
data["gender"] = data.gender.map({"M":0,"F":1})

data["ssc_b"] = data.ssc_b.map({"Others":0,"Central":1})

data["hsc_b"] = data.hsc_b.map({"Others":0,"Central":1})

data["hsc_s"] = data.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})

data["degree_t"] = data.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})

data["workex"] = data.workex.map({"No":0, "Yes":1})

data["status"] = data.status.map({"Not Placed":0, "Placed":1})

data["specialisation"] = data.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})
plt.figure(figsize=(12,8))

sns.heatmap(data.corr(),annot=True)

plt.show()
# Seperating Features and Target

X = data[[ 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p',  'workex','etest_p', 'specialisation', 'mba_p',]]

y = data['status']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score,precision_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import  LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
# Let us now split the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,stratify =y)





print("X-Train:",X_train.shape)

print("X-Test:",X_test.shape)

print("Y-Train:",y_train.shape)

print("Y-Test:",y_test.shape)
logreg = LogisticRegression(solver= 'lbfgs',max_iter=400)

logreg.fit(X_train, y_train)



log_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, y_train) * 100, 2)
random_forest = RandomForestClassifier(n_estimators=200,criterion='gini',

 max_depth= 4 ,

 max_features= 'auto',random_state=42)

random_forest.fit(X_train, y_train)



ran_pred = random_forest.predict(X_test)



random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
plt.subplots(figsize=(20,6))

a_index=list(range(1,50))

a=pd.Series()

x=range(1,50)

#x=[1,2,3,4,5,6,7,8,9,10]

for i in list(range(1,50)):

    model=KNeighborsClassifier(n_neighbors=i) 

    model.fit(X_train, y_train) 

    prediction=model.predict(X_test)

    a=a.append(pd.Series(accuracy_score(y_test,prediction)))

plt.plot(a_index, a,marker="*",color='r')

plt.xticks(x)

plt.show()
knn = KNeighborsClassifier(n_neighbors = 10)

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test) 

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

gau_pred = gaussian.predict(X_test) 

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)



svc_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
results = pd.DataFrame({

    'Model': [ 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes',  

              ' Support Vector Machine'

            ],

    'Train Score': [ acc_knn, acc_log, 

              acc_random_forest, acc_gaussian,  

              acc_linear_svc],

    'Accuracy_score':[round(accuracy_score(y_test,knn_pred) * 100, 2),

                round(accuracy_score(y_test,log_pred) * 100, 2),

                round(accuracy_score(y_test,ran_pred) * 100, 2),

                round(accuracy_score(y_test,gau_pred)* 100, 2),

                round(accuracy_score(y_test,svc_pred)* 100, 2)

                

        

    ]



})

result_df = results.sort_values(by='Accuracy_score', ascending=False)

result_df = result_df.set_index('Accuracy_score')

result_df
plt.subplots(figsize=(10,6))

ax=sns.pointplot(x='Model',y="Accuracy_score",data=results)

labels = (results["Accuracy_score"])

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v+0.2, str(v), horizontalalignment = 'center', size = 15, color = 'red')
randomForestFinalModel = RandomForestClassifier(n_estimators=200,criterion='gini',

 max_depth= 4 ,

 max_features= 'auto',random_state=42)

randomForestFinalModel.fit(X_train, y_train)

predictions_rf = randomForestFinalModel.predict(X_test)



cm_logit = confusion_matrix(y_test, predictions_rf)

print('Confusion matrix for Random Forest\n',cm_logit)



accuracy_logit = accuracy_score(y_test,predictions_rf)

precision_logit =precision_score(y_test, predictions_rf)

recall_logit =  recall_score(y_test, predictions_rf)

f1_logit = f1_score(y_test,predictions_rf)

print('accuracy_random_Forest : %.3f' %accuracy_logit)

print('precision_random_Forest : %.3f' %precision_logit)

print('recall_random_Forest : %.3f' %recall_logit)

print('f1-score_random_Forest : %.3f' %f1_logit)

auc_logit = roc_auc_score(y_test,predictions_rf)

print('AUC_random_Forest: %.2f' % auc_logit)



cf_matrix = confusion_matrix(y_test, predictions_rf)

fig = plt.figure(figsize=(10,7))

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in

                cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in

                     cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in

          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cf_matrix, annot=labels, annot_kws={"size": 16}, fmt='')

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()
a=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

b=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

fig =plt.figure(figsize=(20,12),dpi=50)

fpr, tpr, thresholds = roc_curve(y_test,predictions_rf )

plt.plot(fpr, tpr,color ='blue',label ='random Forest',linewidth=2 )

plt.plot(a,b,color='black',linestyle ='dashed',linewidth=2)

plt.legend(fontsize=15)

plt.xlabel('False Positive Rate',fontsize=15)

plt.ylabel('True Positive Rate',fontsize=15)