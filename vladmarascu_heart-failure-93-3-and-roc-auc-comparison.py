import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.info()
df.isnull().sum()
plt.figure(figsize=(15,3))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.columns
len(df.columns)
plt.figure(figsize=(6,6))
df['DEATH_EVENT'].value_counts().plot(kind='pie', autopct='%1.1f', shadow=True)
df['DEATH_EVENT'].value_counts()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),cmap='coolwarm', annot=True)
sns.set_style('whitegrid')

g = sns.FacetGrid(df, hue="sex", height=6, aspect=2, palette='dark')
g = g.map(plt.hist, "age", bins=30, alpha=0.5)

g.add_legend()
g = sns.FacetGrid(df, hue="DEATH_EVENT", height=6, aspect=2, palette='dark')
g = g.map(plt.hist, "age", bins=30, alpha=0.5)

g.add_legend()
sns.boxplot(x="DEATH_EVENT", y="age", data=df)
plt.figure(figsize=(6,6))
df['anaemia'].value_counts().plot(kind='pie', autopct='%1.1f', shadow=True)
g = sns.FacetGrid(df, hue="DEATH_EVENT", height=6, aspect=2, palette='dark')
g = g.map(plt.hist, "creatinine_phosphokinase", bins=50, alpha=0.5)

g.add_legend()
plt.figure(figsize=(6,6))
df['diabetes'].value_counts().plot(kind='pie', autopct='%1.1f', shadow=True)
g = sns.FacetGrid(df, hue="DEATH_EVENT", height=6, aspect=2, palette='dark')
g = g.map(plt.hist, "ejection_fraction", bins=10, alpha=0.5)

g.add_legend()
sns.boxplot(x="DEATH_EVENT", y="ejection_fraction", data=df)
plt.figure(figsize=(6,6))
df['high_blood_pressure'].value_counts().plot(kind='pie', autopct='%1.1f', shadow=True)
g = sns.FacetGrid(df, hue="DEATH_EVENT", height=6, aspect=2, palette='dark')
g = g.map(plt.hist, "platelets", bins=30, alpha=0.5)

g.add_legend()
g = sns.FacetGrid(df, hue="DEATH_EVENT", height=6, aspect=2, palette='dark')
g = g.map(plt.hist, "serum_creatinine", bins=30, alpha=0.5)

g.add_legend()
sns.boxplot(x="DEATH_EVENT", y="serum_creatinine", data=df)
g = sns.FacetGrid(df, hue="DEATH_EVENT", height=6, aspect=2, palette='dark')
g = g.map(plt.hist, "serum_sodium", bins=30, alpha=0.5)

g.add_legend()
sns.boxplot(x="DEATH_EVENT", y="serum_sodium", data=df)
plt.figure(figsize=(6,6))
df['sex'].value_counts().plot(kind='pie', autopct='%1.1f', shadow=True)
plt.figure(figsize=(6,6))
df['smoking'].value_counts().plot(kind='pie', autopct='%1.1f', shadow=True)
sns.regplot(x='serum_sodium',y='ejection_fraction', data=df)
X=df.drop(['DEATH_EVENT'], axis=1)
y=df['DEATH_EVENT']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
def print_validation_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print("Accuracy : "+ str(acc_sc))
def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  
                cmap="Blues", cbar=False)
    plt.ylabel('true label')
    plt.xlabel('predicted label')
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(max_iter=10000)
lr.fit(X_train,y_train)
p1=lr.predict(X_test)
s1=accuracy_score(y_test,p1)
print("Linear Regression Success Rate :", s1*100,'%')
print_validation_report(y_test,p1)
plot_confusion_matrix(y_test,p1)
importance = abs(lr.coef_[0])
coeffecients = pd.DataFrame(importance, X_train.columns)
coeffecients.columns = ['Coeffecient']
plt.figure(figsize=(15,4))
plt.bar(X_train.columns,importance)
plt.show()
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
p2=rfc.predict(X_test)
s2=accuracy_score(y_test,p2)
print("Random Forrest Accuracy :", s2*100,'%')
plot_confusion_matrix(y_test,p2)
from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train,y_train)
p3=svm.predict(X_test)
s3=accuracy_score(y_test,p3)
print("SVM Accuracy :", s3*100,'%')
plot_confusion_matrix(y_test,p3)
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
p4=knn.predict(X_test)
s4=accuracy_score(y_test,p4)
print("KNN Accuracy :", s4*100,'%')
error_rate = []
scores = []

for i in range(1,40): # check all values of K between 1 and 40
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    score=accuracy_score(y_test,pred_i)
    scores.append(score)
    error_rate.append(np.mean(pred_i != y_test)) # ERROR RATE DEF and add it to the list
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10,6))
plt.plot(range(1,40),scores,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Score vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Score')
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=35)
knn.fit(X_train,y_train)
p4=knn.predict(X_test)
s4=accuracy_score(y_test,p4)
print("KNN Accuracy:", s4*100,'%')
plot_confusion_matrix(y_test,p4)
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train, y_train)
p5 =nb.predict(X_test)
s5=accuracy_score(y_test,p5)
print("Naive-Bayes Accuracy:", s5*100,'%')
plot_confusion_matrix(y_test,p5)
f1_score(y_test,p5)
models = pd.DataFrame({
    'Model': ["LOGISTIC REGRESSION","RANDOM FOREST","SUPPORT VECTOR MACHINE","KNN","NAIVE-BAYES"],
    'Accuracy Score': [s1*100,s2*100,s3*100,s4*100,s5*100]})
models.sort_values(by='Accuracy Score', ascending=False)
print(f1_score(y_test,p1))
print(f1_score(y_test,p2))
print(f1_score(y_test,p3))
print(f1_score(y_test,p4))
print(f1_score(y_test,p5))
from sklearn.metrics import roc_curve,roc_auc_score, auc
fpr1,tpr1, thr1=roc_curve(y_test,p1)
fpr2,tpr2, thr2=roc_curve(y_test,p2)
fpr3,tpr3, thr3=roc_curve(y_test,p3)
fpr4,tpr4, thr4=roc_curve(y_test,p4)
fpr5,tpr5, thr5=roc_curve(y_test,p5)
plt.figure(figsize=(10,6))
plt.plot(fpr1,tpr1, linestyle='--', label='LR')
plt.plot(fpr2,tpr2, linestyle='--', label='RF')
plt.plot(fpr3,tpr3, linestyle='--', label='SVM')
plt.plot(fpr4,tpr4, linestyle='--', label='KNN')
plt.plot(fpr5,tpr5, linestyle='--', label='NB')
plt.legend()
print(roc_auc_score(y_test,p1))
print(roc_auc_score(y_test,p2))
print(roc_auc_score(y_test,p3))
print(roc_auc_score(y_test,p4))
print(roc_auc_score(y_test,p5))