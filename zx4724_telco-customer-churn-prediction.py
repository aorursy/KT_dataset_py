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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.info()
pd.DataFrame({'count':df.isnull().sum(),'percentage':df.isnull().sum()/len(df)*100})
df['Churn'].value_counts()
sns.countplot(x='Churn',data=df)
df.drop(['customerID'],axis=1,inplace=True)
df['gender'].value_counts()
sns.countplot(x='Churn',hue='gender',data=df)
df['SeniorCitizen'].unique()
df['SeniorCitizen'].value_counts()
sns.countplot(x='Churn',hue='SeniorCitizen',data=df)
df['Partner'].unique()
df['Partner'].value_counts()
sns.countplot(x='Churn',hue='Partner',data=df)
df['Dependents'].unique()
df['Dependents'].value_counts()
sns.countplot(x='Churn',hue='Dependents',data=df)
df['tenure'].describe()
sns.boxplot(y='tenure',x='Churn',data=df)
sns.boxplot(y='tenure',x='Churn',hue='SeniorCitizen',data=df)
sns.boxplot(y='tenure',x='Churn',hue='Partner',data=df)
sns.boxplot(y='tenure',x='Churn',hue='Dependents',data=df)
df['PhoneService'].unique()
df['PhoneService'].value_counts()
sns.countplot(x='Churn',hue='PhoneService',data=df)
df['MultipleLines'].unique()
df['MultipleLines'].value_counts()
sns.countplot(x='Churn',hue='MultipleLines',data=df)
df['InternetService'].unique()
df['InternetService'].value_counts()
sns.countplot(x='Churn',hue='InternetService',data=df)
df['OnlineSecurity'].unique()
df['OnlineSecurity'].value_counts()
sns.countplot(x='Churn',hue='OnlineSecurity',data=df)
df['OnlineBackup'].unique()
df['OnlineBackup'].value_counts()
sns.countplot(x='Churn',hue='OnlineBackup',data=df)
df['DeviceProtection'].unique()
df['DeviceProtection'].value_counts()
sns.countplot(x='Churn',hue='DeviceProtection',data=df)
df['TechSupport'].unique()
df['TechSupport'].value_counts()
sns.countplot(x='Churn',hue='TechSupport',data=df)
df['StreamingTV'].unique()
df['StreamingTV'].value_counts()
sns.countplot(x='Churn',hue='StreamingTV',data=df)
df['Contract'].unique()
df['Contract'].value_counts()
sns.countplot(x='Churn',hue='Contract',data=df)
df['PaperlessBilling'].unique()
df['PaperlessBilling'].value_counts()
sns.countplot(x='Churn',hue='PaperlessBilling',data=df)
df['PaymentMethod'].unique()
df['PaymentMethod'].value_counts()
sns.countplot(x='Churn',hue='PaymentMethod',data=df)
df['MonthlyCharges'].describe()
sns.boxplot(y='MonthlyCharges',x='Churn',data=df)
target_0 = df.loc[df['Churn'] == 'No']
target_1 = df.loc[df['Churn'] == 'Yes']
sns.distplot(target_0['MonthlyCharges'],hist=False, rug=True)
sns.distplot(target_1['MonthlyCharges'],hist=False, rug=True)
plt.show()
df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].describe()
sns.boxplot(y='TotalCharges',x='Churn',data=df)
target_0 = df.loc[df['Churn'] == 'No']
target_1 = df.loc[df['Churn'] == 'Yes']
sns.distplot(target_0['TotalCharges'],hist=False, rug=True)
sns.distplot(target_1['TotalCharges'],hist=False, rug=True)
plt.show()
df.drop(['PhoneService','InternetService'],axis=1,inplace=True)
df['MonthlyCharges']=(df['MonthlyCharges']-df['MonthlyCharges'].min())/(df['MonthlyCharges'].max()-df['MonthlyCharges'].min())

df['TotalCharges']=(df['TotalCharges']-df['TotalCharges'].min())/(df['TotalCharges'].max()-df['TotalCharges'].min())

df['tenure']=(df['tenure']-df['tenure'].min())/(df['tenure'].max()-df['tenure'].min())

gle = LabelEncoder()
gender_labels = gle.fit_transform(df['gender'])
gender_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['gender'] = gender_labels
gender_mappings
Partner_labels = gle.fit_transform(df['Partner'])
Partner_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['Partner'] = Partner_labels
Partner_mappings
Dependents_labels = gle.fit_transform(df['Dependents'])
Dependents_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['Dependents'] = Dependents_labels
Dependents_mappings
MultipleLines_labels = gle.fit_transform(df['MultipleLines'])
MultipleLines_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['MultipleLines'] = MultipleLines_labels
MultipleLines_mappings
OnlineSecurity_labels = gle.fit_transform(df['OnlineSecurity'])
OnlineSecurity_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['OnlineSecurity'] = OnlineSecurity_labels
OnlineSecurity_mappings
OnlineBackup_labels = gle.fit_transform(df['OnlineBackup'])
OnlineBackup_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['OnlineBackup'] = OnlineBackup_labels
OnlineBackup_mappings
DeviceProtection_labels = gle.fit_transform(df['DeviceProtection'])
DeviceProtection_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['DeviceProtection'] = DeviceProtection_labels

DeviceProtection_mappings
TechSupport_labels = gle.fit_transform(df['TechSupport'])
TechSupport_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['TechSupport'] = TechSupport_labels
TechSupport_mappings
StreamingTV_labels = gle.fit_transform(df['StreamingTV'])
StreamingTV_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['StreamingTV'] = StreamingTV_labels

StreamingTV_mappings
StreamingMovies_labels = gle.fit_transform(df['StreamingMovies'])
StreamingMovies_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['StreamingMovies'] = StreamingMovies_labels
StreamingMovies_mappings
Contract_labels = gle.fit_transform(df['Contract'])
Contract_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['Contract'] = Contract_labels
Contract_mappings
PaperlessBilling_labels = gle.fit_transform(df['PaperlessBilling'])
PaperlessBilling_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['PaperlessBilling'] = PaperlessBilling_labels
PaperlessBilling_mappings
PaymentMethod_labels = gle.fit_transform(df['PaymentMethod'])
PaymentMethod_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['PaymentMethod'] = PaymentMethod_labels
PaymentMethod_mappings
Churn_labels = gle.fit_transform(df['Churn'])
Churn_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['Churn'] = Churn_labels
Churn_mappings 
SeniorCitizen_labels = gle.fit_transform(df['SeniorCitizen'])
SeniorCitizen_mappings = {index: label for index, label in 
                  enumerate(gle.classes_)}
df['SeniorCitizen'] = SeniorCitizen_labels
SeniorCitizen_mappings
df.describe().T
df1=df.copy()
corr_matrix=df1.corr()
corr_matrix['Churn'].sort_values(ascending=False)
sns.heatmap(corr_matrix, cmap="YlGnBu")
corr_matrix['tenure'].sort_values(ascending=False)
sns.jointplot(x='tenure',y='TotalCharges',data=df,kind="scatter")
sns.jointplot(x='tenure',y='Contract',data=df,kind="scatter")
corr_matrix['TotalCharges'].sort_values(ascending=False)
sns.jointplot(x='TotalCharges',y='MonthlyCharges',data=df,kind="scatter")
df1.drop(['TotalCharges'],axis=1,inplace=True)
sns.heatmap(df1.corr(), cmap="YlGnBu")
x=df1.drop(['Churn'],axis=1)
y=df1['Churn']
from sklearn.neighbors import KNeighborsClassifier

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=24)

log_param_grid={'C':[0.1,0.5,1.0,1.5],'random_state':[0,1,2,3],'max_iter':[50,100,150,200]}

log_reg=LogisticRegression()
log_grid_search=GridSearchCV(log_reg,log_param_grid,cv=5,scoring='neg_mean_squared_error')

log_grid_search.fit(x_train,y_train)
log_grid_search.best_params_
lr = LogisticRegression(C=0.5,random_state=0,max_iter=50)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print("[Logistic regression algorithm] accuracy_score: {:.3f}.".format(acc))
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:",cm)
print(classification_report(y_test, y_pred))
probs = lr.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic regression algorithm Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
KNeighborsClassifier().get_params().keys()

knn_param_grid={'n_neighbors':[2,4,6,8,10.12],'leaf_size':[10,20,30,40,50]}

knn_reg=KNeighborsClassifier()
knn_grid_search=GridSearchCV(knn_reg,knn_param_grid,cv=5,scoring='neg_mean_squared_error')

knn_grid_search.fit(x_train,y_train)
knn_grid_search.best_params_
knn = KNeighborsClassifier(n_neighbors=6,leaf_size=10)

# Fit the classifier to the data
knn.fit(x_train,y_train)

# Predict the labels for the training data X
y_pred = knn.predict(x_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

print('[K-Nearest Neighbors (KNN)] knn.score: {:.3f}.'.format(knn.score(x_test, y_test)))
print('[K-Nearest Neighbors (KNN)] accuracy_score: {:.3f}.'.format(acc))
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:",cm)
print(classification_report(y_test, y_pred))
probs = knn.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('k-NN classifier Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
dt_param_grid={'criterion':('gini', 'entropy'),'max_depth':[2,4,6,8,10.12],'random_state':[1,2,3]}

dt_reg=DecisionTreeClassifier()
dt_grid_search=GridSearchCV(dt_reg,dt_param_grid,cv=5,scoring='neg_mean_squared_error')

dt_grid_search.fit(x_train,y_train)
dt_grid_search.best_params_
dt = DecisionTreeClassifier(max_depth=6, criterion='entropy', random_state=1)


# Fit dt_entropy to the training set
dt.fit(x_train, y_train)

# Use dt_entropy to predict test set labels
y_pred= dt.predict(x_test)

# Evaluate accuracy_entropy
accuracy_entropy = accuracy_score(y_test, y_pred)


# Print accuracy_entropy
print('[Decision Tree -- accuracy_score: {:.3f}.'.format(accuracy_entropy))

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:",cm)
print(classification_report(y_test, y_pred))
probs = dt.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision tree algorithm Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
RandomForestClassifier().get_params().keys()

rf_param_grid={'n_estimators':[60,70,80,90,100,110,120,130],'max_features':('auto', 'log2')}

rf_reg=RandomForestClassifier()
rf_grid_search=GridSearchCV(rf_reg,rf_param_grid,cv=5,scoring='neg_mean_squared_error')

rf_grid_search.fit(x_train,y_train)
rf_grid_search.best_params_
#Create a Gaussian Classifier
rf=RandomForestClassifier(n_estimators=130, max_features='auto')

#Train the model using the training sets y_pred=clf.predict(X_test)
rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)


# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Model Accuracy, how often is the classifier correct?
print("[Randon forest algorithm] accuracy_score: {:.3f}.".format(acc))
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:",cm)
print(classification_report(y_test, y_pred))
probs = rf.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest algorithm Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
feature_imp = pd.Series(rf.feature_importances_,index=x.columns).sort_values(ascending=False)
# Creating a bar plot, displaying only the top k features
k=10
sns.barplot(x=feature_imp[:10], y=feature_imp.index[:k])
sfm = SelectFromModel(rf, threshold=0.03)
# Train the selector
sfm.fit(x_train, y_train)

feat_labels=x.columns

# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feat_labels[feature_list_index])
x_important_train = sfm.transform(x_train)
x_important_test = sfm.transform(x_test)
clf_important = RandomForestClassifier(n_estimators=120, random_state=0, n_jobs=-1)
clf_important.fit(x_important_train, y_train)
y_pred = rf.predict(x_test)
print('[Randon forest algorithm -- Full feature] accuracy_score: {:.3f}.'.format(accuracy_score(y_test, y_pred)))

svc_param_grid={'kernel':('linear','poly','rbf','sigmoid'),'C':[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3],'gamma':('auto', 'scale'),'degree':[1,2,3]}

svc_reg=SVC()
svc_grid_search=GridSearchCV(svc_reg,svc_param_grid,cv=5,scoring='neg_mean_squared_error')

svc_grid_search.fit(x_train,y_train)
svc_grid_search.best_params_
clf=SVC(kernel='linear',degree=1,C=0.6)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)


# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Model Accuracy, how often is the classifier correct?
print("[Support Vector Machines algorithm] accuracy_score: {:.3f}.".format(acc))

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:",cm)
print(classification_report(y_test, y_pred))
clf=GaussianNB()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)


# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Model Accuracy, how often is the classifier correct?
print("[Naive Bayes algorithm] accuracy_score: {:.3f}.".format(acc))

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:",cm)
print(classification_report(y_test, y_pred))

probs = clf.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes algorithm Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
LDA().get_params().keys()

lda_param_grid={'solver':('svd','lsqr','eigen'),'n_components':[1,2,3,4,5,6,7,8]}

lda_reg=LDA()
lda_grid_search=GridSearchCV(lda_reg,lda_param_grid,cv=5,scoring='neg_mean_squared_error')

lda_grid_search.fit(x_train,y_train)
lda_grid_search.best_params_
clf=LDA(n_components=1,solver='svd')

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)


# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Model Accuracy, how often is the classifier correct?
print("[Linear Discriminant Analysis algorithm] accuracy_score: {:.3f}.".format(acc))

cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:",cm)
print(classification_report(y_test, y_pred))
probs = clf.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Linear Discriminant Analysis algorithm Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
clf=MLPClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)


# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Model Accuracy, how often is the classifier correct?
print("[MLPClassifier algorithm] accuracy_score: {:.3f}.".format(acc))
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:",cm)
print(classification_report(y_test, y_pred))
probs = clf.predict_proba(x_test)[:,1]
auc = roc_auc_score(y_test, probs)
print('AUC: %.2f' % auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('MLPClassifier algorithm Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
