# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
data
# output = (rows, columns)
data.shape 
# making a copy of the data and saving it in a variable 
data_temp = data.copy()
data_temp['Churn'].value_counts().plot.pie(autopct='%1.1f%%')
sns.factorplot(x="Contract", y="MonthlyCharges", hue="PaymentMethod", kind="point", data=data_temp)
sns.pairplot(data_temp[['tenure','MonthlyCharges','TotalCharges','Churn','Contract','SeniorCitizen']], hue='Churn')
# shorter the contract period, higher probability of churn? lets check crosstab

cr  = pd.crosstab(data_temp.Contract, data_temp.Churn)
cr.plot(kind='bar')
plt.title('Churn by Contract')
plt.ylabel("Number of chrun")
plt.show()
data_temp.dtypes
tot_charges = data_temp["TotalCharges"]
def convert_to_float(strin):
        try:
            return float(strin)
        except:
            return 0
data_temp['TotalCharges']=data_temp['TotalCharges'].apply(convert_to_float)
data_temp['TotalCharges']
#describe the data
data_temp.describe()
# Checking For NULL 
data_temp.isnull().sum()
#Identifying the rows containing 0 value in Total Charges
zero_value_row = list(data_temp[data_temp['TotalCharges'] == 0].index)
print('0 Value Rows=',zero_value_row, "\ntotal =", len(zero_value_row))

for zero_row in zero_value_row :
    print( data_temp['MonthlyCharges'][zero_row],data_temp['tenure'][zero_row],data_temp['TotalCharges'][zero_row])
# Look at the shape before and after to be sure they were removed
print(data_temp.shape)
data_temp = data_temp.drop(data_temp.index[[488, 753, 936, 1082, 1340, 3331, 3826, 4380, 5218, 6670, 6754]])
print(data_temp.shape)
#coverting Churn from Yes or No to 1 or 0 and saving it in a new column name class
data_temp['class'] = data_temp['Churn'].apply(lambda x : 1 if x == "Yes" else 0)
count_no_churn = len(data_temp[data_temp['class']==0])
count_churn = len(data_temp[data_temp['class']==1])
pct_of_no_churn = count_no_churn/(count_no_churn+count_churn)
print("percentage of no churn is", pct_of_no_churn*100)
pct_of_churn = count_churn/(count_no_churn+count_churn)
print("percentage of churn", pct_of_churn*100)
data_temp.loc[(data_temp.Churn == 'Yes'),'MonthlyCharges'].median()
data_temp.loc[(data_temp.Churn == 'Yes'),'TotalCharges'].median()
data_temp.loc[(data_temp.Churn == 'Yes'),'tenure'].median()
data_temp.loc[(data_temp.Churn == 'Yes'), 'Contract'].value_counts(normalize = True)
data_temp.loc[(data_temp.Churn == 'Yes'),'PaymentMethod'].value_counts(normalize = True)
data_temp['Is_Electronic_check'] = np.where(data_temp['PaymentMethod'] == 'Electronic check',1,0)
data_temp.loc[(data_temp.Churn == 'Yes'),'PaperlessBilling'].value_counts(normalize = True)
data_temp.loc[(data_temp.Churn == 'Yes'), 'Contract'].value_counts(normalize = True)
data_temp['Is_Contract_Month'] = np.where(data_temp['PaymentMethod'] == 'Month-to-month',1,0)
data_temp.loc[(data_temp.Churn == 'Yes'),'DeviceProtection'].value_counts(normalize = True)
data_temp.loc[(data_temp.Churn == 'Yes'),'OnlineBackup'].value_counts(normalize = True)
data_temp.loc[(data_temp.Churn == 'Yes'),'TechSupport'].value_counts(normalize = True)
data_temp.loc[(data_temp.Churn == 'Yes'),'OnlineSecurity'].value_counts(normalize = True)
data_temp= pd.get_dummies(data_temp,columns=['Partner','Dependents',
       'PhoneService', 'MultipleLines','StreamingTV',
       'StreamingMovies','Contract','PaperlessBilling','InternetService'],drop_first=True)
data_temp.info()
data_temp.drop(['StreamingTV_No internet service','StreamingMovies_No internet service'],axis=1,inplace=True)
data_temp.drop('gender',axis=1,inplace=True)
data_temp.drop('class',axis=1,inplace=True)
data_temp.drop('customerID',axis=1,inplace=True)
#data_temp.drop(['tenure','MonthlyCharges'],axis=1,inplace=True)
data_temp.drop(['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','PaymentMethod'],axis=1,inplace=True)
data_temp = pd.get_dummies(data_temp,columns=['Churn'],drop_first=True)
data_temp.info()
X = data_temp.drop('Churn_Yes',axis=1).values.astype('float')
y = data_temp['Churn_Yes']
# train test 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
# create model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# train model
model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,classification_report
print('Model Score:',model.score(X_train,y_train))
y_pred = model.predict(X_test)
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print ('confusion matrix for logistic regression \n ', cnf_matrix)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print ('accuracy for logistic regression  : {0:.2f}'.format(accuracy_score(y_test, model.predict(X_test))))
print ('precision for logistic regression : {0:.2f}'.format(precision_score(y_test, model.predict(X_test))))
print ('recall for logistic regression    : {0:.2f}'.format(recall_score(y_test, model.predict(X_test))))
print(classification_report(y_test,model.predict(X_test)))
y_pred_prob = model.predict(X_test)
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob)
auc = metrics.roc_auc_score(y_test, y_pred_prob)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)
plt.ylabel('True Positive Rate (recall)', fontsize=14)
plt.show()
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)  
from kmodes.kmodes import KModes 
data_temp2 = data_temp.copy()
data_temp=data_temp.drop(['TotalCharges', 'tenure', 'MonthlyCharges'], axis=1)

# random categorical data

km = KModes(n_clusters=4, init='Huang', n_init=5, verbose=1)

clusters = km.fit_predict(data_temp)


km.cluster_centroids_
clusters
data_temp['Cluster_Group']= clusters
data
data_temp.columns
data_temp['Cluster_Group'].value_counts().plot(kind='bar',title='Distribution of Customers across groups')
plt.xlabel("Clusters")
plt.ylabel("Number of Clusters")
ct  = pd.crosstab(data_temp['Cluster_Group'],data_temp.Churn_Yes)
ct.plot(kind='bar')
plt.title('Churn rate of each Cluster Group')
plt.show()
data_temp
pd.DataFrame({
    'clusters':clusters,
    'SeniorCitizen': data_temp['SeniorCitizen'],
    'Churn_Yes': data_temp['Churn_Yes']
}).plot(kind='scatter', x='SeniorCitizen', y='Churn_Yes',c="clusters")
plt.xlabel('SeniorCitizen')
plt.figure(figsize=(13,13))
plt.show()
cgroup = data_temp.groupby('Cluster_Group')
cgroup.apply(lambda x : x.mode())
sns.pairplot(data_temp[['Is_Contract_Month','Contract_One year','Contract_Two year','Cluster_Group']], hue='Cluster_Group')
sns.pairplot(data_temp[['Is_Electronic_check','PaperlessBilling_Yes','Cluster_Group']], hue='Cluster_Group')
subgroup = data_temp[['Is_Contract_Month','Contract_One year','Contract_Two year','Is_Electronic_check','PaperlessBilling_Yes','Cluster_Group']]
cluster_data = subgroup.groupby('Cluster_Group')
cluster_data.plot(subplots=True,)
# Print the cluster centroids
print(km.cluster_centroids_)
X_data = data_temp2[['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','Is_Electronic_check','Is_Contract_Month','StreamingTV_Yes']]
y_data = data_temp2['Churn_Yes']
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
clf = DecisionTreeClassifier(max_depth=2, criterion='entropy')
clf.fit(X_data, y_data)
DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=2,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='best')
y_data
pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100) ], index = X_data.columns, columns = ['Feature Significance in Decision Tree'])
import graphviz
dot_data = tree.export_graphviz(clf,out_file=None, 
                                feature_names=X_data.columns,
                                class_names = ['No', 'Yes'],
                         filled=True, rounded=True,  proportion=True,
                                node_ids=True, #impurity=False,
                         special_characters=True)
graph = graphviz.Source(dot_data) 
graph
