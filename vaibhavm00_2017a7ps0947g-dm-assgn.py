import numpy as np
import math
import pandas as pd
from sklearn import tree,svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
np.random.seed(42)
train_df=pd.read_csv('/kaggle/input/data-mining-assignment-2/train.csv')
test_df=pd.read_csv('/kaggle/input/data-mining-assignment-2/test.csv')
train_df.info()

y_pred_test_df=pd.DataFrame(data=test_df['ID'],columns=['ID'])
y_pred_test_df.head()
train_df.head()
test_df.head()
train_df.drop(labels=['ID'],axis=1,inplace=True)
test_df.drop(labels=['ID'],axis=1,inplace=True)

train_df
train_df.isnull().any().any()#to check null values
train_df.replace({'yes':1,'no':0,'Yes':1,'No':0},inplace=True)
test_df.replace({'yes':1,'no':0,'Yes':1,'No':0},inplace=True)
train_df=pd.get_dummies(train_df,columns=['col2','col37','col56'])
test_df=pd.get_dummies(test_df,columns=['col2','col37','col56'])

from sklearn import preprocessing
#Performing Min_Max Normalization
standard_scaler = preprocessing.StandardScaler()
np_scaled = standard_scaler.fit_transform(train_df.drop(['Class'],axis=1))
train_df_ = pd.DataFrame(np_scaled)
train_df_['Class']=train_df['Class']
train_df=train_df_
train_df.head()
np_scaled =standard_scaler.fit_transform(test_df)
test_df = pd.DataFrame(np_scaled)
test_df.head()
x=train_df.drop(labels='Class',axis=1)
y=train_df['Class']
x_train,x_valid,y_train,y_valid=train_test_split(x,y,test_size=0.2,random_state=42)
x_test=np.array(test_df)

x_train
df_to_be_sample=pd.concat([x_train,y_train],axis=1)
df_to_be_sample.head()
df_train_class1=df_to_be_sample[df_to_be_sample['Class']==1]
df_train_rest=df_to_be_sample[df_to_be_sample['Class']!=1]
df_oversample=df_train_class1.sample(180,random_state=42,replace=True)
df_oversample.head()
train_df_sampled=pd.concat([df_train_rest,df_oversample],ignore_index=False)
train_df_sampled
x_train=train_df_sampled.drop(['Class'],axis=1)
y_train=train_df_sampled['Class']
np.unique(y_train,return_counts=True)
y_valid.value_counts(dropna=False)
#PRE-PROCESSING COMPLETE
from sklearn.ensemble import RandomForestClassifier

score_train_RF = []
score_valid_RF = []

for i in range(5,25,1):
    rf = RandomForestClassifier(n_estimators = 100, max_depth=i,random_state=42)
    rf.fit(x_train, y_train)
    sc_train = f1_score(y_train,rf.predict(x_train),average='macro')
    score_train_RF.append(sc_train)
    sc_valid = f1_score(y_valid,rf.predict(x_valid),average='macro')
    score_valid_RF.append(sc_valid)
plt.figure(figsize=(15,6))
train_score,=plt.plot(range(5,25,1),score_train_RF,color='blue', linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(5,25,1),score_valid_RF,color='red',linestyle='dashed',  marker='o',markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Validation Score"])
plt.title('Fig4. Score vs. MaxDepth')
plt.xlabel('Max Depth')
plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=100,max_depth=7,random_state=42)
rf.fit(x_train,y_train)
acc_valid = rf.score(x_valid,y_valid)                 
acc_valid
y_pred_RF = rf.predict(x_valid)
confusion_matrix(y_valid, y_pred_RF)
print(classification_report(y_valid, y_pred_RF))
from sklearn.ensemble import RandomForestClassifier

score_train_RF = []
score_valid_RF = []

for i in range(5,25,1):
    rf = RandomForestClassifier(n_estimators = 100, min_samples_split=i,random_state=42)
    rf.fit(x_train, y_train)
    sc_train = f1_score(y_train,rf.predict(x_train),average='macro')
    score_train_RF.append(sc_train)
    sc_valid = f1_score(y_valid,rf.predict(x_valid),average='macro')
    score_valid_RF.append(sc_valid)
plt.figure(figsize=(15,6))
train_score,=plt.plot(range(5,25,1),score_train_RF,color='blue', linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(5,25,1),score_valid_RF,color='red',linestyle='dashed',  marker='o',markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Validation Score"])
plt.title('Fig4. Score vs. Min samples split')
plt.xlabel('Min samples split')
plt.ylabel('Score')
from sklearn.ensemble import RandomForestClassifier

score_train_RF = []
score_valid_RF = []

for i in range(5,25,1):
    rf = RandomForestClassifier(n_estimators = 100, min_samples_leaf=i,random_state=42)
    rf.fit(x_train, y_train)
    sc_train = f1_score(y_train,rf.predict(x_train),average='macro')
    score_train_RF.append(sc_train)
    sc_valid = f1_score(y_valid,rf.predict(x_valid),average='macro')
    score_valid_RF.append(sc_valid)
plt.figure(figsize=(15,6))
train_score,=plt.plot(range(5,25,1),score_train_RF,color='blue', linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(5,25,1),score_valid_RF,color='red',linestyle='dashed',  marker='o',markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Validation Score"])
plt.title('Fig4. Score vs. Min samples leaf')
plt.xlabel('Min samples leaf')
plt.ylabel('Score')
from sklearn.model_selection import GridSearchCV

rf_temp = RandomForestClassifier(n_estimators = 100,random_state=42)        #Initialize the classifier object

parameters = {'max_depth':[5,8,13],'min_samples_split':[2,5,16,24],'min_samples_leaf':[1,5,10,19]}    #Dictionary of parameters

scorer = make_scorer(f1_score, average = 'micro')         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(rf_temp, parameters, scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(x_train, y_train)        #Fit the gridsearch object with X_train,y_train

best_rf = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

print(grid_fit.best_params_)
best_rf=RandomForestClassifier(max_depth=13,min_samples_leaf=1,min_samples_split=16,random_state=42)

best_rf.fit(x_train,y_train)


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_rf = best_rf.predict(x_valid)
cfm = confusion_matrix(y_valid, y_pred_rf, labels = [0,1,2,3])
print(cfm)
#entry (i,j) in a confusion matrix is the number of observations actually in group i, but predicted to be in group j.

print("True Positives of Class 0: ", cfm[0][0])
print("False Positives of Class 0 wrt Class 1: ", cfm[1][0]) # Predicted as 0 but actually in 1 
print("False Positives of Class 0 wrt Class 2: ", cfm[2][0])
print("False Negatives of Class 0 wrt Class 1: ", cfm[0][1]) # Precited as 1 but actually in 0
print("False Negatives of Class 0 wrt Class 2: ", cfm[0][2])
print(classification_report(y_valid, y_pred_rf))
# Precision of class 0: Out of all those that you predicted as 0, how many were actually 0
# Recall of Class 0: Out of all those that were actually 0, how many you predicted to be 0
y_pred_test=best_rf.predict(x_test)

y_pred_test_df['Class']=y_pred_test
y_pred_test_df.head()

from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
  csv = df.to_csv(index=False)
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)

create_download_link(y_pred_test_df)
