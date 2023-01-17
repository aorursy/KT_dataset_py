import numpy as np # linear algebra
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import seaborn as sns
# Read the data
df = pd.read_csv("/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
# Print the data
df.head()
# Check for the summary statistics
df.describe()
# Plot the histogram
df.hist(bins=20,figsize=(20,20))
# Check for data types and missing values
df.info()
# Drop unnecessary columns
df.drop(["EmployeeNumber","EmployeeCount","StandardHours","Over18"],axis=1,inplace=True)
# Check for nulls
df.isnull().sum().plot.bar()
# Plot Co-relation matrix
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df.corr(), annot=True)
# Plots
plt.figure(figsize=[20,20])
plt.subplot(611)
sns.countplot(x='JobRole',hue='Attrition',data=df)
plt.subplot(612)
sns.countplot(x='HourlyRate',hue='Attrition',data=df)
plt.subplot(613)
sns.countplot(x='JobInvolvement',hue='Attrition',data=df)
plt.subplot(614)
sns.countplot(x='JobLevel',hue='Attrition',data=df)
plt.subplot(615)
sns.countplot(x='DistanceFromHome',hue='Attrition',data=df)
plt.subplot(616)
sns.countplot(x='Age',hue='Attrition',data=df)
# Plot the count of Attrition
plt.figure(figsize=[12,12])
total = float(len(df)) 
ax=sns.countplot(df["Attrition"])
# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.3, i.get_height()+5,
        str(i.get_height()), fontsize=15,
    color='dimgrey')
        # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+0.3, i.get_height()+35,
            '{:1.2f}%'.format(i.get_height()/total*100), fontsize=15,
    color='red')
show()

# What is the median salary of each job roles?

plt.figure(figsize=(15,10))
sns.boxplot(x=df.MonthlyIncome,y=df.JobRole)
# Does people with higher salary work longer and the vice versa?
plt.figure(figsize=(15,10))
sns.boxplot(y=df.JobRole,x=df.TotalWorkingYears)
# Does years with current manager influence the employee to stay longer?
plt.figure(figsize=(15,10))
sns.boxplot(x=df.YearsWithCurrManager,y=df.YearsAtCompany)
# label encode target variable 
df["Attrition"]=df["Attrition"].astype('category')
df["Attrition"] = df["Attrition"].cat.codes
df["Attrition"]
# encode all categorical columns
Obj_col = df.select_dtypes(include='object')
Obj_col
Obj_col.nunique()
# one line code for one-hot-encoding:
df_encoded=pd.get_dummies(df,columns=Obj_col.columns)
df_encoded.head()
X = df_encoded.loc[:,df_encoded.columns!="Attrition"]
y = df_encoded["Attrition"]
print(X.head())
print(y.head())
# Scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X= scaler.fit_transform(X)
y=y.values.reshape(-1,1)
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25)
print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model_lr= LogisticRegression()
model_lr.fit(X_train,y_train)
y_pred = model_lr.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
print('Accuracy {} %'.format(100* accuracy_score(y_pred,y_test)))
#Getting predicted probabilities
y_score = model_lr.predict_proba(X_test)[:,1]
print('\nRoc value '+ str(roc_auc_score(y_test, y_score)))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot =True)
print(classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
model_rf= RandomForestClassifier()
model_rf.fit(X_train,y_train)
pred = model_rf.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
print('Accuracy {} %'.format(100* accuracy_score(y_pred,y_test)))
y_score = model_rf.predict_proba(X_test)[:,1]
print('\nRoc value '+ str(roc_auc_score(y_test, y_score)))
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot =True)
print(classification_report(y_test,y_pred))
import tensorflow as tf
model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=500,activation='relu',input_shape= (51,)))
model.add(tf.keras.layers.Dense(units=500,activation='relu'))
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

model.summary()
model.compile(optimizer='adam',loss='binary_crossentropy',metrics='accuracy')
epochs_hist = model.fit(X_train,y_train,epochs=100,batch_size=25)
y_pred=model.predict(X_test)
y_pred = (y_pred>0.5)
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('percentage')
plt.legend(['loss','accuracy'])
plt.title('Loss and Accuracy plot')
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
cm = confusion_matrix(y_test,y_pred)
print('Accuracy {} %'.format(100* accuracy_score(y_pred,y_test)))
print('\nRoc value '+ str(roc_auc_score(y_test, y_pred)))
sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_pred))
