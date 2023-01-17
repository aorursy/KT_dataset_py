import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# You have to include the full link to the csv file containing your dataset
dataset = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
dataset.sample(5)
dataset.info()
dataset.describe()
# Let's replace 'Attrition' , 'overtime' , 'Over18' column with integers before performing any visualizations 
dataset['Attrition'] = dataset['Attrition'].apply(lambda x:1 if x == 'Yes' else 0)
dataset['OverTime'] = dataset['OverTime'].apply(lambda x:1 if x == 'Yes' else 0)
dataset['Over18'] = dataset['Over18'].apply(lambda x:1 if x == 'Y' else 0)
dataset.head()
# Let's see if we have any missing data.
sns.heatmap(dataset.isnull(),cmap = 'Blues', cbar = False, yticklabels = False)
dataset.hist(bins=30,figsize=(20,20),color='g')
# Several features such as 'MonthlyIncome' and 'TotalWorkingYears' are tail heavly
# It makes sense to drop 'EmployeeCount' and 'Standardhours' since they do not change from one employee to the other
# It makes sense to drop 'EmployeeCount' , 'Standardhours' and 'Over18' since they do not change from one employee to the other
# Let's drop 'EmployeeNumber' as well
# use inplace = True to change the values in memory.

dataset.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber'],axis = 1, inplace = True)
# Let's see how many employees left the company! 
left_df = dataset[dataset['Attrition'] == 1]
stayed_df = dataset[dataset['Attrition'] == 0]
print('1. Total = {} '.format(len(dataset)))
print('2. Number of employees left the company = {}'.format(len(left_df)))
print('3. Percentage of employees left the company = {}'.format((len(left_df)/len(dataset))*100))
print('4. Number of employees who stayed in the company = {}'.format(len(stayed_df)))
print('5. Percentage of employees stayed the company = {}'.format((len(stayed_df)/len(dataset))*100))
left_df.describe()

#  Let's compare the mean and std of the employees who stayed and left 
# 'age': mean age of the employees who stayed is higher compared to who left
# 'DailyRate': Rate of employees who stayed is higher
# 'DistanceFromHome': Employees who stayed live closer to home 
# 'EnvironmentSatisfaction' & 'JobSatisfaction': Employees who stayed are generally more satisifed with their jobs
# 'StockOptionLevel': Employees who stayed tend to have higher stock option level
stayed_df.describe()
correlations = dataset.corr()
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(correlations, annot=True)
plt.figure(figsize=(25,12))
sns.countplot(x = 'Age', hue = 'Attrition', data = dataset)
plt.figure(figsize=(20,20))
plt.subplot(511)
sns.countplot(x = 'JobRole',hue = 'Attrition', data=dataset)
plt.subplot(512)
sns.countplot(x = 'MaritalStatus',hue = 'Attrition', data=dataset)
plt.subplot(513)
sns.countplot(x = 'JobInvolvement',hue = 'Attrition', data=dataset)
plt.subplot(514)
sns.countplot(x = 'JobLevel',hue = 'Attrition', data=dataset)
plt.subplot(515)
sns.countplot(x = 'OverTime',hue = 'Attrition', data=dataset)
plt.figure(figsize = (12,8))
sns.kdeplot(left_df['DistanceFromHome'], label = 'Employees who left', color = 'r', shade = True)
sns.kdeplot(stayed_df['DistanceFromHome'],label='Employees who stayed',color = 'b',shade=True)
plt.xlabel('Distance from home')
plt.figure(figsize=(12,8))
sns.kdeplot(left_df['YearsWithCurrManager'],shade=True,color='r',label='Employes who left')
sns.kdeplot(stayed_df['YearsWithCurrManager'],shade=True,color='b',label='Employes who stayed')

plt.xlabel('Number of years with the current manager')
plt.title('Number of years with the current manager v/s Atrition')
plt.figure(figsize=(12,8))
sns.kdeplot(left_df['TotalWorkingYears'],label='Employees who left',shade = True, color = 'r')
sns.kdeplot(stayed_df['TotalWorkingYears'],label='Employees who stayed',shade = True, color = 'b')

plt.xlabel('Number of total working years')
plt.title('Number of total working years v/s Attrition')
# Let's see the Gender vs. Monthly Income
sns.boxplot(x='MonthlyIncome',y='Gender',data=dataset)
# Let's see the Jod role vs. Monthly Income
plt.figure(figsize=(10,8))
sns.boxplot(x='MonthlyIncome',y='JobRole',data=dataset)
cat_var = [key for key in dict(dataset.dtypes)
             if dict(dataset.dtypes)[key] in ['object'] ] 
cat_var
X_cat = dataset[['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus']]
X_cat.head()
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
X_cat
X_cat = pd.DataFrame(X_cat)
X_cat.head()
numeric_var = [key for key in dict(dataset.dtypes)
                   if dict(dataset.dtypes)[key]
                       in ['float64','float32','int32','int64']]
numeric_var
X_numerical = dataset[['Age','Attrition','DailyRate','DistanceFromHome','Education','EnvironmentSatisfaction','HourlyRate','JobInvolvement','JobLevel','JobSatisfaction','MonthlyIncome','MonthlyRate','NumCompaniesWorked','OverTime','PercentSalaryHike','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','TotalWorkingYears','TrainingTimesLastYear','WorkLifeBalance','YearsAtCompany','YearsInCurrentRole','YearsSinceLastPromotion','YearsWithCurrManager']]
X_all = pd.concat([X_cat,X_numerical],axis=1)
X_all.head()
# I will now drop the target variable 'Attrition'
X_all.drop('Attrition',axis=1,inplace=True)
X_all.shape
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(X_all)
scaled_data
y = dataset['Attrition']
y.shape
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
X_train,X_test,y_train,y_test = train_test_split(scaled_data,y,test_size = 0.25, random_state=43)
model_LR = LogisticRegression()
model_LR.fit(X_train,y_train)
LR_pred = model_LR.predict(X_test)
print('The accuracy score for Logistic Regression is: {}'.format(100*accuracy_score(LR_pred,y_test)))
cm = confusion_matrix(LR_pred,y_test)
sns.heatmap(cm,annot=True)
print(classification_report(LR_pred,y_test))
from sklearn.ensemble import RandomForestClassifier
model_RF = RandomForestClassifier()
model_RF.fit(X_train,y_train)
RF_pred = model_RF.predict(X_test)
print('The accuracy score for Random Forest is: {}'.format(100*accuracy_score(RF_pred,y_test)))
# Testing Set Performance
cm = confusion_matrix(RF_pred,y_test)
sns.heatmap(cm,annot=True)
print(classification_report(RF_pred,y_test))
import tensorflow as tf
model_NN = tf.keras.models.Sequential()
model_NN.add(tf.keras.layers.Dense(units=500, activation='relu', input_shape=(50, )))
model_NN.add(tf.keras.layers.Dense(units=500, activation='relu'))
model_NN.add(tf.keras.layers.Dense(units=500, activation='relu'))
model_NN.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
model_NN.summary()
model_NN.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
scaled_df = pd.DataFrame(scaled_data,columns=X_all.columns)
scaled_df.head()
X_train_new,X_test_new,y_train_new,y_test_new = train_test_split(scaled_df,y,test_size = 0.25, random_state=43)
epochs_hist = model_NN.fit(X_train_new, y_train_new, epochs = 30, batch_size = 50)
y_pred = model_NN.predict(X_test)
y_pred = (y_pred > 0.5)
plt.plot(epochs_hist.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])
plt.plot(epochs_hist.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Accuracy')
plt.legend(['Training Accuracy'])
# Testing Set Performance
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_pred))
#from imblearn.over_sampling import SMOTE
#oversampler = SMOTE(random_state=0)
#smote_train, smote_target = oversampler.fit_sample(X_train_new, y_train_new)
#epochs_hist = model_NN.fit(smote_train, smote_target, epochs = 10, batch_size = 50)
