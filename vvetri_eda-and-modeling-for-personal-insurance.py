import pandas as pd
import numpy as pi

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

bank_cust_data = pd.read_csv("/kaggle/input/personal-loan-modeling/Bank_Personal_Loan_Modelling.csv")

bank_cust_data.head()

print("Shape of the data :{0}".format(bank_cust_data.shape))
print("Size of the data :{0}".format(bank_cust_data.size))
print("nDim of the data :{0}".format(bank_cust_data.ndim))

bank_cust_data.describe()
# Identify the types and the missing values of the data

print("\nList of Null values in each column")
print(bank_cust_data.isna().sum())

print("\nData types of each column")
print(bank_cust_data.dtypes)

#All fields are numeric and no missing values
print("Unique Values")
print(bank_cust_data['Personal Loan'].unique())

print("\nSeacrching for 'scott' in all column")
print(bank_cust_data.isin(['scott']).any())

sns.boxplot('Personal Loan',data=bank_cust_data);
# Drop un-wanted column
bank_extract_data = bank_cust_data.drop(['ID'],axis=1).copy()

print('Shape of Data: ', bank_extract_data.shape)

print('\nMedian for the data')
print(bank_extract_data.median())

print('\nMode for the data')
print(bank_extract_data.mode())

print('\n Five Point Summary for the data')
bank_extract_data.describe()
# Unique values for the categorical

print('Number of Unique values of ZipCodes: ',bank_extract_data['ZIP Code'].nunique())
print('Unique values of Family: ',bank_extract_data['Family'].unique())
print('Unique values of Education: ',bank_extract_data['Education'].unique())
print('Unique values of Personal Loan: ',bank_extract_data['Personal Loan'].unique())
print('Unique values of Securities Account: ',bank_extract_data['Securities Account'].unique())
print('Unique values of CD Account ',bank_extract_data['CD Account'].unique())
print('Unique values of Online ',bank_extract_data['Online'].unique())
print('Unique values of CreditCard ',bank_extract_data['CreditCard'].unique())


print('\n*** Observation: Education attribute needs "One hot Encoding"')
# Distribution for the non-categorical attributes

fig, axis = plt.subplots(2, 3, figsize=(25, 15), sharex=False)

sns.distplot(bank_extract_data['Age'],bins=10,ax=axis[0,0]);

sns.distplot(bank_extract_data['Experience'],ax=axis[0,1]);

sns.distplot(bank_extract_data['Income'],ax=axis[0,2]);

sns.distplot(bank_extract_data['CCAvg'],ax=axis[1,0],color='orange');

sns.distplot(bank_extract_data['Mortgage'],ax=axis[1,1],color='orange');

sns.distplot(bank_extract_data['Family'],ax=axis[1,2],color='orange');


plt.show()
print("Age Parameter Right Skewed: ", bank_extract_data['Age'].mean() > bank_extract_data['Age'].median() )
print("CCAvg Parameter Right Skewed: ", bank_extract_data['CCAvg'].mean() > bank_extract_data['CCAvg'].median() )
print("Mortgage Parameter Right Skewed: ", bank_extract_data['CCAvg'].mean() > bank_extract_data['CCAvg'].median() )

print("\n*** The skwed data does not impact the algorithm we are going to use")
fig, axis1 = plt.subplots(1, 2, figsize=(20, 6), sharex=False)

sns.boxplot(bank_extract_data['ZIP Code'],ax=axis1[0],orient='h');
sns.boxplot(bank_extract_data['Mortgage'],ax=axis1[1],color='red',orient='v');

plt.show()

#The outlier shown below does not impact the algorithm hence we are leaving out the outliers
fig, axis = plt.subplots(1, 2, figsize=(20, 10), sharex=False)

sns.violinplot(bank_extract_data['Personal Loan'],bank_extract_data['Income'],ax=axis[0]);
sns.violinplot(bank_extract_data['Personal Loan'],bank_extract_data['Age'],ax=axis[1]);

plt.show()
plt.clf()
fig, axis = plt.subplots(1, 2, figsize=(20, 10), sharex=False)

sns.violinplot(bank_extract_data['Personal Loan'],bank_extract_data['Experience'],ax=axis[0]);
sns.violinplot(bank_extract_data['Personal Loan'],bank_extract_data['Mortgage'],ax=axis[1]);

plt.show()

#bank_pairplot_data=bank_extract_data.drop(['Personal Loan',])

#bank_pairplot_data.head();

plot_vars=['Age','Experience','Income','ZIP Code','CCAvg','Mortgage','Online']

sns.pairplot(bank_extract_data,kind='scatter',x_vars=plot_vars,y_vars=plot_vars,hue='Personal Loan');
sns.catplot(x='Income', y='Mortgage', kind='swarm', data=bank_extract_data,aspect=2);

# Deeper look into income/Mortgage out of curiosity to see the linear reference
fig, axis = plt.subplots(2, 1, figsize=(25, 30), sharex=False)
threshold=0.25

sns.heatmap(bank_extract_data.corr(),annot=True,fmt='f',ax=axis[0])

filtered_data =pd.DataFrame(bank_extract_data.corr()>threshold)

print('The correlated data based on threshold: ',threshold)
sns.heatmap(filtered_data,annot=True,fmt='d',ax=axis[1]);
plt.show()


# As per corelation factor only 3 feilds are applicable, but considering all attributes as for now.
# so as the attributes might have other positive impact
# Applying encoding on the Education parameter to generate the even 0/1 parameter
bank_modified = pd.get_dummies(bank_extract_data['Education'],prefix='Edu_') 
bank_modified=pd.concat([bank_extract_data.copy(), bank_modified], axis=1)

parameter_drop=['Education'] # dropping the duplicate param

bank_modified.drop(parameter_drop,axis=1,inplace=True)

bank_modified['Income']=bank_extract_data['Income']/12 # Converting to the monthly income
bank_modified['Mortgage']=bank_extract_data['Mortgage']/10 # reducing the mortgage to normalise data

bank_modified.describe()
#Spliting the data for the model training

axis_x = bank_modified.drop(['Personal Loan'],axis=1)
axis_y = bank_modified['Personal Loan']

#spliting the data into 70/30
x_train,x_test,y_train,y_test = train_test_split(axis_x,axis_y,test_size=0.3,random_state=100)

x_train.head()
# Identidy the true and false for Personal Loan
loan_true = len(bank_modified.loc[bank_modified['Personal Loan'] == 1])
loan_false = len(bank_modified.loc[bank_modified['Personal Loan'] == 0])

print (f"{len(x_train)/len(bank_modified)*100} % data in the Training")
print (f"{len(x_test)/len(bank_modified)*100} % data in the Testing")

print("\nPercent of the Loan offer accepted")
print (f"Accepted: {loan_true} in total {len(bank_modified)} {loan_true/len(bank_modified)*100}%")
print (f"Not Accepted: {loan_false} in total {len(bank_modified)} {loan_false/len(bank_modified)*100}%")

print("\nPercent of the Loan offer accepted in Training data")
print (f"Accepted: {len(y_train.loc[y_train[:]==1])} in total {len(y_train)} {len(y_train.loc[y_train[:]==1])/len(y_train)*100}%")
print (f"Not Accepted: {len(y_train.loc[y_train[:]==0])} in total {len(y_train)} {len(y_train.loc[y_train[:]==0])/len(y_train)*100}%")

print("\nPercent of the Loan offer accepted in Test data")
print (f"Accepted: {len(y_test.loc[y_test[:]==1])} in total {len(y_test)} {len(y_test.loc[y_test[:]==1])/len(y_test)*100}%")
print (f"Not Accepted: {len(y_test.loc[y_test[:]==0])} in total {len(y_test)} {len(y_test.loc[y_test[:]==0])/len(y_test)*100}%")

from sklearn.linear_model import LogisticRegression

#model train is set
model = LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)

thres_predict_train=model.predict_proba(x_train)[:,1] > 0.8 # Threshold changes the confusion matrix
thres_predict_test=model.predict_proba(x_test)[:,1] > 0.8 

print(f"Training Accuracy Score: ",metrics.accuracy_score(y_train,thres_predict_train))
print(f"Test Accuracy Score: ",metrics.accuracy_score(y_test,thres_predict_test))

print("\nClassification Report")
report_logistic=classification_report(y_test,thres_predict_test,labels=[1,0])
print(report_logistic)

confusin_matix_logistic = metrics.confusion_matrix(y_test,thres_predict_test,labels=[1,0])
confusin_matix_logistic = pd.DataFrame(confusin_matix_logistic,index=['Accepted','NotAccepted'],
                              columns=['Pred_Accepted', 'Pred_NotAccepted'])

plt.figure(figsize=(10,5))
sns.heatmap(confusin_matix_logistic,annot=True,fmt='g');
plt.show()
from sklearn.naive_bayes import GaussianNB 

data_model = GaussianNB()

data_model.fit(x_train,y_train.ravel()) #ravel convert to an one dimensional array

naive_train_predict = data_model.predict(x_train)
naive_test_predict = data_model.predict(x_test)

print(f"Training Accuracy Score: ",metrics.accuracy_score(y_train,naive_train_predict))
print(f"Test Accuracy Score: ",metrics.accuracy_score(y_test,naive_test_predict))

print("\nClassification Report")
report_naiveBayes=classification_report(y_test,naive_test_predict,labels=[1,0])
print(report_naiveBayes)

confusin_matix_Naive = metrics.confusion_matrix(y_test,naive_test_predict,labels=[1,0])
confusin_matix_Naive = pd.DataFrame(confusin_matix_Naive,index=['Accepted','NotAccepted'],
                              columns=['Pred_Accepted', 'Pred_NotAccepted'])

plt.figure(figsize=(10,5))
sns.heatmap(confusin_matix_Naive,annot=True,fmt='g');
plt.show()
from sklearn.neighbors import KNeighborsClassifier


KNN_model = KNeighborsClassifier(n_neighbors=5,weights='distance',metric='euclidean') # 

KNN_model.fit(x_train,y_train)

KNN_predict_train = KNN_model.predict(x_train)
KNN_predict_test = KNN_model.predict(x_test)

print(f"Training Accuracy Score: ",metrics.accuracy_score(y_train,KNN_predict_train))
print(f"Test Accuracy Score: ",metrics.accuracy_score(y_test,KNN_predict_test))

print("\nClassification Report")
report_KNN = classification_report(y_test,KNN_predict_test,labels=[1,0])
print(report_KNN)

confusin_matix_KNN = metrics.confusion_matrix(y_test,KNN_predict_test,labels=[1,0])
confusin_matix_KNN = pd.DataFrame(confusin_matix_KNN,index=['Accepted','NotAccepted'],
                              columns=['Pred_Accepted', 'Pred_NotAccepted'])

plt.figure(figsize=(10,5))
sns.heatmap(confusin_matix_KNN,annot=True,fmt='g');
plt.show()
from sklearn import svm

svm_model = svm.SVC(gamma=100,C=3)
svm_model.fit(x_train,y_train)


svm_predic_train = svm_model.predict(x_train)
svm_predic_test = svm_model.predict(x_test)

print(f"Training Accuracy Score: ",metrics.accuracy_score(y_train,svm_predic_train))
print(f"Test Accuracy Score: ",metrics.accuracy_score(y_test,svm_predic_test))

print("\nClassification Report")
report_svm=classification_report(y_test,svm_predic_test,labels=[1,0])
print(report_svm)

confusin_matix_svm = metrics.confusion_matrix(y_test,svm_predic_test,labels=[1,0])
confusin_matix_svm = pd.DataFrame(confusin_matix_svm,index=['Accepted','NotAccepted'],
                              columns=['Pred_Accepted', 'Pred_NotAccepted'])

plt.figure(figsize=(10,5))
sns.heatmap(confusin_matix_svm,annot=True,fmt='g');
plt.show()
# Data is imbalanced hence model accuracy alone does not alone, the below parater to be verifed for selecting a model

print("\nLogisitic Regression\n",report_logistic)
print("\nNaive Bayes\n",report_naiveBayes)
print("\nKNN Classifier\n",report_KNN)
print("\nSVM\n",report_svm)