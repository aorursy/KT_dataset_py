import pandas as pd

gdata=pd.read_csv("../input/Admission_Predict_Ver1.1.csv")

col_names=gdata.columns.tolist()

print("Column names:")

print(col_names)

print("\nSample Data:")

print(gdata.head())

gdata=gdata.rename(columns={'Serial No.':'no','GRE Score':'gre','TOEFL Score':'toefl','University Rating':'rating','SOP':'sop','LOR ':'lor',

                           'CGPA':'gpa','Research':'research','Chance of Admit ':'chance'})
gdata.dtypes
print('Shape of the data:')

gdata.shape
print('Missing values in columns:')

gdata.isnull().any()
gdata.describe()
gdata.groupby('rating').mean()
gdata[gdata['chance']>0.82].groupby('chance').mean()
import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

plt.hist(gdata['chance'],bins=10,color="orange")

plt.title('Histogram of Admission Chance')

plt.xlabel('Admission Chance')

plt.ylabel('Frequency of Chance')

plt.show()
plt.figure(figsize=(12,8))

plt.plot(range(len(gdata[gdata['research']==1])), gdata[gdata['research']==1]['chance'], color='orange')

plt.plot(range(len(gdata[gdata['research']==0])), gdata[gdata['research']==0]['chance'], color='olive')

plt.show()
gdata.boxplot(column='chance',by='rating',grid=False,figsize=(12,8))

plt.title('The Chance of Admission for University Ratings')

plt.xlabel('University Rating')

plt.ylabel('Chance of Admission')

plt.show()
gdata.hist(bins=10, figsize=(20,15))

plt.show()
gdata.drop(['no'],axis=1,inplace=True)

var=gdata.columns.values.tolist()

y=gdata['chance']

x=[i for i in var if i not in ['chance']]

x=gdata[x]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
from sklearn.preprocessing import MinMaxScaler

xs=MinMaxScaler()

x_train[x_train.columns] = xs.fit_transform(x_train[x_train.columns])

x_test[x_test.columns] = xs.transform(x_test[x_test.columns])
import numpy as np

cy_train=[1 if chance > 0.82 else 0 for chance in y_train]

cy_train=np.array(cy_train)



cy_test=[1 if chance > 0.82 else 0 for chance in y_test]

cy_test=np.array(cy_test)



# Fitting logistic regression model

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, cy_train)





# Printing accuracy score & confusion matrix

from sklearn.metrics import accuracy_score

print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(cy_test, lr.predict(x_test))))

print('--------------------------------------')

from sklearn.metrics import classification_report

print(classification_report(cy_test, lr.predict(x_test)))



cy = lr.predict(x_test)

from sklearn.metrics import confusion_matrix

import seaborn as sns

lr_confm = confusion_matrix(cy, cy_test, [1,0])

sns.heatmap(lr_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.title('Logistic Regression')

plt.show()
# Fitting random forest model

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_train, cy_train)



# Printing accuracy score & confusion matrix

print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(cy_test, rf.predict(x_test))))

print('--------------------------------------')

print(classification_report(cy_test, rf.predict(x_test)))



cy = rf.predict(x_test)

rf_confm = confusion_matrix(cy, cy_test, [1,0])

sns.heatmap(rf_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.title('Random Forest')

plt.show()
# Fitting support vector machine model

from sklearn.svm import SVC

svc = SVC()

svc.fit(x_train, cy_train)



# Printing accuracy score & confusion matrix

print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(cy_test, svc.predict(x_test))))

print('--------------------------------------')

print(classification_report(cy_test, svc.predict(x_test)))



cy = svc.predict(x_test)

svc_confm = confusion_matrix(cy, cy_test, [1,0])

sns.heatmap(svc_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.title('Support Vector Machine')

plt.show()
f_imp=pd.Series(rf.feature_importances_,index=x_train.columns).sort_values(ascending=False)

print(f_imp)