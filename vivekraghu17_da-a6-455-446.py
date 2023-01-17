
#Team Members
#Vinay Kumar S R - 01FB16ECS446
#Vivek R - 91FB16ECS455

import numpy as np 

import pandas as pd

File=pd.read_csv("../input/Absenteeism_at_work.csv")
File.head()
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
File.describe()
from sklearn.tree import DecisionTreeClassifier
#Preprocessing data - stage 1 (Removing outliers in label)
sns.boxplot(File['Absenteeism time in hours'])

median = np.median(File['Absenteeism time in hours'])
q75, q25 = np.percentile(File['Absenteeism time in hours'], [75 ,25])
iqr = q75 - q25
print("Lower outlier bound:",q25 - (1.5*iqr))
print("Upper outlier bound:",q75 + (1.5*iqr))

File= File[File['Absenteeism time in hours']<=17]
File= File[File['Absenteeism time in hours']>=-7]
print("count for Output class:")
File['Absenteeism time in hours'].value_counts(sort = False)
fig, ax = plt.subplots(figsize=(20, 20)) 
sns.heatmap(File.corr(), annot = True, ax = ax)
#Splitting data into training and testing
from sklearn.model_selection import train_test_split
y=File['Absenteeism time in hours']
X=File.drop('Absenteeism time in hours',axis=1)#Extracting only the features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
print(File.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("Number of unique ouput classes after preprocessing:",((np.unique(y_train))))

#Dropping the following attributes due to multi-collinearity
X_train=X_train.drop('Service time',axis=1)
X_test=X_test.drop('Service time',axis=1)
#Normalizing features
from sklearn import preprocessing
X_scaled_train = preprocessing.scale(X_train)
X_scaled_test = preprocessing.scale(X_test)
X_scaled_train.shape
# 1 ->Classification technique using SVM (SV classifier)
from sklearn import metrics, svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report


svm=svm.SVC()
svm.fit(X_scaled_train, y_train)
y_pred = svm.predict(X_scaled_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
#  2 -> Classification technique using KNN 

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
error=[]
accuracy=[]
for i in range(1, 40):    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error.append(np.mean(y_pred != y_test))
    accuracy.append(metrics.accuracy_score(y_test,y_pred))

print("Error Rate:\n",error)
print("Accuracy Score:\n",accuracy)
#For choosing the K in the KNN, an iterative approach was practised.
#The K value and the model with the lowest error was choosen from 1 - 40 to determine 
#the required output
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error') 
k = np.argmin(error) + 1
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix  
print("Accuracy when k={}: ".format(k),metrics.accuracy_score(y_test, y_pred)) 
print("Confusion matrix: \n",confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
#3 ->Classification technique using Decision Tree 
from sklearn.tree import DecisionTreeClassifier 

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_scaled_train, y_train) 
y_pred = dtree_model.predict(X_scaled_test) 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
#Conclusion

# 1 - SVM (SVC)
#The accuracy for SV classifier is the highest amongst all other models.
# Justification - The kernel used for SVC is Radial Basis Function(RBF).
#One property of the RBF kernel is that it is infinitely smooth
#They are relatively easy to calibrate, as opposed to other kernels.
#It has localized and finite response along the entire x-axis.

# Higher Overall Average Precison over majority of output classes
# Higher Overall F1-Score over majority of Output classes.
#Also,
#Takes lesser computation power(time).
#Simpler and better for a multi-class classification problem

#Hence the Best model for this dataset would be the SV clasiffier

