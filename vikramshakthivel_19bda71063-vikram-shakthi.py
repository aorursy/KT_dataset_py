# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns                                 #imported to plot different plots

import matplotlib.pyplot as plt                       #imported for visualization

from sklearn import preprocessing                     #Preprocessing

from sklearn.model_selection import train_test_split  #module imported to split the data for training

from sklearn.linear_model import LogisticRegression   #fitting Logistic Regression model  

from sklearn.tree import DecisionTreeClassifier       #fitting Decision Tree classifier 

from sklearn import metrics                           #To evaluate model performance

data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")    #reading the data

data
sns.boxplot(data['currentBack']) #boxplot of 'currentBack' variable

                                 #shows a huge amount of outliers which is to be checked if it has to be included in the model
sns.boxplot(data['velocityBack']) #boxplot of 'velocityBack' variable

                                  #This also shows a huge spread in the data.
sns.boxplot(data['trackingDeviationFront'])  #boxplot of 'trackingDeviationFront' variable

                                             #detection of zero outliers
sns.boxplot(data['motorTempBack'])    #boxplot of 'trackingDeviationFront' variable

                                      # shows negligible amount of outliers in the data
#To compute the correlation matrix

d=data.drop(['flag','timeindex'],axis=1) #dropping the categorical variable and index variable to find correlation between continuous variable

c=d.corr()                               #finding the correlation matrix

c
#plotting heat map for the correlation matrix above for visualization

plt.figure(figsize=(14,7)) #To make the visualization clear

sns.heatmap(data=c)

#seems to be a lot of multicollinearity present in the data
#data being seperated accoding to the categories of target variable, so as to view it on the basis of it on different variables

d_0=data[data['flag']==0] #data containing the category of target variable as 0

d_1=data[data['flag']==1] #data containing the category of target variable as 1

data.columns
plt.figure(figsize=(8,5))

#KDE plot to check the effet of categories on the data

#Here, the movement of currentBack is observed according to the flag variable 

sns.kdeplot(data=d_0['currentBack'], label="flag-0", shade=True)

sns.kdeplot(data=d_1['currentBack'], label="flag-1", shade=True)

plt.xlabel("currentBack")

plt.ylabel('density')

#The plot observes that the CurrentBack variable differs according to the flag variable differently

#we can make a statement that the variable is an influencing factor to the flag variable
plt.figure(figsize=(8,5))

#Here, the movement of 'motorTempBack' is observed according to the flag variable 

sns.kdeplot(data=d_0['motorTempBack'], label="flag-0", shade=True)

sns.kdeplot(data=d_1['motorTempBack'], label="flag-1", shade=True)

plt.xlabel("motorTempBack")

plt.ylabel('density')

#Again, we observe that the variable differs along the flag categories

#we can make a statement that the variable is an influencing factor to the flag variable
plt.figure(figsize=(8,5))

#Here, the movement of 'trackingDeviationFront' is observed according to the flag variable 

sns.kdeplot(data=d_0['trackingDeviationFront'], label="flag-0", shade=True)

sns.kdeplot(data=d_1['trackingDeviationFront'], label="flag-1", shade=True)

plt.xlabel("trackingDeviationFront")

plt.ylabel('density')

#This plot shows the overlap of categories on the 'trackingDeviationFront' variable

#this variable might not influence the flag variable and hence not including in the model
plt.figure(figsize=(8,5))

#Here, the movement of 'motorTempFront' is observed according to the flag variable 

sns.kdeplot(data=d_0['motorTempFront'], label="flag-0", shade=True)

sns.kdeplot(data=d_1['motorTempFront'], label="flag-1", shade=True)

#This variable shows an influence on the flag variable but also has strong correlation towards the 'motorTempBack' variable and hence,

#the problem of multicollinearity rises. To overcome that, one of these variables are proceeded with the analysis.
#Similarly, KDE plot for each variable was read and only those were selected which had an influence to flag variable and was proceeded for analysis

#By this, we find out the influential variables and also we try to reduce the multicollinearity by taking only those set of variables.
plt.figure(figsize=(8,5))

#the plot between the variable and log odds so as to check if there is a sigmoid curve in the plot to test the assumption of Logistic regression

sns.regplot(x=d['currentBack'],y=data['flag'],logistic=True)

plt.title("currentBack log odds linear plot")

#the plot remembles almost a sigmoid representing the relationship between the variable and log of odds of flag variable

#similarly, the upcoming plots are also observed according the the above criteria anthe features have been selected
plt.figure(figsize=(8,5))



#the plot between the variable and log odds so as to check if there is a sigmoid curve in the plot to test the assumption of Logistic regression

plt.title("velocityBack log odds linear plot")



sns.regplot(x=d['velocityBack'],y=data['flag'],logistic=True)

#the plot remembles almost a line representing the relationship between the variable and log of odds of flag variable.

#The variable might or might not be affecting the target variable.
plt.figure(figsize=(8,5))

plt.title("currentFront log odds linear plot")

sns.regplot(x=d['currentFront'],y=data['flag'],logistic=True)

#This variable shows an inreasing/ linear relationship between 'currentFront' and log odds of target variable
plt.figure(figsize=(8,5))

plt.title("motorTempBack log odds linear plot")

sns.regplot(x=d['motorTempBack'],y=data['flag'],logistic=True)

#The plot shows kind of a u-shaped cure making it somewhat eligible for logistic model
X=data[['currentBack','velocityBack','motorTempBack','currentFront']] #Taking only the important features from KDE plot

y=data['flag']                                                        #Target variable
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0) #Splitting the data into train and test with 30% of the data as testdata

#Calling the function of the model

logreg = LogisticRegression()



# fit the model with data

logreg.fit(X_train,y_train)



# predicting the values

y_pred=logreg.predict(X_test)
print(metrics.roc_auc_score(y_test, y_pred))

#cheking the roc_auc_score which tells the confidence with which the model can predict the classes of the target variable
print(metrics.classification_report(y_test, y_pred))

#checking the classification report

#we see taht f1 score for 1 is lesser and hence, we proceed with a different model.
#To improve the analysis, different model is being applied

#Decision trees classifier is being used as the next model

# Create Decision Tree classifer object

clf = DecisionTreeClassifier(criterion="entropy",max_depth=8)



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

test=pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv') #importing the test data

test.drop(['timeindex'],axis=1,inplace=True)                             #dropping timeindex

test=test[['currentBack','velocityBack','motorTempBack','currentFront']] #selecting the appropriate features needed for prediction
test['a_flag']=clf.predict(test) #predicting the values from test data
sample_submission9=pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv') #importing thesample submission file
sample_submission9['flag']=test['a_flag'] #overwriting the target variable into sample submission
sample_submission9.to_csv('sample_submission9.csv',index=False) #submitting the predictions