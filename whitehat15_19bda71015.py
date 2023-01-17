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
#importing package called pandas

import pandas as pd 



#importing package called numpy

import numpy as np



#importing package called matplotlib

import matplotlib.pyplot as plt



#importing package called seaborn

import seaborn as sns
#reading train csv using pandas

train=pd.read_csv('/kaggle/input/bda-2019-ml-test/Train_Mask.csv') 



#reading test csv using pandas

test=pd.read_csv('/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv')



#reading sample csv using pandas

sample=pd.read_csv('/kaggle/input/bda-2019-ml-test/Sample Submission.csv')
#viewing first 5 rows of train dataset using head function

train1.head()
#gives no. of rows and no. of columns in a list

train1.shape
corr=train1.corr()            #finding correlation matrix

#plotting heatmap for correlation matrix to obtain correlation plot

sns.heatmap(corr,vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
train.head()  #viewing first 5 rows of train csv
#plotting distribution plot for currentback

sns.distplot(train['currentBack'])
#plotting distribution plot for motortempback

sns.distplot(train['motorTempBack'])
#plotting distribution plot for positionback

sns.distplot(train['positionBack'])
#plotting distribution plot for refpositionback

sns.distplot(train['refPositionBack'])
#plotting distribution plot for velocityback

sns.distplot(train['velocityBack'])
#plotting distribution plot for refvelocityback

sns.distplot(train['refVelocityBack'])
#plotting distribution plot for trackingDeviationback

sns.distplot(train['trackingDeviationBack'])
#plotting distribution plot for currentfront

sns.distplot(train['currentFront'])
#plotting distribution plot for motortempfront

sns.distplot(train['motorTempFront'])
#plotting distribution plot for positionfront

sns.distplot(train['positionFront'])
#plotting distribution plot for refpositionfront

sns.distplot(train['refPositionFront'])
#plotting distribution plot for refvelocityfront

sns.distplot(train['refVelocityFront'])
#plotting distribution plot for trackingDeviationFront

sns.distplot(train['trackingDeviationFront'])
#plotting distribution plot for velocityFront

sns.distplot(train['velocityFront'])
#checking for null values using isnull and printing number of null values for each columns

train.isnull().sum()
#gives of no.of both classes in the train dataset

train['flag'].value_counts()
# gives different statistical values of the train dataset

train.describe()
#gives the non null count and datatype for each columns

train.info()
#plotting boxplot for the feature to detect the outliers

sns.boxplot(train['currentBack'])
#imputing nan to the values greater the maximum value(Q3+1.5*IQR)

train.loc[train['currentBack']>(90+1.5*107),'currentBack']=np.nan
#filling nan with 95th percentile

train['currentBack'].fillna(np.array(train['currentBack'].quantile(.95)),inplace=True)
#plotting boxplot for the feature after removing the outliers 

sns.boxplot(train['currentBack'])
#plotting boxplot for the feature to detect the outliers

sns.boxplot(train['motorTempBack'])
#imputing nan to the values less than the minimum value(Q1-1.5*IQR)

train.loc[train['motorTempBack']<(39-1.5*5),'motorTempBack']=np.nan
#filling nan with 5th percentile

train['motorTempBack'].fillna(np.array(train['motorTempBack'].quantile(.05)),inplace=True)
#plotting boxplot for the feature after removing the outliers 

sns.boxplot(train['motorTempBack'])
#plotting boxplot for the feature to detect the outliers

sns.boxplot(train['positionBack'])
#imputing nan to the values less than the minimum value(Q1-1.5*IQR)

train.loc[train['positionBack']<(384-1.5*195),'positionBack']=np.nan
#filling nan with 5th percentile

train['positionBack'].fillna(np.array(train['positionBack'].quantile(.05)),inplace=True)
#plotting boxplot for the feature after removing the outliers 

sns.boxplot(train['positionBack'])
#plotting boxplot for the feature to detect the outliers

sns.boxplot(train['refPositionBack'])
#imputing nan to the values less than the minimum value(Q1-1.5*IQR)

train.loc[train['refPositionBack']<(389-1.5*190),'refPositionBack']=np.nan
#filling nan with 5th percentile

train['refPositionBack'].fillna(np.array(train['refPositionBack'].quantile(.05)),inplace=True)

#plotting boxplot for the feature after removing the outliers 

sns.boxplot(train['refPositionBack'])
#plotting boxplot for the feature to detect the outliers

sns.boxplot(train['currentFront'])
#imputing nan to the values greater than the maximum value(Q3+1.5*IQR)

train.loc[train['currentFront']>(191+155*1.5),'currentFront']=np.nan
#filling nan with 95th percentile

train['currentFront'].fillna(np.array(train['currentFront'].quantile(.95)),inplace=True)
#plotting boxplot for the feature after removing the outliers 

sns.boxplot(train['currentFront'])
#plotting boxplot for the feature to detect the outliers

sns.boxplot(train['motorTempFront'])
#imputing nan to the values less than the minimum value(Q1-1.5*IQR)

train.loc[train['motorTempFront']<(41-1.5*4),'motorTempFront']=np.nan
#filling nan with 5th percentile

train['motorTempFront'].fillna(np.array(train['motorTempFront'].quantile(.05)),inplace=True)
#plotting boxplot for the feature after removing the outliers 

sns.boxplot(train['motorTempFront'])
#plotting boxplot for the feature to detect the outliers

sns.boxplot(train['positionFront'])
#imputing nan to the values less than the minimum value(Q1-1.5*IQR)

train.loc[train['positionFront']<(761-204*1.5),'positionFront']=np.nan
#filling nan with 5th percentile

train['positionFront'].fillna(np.array(train['positionFront'].quantile(.05)),inplace=True)
#plotting boxplot for the feature after removing the outliers 

sns.boxplot(train['positionFront'])
#plotting boxplot for the feature to detect the outliers

sns.boxplot(train['refPositionFront'])
#imputing nan to the values less than the minimum value(Q1-1.5*IQR)

train.loc[train['refPositionFront']<(772-199*1.5),'refPositionFront']=np.nan
#filling nan with 5th percentile

train['refPositionFront'].fillna(np.array(train['refPositionFront'].quantile(.05)),inplace=True)
#plotting boxplot for the feature after removing the outliers 

sns.boxplot(train['refPositionFront'])
#ob tainind different statistical values of train dataset

train.describe()
y=train['flag']                           #assigning the target variable

x=train.drop('flag',axis='columns')     #dropping target variable from the train data
# viewing first 5 rows of y

y.head()
# viewing first 5 rows of x

x.head()
#taking square root of the whole data

x.transform([np.sqrt]).describe()       
from sklearn import model_selection     #importing model selection from scikit learn
#viewing first 5 rows of the test data

test.head()
#viewing first 5 rows of test csv

test.head()
#viewing first 5 rows of the sample data

sample.head()
from sklearn.model_selection import train_test_split      #importing train test split from scikit learn model selection

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier                      #importing RandomForestClassifier from scikit learn ensemble

classifier = RandomForestClassifier(n_estimators=90, random_state=0)      #assigning function into the classifier variable

classifier.fit(x_train, y_train)                                        #fitting the model

y_pred2 = classifier.predict(x_test)                                     #predicting the values for xtest
#values of predicted

y_pred2 
#taking square root of the test data

test.transform([np.sqrt]).describe()
#predictind values for the given test dataset

pred=classifier.predict(test)
#replacing the flag values in sample csv with predicted values

sample['flag']=pred
#converting the sample dataframe into a csv file

sample.to_csv('sample_submission.csv',index=False)
#shows first 5 rows of the sample

sample.head()
from sklearn.metrics import accuracy_score,f1_score                 #importing accuracy_score,f1_score fromsklearn metrics

#obtaining f1 score using f1_score function

#f1 score(harmonic mean of precision and recall)

f1_score(y_test,y_pred2)
#obtaining accuracy of the model using accuracy_score function

accuracy_score(y_test,y_pred2)