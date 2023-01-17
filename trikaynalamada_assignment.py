# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_AV3.csv')
test = pd.read_csv('../input/test_AV3.csv')
train.head()
train.dtypes
train.shape
train.isnull().sum(axis=0)   #checking all the missing values in each column
train.Gender.mode() # mode is a good option tofill missing values
train.Dependents.mode()
train.Loan_Amount_Term.plot()
import matplotlib.pyplot as plt
%matplotlib inline
plt.bar(np.arange(0,614),train.Loan_Amount_Term)
train.Loan_Amount_Term.mean()
train.Loan_Amount_Term.median() 
train.Gender.fillna(value='Male',inplace=True)
train.isnull().sum(axis=0)
train.Gender.value_counts()
train.Gender.str.replace('train.Gender.mode()','Male')
train =pd.read_csv('../input/train_AV3.csv')
train
train.Gender.fillna(value='Male',inplace=True)
train.Gender.value_counts()
train.Married.fillna(value='Yes',inplace=True)
train.isnull().sum(axis=0)
train.Loan_Amount_Term.fillna(value=360.0,inplace=True) # filling loan amount terms also with mode as most people prefer to take 360
train.Self_Employed.value_counts()
train.Self_Employed.fillna(value='No',inplace= True)
train.isnull().sum(axis=0)
train.Credit_History.value_counts()
train.Dependents.value_counts()
import fancyimpute

temp = train
temp = fancyimpute.KNN(k=3).complete(temp.iloc[:,[6,7,8,9,13,14,15]]) # using knn to fill loan amounts as others dont take into account other important features
temp['Gender_Male']= temp.Gender.map({'Male':1,'Female':0})
temp['Married_Yes']= temp.Married.map({'Yes':1,'No':0})
temp['Education_Yes']= temp.Education.map({'Graduate':1,'Not Graduate':0}) # creating a 1,0 map for categorical variables to be used for knn
temp
temp = pd.DataFrame(temp,columns=['Applicant_Income','Coapplicant_Income','Loan_Amount','Loan_Term','Gender_Male','Married_Yes','Education_Yes'])
temp
train1=train

trainmain = pd.concat([train1,temp],axis=1)
trainmain.drop('Gender_Male',axis=1,inplace=True)

trainmain.drop('ApplicantIncome',axis=1,inplace=True)
trainmain.drop('CoapplicantIncome',axis=1,inplace=True)
trainmain.drop('LoanAmount',axis=1,inplace=True)
trainmain.drop('Loan_Amount_Term',axis=1,inplace=True)
trainmain
trainmain.isnull().sum()
temp = pd.concat([temp,trainmain.loc[:,'Credit_History']],axis=1)
temp = fancyimpute.KNN(k=3).complete(temp) # knn for credit history
temp = pd.DataFrame(temp,columns=['0','1','2','3','4','5','6','CreditHistory'])
trainmain = pd.concat([trainmain,temp.loc[:,'CreditHistory']],axis=1)
trainmain.drop('Credit_History',axis=1,inplace=True)
trainmain.Dependents.fillna(value=0,inplace=True)
trainmain.isnull().sum()
train = trainmain
train
train.Applicant_Income.describe()

train['Gender_Male']= train.Gender.map({'Male':1,'Female':0})
train['Married_Yes']= train.Married.map({'Yes':1,'No':0})
train['Education_Yes']= train.Education.map({'Graduate':1,'Not Graduate':0})
train
train['Category']='a'
txt=train
train.Category[(train.Applicant_Income > 2877.5)&(train.Applicant_Income <3812.5)] = 'Lower Middle Class' # creating a category column to quickly convert continuous variable to a categorical one
train
txt = trainmain 
from sklearn.cluster import DBSCAN
temp.columns= ['ApplicantIncome','CoapplicantIncome','LoanAmount','LoanTerm','Gender_Male','Married_Yes','Education_Yes','CreditHistory']
model = DBSCAN(eps=5000,min_samples=3).fit(temp) #using dbscan to detect outliers
print (model)
outliers_df = pd.DataFrame(temp)
from collections import Counter
Counter(model.labels_)
outliers_df[model.labels_==-1]


