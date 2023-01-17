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
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline
data = pd.read_csv("/kaggle/input/HR_Data.csv")
data.head()
data = data.rename(columns={'Joining.Bonus':'JoiningBonus','Candidate.relocate.actual':'CandidatRelocateActual','Candidate.Source':'CandidateSource','Rex.in.Yrs':'RexInYrs','Candidate.Ref': 'CandidateRef', 'Duration.to.accept.offer': 'Durationtoacceptoffer','DOJ.Extended': 'DOJExtended', 'Notice.period': 'NoticePeriod','Offered.band':'OfferedBand','Pecent.hike.expected.in.CTC':'PecentHikeExpectedInCTC','Percent.hike.offered.in.CTC':'PercentHikeOfferedInCTC','Percent.difference.CTC':'PercentDifferenceCTC'})
data.tail()
#Shape of data
data.shape
data.isnull().sum()
data.info()
left = data.groupby('Status')
left.mean()
data.describe()
sns.countplot(x=data.DOJExtended ,data = data)
plt.show()
sns.countplot(x=data.OfferedBand ,data = data)
plt.show()
sns.countplot(x=data.JoiningBonus ,data = data)
plt.show()
sns.countplot(x=data.CandidatRelocateActual ,data = data)
plt.show()
sns.countplot(x=data.Gender ,data = data)
plt.show()
sns.countplot(x=data.CandidateSource ,data = data)
plt.show()
plt.figure(figsize=(20,10))
sns.countplot(x=data.LOB ,data = data)
plt.show()
print("Location wise employee")
plt.figure(figsize=(20,10))
sns.countplot(x=data.Location ,data = data)
plt.show()
print("Employes joined or not")
sns.countplot(x=data.Status,data = data)
plt.show()
print("Gender wise Joined or Not joined Chart")
sns.countplot(x = "Status",data=data,hue="Gender")
plt.show()
obj_col = []
num_col = []
for col in data.columns:
    if data[col].dtype=='O':
        obj_col.append(col)
    else:
        num_col.append(col)
print("This is all the numerical data columns ")
num_col
# Vilon plot
def TreatOutliners(col,data):
    ax = sns.violinplot(x=data[col])
    plt.show()
for i in num_col:
    TreatOutliners(i,data)
def TreatOutliner(col,data):
    data.boxplot(column=[col])
    plt.show()
    
for i in num_col:
    TreatOutliner(i,data)
#taking care of Durationintoacceptoffer outliers .
per99 = np.percentile(data.Durationtoacceptoffer,[99])[0]
data = data[(data.Durationtoacceptoffer < per99)]
data.boxplot(column=['Durationtoacceptoffer'])
plt.show()
print(data.shape)
#taking care of NoticePeriod outliers 
UpperValue = np.percentile(data.NoticePeriod,[99])[0]
data = data[(data.NoticePeriod<UpperValue)]
data.boxplot(column=['NoticePeriod'])
plt.show()
print(data.shape)
#taking care of PecentHikeExpectedInCTC outliers 
lower_value = np.percentile(data.PecentHikeExpectedInCTC,[1])[0]
data = data[(data.PecentHikeExpectedInCTC>lower_value)]
upper_value = np.percentile(data.PecentHikeExpectedInCTC,[99])[0]
data = data[(data.PecentHikeExpectedInCTC < upper_value)]
data.boxplot(column=['PecentHikeExpectedInCTC'])
plt.show()
# taking care of .PercentHikeOfferedInCTC outliers 
upperValue = np.percentile(data.PercentHikeOfferedInCTC,[99])[0]
lowerValue = np.percentile(data.PercentHikeOfferedInCTC,[1])[0]
data = data[(data.PercentHikeOfferedInCTC <upperValue)]
data = data[(data.PercentHikeOfferedInCTC >lowerValue)]
data.boxplot(column=['PercentHikeOfferedInCTC'])
plt.show()
print(data.shape)
# taking care of .PercentDifferenceCTC outliers 
upperValue = np.percentile(data.PercentDifferenceCTC,[95])[0]
lowerValue = np.percentile(data.PercentDifferenceCTC,[3])[0]
data = data[(data.PercentDifferenceCTC <upperValue)]
data = data[(data.PercentDifferenceCTC >lowerValue)]
data.boxplot(column=['PercentDifferenceCTC'])
plt.show()
data.shape
# taking care of RexInYrs outliers 
upperValue = np.percentile(data.RexInYrs,[99])[0]
data = data[(data.RexInYrs < upperValue)]
data.boxplot(column=['RexInYrs'])
plt.show()
data.describe()
data = data.drop(['SLNO'],axis=1)
data.head()
data1 = data.drop(['CandidateRef'],axis=1)
data1.head()
data1.shape
obj_col
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
data1['DOJExtended'] = labelEncoder_X.fit_transform(data1['DOJExtended'])
data1['OfferedBand'] = labelEncoder_X.fit_transform(data1['OfferedBand'])
data1['CandidatRelocateActual'] = labelEncoder_X.fit_transform(data1['CandidatRelocateActual'])
data1['Gender'] = labelEncoder_X.fit_transform(data1['Gender'])
data1['CandidateSource'] = labelEncoder_X.fit_transform(data1['CandidateSource'])
data1['LOB'] = labelEncoder_X.fit_transform(data1['LOB'])
data1['Location'] = labelEncoder_X.fit_transform(data1['Location'])
data1['JoiningBonus'] = labelEncoder_X.fit_transform(data1['JoiningBonus'])
#Attriton is dependent var
from sklearn.preprocessing import LabelEncoder
label_encoder_y=LabelEncoder()
data1['Status']=label_encoder_y.fit_transform(data['Status'])
data1.head()
corr_cols = data1[['DOJExtended', 'Durationtoacceptoffer', 'NoticePeriod', 'OfferedBand',
       'PecentHikeExpectedInCTC', 'PercentHikeOfferedInCTC',
       'PercentDifferenceCTC', 'JoiningBonus', 'CandidatRelocateActual',
       'Gender', 'CandidateSource', 'RexInYrs', 'LOB', 'Location', 'Age',
       'Status' ]]
corr = corr_cols.corr()
plt.figure(figsize=(18,10))
sns.heatmap(corr, annot = True)
plt.show()
X = data1.iloc[:,:-1]
y = data1.iloc[:,-1]
X.head()
from sklearn.preprocessing import scale
X = scale(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20,random_state = 30)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
predict = model.predict(X_test)
data1.columns
data = ['']
print(predict)
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_test,predict)
