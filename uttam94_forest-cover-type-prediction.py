# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! ls ../input/forest-cover-type-prediction
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')

test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')
print(train.shape,test.shape)
train.isnull().sum()
train.dtypes

# all datatypes are int64
train.columns

#cover type is predictin values 
train.describe()

# No attribute is missing as count is 15120 for all attributes. Hence, all rows can be used

# Negative value(s) present in Vertical_Distance_To_Hydrology. Hence, some tests such as chi-sq cant be used.

# Wilderness_Area and Soil_Type are one hot encoded. Hence, they could be converted back for some analysis

# Attributes Soil_Type7 and Soil_Type15 can be removed as they are constant

# Scales are not the same for all. Hence, rescaling and standardization may be necessary for some algos
train.skew()
train.groupby('Cover_Type').size()
# Correlation tells relation between two attributes.

# Correlation requires continous data. Hence, ignore Wilderness_Area and Soil_Type as they are binary

plt.figure(figsize=(10,10))

size =10

data = train.iloc[:,:size]

corr = data.corr()

sns.heatmap(corr,annot=True)
data.columns
def plot(used_col):

    for i in range(len(used_col)):

        plt.figure(figsize=(7,7))

        sns.distplot(data[used_col[i]])

used_col = ['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Hillshade_9am','Hillshade_Noon','Hillshade_3pm']
plot(used_col)
data.columns
sns.boxplot(x="Cover_Type", y="Aspect",data=train )
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    train.drop('Cover_Type', axis=1), train['Cover_Type'],

    test_size=0.3, random_state=101)
rf = RandomForestClassifier(n_estimators=300,class_weight='balanced',n_jobs=-1,random_state=42)

rf.fit(X_train,y_train)
pred = rf.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,pred)

cm
ct = rf.predict(test)
Id=test['Id']

output = pd.DataFrame(Id)

output['Cover_Type']=ct

output.head()
output.to_csv("output.csv",index=False)