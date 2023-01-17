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
df=pd.read_csv('/kaggle/input/diabetes/diabetic_data.csv')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.utils import resample

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score,precision_recall_curve,roc_curve

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import time

import matplotlib.patches as mpatches

from sklearn.metrics import confusion_matrix

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score
df.info()
df.head()
#we need to check whether a patient admitted within 30 days or not

df['target']=(df['readmitted']=='<30').astype('int')



#dropping readmitted column

df.drop(['readmitted'],axis=1,inplace=True)
x=df['target'].value_counts().values

sns.barplot([0,1],x)

plt.title('Target variable count')
Count_Target_0 = len(df[df["target"]==0])

Count_Target_1 = len(df[df["target"]==1])



Percentage_of_Target_0 = Count_Target_0/(Count_Target_0+Count_Target_1)

print("percentage of Target 0 is",Percentage_of_Target_0*100)



Percentage_of_Target_1= Count_Target_1/(Count_Target_0+Count_Target_1)

print("percentage of Target 1 is",Percentage_of_Target_1*100)
not_readmitted=df[df.target==0]

readmitted=df[df.target==1]
#upsample minority

readmitted_upsampled = resample(readmitted,

                          replace=True, # sample with replacement

                          n_samples=len(not_readmitted), # match number in majority class

                          random_state=27) # reproducible results

# combine majority and upsampled minority

upsampled = pd.concat([not_readmitted, readmitted_upsampled])



# check new class counts

upsampled.target.value_counts()
y=upsampled.target.value_counts()

sns.barplot(y=y,x=[0,1])

plt.title('upsampled data class count')

plt.ylabel('count')
not_readmitted_downsampled = resample(not_readmitted,

                                replace = False, # sample without replacement

                                n_samples = len(readmitted), # match minority n

                                random_state = 27) # reproducible results



# combine minority and downsampled majority

downsampled = pd.concat([not_readmitted_downsampled, readmitted])



# checking counts

downsampled.target.value_counts()
y=downsampled.target.value_counts()

sns.barplot(y=y,x=[0,1])

plt.title('downsampled data class count')

plt.ylabel('count')