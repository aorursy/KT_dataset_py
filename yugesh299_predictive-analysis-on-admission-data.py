# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.cluster import KMeans









# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1 = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")

df  = df1.dropna()

df.head(10)
data = df.drop(columns = ['Serial No.'],axis = 1)
data.describe()
data.corr()
#Normalzied Dataset

Ndata = data

Ndata['GRE_Score']   = (data['GRE Score']/340)*100

Ndata['TOEFL_Score'] = (data['TOEFL Score']/120)*100

Ndata.drop(columns = ['GRE Score','TOEFL Score'],axis = 1).head()
plt.scatter(Ndata['GRE_Score'],Ndata['TOEFL_Score'])

plt.grid()

plt.figure()
plt.hist(Ndata['GRE_Score'])

plt.grid()

plt.figure()
plt.hist(Ndata['TOEFL_Score'])

plt.grid()

plt.figure()