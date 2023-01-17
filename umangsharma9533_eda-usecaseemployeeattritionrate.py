# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/traindata/Train.csv')
#pd.options.display.max_rows=None

#pd.options.display.max_columns=None

df.head()
df_filled_NaN=df.fillna(df.median())
df_mean=pd.Series(df_filled_NaN['Compensation_and_Benefits']).value_counts()

#df_dummies=pd.get_dummies(df_filled_NaN['Compensation_and_Benefits'])
df_mean
print(df_mean/len(df_filled_NaN))
df_filled_NaN['Compensation_and_Benefits']=df_filled_NaN['Compensation_and_Benefits'].map({'type2':5,'type3':4,'type4':3,'type0':2,'type1':1})
df_mean_hht=pd.Series(df_filled_NaN['Hometown']).value_counts()
df_mean_hht
print(df_mean_hht/len(df_filled_NaN))
df_filled_NaN['Hometown']=df_filled_NaN['Hometown'].map({'Lebanon':5,'Springfield':4,'Franklin':3,'Washington':2,'Clinton':1})
df_filled_NaN['Relationship_Status']=df_filled_NaN['Relationship_Status'].map({'Single':0,'Married':1})
df_filled_NaN
df_filled_NaN['Gender']=df_filled_NaN['Gender'].map({'F':1,'M':0})
df_filled_NaN
df_dummies_unit=pd.get_dummies(df_filled_NaN['Unit'])

df_dummies_decision=pd.get_dummies(df_filled_NaN['Decision_skill_possess'])

df_filled_NaN_drop_drop=df_filled_NaN.join([df_dummies_unit,df_dummies_decision])

df_filled_NaN_drop_drop_final=df_filled_NaN_drop_drop.drop(['Unit','Decision_skill_possess'],axis=1)
df_filled_NaN_drop_drop_final
df_filled_NaN_drop_drop_final.columns
df_filled_NaN_drop_drop_final
columns_names=['Employee_ID', 'Gender', 'Age', 'Education_Level',

       'Relationship_Status', 'Hometown', 'Time_of_service',

       'Time_since_promotion', 'growth_rate', 'Travel_Rate', 'Post_Level',

       'Pay_Scale', 'Compensation_and_Benefits', 'Work_Life_balance', 'VAR1',

       'VAR2', 'VAR3', 'VAR4', 'VAR5', 'VAR6', 'VAR7',

       'Accounting and Finance', 'Human Resource Management', 'IT',

       'Logistics', 'Marketing', 'Operarions', 'Production', 'Purchasing',

       'Quality', 'R&D', 'Sales', 'Security', 'Analytical', 'Behavioral',

       'Conceptual', 'Directive','Attrition_rate']
df_filled_NaN_drop_drop_finall= df_filled_NaN_drop_drop_final[columns_names]
df_filled_NaN_drop_drop_finall
X=df_filled_NaN_drop_drop_finall.iloc[:,1:-1]
Y=df_filled_NaN_drop_drop_finall.iloc[:,-1]
import seaborn as sns

#get correlations of each features in dataset

corrmat = df_filled_NaN_drop_drop_finall.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(50,50))

#plot heat map

g=sns.heatmap(df_filled_NaN_drop_drop_finall[top_corr_features].corr(),annot=True,cmap="RdYlGn")