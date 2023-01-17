#codes from Rodrigo Lima  @rodrigolima82

from IPython.display import Image

Image(url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRRVgBbOPCLaF8fMiyIdM-MuJWtVFW3gl-MoM_-0_5zsKnQRTfU&usqp=CAU',width=400,height=400)
import pandas as pd

import numpy as np

import seaborn as sns                       #visualisation

import matplotlib.pyplot as plt             #visualisation

%matplotlib inline     

sns.set(color_codes=True)
df = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

# To display the top 5 rows 

df.head(5)
df.tail(5)  
df.dtypes
df = df.drop(['Pregnancies', 'BloodPressure', 'Age', 'Outcome'], axis=1)

df.head(5)
df.shape
duplicate_rows_df = df[df.duplicated()]

print("number of duplicate rows: ", duplicate_rows_df.shape)
df.count()    
df = df.drop_duplicates()

df.head(5)
df.count()
sns.boxplot(x=df['BMI'])
sns.boxplot(x=df['Glucose'])
Q1 = df.quantile(0.25)

Q3 = df.quantile(0.75)

IQR = Q3 - Q1

print(IQR)
df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]

df.shape
plt.figure(figsize=(10,5))

c= df.corr()

sns.heatmap(c,cmap="BrBG",annot=True)

c