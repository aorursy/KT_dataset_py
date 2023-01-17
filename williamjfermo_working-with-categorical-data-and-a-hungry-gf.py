#First import necessary libraries

import pandas as pd

import numpy as np
#import data into df

df =pd.read_csv("../input/Decisiontree.csv")
#Look at initial df

df.head()
#Get info regarding df

print(df.info())
#Seperating dtypes into own df

print(df.select_dtypes(include=['object']).head())

print(df['Fastfood'].value_counts())  
##Check categories for unique values and counts to find out cardinality 

print(df['Do_you_want_to_eat?'].value_counts())
label = {'y':1, 'n':0}

df['Do_you_want_to_eat?'] = df['Do_you_want_to_eat?'].map(label)

df.head()
#Importing category encoder library

import category_encoders as ce
##Check categories for unique values and counts to find out cardinality 

print(df['What_do_you_want_to_eat?'].value_counts())
encoder = ce.OneHotEncoder(cols=['What_do_you_want_to_eat?'])

df= encoder.fit_transform(df)

df.head()
##Check categories for unique values and counts to find out cardinality 

print(df['Fastfood'].value_counts())
encoder = ce.BinaryEncoder(cols=['Fastfood'])

df = encoder.fit_transform(df)



df.head()
##Check categories for unique values and counts to find out cardinality 

print(df['Restaurant'].value_counts())
encoder = ce.OrdinalEncoder(cols = ['Restaurant'])

# ce_leave.fit(X3, y3['outcome'])        

# ce_leave.transform(X3, y3['outcome']) 

df = encoder.fit_transform(df)



df.head()
##Check categories for unique values and counts to find out cardinality 

print(df['Choice'].value_counts())