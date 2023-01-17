import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sb
df=pd.read_csv("../input/Suicides in India 2001-2012.csv")

df.sample(10)
df.info()
df.Year.unique()
df.Type_code.unique()
def plotbar(df,column,vertical):

    count=[]

    for x in df[column].unique():

        c=df[df[column]==x].Total.sum()

        count.append(c)

    

    plt.figure(figsize=(10,10))

    plt.bar(df[column].unique(),count)

    if(vertical):

        plt.xticks(range(df[column].nunique()),rotation='vertical')

    plt.show()
df_suicides=df[df.Type_code=='Causes'] # data may overlap with other type codes, so selected just one type code

df_suicides.head(2)
plotbar(df_suicides,"Year",False)
plotbar(df_suicides,"State",True)
plotbar(df_suicides,"Gender",False)
plotbar(df_suicides,"Age_group",False)
def makebar(df):

    count=[]

    for means in df.Type.unique():

        c=df[df.Type==means].Total.sum()

        count.append(c)

    

    plt.figure(figsize=(10,10))

    plt.bar(df.Type.unique(),count)

    plt.xticks(range(df.Type.nunique()),rotation='vertical')

    plt.show()
df_means=df[df.Type_code=='Means_adopted']

makebar(df_means)
df_causes=df[df.Type_code=='Causes']

makebar(df_causes)
df_education=df[df.Type_code=='Education_Status']

makebar(df_education)
df_profession=df[df.Type_code=='Professional_Profile']

makebar(df_profession)
df_social=df[df.Type_code=='Social_Status']

makebar(df_social)