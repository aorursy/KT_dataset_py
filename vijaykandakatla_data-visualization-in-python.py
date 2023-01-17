# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/heart-disease-uci/heart.csv")

print("number of rows in data set are:",df.shape[0])

print("number of columns in data set are:",df.shape[1])
df.head()

df.info()
df.describe()
df.corr().style.format("{:.3}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
fig,ax=plt.subplots(figsize=(16,12))

plt.subplot(221)

b1=sns.boxplot(x="sex",y="age",hue="target",data=df,palette="BuGn")

b1.set_title("0: female, 1:male | 0:NO Heart Disease,1:Heart Disease")





plt.subplot(222)

b2=sns.boxplot(x="sex",y="chol",hue="target",data=df,palette="YlGn")

b2.set_title("0: female, 1:male | 0:NO Heart Disease,1:Heart Disease")





plt.subplot(223)

b3=sns.boxplot(x="cp",y="age",hue="sex",data=df,palette="PRGn")

b3.set_title("0: female, 1:male | chest pain type 0,1,2,3")





plt.subplot(224)

b4=sns.boxplot(x="exang",y="thalach",hue="target",data=df,palette="BuGn")

b4.set_title("0: No Angina, 1:Angina | 0:NO Heart Disease,1:Heart Disease")



    
fig,ax=plt.subplots(figsize=(19,10))



plt.subplot(241)

s1=sns.countplot(x='target',data=df,hue='sex',palette='BuPu',linewidth=3)

s1.set_title('0: femle, 1:male | 0:NO Hrt Dis,1:Hrt Dis')



plt.subplot(242)

s2=sns.countplot(x='cp',data=df,hue='target',palette='BuPu',linewidth=3)

s2.set_title('Chest pain type')



plt.subplot(243)

s3=sns.countplot(x='thal',data=df,hue='target',palette='BuPu',linewidth=3)

s3.set_title('Thal')





plt.subplot(244)

s4=sns.countplot(x='slope',data=df,hue='target',palette='BuPu',linewidth=3)

s4.set_title('slope of the peak exercise')



plt.subplot(245)

s5=sns.countplot(x='sex',data=df,hue='target',palette='BuPu',linewidth=3)

s5.set_title('Sex 0:Female,1:Male')



plt.subplot(246)

s6=sns.countplot(x='fbs',data=df,hue='target',palette='BuPu',linewidth=3)

s6.set_title('Fasting blood sugar>120mg/dl')



plt.subplot(247)

s7=sns.countplot(x='ca',data=df,hue='target',palette='BuPu',linewidth=3)

s7.set_title('Number of major vessels coloured')



plt.subplot(248)

s8=sns.countplot(x='restecg',data=df,hue='target',palette='BuPu',linewidth=3)

s8.set_title('Resting electrocardiographic results ')
fig,ax=plt.subplots(figsize=(33,30))

plt.subplot(611)

bx_1 = sns.barplot(x="trestbps", y="restecg", hue="target", data=df,palette="PuBu")

bx_1.set_title("Target 0:no dis,1:dis")



plt.subplot(612)

bx_2 = sns.barplot(x="trestbps", y="restecg", hue="sex", data=df,palette="PuBu")

bx_2.set_title("sex 0:female,1:male")



plt.subplot(613)

bx_3 = sns.barplot(x="age", y="chol", hue="sex", data=df,palette="PuBu")

bx_3.set_title("sex 0:female,1:male")



plt.subplot(614)

bx_4 = sns.barplot(x="oldpeak", y="thal", hue="target", data=df,palette="PuBu")

bx_4.set_title("Target 0:no dis,1:dis")



plt.subplot(615)

bx_5 = sns.barplot(x="oldpeak", y="restecg", hue="target", data=df,palette="PuBu")

bx_5.set_title("Target 0:no dis,1:dis")



plt.subplot(616)

bx_6 = sns.barplot(x="oldpeak", y="age", hue="exang", data=df,palette="PuBu")

bx_6.set_title("Exang 0:No Angina,1:Yes Angina")

fig,ax=plt.subplots(figsize=(23,16))

plt.subplot(231)

sp_1 = sns.scatterplot(x="age", y="thalach",hue="target",data=df,palette="cool")

sp_1.set_title("Target 0:no dis,1:dis")

plt.subplot(232)

sp_2 = sns.scatterplot(x="chol", y="thalach",hue="target",data=df,palette="cool")

sp_2.set_title("Target 0:no dis,1:dis")

plt.subplot(233)

sp_3 = sns.scatterplot(x="chol", y="thalach",hue="cp",data=df,palette="rainbow")

sp_3.set_title("Target 0:no dis,1:dis")

plt.subplot(234)

sp_3 = sns.scatterplot(x="oldpeak", y="thalach",hue="target",data=df,palette="spring")

sp_3.set_title("Target 0:no dis,1:dis")

plt.subplot(235)

sp_3 = sns.scatterplot(x="ca", y="trestbps",hue="sex",data=df,palette="cool")

sp_3.set_title("Target 0:no dis,1:dis")

plt.xticks([0,1, 2, 3, 4])

plt.subplot(236)

sp_3 = sns.scatterplot(x="ca", y="trestbps",hue="target",data=df,palette="cool")

sp_3.set_title("Target 0:no dis,1:dis")

plt.xticks([0,1, 2, 3, 4])



sns.set(style="darkgrid")

f,ax = plt.subplots(figsize = (25,15))

sns.pointplot(x = "thalach",y = 'restecg',data = df,color = "green",alpha = .6)

plt.xticks(rotation=90)



sns.lmplot(x="trestbps", y="thalach", col="fbs", hue="target",

           data=df)

sns.lmplot(x="trestbps", y="thalach", col="sex", hue="target",

           data=df)

sns.lmplot(x="chol", y="age", col="exang", hue="target",

           data=df)

sns.lmplot(x="thalach", y="oldpeak", col="exang", hue="target",

           data=df)

sns.lmplot(x="thalach", y="oldpeak", col="exang", hue="sex",

           data=df)
sns.pairplot(df,hue="target",vars=["age","chol","thalach","trestbps"],palette="husl")
sns.pairplot(df,hue="cp",vars=["age","chol","thalach","trestbps"])
sns.pairplot(df,hue="sex",vars=["age","chol","thalach","trestbps"])
sns.pairplot(df,hue="exang",vars=["age","oldpeak","thalach","slope"])
sns.pairplot(df,hue="target",vars=["age","thalach","restecg","trestbps"],palette="husl")
sns.set(style="white")

sns.jointplot(x="chol",y="age",kind="kde",color="g",data=df)





sns.set(style="white")

sns.jointplot(x="thalach",y="cp",kind="hex",color="g",data=df)



sns.set(style="white")

sns.jointplot(x="trestbps",y="ca",kind="hex",color="g",data=df)





sns.set(style="white")

sns.jointplot(x="oldpeak",y="thal",kind="kde",color="g",data=df)





sns.set(style="white")

sns.jointplot(x="ca",y="thal",kind="kde",color="g",data=df)





sns.set(style="white")

sns.jointplot(x="ca",y="cp",kind="kde",color="g",data=df)



sns.catplot(x="cp", y="age",hue="target", col="sex",data=df, kind="swarm",height=5.7, aspect=.5)



sns.catplot(x="ca", y="age",hue="target", col="sex",data=df, kind="swarm",height=5.5, aspect=.5)