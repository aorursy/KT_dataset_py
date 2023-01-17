import numpy as np

import seaborn as sns

import pandas as pd
df = pd.read_csv("../input/cicekler/iris.csv")
df.head()
df.shape
df.info
df.describe().T
df.isna().sum() 
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df["variety"].unique()
df["variety"].nunique()
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df);
sns.jointplot(x = "sepal.width", y = "sepal.length", data = df, color="purple");
sns.scatterplot(x = "sepal.width", y = "sepal.length", data = df,hue = "variety");
df["variety"].value_counts()
sns.violinplot(y = "sepal.width", data = df);
sns.distplot(df["sepal.width"], bins=16, color="purple");
sns.violinplot(x = "variety",y = "sepal.length", data = df);
sns.countplot(x = "variety", data = df);
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df, color="purple");
sns.jointplot(x = "sepal.length", y = "sepal.width", data = df, kind = "kde", color="purple");
sns.scatterplot(x = "petal.length", y = "petal.width", data = df);
sns.scatterplot(x = "petal.length", y = "petal.width",hue = "variety", data = df);
sns.lmplot(x = "petal.length", y = "petal.width", data = df);
df.corr()["petal.length"]["petal.width"]
df['total.length'] = df['petal.length'] * df['sepal.length']
df["total.length"].mean()
df["total.length"].std()
df["total.length"].max()
df[(df['sepal.length'] > 5.5) & (df['variety'] == "Setosa")]
df[(df['petal.length'] < 5) & (df['variety'] == "Virginica")][["sepal.length","sepal.width"]]
df.groupby(["variety"]).mean()
df.groupby(["variety"])["petal.length"].std()