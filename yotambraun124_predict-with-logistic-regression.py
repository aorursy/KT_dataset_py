import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report


sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
df = pd.read_csv("/kaggle/input/forbes-celebrity-100-since-2005/forbes_celebrity_100.csv")
df.head()
df.info()
df.index
df= df.set_index("Name")
df.head()
biggest_ten_celeb = df["Pay (USD millions)"].nlargest(10)
biggest_ten_celeb
biggest_ten_celeb.plot(kind="bar",colormap='Paired')
df_grup_by_Category = df.groupby(["Category","Name"])["Pay (USD millions)"].mean()
df_grup_by_Category
df_grup_by_Category.nlargest(10)
df_grup_by_Category.nlargest(10).plot(kind="bar")

df_grup_by_Category_year = df.groupby(["Year","Category","Name"])["Pay (USD millions)"].mean()
df_grup_by_Category_year.nlargest(10)
df_grup_by_Category_year.nlargest(10).plot(kind="bar",color="g")
spearmanr(df["Category"],df["Pay (USD millions)"])
spearmanr(df["Year"],df["Pay (USD millions)"])
df_grup_by_Category_year.tail(10).plot(kind="bar",color="g")
df.Category.unique()
df_with_dummy_var = pd.get_dummies(df, columns=['Category'])
df_with_dummy_var.head()
X = df_with_dummy_var[["Pay (USD millions)","Year"]]
y = df_with_dummy_var['Category_Actors']
y.value_counts()
       
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

predictions  = log_reg.predict(X_test)

print(classification_report(y_test,predictions))

score = log_reg.score(X_test, y_test)
print(score)
