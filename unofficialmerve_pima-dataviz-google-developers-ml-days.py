import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.describe()
df.info()
df.isna().any()
df.notna().any()
df.isna().all()
df.notna().all()
df.notna().sum()
df["Overweight"] = [1 if x > 25 else 0 for x in df.BMI]
df.head()
plt.rcParams.update({'font.size': 25})
sns.set_context("paper")
plt.scatter(df.Age, df.Insulin, c=df.Overweight, s=389,
            alpha=0.2, cmap="viridis") #cmap renk paleti
plt.colorbar(); #hangi rengin hangi değere denk geldiğini gösteren yandaki ölçek
plt.xlabel("Age") #eksen ismi
plt.ylabel("Insulin") 
plt.title("Relationship between Age and Insulin") #plot ismi
plt.show()
fig, ax = plt.subplots(1,2)
plt.show()
fig, ax = plt.subplots()
ax.scatter(df.Age, df.Insulin, c=df.Overweight, cmap="viridis")
ax.set_xlabel("Age")
ax.set_ylabel("Insulin")
ax.set_title("Relationship between Age and Insulin")
plt.show()
fig, ax = plt.subplots()
ax.hist(df.Age, label="Age", bins=10)
ax.set_xlabel("Age")
ax.set_ylabel("Number of Observations")
plt.show()
bins=[20, 30, 40, 50, 60, 70, 80]
fig, ax = plt.subplots()
ax.hist(df.Age, label="Age Bins", bins=bins)
ax.set_xlabel("Age")
ax.set_ylabel("Number of Observations")
plt.show()
fig, ax = plt.subplots()
ax.bar(df.Outcome, df.Insulin)
ax.set_xlabel("Outcome")
ax.set_ylabel("Insulin")
plt.show()
fig, ax = plt.subplots()
ax.bar(df.Age, df.Insulin)
ax.set_xticklabels(df.Age, rotation=45)
fig.savefig("Age.png")
sns.set_palette("RdBu")
sns.countplot(x="Age", data=df)
plt.show()
sns.catplot(x="Age", aspect=3, data=df, kind="count")
plt.show()
g = sns.catplot(x="Age", aspect=3, data=df, kind="count")
g.fig.suptitle("Age Counts", y=1.04) #ismi yukarı çıkarıyor
plt.show()
g = sns.catplot(x="Age", aspect=3, data=df, kind="count")
plt.xticks(rotation=30)
plt.show()
sns.scatterplot(x="Age", y="Insulin",data=df, hue="Outcome")
plt.show()
sns.relplot(x="Age", y="Insulin",data=df, hue="Outcome", 
            kind="scatter")
plt.show()
sns.relplot(x="Glucose", y="Insulin", data=df, kind="line", ci="sd", markers=True, dashes=True)
plt.show()
sns.relplot(x="Age", y="Insulin", data=df, kind="line", aspect=4, ci="sd")
plt.show()
sns.relplot(x="Age", y="Insulin", data=df, kind="line", aspect=4, ci=None)
plt.show()
sns.relplot(x="Insulin", y="Glucose", data=df, kind="scatter", row="Outcome")
plt.show()
sns.relplot(x="Insulin", y="Glucose", data=df, kind="scatter", col="Outcome", row="Overweight")
plt.show()
sns.set_palette("RdBu")
correlation=df.corr()
sns.heatmap(correlation, annot=True)
plt.show()
sns.catplot(x="Outcome",y="Insulin",data=df, kind="bar")
plt.show()
sns.catplot(x="Outcome",y="Age",data=df, kind="box")
plt.show()
sns.catplot(x="Outcome",data=df, kind="count")
plt.show()
sns.set_style("dark")
sns.catplot(x="Outcome",data=df, kind="count")
plt.show()
## white, dark, whitegrid, darkgrid, ticks
sns.set_context("notebook")
sns.set_palette("Greys_r")
sns.catplot(x="Outcome",data=df, kind="count")
plt.show()
# categorical plotlarda “RdBu”, “PRGn”,”RdBu_r”,”PRGn_r”
# continuous plotlarda Greys, Blues, PuRd, GnBu
sns.set_context("paper")
plt.show()
# paper,poster, talk