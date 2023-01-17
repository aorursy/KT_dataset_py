import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style("darkgrid")
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
data.describe()
sns.pairplot(data)
sns.heatmap(data.corr(),annot=True,cmap="Blues")
passing = data.copy()
passing_score = 75
passing["math score"] = passing["math score"].apply(lambda x: x >= passing_score)
passing["reading score"] = passing["reading score"].apply(lambda x: x >= passing_score)
passing["writing score"] = passing["writing score"].apply(lambda x: x >= passing_score)
passing.head()
passing[["math score","reading score","writing score"]].sum()/passing[["math score","reading score","writing score"]].count()
sns.heatmap(passing.corr(),annot=True,cmap="Blues")
factor = "race/ethnicity"

plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
subject = "math score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,2)
subject = "reading score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,3)
subject = "writing score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.tight_layout()
factor = "parental level of education"

plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
subject = "math score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.xticks(rotation = 45)

plt.subplot(1,3,2)
subject = "reading score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.xticks(rotation = 45)

plt.subplot(1,3,3)
subject = "writing score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.xticks(rotation = 45)
plt.tight_layout()
factor = "lunch"

plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
subject = "math score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,2)
subject = "reading score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,3)
subject = "writing score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.tight_layout()
factor = "test preparation course"

plt.figure(figsize=(16,4))
plt.subplot(1,3,1)
subject = "math score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,2)
subject = "reading score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)

plt.subplot(1,3,3)
subject = "writing score"
sns.barplot(data=(passing[passing[subject] == True].groupby(factor)[subject].count()/passing.groupby(factor)[subject].count()).reset_index().sort_values(by=subject,ascending=False),x=factor,y=subject)
plt.tight_layout()
z = 75 # Chosen passing score
data["all pass"] = (data["math score"] >= z) & (data["reading score"] >= z) & (data["writing score"] >= z)
data[data["all pass"] == True].head()
data[data["all pass"] == True]["all pass"].count()/data["all pass"].count()
factor = "race/ethnicity"
plt.figure(figsize=(8,6))
sns.barplot(data = (data[data["all pass"] == True].groupby(factor).count()["math score"]/data.groupby(factor).count()['math score']).reset_index(),x=factor,y="math score")
plt.title("Percent of students passed by "+factor)
plt.ylabel("% pass all three")
plt.tight_layout()
factor = "race/ethnicity"

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
sns.boxplot(data=data,x=factor,y="math score",hue="all pass",order=["group A","group B","group C","group D","group E"])

plt.subplot(1,3,2)
sns.boxplot(data=data,x=factor,y="reading score",hue="all pass",order=["group A","group B","group C","group D","group E"])

plt.subplot(1,3,3)
sns.boxplot(data=data,x=factor,y="writing score",hue="all pass",order=["group A","group B","group C","group D","group E"])
plt.tight_layout()
factor = "parental level of education"

plt.figure(figsize=(8,6))
sns.barplot(data = (data[data["all pass"] == True].groupby(factor).count()["math score"]/data.groupby(factor).count()['math score']).reset_index(),x=factor,y="math score",
            order=["some high school","high school","associate's degree","some college","bachelor's degree","master's degree"])
plt.xticks(rotation = 45)
plt.title("Percent of students passed by "+factor)
plt.ylabel("% pass all three")
plt.tight_layout()

factor = "parental level of education"

plt.figure(figsize=(15,6))
plt.subplot(1,3,1)
sns.boxplot(data=data,x=factor,y="math score",hue="all pass",
           order=["some high school","high school","associate's degree","some college","bachelor's degree","master's degree"])
plt.xticks(rotation = 45)

plt.subplot(1,3,2)
sns.boxplot(data=data,x=factor,y="reading score",hue="all pass",
           order=["some high school","high school","associate's degree","some college","bachelor's degree","master's degree"])
plt.xticks(rotation = 45)

plt.subplot(1,3,3)
sns.boxplot(data=data,x=factor,y="writing score",hue="all pass",
           order=["some high school","high school","associate's degree","some college","bachelor's degree","master's degree"])
plt.xticks(rotation = 45)
plt.tight_layout()
factor = "lunch"

plt.figure(figsize=(8,6))
sns.barplot(data = (data[data["all pass"] == True].groupby(factor).count()["math score"]/data.groupby(factor).count()['math score']).reset_index(),x=factor,y="math score")
plt.title("Percent of students passed by "+factor)
plt.ylabel("% pass all three")
plt.tight_layout()
factor = "lunch"

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.boxplot(data=data,x=factor,y="math score",hue="all pass")

plt.subplot(1,3,2)
sns.boxplot(data=data,x=factor,y="reading score",hue="all pass")

plt.subplot(1,3,3)
sns.boxplot(data=data,x=factor,y="writing score",hue="all pass")
plt.tight_layout()
factor = "test preparation course"

plt.figure(figsize=(8,6))
sns.barplot(data = (data[data["all pass"] == True].groupby(factor).count()["math score"]/data.groupby(factor).count()['math score']).reset_index(),x=factor,y="math score")
plt.title("Percent of students passed by "+factor)
plt.ylabel("% pass all three")
plt.tight_layout()
factor = "test preparation course"

plt.figure(figsize=(16,5))
plt.subplot(1,3,1)
sns.boxplot(data=data,x=factor,y="math score",hue="all pass")

plt.subplot(1,3,2)
sns.boxplot(data=data,x=factor,y="reading score",hue="all pass")

plt.subplot(1,3,3)
sns.boxplot(data=data,x=factor,y="writing score",hue="all pass")
plt.tight_layout()
