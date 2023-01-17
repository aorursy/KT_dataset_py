# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/KaggleV2-May-2016.csv")
df.head()
df.info()
df.rename(columns = {"PatientId":"PatientID", "Hipertension":"Hypertension", "Handcap":"Handicap"}, inplace = True)
df.head()
df["PatientID"] = df["PatientID"].astype("int64")
df.head()
df.info()
df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"])
df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"])
df.head()
#Age
np.sort(df["Age"].unique())
df[df["Age"] == -1].count()["Age"]
df.loc[df["Age"] == -1]
df.drop(index = 99832, inplace = True)
df.info()
np.sort(df["Age"].unique())
print("Scholarship:")
print(df["Scholarship"].unique())
print("Hypertension:")
print(df["Hypertension"].unique())
print("Handicap:")
print(df["Handicap"].unique())
print("Diabetes:")
print(df["Diabetes"].unique())
print("SMS_received:")
print(df["SMS_received"].unique())
print("Alcoholism:")
print(df["Alcoholism"].unique())
#Count the number of each gender
df["Gender"].value_counts()
import matplotlib.pyplot as plt
plt.pie([71829, 38687], labels = ["Female", "Male"], colors = ["blue", "red"])
plt.axis("equal")
plt.title("Number of Female and Male Patient")
gender_counts = df.groupby(["Gender", "No-show"]).count()["Age"]
gender_counts
female_show_up_proportion = gender_counts["F", "No"]/df["Gender"].value_counts()["F"]
female_show_up_proportion
male_show_up_proportion = gender_counts["M","No"]/df["Gender"].value_counts()["M"]
male_show_up_proportion
plt.bar([1, 2],[female_show_up_proportion, male_show_up_proportion], width = 0.7, color = ['lightblue', 'red'])
plt.xlabel("Gender")
plt.ylabel("Proportion")
plt.xticks([1, 2],["Female", "Male"])
plt.title("Proportion of Patient that showed up by gender")
plt.show()
#Age
df["Age"].describe()
bin_edges = [0, 20, 40, 60, 80, 115]
bin_names = ["<20", "20-40", "40-60", "60-80", "80-115"]
df["AgeGroup"] = pd.cut(df["Age"], bin_edges, labels = bin_names, right = False, include_lowest = True)
df.head()
Age_counts = df.groupby(["AgeGroup", "No-show"]).count()["Age"]
Age_counts
below_twenty = Age_counts["<20", "No"]/df["AgeGroup"].value_counts()["<20"]
twenty_to_thirtynine = Age_counts["20-40", "No"]/df["AgeGroup"].value_counts()["20-40"]
forty_to_fiftynine = Age_counts["40-60", "No"]/df["AgeGroup"].value_counts()["40-60"]
sixty_to_seventynine = Age_counts["60-80", "No"]/df["AgeGroup"].value_counts()["60-80"]
eighty_and_above = Age_counts["80-115", "No"]/df["AgeGroup"].value_counts()["80-115"]

proportions = [below_twenty, twenty_to_thirtynine, forty_to_fiftynine, sixty_to_seventynine, eighty_and_above]
plt.bar([1,2,3,4,5], proportions, width = 0.2)
plt.xlabel("AgeGroup")
plt.ylabel("Proportion")
plt.xticks([1,2,3,4,5],["<20","20-40","40-60","60-80","80-115"])
plt.title("Proportion of patients that show up accordding to age group")
plt.show()
scholarship_counts = df.groupby(["Scholarship", "No-show"]).count()["Age"]
scholarship_counts
not_enrolled_showed_up = scholarship_counts[0, "No"]/df["Scholarship"].value_counts()[0]
not_enrolled_showed_up
enrolled_showed_up = scholarship_counts[1, "No"]/df["Scholarship"].value_counts()[1]
enrolled_showed_up
plt.bar([1, 2], [not_enrolled_showed_up, enrolled_showed_up], width = 0.6)
plt.xlabel("Enrolled or not")
plt.ylabel("Proporgation")
plt.xticks([1, 2],["Not_enrolled", "Enrolled"])

hypertension_counts = df.groupby(["Hypertension", "No-show"]).count()["Age"]
hypertension_counts
no_hypertension_showed_up = hypertension_counts[0, "No"]/df["Hypertension"].value_counts()[0]
no_hypertension_showed_up
hypertension_showed_up = hypertension_counts[1, "No"]/df["Hypertension"].value_counts()[1]
hypertension_showed_up
plt.bar([1, 2], [no_hypertension_showed_up, hypertension_showed_up])
plt.xlabel("Hypertension or not")
plt.ylabel("Proporgation")
plt.xticks([1, 2], ["No Hypertension", "Hypertension"])
plt.show()
diabetes_counts = df.groupby(["Diabetes", "No-show"]).count()["Age"]
diabetes_counts
no_diabetes_showed_up = diabetes_counts[0, "No"]/df["Diabetes"].value_counts()[0]
diabetes_showed_up = diabetes_counts[1, "No"]/df["Diabetes"].value_counts()[1]
alcoholism_counts = df.groupby(["Alcoholism", "No-show"]).count()["Age"]
alcoholism_counts
no_alcoholism_showed_up = alcoholism_counts[0, "No"]/df["Alcoholism"].value_counts()[0]
alcoholism_showed_up = alcoholism_counts[1, "No"]/df["Alcoholism"].value_counts()[1]
ind = np.array([1, 2, 3])
width = 0.3
plt.bar(ind, [no_hypertension_showed_up, no_diabetes_showed_up, no_alcoholism_showed_up], width = width, label = "Without the condition")
plt.bar(ind+width, [hypertension_showed_up, diabetes_showed_up, alcoholism_showed_up], width = width, label = "With the condition")
plt.xlabel("Conditions")
plt.ylabel("Proporgation that show up")
locations = ind+width/2
plt.xticks(locations, ["Hypertension", "Diabetes", "Alcoholism"])
plt.legend(bbox_to_anchor=(1,1));
df.head()
df.info()
df.iloc[0,-1]
df.Gender.apply(lambda x: 1 if(x == "M") else 0)
df.Gender = df.Gender.apply(lambda x: 1 if(x == "M") else 0)
df.head()
Age = ["<20", "20-40", "40-60", "60-80", "80-115"]
df.AgeGroup = df.AgeGroup.apply(lambda x: Age.index(x))
df.rename(columns = {"No-show": "NoShow"}, inplace = True)

df.head()
df.NoShow = df.NoShow.apply(lambda x:0 if(x == "No") else 1)
df.head()
df.info()
df.AgeGroup.unique()
df.dropna(axis = 0, inplace = True)
df.info()
sorted(df.NoShow.unique())
data = df[["Gender", "Age", "Scholarship", "Hypertension", "Diabetes", "Alcoholism", "Handicap", "AgeGroup"]].values
target = df.NoShow.values
target
from sklearn.model_selection import train_test_split
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size = 0.1)
print(len(train_data), len(test_data))
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(train_data, train_target)
print("Accuracy: ", round(accuracy_score(test_target, model.predict(test_data)), 4)*100, "%")
