import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("../input/fce_data.csv")

df.head()

df = df.drop(columns=['Semester', 'Hrs Per Week 5', 'Hrs Per Week 8'])

df = df[(df["Possible Respondents"] >= 25) & (df["Level"] == "Undergraduate") & (df["Year"] >= 2014) & (df["Response Rate %"] > 33)]

statml = df[(df["Dept"] == "STA") | (df["Dept"] == ("MLG"))]

statml = statml[['Year', 'Name','Hrs Per Week', 'Course ID','Interest in student learning', 'Clearly explain course requirements', 'Clear learning objectives & goals',

                'Instructor provides feedback to students to improve', 'Demonstrate importance of subject matter', 'Explains subject matter of course',

                "Overall teaching rate", 'Overall course rate']]

statml = statml.fillna(statml.mean())

statml.describe()
statml_avg_by_prof = statml.groupby(["Name"], as_index=False).mean()
statml_avg_by_prof.nlargest(10, "Overall teaching rate")
# Find the most time-consuming STATML course! 

statml_by_course = statml.groupby(["Course ID"], as_index=False).mean()

statml_by_course.nlargest(10, "Hrs Per Week")
# Find the least time-consuming STATML course! 

statml_by_course.nsmallest(10, "Hrs Per Week")
ax1 = statml.plot.scatter(x='Explains subject matter of course', y='Overall course rate')

ax2 = statml.plot.scatter(x="Clear learning objectives & goals", y='Overall course rate')

statml_subset = subset[['Explains subject matter of course', 'Overall course rate']]

statml_subset.corr(method="pearson")

statml_corr = subset.corr(method="pearson").style.background_gradient(cmap='coolwarm')

statml_corr