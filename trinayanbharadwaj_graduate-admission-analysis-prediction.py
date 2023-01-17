# importing the necessary libraries/packages.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Reading the data.

location = "../input/graduate-admissions/Admission_Predict.csv"
data = pd.read_csv(location)
data
# Checking if our data has null values or NAN's.

data.isnull().any()
# Checking the data type of all the columns.

data.dtypes
# Replacing the whitespaces with underscore to prevent errors while processing data.

data.columns = data.columns.str.replace(" ", "_")
data.columns
# Plotting the graph between GRE_Score and Chance_of_Admit_.

plt.figure(figsize=[10,10])
sns.regplot(x=data["GRE_Score"], y=data["Chance_of_Admit_"])
plt.title("Graph showing chances of admission VS GRE scores")
# Plotting the graph between TOEFL_Score and Chance_of_Admit_.

plt.figure(figsize=[10,10])
sns.regplot(x=data["TOEFL_Score"], y=data["Chance_of_Admit_"])
plt.title("Graph showing chances of admission VS TOEFL Score")
# Checking the University_Rating column. 

data["University_Rating"].describe()
# Plotting the graph between University_Rating and Chance_of_Admit_.

data_uni_rank = data.groupby("University_Rating", as_index=False).Chance_of_Admit_.mean()

plt.figure(figsize=(15,10))
sns.barplot(x=data_uni_rank["University_Rating"], y=data_uni_rank["Chance_of_Admit_"])
plt.title("Graph showing chance of admission VS University rating", size=15)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel("University_Rating", size=20)
plt.ylabel("Chance_of_Admit_", size=20)
# Plotting the graph between SOP and Chance_of_Admit_.

data_SOP = data.groupby("SOP",as_index=False).Chance_of_Admit_.mean()

plt.figure(figsize=(13,8))
sns.lineplot(data = data_SOP["SOP"])
plt.title("Graph showing chance of admission VS SOP rating", size=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel("Chance_of_Admit_", size=15)
plt.ylabel("SOP rating", size=15)
# Plotting the graph between LOR and Chance_of_Admit_.

plt.figure(figsize=(13,8))
sns.regplot(x = data["LOR_"], y = data["Chance_of_Admit_"])
plt.title("Graph showing chance of admission VS LOR rating", size=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel("LOR rating", size=15)
plt.ylabel("Chance_of_Admit_", size=15)
# Plotting the graph between CGPA and Chance_of_Admit_.

plt.figure(figsize=(13,8))
sns.regplot(x = data["CGPA"], y = data["Chance_of_Admit_"])
plt.title("Graph showing chance of admission VS CGPA", size=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel("CGPA", size=15)
plt.ylabel("Chance_of_Admit_", size=15)
# Checking the Research column.

data["Research"].describe()
# Plotting the graph between Research and Chance_of_Admit_.

plt.figure(figsize=(13,8))
sns.barplot(x=data["Research"], y=data["Chance_of_Admit_"])
plt.title("Graph showing chance of admission VS research", size=15)
plt.xticks(size=15)
plt.yticks(size=15)
plt.xlabel("research(1 is yes, 0 is none)", size=15)
plt.ylabel("Chance_of_Admit_", size=15)
# Distributing the data into X and y variable. 

y = data["Chance_of_Admit_"]
y = (y*100).astype(int)

X = data.drop("Chance_of_Admit_", axis=1)
X = X.astype(int)

# Splitting the data into training and testing data.

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
# Defining the model.

model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=200, normalize=True)

# Fitting the model with training data.

model.fit(train_X, train_y)

# Checking the accuracy of our model.

model.score(test_X, test_y)*100