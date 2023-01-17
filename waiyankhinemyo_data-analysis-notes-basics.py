import numpy as np # linear algebra - arrays & matrices
import pandas as pd # for data structures & tools, data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp #for integrals, solving differential equations, optimization
import matplotlib.pyplot as plt #for plots, graphs - data visualization
%matplotlib inline 
import seaborn as sns #plots - heat maps, time series & violin plots
import sklearn as sklearn #machine learning models
import statsmodels as stmodels #explore data, estimate statistical models, & perform statistical test

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load train dataset
data = '../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv'
dataset = pd.read_csv(data) #for those datasets without headers can use dataset = pd.read_csv(data, header = None)
dataset_withoutheaders = pd.read_csv(data, header = None)
dataset.shape
dataset.columns #finding name of columns of the dataset
dataset.head()
dataset_withoutheaders.head()
#checking datatypes of features

dataset.dtypes
dataset.describe() #returns a statistical summary - but this method does not include object datatype columns, basically skip rows & columns that do not contain numbers
#for full summary with every column & row
dataset.describe(include = "all")
dataset[['sl_no', 'salary', 'gender']].describe(include = "all") #describing only sl_no & salary columns in the dataset, notice I can arrange the way columns appear sl_no --> salary --> gender

#include = "all" is added to display gender column since it is an object type, else not needed
dataset.info()
#DB-API
from dbmodule import connect

#Create a connection object
connection = connect('databasename', 'username', 'password')

#Create a cursor object
cursor = connection.cursor()

#Run queries
cursor.execute('select * from tablename')
results = cursor.fetchall()

#Free resources
Cursor.close()
connection.close()
#output missing data
missing_data = dataset.isnull()
missing_data.head(10)
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    
dataset.head(10)
dataset.dropna(subset=["salary"], axis = 0, inplace = True) #axis = 0 will drop the entire row with NaN value & axis = 1 will drop the entire column with NaN value, inplace = True will write the result back to the original dataset

#dataset.dropna(subset=["salary"], axis = 0, inplace = True) is the same as dataset = dataset.dropna(subset=["salary"], axis = 0)
#if want to drop all rows with NaN value regardless of column/variable just use dataset.dropna(axis = 0, inplace = True)
dataset.head(10)
#reset the dataset to original
dataset = pd.read_csv(data)
dataset.head(10)
#dataframe.replace(missing_value, new_value)
#replacing with mean value
mean = dataset["salary"].mean()
dataset["salary"].replace(np.nan, mean, inplace = True)
dataset.head(10)
dataset[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']].head(10)
#Simple Feature Scaling
dataset["salary"] = dataset["salary"]/data["salary"].max()

#Min-Max Scaling
dataset["salary"] = (dataset["salary"] - dataset["salary"].min())/(dataset["salary"].max()-dataset["salary"].min())

#Z-Score Sacling
dataset["salary"] = (dataset["salary"]-dataset["salary"].mean())/dataset["salary"].std()
bins = np.linspace(min(dataset["salary"]), max(dataset["salary"]),4) #to have 3 bins, need 4 equally spaced numbers hence 4 in code
group_names = ["Low", "Medium", "High"]
dataset["salary-binned"] = pd.cut(dataset["salary"], bins, labels = group_names, include_lowest = True)
dataset.head()
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(dataset["salary"])

# set x/y labels and plot title
plt.pyplot.xlabel("Salary")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Salary Bins")
pyplot.bar(group_names, dataset["salary-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("Salary")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Salary Bins")
dataset.gender.unique()
#To see which values are present in a particular column, we can use the ".value_counts()" method:
dataset['gender'].value_counts()
#We can see that males are the most common type. We can also use the ".idxmax()" method to calculate for us the most common type automatically:
dataset['gender'].value_counts().idxmax()
dummy_variable_1 =  pd.get_dummies(dataset["gender"])
dummy_variable_1.head()
dataset.head(10)
# merge data frame "dataset" and "dummy_variable_1" 
dataset = pd.concat([dataset, dummy_variable_1], axis=1)

# drop original column "gender" from "dataset"
dataset.drop("gender", axis = 1, inplace=True)
dataset.head(10)
#reset the dataset to original
dataset = pd.read_csv(data)
dataset.head(10)
#basic descriptive statistics function
dataset.describe(include = "all")
#summarizing categorical data
print(dataset["gender"].value_counts())
print()
print(dataset["ssc_b"].value_counts())
print()
print(dataset["hsc_b"].value_counts())
print()
print(dataset["hsc_s"].value_counts())
print()
print(dataset["degree_t"].value_counts())
print()
print(dataset["workex"].value_counts())
print()
print(dataset["specialisation"].value_counts())
print()
print(dataset["status"].value_counts())
#Box Plots
sns.boxplot(x="gender", y="salary", data=dataset)
#Scatter Plots
x = dataset["etest_p"]
y = dataset["salary"]
plt.scatter(x,y)

plt.title("Employment Test Percentage vs Salary offered by corporate to candidates")
plt.xlabel("Employment Test Percentage")
plt.ylabel("Salary Offered")
#Groupby Function
dataset_test = dataset[['gender', 'degree_t', 'salary']]
dataset_grp = dataset_test.groupby(['gender', 'degree_t'], as_index = False).mean() #.mean() is used to see how average salary differ across different groups
dataset_grp
#Pivot Table
dataset_pivot = dataset_grp.pivot(index = 'gender', columns = 'degree_t')
dataset_pivot
#Heatmap Plot
fig, ax = plt.subplots()
im = ax.pcolor(dataset_pivot, cmap='RdBu')

#label names
row_labels = dataset_pivot.columns.levels[1]
col_labels = dataset_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(dataset_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(dataset_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()
#Correlation
sns.regplot (x="etest_p", y="salary", data=dataset)
plt.ylim(0,)
from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataset['etest_p'], dataset['salary'])
#replacing with mean value
etest_mean = dataset["etest_p"].mean()
dataset["etest_p"].replace(np.nan, etest_mean, inplace = True)
salary_mean = dataset["salary"].mean()
dataset["salary"].replace(np.nan, salary_mean, inplace = True)
from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataset['etest_p'], dataset['salary'])
print("Correlation Coefficient is " + str(pearson_coef))
print("p_value is " + str(p_value))
#finding correlation value between multiple features
dataset[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']].corr()
#Correlation Heatmap
corr_mat = dataset.corr()

plt.figure(figsize = (13,5))
sns_plot = sns.heatmap(data = corr_mat, annot = True, cmap='GnBu')
plt.show()
dataset.specialisation.unique()
dataset_anova = dataset[["specialisation", "salary"]]
grouped_anova = dataset_anova.groupby(["specialisation"])

f_val, p_val = stats.f_oneway(grouped_anova.get_group("Mkt&HR")["salary"],grouped_anova.get_group("Mkt&Fin")["salary"])
print( "ANOVA results: F=", f_val, ", P =", p_val)   