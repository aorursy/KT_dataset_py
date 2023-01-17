# YUSUF BIN MOHD SUHAIR
# DATA MINING
# NIM : 3820176110357
### Import the Required Modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/property-listings-in-kuala-lumpur/data_kaggle.csv')
data.head()         #Inspecting the first 5 rows
data.shape
data.describe()       # Summary Statistics
data.boxplot()
data.hist()
data.info()
### Data Cleaning
#### Count the number of missing values in the Dataframe
data.isnull()
# Count the number of missing values in each column
data.isnull().sum()
### Check how many ratings are more than 5 - Outliers
data[data.Bathrooms > 2]
data.drop([10472],inplace=True)
data[10470:10475]
data.boxplot()
data.hist()
threshold = len(data)* 0.1
threshold
data.dropna(thresh=threshold, axis=1, inplace=True)
print(data.isnull().sum())
### Data Imputation and Manipulation
#Define a function impute_median
def impute_median(series):
    return series.fillna(series.median())
data.Rating = data['Bathrooms'].transform(impute_median) 
#count the number of null values in each column
data.isnull().sum()
# modes of categorical values
print(data['Location'].mode())
print(data['Bathrooms'].mode())
print(data['Rooms'].mode())
# Fill the missing categorical values with mode
data['Location'].fillna(str(data['Location'].mode().values[0]), inplace=True)
data['Bathrooms'].fillna(str(data['Bathrooms'].mode().values[0]), inplace=True)
data['Rooms'].fillna(str(data['Rooms'].mode().values[0]), inplace=True)
#count the number of null values in each column
data.isnull().sum()
data.head(10) 
data.describe()
### Data Visualization
grp = data.groupby('Property Type')
x = grp['Rooms'].agg(np.sum)
y = grp['Car Parks'].agg(np.sum)
z = grp['Rooms'].agg(np.sum)
print(x)
print(y)
print(z)
plt.figure(figsize=(12,5))
plt.plot(x, "ro", color='g')
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(16,5))
plt.plot(x,'ro', color='r')
plt.xticks(rotation=90)
plt.title('Category wise Rating')
plt.xlabel('Categories-->')
plt.ylabel('Rating-->')
plt.show()
plt.figure(figsize=(16,5))
plt.plot(y,'r--', color='b')
plt.xticks(rotation=90)
plt.title('Category wise Pricing')
plt.xlabel('Categories-->')
plt.ylabel('Prices-->')
plt.show()
plt.figure(figsize=(16,5))
plt.plot(z,'bs', color='g')
plt.xticks(rotation=90)
plt.title('Category wise Reviews')
plt.xlabel('Categories-->')
plt.ylabel('Reviews-->')
plt.show()
