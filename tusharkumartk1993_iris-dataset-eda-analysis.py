# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import the libraries

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import pandas_profiling as pp

%matplotlib inline
#read the data

iris = pd.read_csv('/kaggle/input/iris/Iris.csv')
#getting the top 5 rows Of the dataset

iris.head()
#info() looks into the datatypes and non null value counts

iris.info()
#describe() looks into the statistical measures of the numerical variables present in the dataset

iris.describe()
#for Categorical variables 

iris['Species'].describe()
#Full profile report of the dataset

pp.ProfileReport(iris)
#total no rows

iris.index
#list of columns

iris.columns
#drop unwanted columns

iris.drop(['Id'],axis=1)
#check for any null values

iris.isnull().any()
#check the total number of null values in each column

iris.isnull().sum()
#no of unique values in a specific column

iris['Species'].nunique()
#array of unique values in a specific Column

iris['Species'].unique()
#count of each species(non graphical for categorical variable)

species=iris['Species'].value_counts()

species
#Graphical for categorical variable(pie chart)

plt.pie(species,labels=species.index,autopct='%1.1f%%')

plt.title('Percentage Distribution of each Species')
#Graphical for categorical variable(count or bar chart)

sns.countplot(x='Species',data=iris)

plt.title('Count of each Species')
#kernel density or histogram(Numerical variable)

sns.kdeplot(iris['SepalLengthCm'])

plt.title('Distribution of Sepal Length in cm for every Species')
#Use of FacetGrid for mapping histogram(numerical variable) with Categorical variable

sns.FacetGrid(iris,hue='Species',size=5).map(sns.kdeplot,'SepalLengthCm').add_legend()
#boxplot helps in determining Median,25th percentile,75th percentile and outliers

sns.boxplot(x='Species',y='SepalLengthCm',data=iris)

plt.title('Boxplot distribution for each species')
#use of boxplot with points distribution

sns.boxplot(x='Species',y='SepalLengthCm',data=iris)

sns.stripplot(x='Species',y='SepalLengthCm',data=iris,jitter=True)

plt.title('Boxplot with points distribution of each species')
#better to use violin plot if you dont want to use the above plots

#tells us about the density of points

#fatter where more points

#thinner where less points

sns.violinplot(x='Species',y='SepalLengthCm',data=iris)

plt.title('violin density plot for each Species')
#scatterplot to compare two numerical variables

iris.plot(kind='scatter',x='SepalWidthCm',y='SepalLengthCm') 

plt.title('Distribution of SepalWidth vs SepalLength for all Species')
#use of jointplot to see scatter plot as well as histogram for each numerical variable

sns.jointplot(x='SepalWidthCm',y='SepalLengthCm',data=iris)
#use of FacetGrid for comparing 2 Numerical variables with 1 Categorical variable

sns.FacetGrid(data=iris,hue='Species',size=6).map(plt.scatter,'SepalWidthCm','SepalLengthCm').add_legend()
#use of pairplot to see scatter plot between each numerical variable 

sns.pairplot(iris.drop(['Id'],axis=1),hue='Species')
#encoding of categorical variable

species_Cat=pd.get_dummies(iris['Species'],columns=['Species'])
#join encoded dataframe to main dataframe

iris_df=iris.join(species_Cat)
iris_df
iris_df.drop(['Id','Species'],axis=1)
# finding correlation matrix

corr=iris_df.corr()
#correlation matrix

corr
#heatmap to find correlation between variables using correlation matrix

sns.heatmap(corr,annot=True)

plt.title("heatmap(correlation between variables)")