# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # Data frame manipulation

import seaborn as sns  # corelation plt

import matplotlib.pyplot as plt  # graphs
import pandas as pd

iris = pd.read_csv("../input/iris/Iris.csv")
iris.head()
iris.columns 
iris.info()
iris.drop('Id', axis = 1, inplace = True)

# axis =1 is for deleating the column name, and inplace = true is for doing the change in the original data set rather than creating a new one.

iris.columns
iris.describe()
iris.describe(include = 'object')
iris.describe(include = 'all')
iris['Species'].value_counts()  

# here since the data is evenly distributed so we have a balanced data set
sns.countplot('Species', data = iris). set_title('count for each species') 
iris['Species'].value_counts().plot.pie().set_title('species count') 
sns.scatterplot(x= 'SepalLengthCm',

                y= 'SepalWidthCm',

                data= iris).set_title('scatter plot based on species')
sns.scatterplot(x= 'SepalLengthCm',

                y= 'SepalWidthCm',

                hue = 'Species',

                data= iris).set_title('scatter plot based on species')
sns.pairplot(iris, hue= 'Species')
sns.pairplot(iris, hue= 'Species',

            x_vars = ['SepalLengthCm', 'SepalWidthCm'] ,

            y_vars = ['PetalLengthCm', 'PetalWidthCm'])
sns.FacetGrid(iris, col = 'Species').map(plt.hist, 'PetalLengthCm')
sns.catplot(x= 'PetalLengthCm' , kind= 'count', data = iris, col ='Species',col_wrap = 1,aspect =4,height =2.5)

sns.FacetGrid(iris, hue = 'Species').map(plt.hist, 'PetalLengthCm').add_legend()
sns.FacetGrid(iris, hue = 'Species').map(sns.distplot, 'PetalWidthCm')
sns.FacetGrid(iris, hue = 'Species').map(sns.distplot, 'PetalLengthCm').add_legend()
sns.FacetGrid(iris, hue = 'Species').map(sns.distplot, 'SepalLengthCm').add_legend()
sns.FacetGrid(iris, hue = 'Species').map(sns.distplot, 'SepalWidthCm')
plt.figure(figsize = (15,10))

plt.subplot(2,2,1)

sns.boxplot(x = 'Species', y = 'SepalLengthCm', data = iris)



plt.subplot(2,2,2)

sns.boxplot(x = 'Species', y = 'SepalWidthCm', data = iris)



plt.subplot(2,2,3)

sns.boxplot(x = 'Species', y = 'PetalLengthCm', data = iris)



plt.subplot(2,2,4)

sns.boxplot(x = 'Species', y = 'PetalWidthCm', data = iris)
sns.pairplot(iris,

            kind = 'reg',

            )
#   Mean sepal length per species

setosa = iris[iris.Species == 'Iris-setosa']

versicolor = iris[iris.Species == 'Iris-versicolor']

virginica = iris[iris.Species == 'Iris-virginica']

print(np.mean(setosa['SepalWidthCm']))

print(np.mean(versicolor['SepalWidthCm']))

print(np.mean(virginica['SepalWidthCm']))
iris.groupby('Species').mean()