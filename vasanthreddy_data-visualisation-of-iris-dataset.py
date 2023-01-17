# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

import seaborn as sns





# Any results you write to the current directory are saved as output.

iris=pd.read_csv('../input/Iris.csv')

#now use this to see the preview values

iris.head()

#as seenn in the above id is not useful in the plotting of data,so drop that coloumn 

iris=iris.drop('Id',axis=1)
#see how many no of species are there and what kind of species are there to dealth with

iris['Species'].value_counts()
#Here we are plotting the data without using seaborn and just using the basic things we know

#as we go from values 0 to 150 ,for the species-iris:setosa--it plots the respective values with red color

#and same for the remaining two

for n in range(0,150):

 if iris['Species'][n] == 'Iris-setosa':

    

    plt.scatter(iris['SepalLengthCm'][n],iris['SepalWidthCm'][n],color='red')

    plt.xlabel('SepalLengthCm')

    plt.ylabel('SepalWidthCm')

    

 elif iris['Species'][n] == 'Iris-versicolor':

    

    plt.scatter(iris['SepalLengthCm'][n],iris['SepalWidthCm'][n],color='blue')

    

 else:

    

    plt.scatter(iris['SepalLengthCm'][n],iris['SepalWidthCm'][n],color='green')



    

#Here we enter into seaborn

#joint plot shows bivariate scatterplots and univariate histograms and we can use kde instead of histogram to represent

#kde by using kind='kde' in below

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris)

#as we see in the above species aren't differentiated,by using facegrid we can differentiate 

i=sns.FacetGrid(iris, hue="Species", size=5)

i=(i.map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend())



#here we can plot the data with the regression line/boundary line

#and the species are differentiated using hue

sns.lmplot(x='SepalLengthCm',y='SepalWidthCm',data=iris)

sns.lmplot(x='SepalLengthCm',y='SepalWidthCm',data=iris,hue='Species')

sns.residplot(x='SepalLengthCm',y='SepalWidthCm',data=iris)
#this is same as above but plotting three species in differnt axis using col 

sns.lmplot(x='SepalLengthCm',y='SepalWidthCm',data=iris,hue='Species',col='Species')
#plotting data in strip 

plt.subplot(2,2,1)

sns.stripplot(x="Species", y="SepalLengthCm", data=iris, jitter=True)

plt.subplot(2,2,2)

sns.stripplot(x="Species", y="SepalWidthCm", data=iris, jitter=True)

plt.subplot(2,2,3)

sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True)

plt.subplot(2,2,4)

sns.stripplot(x="Species", y="PetalWidthCm", data=iris, jitter=True)
#here we plot them using boxes 

plt.style.use('ggplot')

plt.subplot(2,2,1)

sns.boxplot(x="Species", y="SepalLengthCm", data=iris)

plt.subplot(2,2,2)

sns.boxplot(x="Species", y="SepalWidthCm", data=iris)

plt.subplot(2,2,3)

sns.boxplot(x="Species", y="PetalLengthCm", data=iris)

plt.subplot(2,2,4)

sns.boxplot(x="Species", y="PetalWidthCm", data=iris)
#here it is another form to represent data in violin form 

plt.style.use('ggplot')

plt.subplot(2,2,1)

sns.violinplot(x="Species", y="SepalLengthCm", data=iris)

plt.subplot(2,2,2)

sns.violinplot(x="Species", y="SepalWidthCm", data=iris)

plt.subplot(2,2,3)

sns.violinplot(x="Species", y="PetalLengthCm", data=iris)

plt.subplot(2,2,4)

sns.violinplot(x="Species", y="PetalWidthCm", data=iris)
#with these pairplot we can relate each pair of feature with other features

#here we can kde form instead of histogram just by changing the kind to kde

sns.pairplot(iris,hue="Species")