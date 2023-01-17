# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

from scipy import stats, linalg

import seaborn as sns
#### now we import the data using PD and parse the dates (seen this in examples)
mydata = pd.read_csv("../input/kc_house_data.csv", parse_dates = ['date'])

#make a table of the data!

#categorize/clear the data

#zip codes are strings
mydata['zipcode'] = mydata['zipcode'].astype(str)
#the other non ratio data are "categories" thanks pandas :D
mydata['waterfront'] = mydata['waterfront'].astype('category',ordered=True)
mydata['condition'] = mydata['condition'].astype('category',ordered=True)
mydata['view'] = mydata['view'].astype('category',ordered=True)
mydata['grade'] = mydata['grade'].astype('category',ordered=False)

#drop ID
mydata = mydata.drop('id',axis=1)

#display a table of all the data for refernce (handy)
df = pd.DataFrame(data = mydata)
df.head(3)
#time to figure out basic stats
###this is simplified after reading the dataframe documentation, there's a way to calculate it all at once :D
mydata.describe()

#unfortunatly there seems to be a few outliers (33 bedrooms?)
#check normal
interestingCol =['price','bedrooms','bathrooms','sqft_above','sqft_living']
interestingData = mydata[interestingCol]
for col in interestingData.columns:
    sns.boxplot(x=col,data = interestingData, orient = 'h',showmeans=True)
    plt.show()
    

#would be handier if I understood what this meant....
#time to graph some data
with sns.plotting_context("notebook",font_scale=2):
    plotter = sns.pairplot(mydata[['price','bedrooms','bathrooms','sqft_above','sqft_living']],hue='bathrooms',size=5)
plotter.set(xticklabels=[]);
#correlation time!
interestingData.corr()
#now it's PCA time!
#first standardize
intDataStand = (interestingData -interestingData.mean()) / interestingData.std()
X = intDataStand.as_matrix()
Y = X - np.ones((X.shape[0],1))*X.mean(0)
U,S,V = linalg.svd(Y,full_matrices=False)
V=V.T
#calculate the variance from principle components
rho = (S*S)/(S*S).sum()

cumsumrho = np.cumsum(rho)

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'o-',label='Variation')
plt.plot(range(1,len(cumsumrho)+1),cumsumrho, label='cumulative Variation')
plt.title('Variation explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variation explained')
plt.legend()
plt.show()

#visualize the PCA

#make lables (zip code)
y = mydata['zipcode']

# grab 100 random points
randomData =intDataStand.sample(n=50)
X1 = randomData.as_matrix()
Y1 = X1 - np.ones((X1.shape[0],1))*X1.mean(0)
U1,S1,V1 = linalg.svd(Y1,full_matrices=False)

Z = U1*S1
#pd.plotting.parallel_coordinates(randomData,'bedrooms')
#project data onto first 3 components of the PCA
reducedData = intDataStand.dot(V)
reducedData = reducedData.as_matrix()

princCompX1 = [[0],[0]]
princCompY1 = [[0],[10]]

princCompX2 = [[0],[10]]
princCompY2 = [[0],[0]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(reducedData[:,0],reducedData[:,1], marker='.')
ax.plot(princCompX1,princCompY1, c='r')
ax.plot(princCompX2,princCompY2, c='r')

plt.title('Data plotted along two PCA components')
plt.xlabel('PCA1')
plt.ylabel('PCA2')

plt.show()

