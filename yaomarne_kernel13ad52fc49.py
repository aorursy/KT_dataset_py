import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cluster import KMeans



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt # plotting

import pandas

from collections import Counter
df1 = pd.read_csv('../input/Mall_Customers.csv')

df1.head(5)
#show distribution

import seaborn as sns



def plotDistriCorre(df):

    # Basic correlogram

    sns.pairplot(df)

    

    # distribution

    for name in df1: 

        mydf = df1[name]

        if mydf.dtypes == object:

            letter_counts = Counter(mydf)

            df = pandas.DataFrame.from_dict(letter_counts, orient='index')

            df.plot(kind='bar')

        else:

            plt.hist(mydf)

            plt.xlabel(name)

            plt.ylabel('number')

            plt.show()

            #plt.title(r'Histogram of ',name)
plotDistriCorre(df1)
len = 10

wcss = []

data = df1.iloc[:,3:5]

for i in range(1,len):

    km = KMeans(n_clusters = i,init='k-means++', random_state=0)

    km.fit(data)

    wcss.append(km.inertia_)
plt.plot(range(1,len),wcss)

plt.xlabel('k value')

plt.ylabel('wcss')

plt.grid()
km = KMeans(n_clusters = 5,init='k-means++', random_state=0)

df1['label_income_spend'] = km.fit_predict(data)

df1.head()
plt.subplot(121)

plt.scatter(data.iloc[:,0],data.iloc[:,1])

plt.ylabel('Annual income(k$)')

plt.xlabel('Spending score')



plt.subplot(122)

plt.scatter(data.iloc[:,0],data.iloc[:,1],c=df1['label_income_spend'])

plt.ylabel('Annual income(k$)')

plt.xlabel('Spending score')
df1.groupby(['label_income_spend','Gender']).count().reset_index()

#plt.hist(gender['CustomerID'])