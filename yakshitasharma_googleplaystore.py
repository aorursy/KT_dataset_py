import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

google_data = pd.read_csv('../input/googleplaystore-acsv/googleplaystore_a.csv')

google_data.head()
google_data.shape
google_data.describe()
google_data.tail(10)
google_data.boxplot()
google_data.hist()
google_data.info()
google_data.isnull()
google_data.isnull().sum()
google_data[google_data.Rating>2]
google_data[100:105]
google_data.boxplot()
grp = google_data.groupby('Category')

x = grp['Rating'].agg(np.mean)

y = grp['Reviews'].agg(np.mean)

print(x)

print(y)
plt.plot(x)
plt.plot(x, 'ro')
plt.figure(figsize=(16,5))

plt.plot(x,'ro',color='r')

plt.xticks(rotation=90)

plt.title('Category wise Rating')

plt.xlabel("Categories-->")

plt.ylabel('Rating-->')

plt.show()