%matplotlib inline

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



plt.style.use('ggplot') 

plt.rcParams['figure.figsize'] = (10, 5) # some styles for our graph
df = pd.read_csv("../input/rubella.csv")
df.head()
a = df[['state', 'cases']]

a.head()
b = a['state'].value_counts()



plt.xlabel(u'State')

plt.ylabel(u'Number of cases')

plt.title(u'Rubella in USA')

b.plot(kind='bar')