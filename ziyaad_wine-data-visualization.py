import matplotlib.pyplot as plt

import pandas as pd
data = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")



data.head()
data.isnull().sum()
data.describe()
data['province'].value_counts()
info = data['province'].value_counts().head(10)/len(data)



fig,ax = plt.subplots(figsize = (12,6))



ax = plt.bar(info.keys(),info, width=.8)



plt.xlabel("Province")

plt.ylabel("Count")



fig.autofmt_xdate()

plt.show()
(data['province'].value_counts().head(10)/len(data)).plot.bar()
info = data['points'].value_counts()



fig,ax = plt.subplots(figsize = (12,6))



ax = plt.bar(info.keys(),info, alpha=.4, color='g')



plt.show()
data['points'].value_counts().sort_index().plot.line()









data['points'].value_counts().sort_index().plot.area()
data[data['price']<200]['price'].plot.hist()
data['price'].plot.hist()
data[data['price']>1500]
data['points'].plot.hist()
data[data['price']<100].sample(100).plot.scatter(x='price' , y='points')
data[data['price']<100].plot.hexbin(x='price',y='points', gridsize=15)
data = pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv")



data.head()
data.set_index('points',inplace=True)
data
data.plot.bar(stacked=True,figsize = (12,6))
data.plot.area()
data.plot.line()