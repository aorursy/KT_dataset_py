import pandas as pd
# Reading the data:
data = pd.read_csv("../input/retailtransactiondata/Retail_Data_Transactions.csv")
data.head()
# Defining Recency as two of the three columns in data:
recency = data[['trans_date', 'customer_id']]
recency.apply(pd.Series.nunique)
recency.shape
import time

recency['trans_date'] = pd.to_datetime(recency.trans_date)
# now refers to the latest date available in the data, to which we will peg our rececny dimensions on:
now = max(recency['trans_date'])
recency = recency.groupby(['customer_id']).max()
recency_days = now - recency['trans_date']
recency_days = pd.DataFrame(recency_days)
recency_days.head()
monetary = data[['customer_id', 'tran_amount']]
monetary = monetary.groupby(['customer_id']).sum()
monetary.head()
frequency = data[['customer_id', 'trans_date']]
frequency = frequency.groupby(['customer_id']).count()
frequency.head()
recency = pd.DataFrame(recency_days['trans_date'].astype('timedelta64[D]'))
recency.columns = ['recency']
recency.head()

rfm = pd.concat([recency, frequency, monetary], axis=1)
# Defining the columns:
rfm.columns=['recency', 'frequency', 'monetary']

rfm.head()
# Plotting for the last day since the customer made a purchase:

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))
sns.set_context("poster")
sns.set_palette(['skyblue'])
sns.distplot(rfm['recency'])
plt.xlabel('Days since last purchase')
# Plotting the number of times the customer has made a purchase:

plt.figure(figsize=(8,8))
sns.set_context("poster")
sns.set_palette(['pink'])
sns.distplot(rfm['frequency'])

# Plotting the total revenue that the particular customer brought in to the shop:

plt.figure(figsize=(8,8))
sns.set_context("poster")
sns.set_palette(['orange'])
sns.distplot(rfm['monetary'])
plt.xlabel('Dollars')
# Making a copy so that I don't lose my temper over messing up a previously perfect dataframe.
RFM = rfm.copy()

# Importing KMeans and finding clusters:
from sklearn.cluster import KMeans

km = KMeans(n_clusters = 4, init = 'k-means++')

nclusters = km.fit_predict(RFM)

clusters = pd.DataFrame(nclusters, columns = ['clusters'], index = RFM.index)

# Concatenating the clusters with the RFM dataframe:
rfmK= pd.concat([RFM, clusters], axis=1)
plt.figure(figsize=(8,8))
sns.scatterplot(data = rfmK, x='frequency', y='monetary', hue='clusters')
RFMKMEANS = rfmK.copy()

RFMKMEANS['clusters'] = ['SuperFans' if x == 3 else x for x in RFMKMEANS['clusters']]
RFMKMEANS['clusters'] = ['UsualCustomers' if x == 2 else x for x in RFMKMEANS['clusters']]
RFMKMEANS['clusters'] = ['FrequentCustomers' if x == 0 else x for x in RFMKMEANS['clusters']]
RFMKMEANS['clusters'] = ['Thrifters' if x == 1 else x for x in RFMKMEANS['clusters']]
plt.figure(figsize=(15,7))
sns.set_context("poster", font_scale=0.7)
sns.set_palette('twilight')
sns.countplot(RFMKMEANS['clusters'])
superfans_df = rfmK.loc[rfmK['clusters'] == 2]
superfans_df.head()
# Dropping the clusters column because there's just a single value in the whole column:
superfans_df.drop(['clusters'], axis= 1, inplace=True)
# Lets take a look at the quantile:
superfans_df.quantile([.25, .5, .75, 1], axis=0)
# Copying the RFM dataset so that it isn't affected by the changes:
RFMscores = superfans_df.copy()
# Converting Recency:

RFMscores['recency'] = [4 if x <= 14 else x for x in RFMscores['recency']]
RFMscores['recency'] = [3 if 14 < x <= 37 else x for x in RFMscores['recency']]
RFMscores['recency'] = [2 if 37 < x <= 70 else x for x in RFMscores['recency']]
RFMscores['recency'] = [1 if x > 70 else x for x in RFMscores['recency']]
# Converting Frequency:

RFMscores['frequency'] = [1 if a <= 24 else a for a in RFMscores['frequency']]
RFMscores['frequency'] = [2 if a == 25 else a for a in RFMscores['frequency']]
RFMscores['frequency'] = [3 if 27 > a > 25 else a for a in RFMscores['frequency']]
RFMscores['frequency'] = [4 if a >= 27 else a for a in RFMscores['frequency']]
# Converting Monetary:

RFMscores['monetary'] = [1 if x < 1721 else x for x in RFMscores['monetary']]
RFMscores['monetary'] = [2 if 1721 <= x < 1808 else x for x in RFMscores['monetary']]
RFMscores['monetary'] = [3 if 1808 <= x < 1954 else x for x in RFMscores['monetary']]
RFMscores['monetary'] = [4 if 1954 <= x else x for x in RFMscores['monetary']]
RFMscores.apply(pd.Series.nunique)
RFMscores.head()
score = pd.DataFrame((RFMscores['recency'] + RFMscores['frequency'] + RFMscores['monetary'])/3, columns=['AggrScore'])

# Concatenating the two:
RFMscores = pd.concat([RFMscores, score], axis = 1)
RFMscores.head()
# Using Quantiles we find the limit for our top 25% customers:
RFMscores.quantile([.75], axis=0)
topcustomers = RFMscores['AggrScore'].iloc[[x >= 3 for x in RFMscores['AggrScore']]]
MVCs = pd.DataFrame(topcustomers, columns = ['AggrScore'])
MostValuableCustomers = list(MVCs.index)
MostValuableCustomers
rfmK.loc[MVCs.index]