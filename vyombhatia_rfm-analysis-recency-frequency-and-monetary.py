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
sns.distplot(rfm['recency'])
plt.xlabel('Days since last purchase')
# Plotting the number of times the customer has made a purchase:

plt.figure(figsize=(8,8))
sns.set_context("poster")
sns.distplot(rfm['frequency'])

# Plotting the total revenue that the particular customer brought in to the shop:

plt.figure(figsize=(8,8))
sns.set_context("poster")
sns.distplot(rfm['monetary'])
plt.xlabel('Dollars')
rfm.quantile([.25, .5, .75, 1], axis=0)
# copying the rfm dataset so that it isn't affected by the changes:

RFMscores = rfm.copy()
# Converting Recency:

RFMscores['recency'] = [4 if x <= 22 else x for x in RFMscores['recency']]
RFMscores['recency'] = [3 if 22 < x <= 53 else x for x in RFMscores['recency']]
RFMscores['recency'] = [2 if 53 < x <= 111 else x for x in RFMscores['recency']]
RFMscores['recency'] = [1 if x > 111 else x for x in RFMscores['recency']]
# Converting Frequency:

RFMscores['frequency'] = [1 if a < 14 else a for a in RFMscores['frequency']]
RFMscores['frequency'] = [2 if 18 > a >= 14 else a for a in RFMscores['frequency']]
RFMscores['frequency'] = [3 if 22 > a >= 18 else a for a in RFMscores['frequency']]
RFMscores['frequency'] = [4 if a >= 22 else a for a in RFMscores['frequency']]
# Converting Monetary:

RFMscores['monetary'] = [1 if x < 781 else x for x in RFMscores['monetary']]
RFMscores['monetary'] = [2 if 781 <= x < 1227 else x for x in RFMscores['monetary']]
RFMscores['monetary'] = [3 if 1227 <= x < 1520 else x for x in RFMscores['monetary']]
RFMscores['monetary'] = [4 if 1520 <= x else x for x in RFMscores['monetary']]
RFMscores.apply(pd.Series.nunique)
RFMscores.head()
score = pd.DataFrame((RFMscores['recency'] + RFMscores['frequency'] + RFMscores['monetary'])/3, columns=['AggrScore'])

# Concatenating the two:
RFMscores = pd.concat([RFMscores, score], axis = 1)
RFMscores.head()
# Using Quantiles we find the limit for our top 25% customers:
RFMscores.quantile([.75, 1], axis=0)
topcustomers = RFMscores['AggrScore'].iloc[[x >= 3.333333 for x in RFMscores['AggrScore']]]
MVCs = pd.DataFrame(topcustomers, columns = ['AggrScore'])
MostValuableCustomers = list(MVCs.index)
MostValuableCustomers