import datetime as dt

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

import seaborn as sns



import squarify

import matplotlib.pyplot as plt

%matplotlib inline
# Import datasets

data = pd.read_csv('../input/online-retail-customer-clustering/OnlineRetail.csv', sep=",", encoding="ISO-8859-1", header=0)

data.head()
data.isnull().sum()
data = data.dropna()

data.shape
#checking for duplicates

data.duplicated().sum()
#removing duplicates

data.drop_duplicates(keep='first', inplace=True)

data.shape
#let's do a copy of our df for next manipulations

retail = data.copy()
#calculate revenue per row and add new column

retail['Revenue'] = retail['Quantity'] * retail['UnitPrice']
retail.InvoiceDate = pd.to_datetime(retail['InvoiceDate'], format='%d-%m-%Y %H:%M')
# Let's visualize the top grossing months

retail_month = retail[retail.InvoiceDate.dt.year==2011]

monthly_gross = retail_month.groupby(retail_month.InvoiceDate.dt.month).Revenue.sum()



plt.figure(figsize=(8,4))

sns.set_context("talk")

sns.set_palette("PuBuGn_d")

sns.lineplot(y=monthly_gross.values,x=monthly_gross.index, marker='o')

plt.xticks(range(1,13))

plt.title("Revenue per month in 2011")

plt.show()
#amount of transactions per month

plt.figure(figsize=(8,4))

retail[retail.InvoiceDate.dt.year==2011].InvoiceDate.dt.month.value_counts(sort=False).plot(kind='bar')

plt.title("Amount of transactions per month in 2011")

plt.show()
# Let's visualize some top products from the whole range

top_products = retail['Description'].value_counts()[:20]

plt.figure(figsize=(8,6))

sns.set_context("paper", font_scale=1.5)

sns.barplot(y = top_products.index,

            x = top_products.values, 

           palette='PuBuGn_d')

plt.title("Top selling products")

plt.show()

plt.savefig('top_products.png')
#creating invoice month column to see first month when customer purchased 

retail['InvoiceMonth'] = retail['InvoiceDate'].apply(lambda x: dt.datetime(x.year, x.month, 1))
grouping = retail.groupby('CustomerID')['InvoiceMonth']

#assign smallest invoice value to each customer

retail['CohortMonth'] = grouping.transform('min')

retail.head()
#function to extract year, month, day as integers

def get_date_int(df, column):

    year = df[column].dt.year

    month = df[column].dt.month

    day = df[column].dt.day

    return year, month, day
#extract month

invoice_year, invoice_month, _ = get_date_int(retail, 'InvoiceMonth')

cohort_year, cohort_month, _ = get_date_int(retail, 'CohortMonth')
years_diff = invoice_year - cohort_year

months_diff = invoice_month - cohort_month
# Extract the difference in days from all previous values

retail['CohortIndex'] = years_diff * 12 + months_diff + 1

retail.head()
#count monthly active customers from each cohort

grouping = retail.groupby(['CohortMonth', 'CohortIndex'])

cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)

cohort_data = cohort_data.reset_index()

cohort_counts = cohort_data.pivot(index='CohortMonth', columns = 'CohortIndex', values='CustomerID')
#Customer retention

cohort_sizes = cohort_counts.iloc[:,0]

retention = cohort_counts.divide(cohort_sizes, axis=0)

retention = retention.round(3) * 100

retention.head(20)
month_list = ["Dec '10", "Jan '11", "Feb '11", "Mar '11", "Apr '11",\

              "May '11", "Jun '11", "Jul '11", "Aug '11", "Sep '11", \

              "Oct '11", "Nov '11", "Dec '11"]



plt.figure(figsize=(15,8))

plt.title('Retention by Monthly Cohorts')

sns.heatmap(data=retention,

            annot = True,

            cmap = "Greens",

            vmin = 0.0,

            vmax = list(retention.max().sort_values(ascending = False))[1]+3,

            fmt = '.1f',

            linewidth = 0.3,

            yticklabels=month_list)



plt.show()
#12 months of data

print('Min:{}; Max:{}'.format(min(retail.InvoiceDate), max(retail.InvoiceDate)))
#calculate revenue per row and add new column

retail['MonetaryValue'] = retail['Quantity'] * retail['UnitPrice']
#let's look at amount spend per customer (revenue contributed) M-Monetary

retail_mv = retail.groupby(['CustomerID']).agg({'MonetaryValue': sum}).reset_index()

retail_mv.head()
#F-frequency (how many purchases each customer made)

retail_f = retail.groupby('CustomerID')['InvoiceNo'].count()

retail_f = retail_f.reset_index()

retail_f.head()
#merge previous dataframes together (mv+f)

retail_mv_f = pd.merge(retail_mv, retail_f, on='CustomerID', how='inner')

retail_mv_f.head()
#R-recency 

#last transaction date 



retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],format='%d-%m-%Y %H:%M')

max_date = max(retail['InvoiceDate'])



#difference between last date and transaction date

retail['Diff'] = max_date - retail['InvoiceDate']

retail.head()
#recency per customer (last transaction date)

retail_r = retail.groupby('CustomerID')['Diff'].min()

retail_r = retail_r.reset_index()



# Extract number of days only

retail_r['Diff'] = retail_r['Diff'].dt.days
#merge R dataframe with FM



retail_rfm = pd.merge(retail_mv_f, retail_r, on='CustomerID', how='inner')

retail_rfm.columns = ['CustomerID', 'MonetaryValue', 'Frequency', 'Recency']

retail_rfm.head()
cols = retail_rfm.columns.tolist()

cols
#changed columns order

cols = ['CustomerID', 'Recency', 'Frequency', 'MonetaryValue']

retail_rfm = retail_rfm[cols]

retail_rfm.head()
# create labels and assign them to tree percentile groups 

r_labels = range(4, 0, -1)

r_groups = pd.qcut(retail_rfm.Recency, q = 4, labels = r_labels)

f_labels = range(1, 5)

f_groups = pd.qcut(retail_rfm.Frequency, q = 4, labels = f_labels)

m_labels = range(1, 5)

m_groups = pd.qcut(retail_rfm.MonetaryValue, q = 4, labels = m_labels)
# make a new column for group labels

retail_rfm['R'] = r_groups.values

retail_rfm['F'] = f_groups.values

retail_rfm['M'] = m_groups.values

# sum up the three columns

retail_rfm['RFM_Segment'] = retail_rfm.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']), axis = 1)

retail_rfm['RFM_Score'] = retail_rfm[['R', 'F', 'M']].sum(axis = 1)

retail_rfm.head()
# assign labels from total score

score_labels = ['Green', 'Bronze', 'Silver', 'Gold']

score_groups = pd.qcut(retail_rfm.RFM_Score, q = 4, labels = score_labels)

retail_rfm['RFM_Level'] = score_groups.values

retail_rfm.sort_values(by='RFM_Score', ascending=False)

retail_rfm.head(10)
retail_rfm_levels = retail_rfm.groupby('RFM_Level')['CustomerID'].count().reset_index(name='counts')

retail_rfm_levels.head()
#let's exclude others segment for visualization

levels = list(retail_rfm_levels.RFM_Level)

score = list(retail_rfm_levels.counts)

plt.figure(figsize=(12,8))

plt.title('Customer Levels distribution')

squarify.plot(sizes=score, label=levels)



plt.show()
#let's try to do more detailed segmentation

segment_dict = {    

    'Best Customers':'444',      # Highest frequency as well as monetary value with least recency

    'Loyal Customers':'344',     # High frequency as well as monetary value with good recency

    'Potential Loyalists':'434', # High recency and monetary value, average frequency

    'Big Spenders':'334',        # High monetary value but good recency and frequency values

    'At Risk Customers':'244',   # Customer's shopping less often now who used to shop a lot

    'Can’t Lose Them':'144',      # Customer's shopped long ago who used to shop a lot.

    'Recent Customers':'443',    # Customer's who recently started shopping a lot but with less monetary value

    'Lost Cheap Customers':'122' # Customer's shopped long ago but with less frequency and monetary value

}
# Swap the key and value of dictionary

dict_segment = dict(zip(segment_dict.values(),segment_dict.keys()))



# Allocate segments to each customer as per the RFM score mapping

retail_rfm['Segment'] = retail_rfm.RFM_Segment.map(lambda x: dict_segment.get(x))
# Allocate all remaining customers to others segment category

retail_rfm.Segment.fillna('others', inplace=True)
retail_rfm.sample(10)
retail_rfm_segments = retail_rfm[retail_rfm.Segment!='other'].groupby('Segment')['CustomerID'].count().reset_index(name='counts')

retail_rfm_segments.iloc[:8]
#let's exclude others segment for visualization

segment = list(retail_rfm_segments.iloc[:8].Segment)

score = list(retail_rfm_segments.iloc[:8].counts)

color_list = ["#248af1", "#eb5d50", "#8bc4f6", "#8c5c94", "#a170e8", "#fba521", "#75bc3f"]

plt.figure(figsize=(12,8))

plt.title('Customer Segments distribution')

squarify.plot(sizes=score, label=segment,color=color_list, alpha=0.7)



plt.show()