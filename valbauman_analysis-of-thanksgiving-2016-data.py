import numpy as np 

import pandas as pd 

from google.cloud import bigquery



import matplotlib.pyplot as plt 



client= bigquery.Client()

dataset_ref= client.dataset('google_analytics_sample', project= 'bigquery-public-data')

dataset= client.get_dataset(dataset_ref)



query_summary_data= """

SELECT date, SUM(totals.visits) AS visits, SUM(totals.pageviews) AS pageviews, SUM(totals.transactions) AS transactions, SUM(totals.transactionRevenue)/1000000 AS revenue

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

WHERE _TABLE_SUFFIX BETWEEN '20161124' AND '20161128'

GROUP BY date

ORDER BY date ASC

"""



# run the query if it uses < 1GB

safe_config= bigquery.QueryJobConfig(maximum_bytes_billed= 1000000000)

safe_query_job= client.query(query_summary_data, job_config= safe_config)

summary_df= safe_query_job.to_dataframe()

print(summary_df)
rev_heights= summary_df['revenue']

pos= np.arange(len(rev_heights))

plt.bar(pos, rev_heights, align= 'center')

plt.xticks(pos, summary_df['date'])

plt.xlabel('Date'); plt.ylabel('Revenue (USD $)')

plt.title('Revenue for each day during the 2016 Thanksgiving weekend')



plt.figure()

view_heights= summary_df['pageviews']

plt.bar(pos, view_heights, align= 'center')

plt.xticks(pos, summary_df['date'])

plt.xlabel('Date'); plt.ylabel('Number of page views')

plt.title('Page views for each day during the 2016 Thanksgiving weekend')
# create query that gets running total of revenue for each of the traffic sources using only instances where a purchase was made (revenue > 0)

query_running_tots= """

SELECT date, trafficSource.source AS source, SUM(totals.transactionRevenue/1000000)

OVER(

PARTITION BY trafficSource.source, date 

ORDER BY date, trafficSource.source

ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW

) as running_total_revenue



FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

WHERE _TABLE_SUFFIX BETWEEN '20161124' AND '20161128' AND totals.transactionRevenue > 0



"""



# run the query if it uses < 1GB

safe_config= bigquery.QueryJobConfig(maximum_bytes_billed= 1000000000)

safe_query_job= client.query(query_running_tots, job_config= safe_config)

running_tots_df= safe_query_job.to_dataframe()
# visualize distribution of traffic sources that resulted in a revenue > 0

source_transactions= running_tots_df['source'].value_counts()

plt.figure()

plt.pie(x= source_transactions, labels= source_transactions.index, labeldistance= None)

plt.legend(loc= "center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('Distribution of instances where revenue > 0 by traffic source')
# plot of cumulative sum of revenue for each day

direct_traffic= running_tots_df[running_tots_df['source'] == '(direct)'].sort_values(by= ['date'])

plt.plot(direct_traffic['date'], direct_traffic['running_total_revenue'], 'x')

plt.xlabel('Date'); plt.ylabel('Cumulative sum - revenue ($)')

plt.title("Cumulative revenue for each day, '(direct)' traffic source")
bounce_query= """

WITH bounce_counts AS 

(SELECT COUNT(trafficSource.source) AS visits, SUM(totals.bounces) AS num_bounces, SUM(totals.transactions) as num_transactions, trafficSource.source AS source

FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`

WHERE _TABLE_SUFFIX BETWEEN '20161124' AND '20161128'

GROUP BY source

)



SELECT source, num_transactions, visits, num_bounces, (100*(num_bounces / visits)) AS bounce_rate

FROM bounce_counts

ORDER BY visits DESC



"""

# run the query if it uses < 1GB

safe_config= bigquery.QueryJobConfig(maximum_bytes_billed= 1000000000)

safe_query_job= client.query(bounce_query, job_config= safe_config)

bounce_df= safe_query_job.to_dataframe()

bounce_df.fillna(0, inplace= True)

print(bounce_df.head(), np.shape(bounce_df))
bad_source= bounce_df[(bounce_df['visits'] < 6) | (bounce_df['bounce_rate'] >= 85)]

print(bad_source['source'])

print('\n','Total number of bad traffic sources:', len(bad_source))