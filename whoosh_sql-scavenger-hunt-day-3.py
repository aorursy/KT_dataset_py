# boilerplate from the original notebook
import bq_helper
traffic_fatalities = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")

# for visualization
import matplotlib.pyplot as plt
import seaborn as sns
# First we estimate costs
query = """
SELECT EXTRACT(HOUR FROM timestamp_of_crash) AS hour, count(*) AS accidents
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_*`
GROUP BY Hour
ORDER BY Hour
"""
traffic_fatalities.estimate_query_size(query)
accident_per_hour = traffic_fatalities.query_to_pandas_safe(query)
accident_per_hour.plot(x='hour', y='accidents', figsize=(11.7, 8.27), title='Accidents per hour (Over 2015 and 2016)')
# First we estimate costs
query = """
SELECT DISTINCT registration_state_name
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_*`
"""
traffic_fatalities.estimate_query_size(query)
unique_states = traffic_fatalities.query_to_pandas_safe(query)
', '.join(unique_states['registration_state_name'].tolist())
# First we estimate costs
query = """
SELECT registration_state_name AS state, count(*) AS hit_and_runs
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_*`
WHERE hit_and_run = 'Yes' AND NOT REGEXP_CONTAINS('^(No(t| )|Unknown|Other)', registration_state_name)
GROUP BY state
ORDER BY hit_and_runs DESC
LIMIT 5
"""
traffic_fatalities.estimate_query_size(query)
state_most_hitruns = traffic_fatalities.query_to_pandas_safe(query)
plt.subplots(figsize=(11.7, 8.27))
sns.barplot(x='state', y='hit_and_runs', data=state_most_hitruns)
plt.title("Top 5 states in hit and runs (over 2015 and 2016)")