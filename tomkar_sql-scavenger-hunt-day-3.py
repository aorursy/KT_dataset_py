# import package with helper functions 
import bq_helper

# create a helper object for this dataset
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# print the first couple rows of the "accident_2015" table
accidents.head("accident_2015")
# query to find out the number of accidents which 
# happen on each hour of the day
query = """SELECT 
            EXTRACT(HOUR FROM timestamp_of_crash),
            COUNT(consecutive_number) as cnt
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

accidents_by_hour = accidents.query_to_pandas_safe(query)

accidents_by_hour
import matplotlib.pyplot as plot
import matplotlib.ticker

p1 = plot.bar(accidents_by_hour.f0_, accidents_by_hour.cnt, 0.4)

plot.ylabel('# of accidents')
plot.title('Number of accidents by time of day')

plot.show()
query = """SELECT 
            EXTRACT(HOUR FROM timestamp_of_crash),
            SUM(number_of_fatalities) fatal,
            COUNT(consecutive_number) as cnt
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """

fatalities_by_hour = accidents.query_to_pandas_safe(query)

fatalities_by_hour
p1 = plot.bar(fatalities_by_hour.f0_, fatalities_by_hour.fatal, 0.4)
p2 = plot.bar(fatalities_by_hour.f0_, fatalities_by_hour.cnt, 0.4)

plot.ylabel('# of accidents')
plot.title('Number of accidents by time of day')

plot.legend((p1[0], p2[0]), ('# fatalities','# accidents'))

plot.show()
query = """SELECT 
            EXTRACT(DAY FROM timestamp_of_crash),
            EXTRACT(HOUR FROM timestamp_of_crash),
            COUNT(consecutive_number) as cnt
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY EXTRACT(DAY FROM timestamp_of_crash),EXTRACT(HOUR FROM timestamp_of_crash)
        """

accidents_by_hour = accidents.query_to_pandas_safe(query)
#accidents_by_hour.sort_values(f0_,f1_)
accidents_by_hour['day_hour'] = accidents_by_hour.f0_.astype(str) + '_' + accidents_by_hour.f1_.astype(str)

accidents_by_hour
plot.rcParams["figure.figsize"] = (18,6)
p1 = plot.bar(accidents_by_hour.day_hour, accidents_by_hour.cnt, 0.4)

plot.ylabel('# of accidents')
plot.title('Number of accidents by time of day')
plot.show()
query = """SELECT 
            registration_state_name,
            SUM(CASE WHEN hit_and_run = 'Yes' THEN 1 ELSE 0 END) as hit_and_run ,
            COUNT(consecutive_number) as cnt
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
        """

hit_run = accidents.query_to_pandas_safe(query)

hit_run.sort_values("hit_and_run",ascending=False)

query = """SELECT 
            registration_state_name,
            SUM(CASE WHEN hit_and_run = 'Yes' THEN 1 ELSE 0 END) as hit_and_run ,
            COUNT(consecutive_number) as cnt
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY registration_state_name
        """

hit_run = accidents.query_to_pandas_safe(query)

hit_run["percent"] = hit_run["hit_and_run"]/hit_run["cnt"]*100

hit_run.sort_values("percent",ascending=False)
