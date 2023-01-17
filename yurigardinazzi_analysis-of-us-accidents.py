#First of all we have to import all the libraries that we need and set up our database
import bq_helper
import matplotlib.pyplot as plt #this will help us to plot the datas

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents_query =  """
                     SELECT COUNT(consecutive_number) AS crashes, state_name
                     FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` 
                     GROUP BY state_name
                     ORDER BY crashes DESC                   
                   """
state_with_more_accidents = accidents.query_to_pandas_safe(accidents_query)

plt.bar( state_with_more_accidents.state_name.head(), state_with_more_accidents.crashes.head())
plt.ylabel("Number of accidents")
plt.title("States with more accidents in 2016")
role_query =  """
                SELECT person_type_name AS role , COUNT(person_type_name) as number
                FROM   `bigquery-public-data.nhtsa_traffic_fatalities.person_2016` 
                GROUP BY role
                ORDER BY number DESC
              """
roles = accidents.query_to_pandas_safe(role_query)
roles
pedestrians_query= """
                    SELECT a.state_name, COUNT(a.consecutive_number) as people_involved
                    FROM  `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` AS a
                    INNER JOIN `bigquery-public-data.nhtsa_traffic_fatalities.person_2016` AS p
                          ON a.consecutive_number = p.consecutive_number
                    WHERE p.person_type_name = 'Pedestrian'
                    GROUP BY a.state_name
                    ORDER BY people_involved DESC
                   """
pedestrians_per_state = accidents.query_to_pandas_safe(pedestrians_query)
plt.bar(pedestrians_per_state.state_name.head(), pedestrians_per_state.people_involved.head())
plt.ylabel("number of people involved")
plt.title("Pedestrians involved in accidents in 2016")
status_query = """
                 SELECT driver_distracted_by_name AS status, COUNT(consecutive_number) AS total
                 FROM `bigquery-public-data.nhtsa_traffic_fatalities.distract_2016`
                 GROUP BY status
                 ORDER BY total DESC                 
               """
driver_status = accidents.query_to_pandas_safe(status_query)
driver_status.head()

maneuver_query = """
                    SELECT driver_maneuvered_to_avoid_name AS maneuvered_to_avoid, 
                           COUNT(consecutive_number) AS number
                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.maneuver_2016`
                    
                    GROUP BY maneuvered_to_avoid
                    ORDER BY number DESC
                 """
maneuvers = accidents.query_to_pandas_safe(maneuver_query)
maneuvers.head()
month_query = """
                SELECT month_of_crash, COUNT(consecutive_number) as total
                FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`
                GROUP BY month_of_crash
                ORDER BY total  DESC  
             """
accidents_per_month = accidents.query_to_pandas_safe(month_query)

plt.rcParams["figure.figsize"] = (20,10)
plt.bar(accidents_per_month.month_of_crash , accidents_per_month.total)
plt.title("accidents per month")
accidents_per_month.total.mean()  #Average of accidents per month