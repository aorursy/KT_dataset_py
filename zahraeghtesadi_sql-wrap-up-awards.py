import bq_helper
import pandas as pd
import matplotlib.pylab as plt
%matplotlib inline

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
query="""
         SELECT COUNT(accident.consecutive_number) AS total,
             vehicle.body_type_name AS bodytype,
             vehicle.vehicle_make_name AS manufacturer
         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` AS accident
         INNER JOIN `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016` AS vehicle
         ON accident.consecutive_number=vehicle.consecutive_number
         WHERE vehicle.body_type NOT IN (9,98,99)
             -- 9 : Other or Unknown Automobile Type (Since 1994)
             -- 98 : Not Reported
             -- 99 : Unknown Body Type
                 AND vehicle.vehicle_make != 99
             -- 99 : Unknown make
         GROUP BY bodytype, manufacturer
         ORDER BY total DESC
     """
accidents.estimate_query_size(query)
accidents_by_type_make=accidents.query_to_pandas_safe(query)
accidents_by_type_make.head(20)
query = """ SELECT COUNT(accident.consecutive_number) AS total,
                person.police_reported_drug_involvement AS envolvement,
                person.method_of_drug_determination_by_police AS method
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` AS accident
            INNER JOIN `bigquery-public-data.nhtsa_traffic_fatalities.person_2016` AS person
            ON accident.consecutive_number = person.consecutive_number
            WHERE --person.police_reported_drug_involvement='Yes (Drugs Involved)'
                 person.method_of_drug_determination_by_police != 'Not Reported'
            GROUP BY envolvement, method
            ORDER BY total DESC
        """
accidents.estimate_query_size(query)
accidents_drug=accidents.query_to_pandas_safe(query)
accidents_drug.head(20)
#Question 2: common drug type
query = """ SELECT COUNT(accident.consecutive_number) AS total,
                person.drug_test_type4_name AS type4--,
                --person.drug_test_type5_name AS type5,
                --person.drug_test_type6_name AS type6
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016` AS accident
            INNER JOIN `bigquery-public-data.nhtsa_traffic_fatalities.person_2016` AS person
            ON accident.consecutive_number = person.consecutive_number
            WHERE person.police_reported_drug_involvement='Yes (Drugs Involved)'
                 AND person.method_of_drug_determination_by_police != 'Not Reported'
                 AND person.drug_test_type4_name NOT IN ('Not Tested for Drugs',
                         'No Drugs Reported/Negative','Not Reported')
            GROUP BY type4 --, type5, type6
            ORDER BY total DESC
        """
accidents.estimate_query_size(query)
accidents_drug_type=accidents.query_to_pandas_safe(query)
accidents_drug_type.head(20)
plt.pie(x=accidents_drug_type.total,labels=accidents_drug_type.type4);
query = """
WITH vehicle AS (
        SELECT consecutive_number, vehicle_number, first_harmful_event,
            first_harmful_event_name,most_harmful_event,critical_event_precrash,
            critical_event_precrash_name
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
                ),
     person AS (
         SELECT consecutive_number,vehicle_number,person_number,injury_severity_name,
             number_of_motor_vehicle_striking_non_motorist, person_type
         FROM `bigquery-public-data.nhtsa_traffic_fatalities.person_2016`
         WHERE injury_severity IN (3,4) --i.e., there is a severe injury reported
               )
SELECT COUNT(person.person_number) AS total_injured_people,
    person.injury_severity_name AS severity,
    vehicle.first_harmful_event_name AS FHE,
    vehicle.most_harmful_event AS MHE,
    vehicle.critical_event_precrash_name AS CPE
FROM vehicle
INNER JOIN person
ON vehicle.consecutive_number=person.consecutive_number
    AND (IF(person.vehicle_number != 0, vehicle.vehicle_number=person.vehicle_number,
            vehicle.vehicle_number=person.number_of_motor_vehicle_striking_non_motorist))
        --per manual Non-Occupants have veicle_number = 0, in this case number_of_motor_vehicle_striking_non_motorist 
        --reflects the veicle_number in the pearson_ table.
GROUP BY severity,FHE, MHE,CPE
ORDER BY total_injured_people DESC
"""
accidents.estimate_query_size(query)

accident_event_severity=accidents.query_to_pandas_safe(query)
print(accident_event_severity.shape)
accident_event_severity.head(20)

import bq_helper
import pandas as pd
import matplotlib.pylab as plt

accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
# accidents.table_schema('accident_2015')
# query to find out the number of accidents which 
# happen on each day of the week
query = """SELECT COUNT(consecutive_number) AS num_accidents, 
                  EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS weekday
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY weekday
            ORDER BY num_accidents DESC
        """
accidents.estimate_query_size(query)
accidents_by_day = accidents.query_to_pandas_safe(query)
accidents_by_day.head(10)
# Here I tried to create an ordered cathegorical index for weekday for nicer plotting
weekday_map={1:'Sunday',2:'Monday',3:'Tuesday',4:'Wednesday',5:'Thursday',6:'Friday',7:'Saturday'}
new_index=(pd.CategoricalIndex(accidents_by_day.weekday.map(weekday_map)).
           reorder_categories(new_categories=['Monday','Tuesday','Wednesday','Thursday',
                                              'Friday','Saturday','Sunday'],
                              ordered=True))
new_index
# I could do inplace but I don't want to loose the original data from the query
copy1=accidents_by_day.copy()
copy1.set_index(new_index,drop=True,inplace=True)
copy1
copy1.sort_index().num_accidents.plot(kind='bar')
plt.title("Number of Accidents by Rank of Day \n (Most to least dangerous)");
# This gives me tuple index out of range which I don't underestand 
##plt.plot(copy1.num_accidents)
query = """SELECT COUNT(consecutive_number) AS num_accidents, 
                  EXTRACT(HOUR FROM timestamp_of_crash) AS hour
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            GROUP BY hour
            ORDER BY num_accidents DESC
        """
accidents.estimate_query_size(query)
accidents_by_hours=accidents.query_to_pandas_safe(query)
accidents_by_hours.head()
query = """SELECT COUNTIF(hit_and_run="Yes") AS num_hitrun, 
                  registration_state_name AS state
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
            GROUP BY state
            ORDER BY num_hitrun DESC
        """
accidents.estimate_query_size(query)
vehicles_hAr_by_state=accidents.query_to_pandas_safe(query)
vehicles_hAr_by_state.head(10)