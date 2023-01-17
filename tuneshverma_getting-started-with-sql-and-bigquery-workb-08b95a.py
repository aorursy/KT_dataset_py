import bq_helper
# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
chicago_crime.list_tables()
chicago_crime.table_schema('crime')
chicago_crime.head('crime')
query= """SELECT case_number 
          FROM `bigquery-public-data.chicago_crime.crime`
          WHERE iucr = '1110' """

query1= """SELECT case_number
          FROM `bigquery-public-data.chicago_crime.crime`
          WHERE location_description = 'RESIDENCE' """

chicago_crime.estimate_query_size(query)
chicago_crime.estimate_query_size(query1)
case_no_iucr=chicago_crime.query_to_pandas_safe(query)
chicago_crime.query_to_pandas_safe(query1)

