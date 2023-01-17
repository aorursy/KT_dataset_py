import bq_helper

# create a helper object for our bigquery dataset
chicago_crime = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "chicago_crime")
#print a list of all tables in chicago crime dataset
chicago_crime.list_tables()
#print all info in crime table in the dataset
chicago_crime.table_schema("crime")
#check just the first couple of lines of "crime " table if its right.Preview of first 3 columns
chicago_crime.head("crime",3)
#check for a particular column in crime table . eg: primary_type : primary description of the crime as per iucr
chicago_crime.head("crime", selected_columns= "primary_type", num_rows=10)
#running query in crime table and checking size of query.
query = """SELECT primary_type
           FROM `bigquery-public-data.chicago_crime.crime`
           WHERE arrest = True """

#estimate size
chicago_crime.estimate_query_size(query)
#check if size is <0.5 GB and then return a pandas dataframe
chicago_crime.query_to_pandas_safe(query, max_gb_scanned=0.05)
arrest_score = chicago_crime.query_to_pandas_safe(query)
arrest_score.head(10)
arrest_score.primary_type.count()