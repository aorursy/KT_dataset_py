# Set up feedack system

from learntools.core import binder

binder.bind(globals())

from learntools.sql.ex1 import *

print("Setup Complete")
from google.cloud import bigquery

import pandas as pd



# Create a "Client" object

client = bigquery.Client()



# Construct a reference to the "chicago_crime" dataset

dataset_ref = client.dataset("chicago_crime", project="bigquery-public-data")



# API request - fetch the dataset

dataset = client.get_dataset(dataset_ref)
# Write the code you need here to figure out the answer

num_tables = len(list(client.list_tables(dataset)))  # Store the answer as num_tables and then run this cell



print(num_tables)

q_1.check()
#q_1.hint()

#q_1.solution()
# Write the code to figure out the answer
crime_table_ref = dataset_ref.table("crime")

crime_table = client.get_table(crime_table_ref)

crime_table.schema

num_timestamp_fields = 2 # Put your answer here



q_2.check()
#q_2.hint()

#q_2.solution()
# Write the code here to explore the data so you can find the answer

df = client.list_rows(crime_table, max_results=25).to_dataframe()

for column in df.columns:

    print(column)





fields_for_plotting = ['latitude', 'longitude'] # Put your answers here



q_3.check()
#q_3.hint()

#q_3.solution()
# Scratch space for your code



df = client.list_rows(crime_table, max_results=25).to_dataframe()

df.loc[:,['latitude','longitude','location']]