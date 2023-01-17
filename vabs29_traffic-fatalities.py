# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset
traffic_fatalities = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "nhtsa_traffic_fatalities")
# print a list of all the tables in the nhtsa_traffic_fatalities
traffic_fatalities.list_tables()
# print information on all the columns in the "accident_2015" table
# in the nhtsa_traffic_fatalities dataset
traffic_fatalities.table_schema("accident_2015")
# preview the first couple lines of the "accident_2015" table
traffic_fatalities.head("accident_2015")	
# preview the first ten entries in the timestamp_of_crashcolumn of the accident_2015
traffic_fatalities.head("accident_2015", selected_columns="timestamp_of_crash", num_rows=10)
# this query looks in the accident_2015 table  in the nhtsa_traffic_fatalities 
# dataset, then gets the city column from every row where 
# the state_name column has "Iowa" in it.
query = """SELECT city
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
            WHERE state_name = 'Iowa'  """

# check how big this query will be
traffic_fatalities.estimate_query_size(query)