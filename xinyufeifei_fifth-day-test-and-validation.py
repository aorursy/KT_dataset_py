import sys
!{sys.executable} -m pip install csvvalidator
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
from io import StringIO
import csv

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/restaurant-scores-lives-standard.csv')
sample_data = data[['business_id', 'business_name', 'business_postal_code', 'inspection_id', 
                    'inspection_date', 'inspection_score', 'risk_category']][:2000].dropna()
sample_data.head()
from csvvalidator import *

# Specify which fields (columns) your .csv needs to have
# You should include all fields you use in your dashboard
field_names = ('business_id', 
               'business_name', 
               'business_postal_code', 
               'inspection_id',
               'inspection_date',
               'inspection_score', 
               'risk_category'
               )

# create a validator object using these fields
validator = CSVValidator(field_names)

# write some checks to make sure specific fields 
# are the way we expect them to be
validator.add_value_check('inspection_id', # the name of the field
                          int, # a function that accepts a single argument and 
                               # raises a `ValueError` if the value is not valid.
                               # Here checks that "key" is an integer.
                          'EX1', # code for exception
                          'inspection_id must be an integer'# message to report if error thrown
                         )
validator.add_value_check('inspection_date', 
                          # check for a date with the sepcified format
                          datetime_string('%Y-%m-%dT%H:%M:%S'), 
                          'EX2',
                          'invalid date'
                         )
validator.add_value_check('inspection_score',
                          # makes sure the number of units sold is an integer
                          float,
                          'EX3',
                          'inspection_score not an number'
                         )
validator.add_value_check('risk_category', 
                          # check that the "store" field only has
                          # "store1" and "store2" in it
                          enumeration('Low Risk', 'High Risk', 'Moderate Risk'),
                          'EX4', 
                          'risk_category not recognized')
sample_data.shape
# convert pandas data frame to list of list in order to be validated by csv validator
sample_values = sample_data.values.tolist()
validator.validate(sample_values)
