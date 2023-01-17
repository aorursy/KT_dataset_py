# Load Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # ploting


"""
List of collumns in the data

date_received                   object
product                         object
sub_product                     object
issue                           object
sub_issue                       object
consumer_complaint_narrative    object
company_public_response         object
company                         object
state                           object
zipcode                         object
tags                            object
consumer_consent_provided       object
submitted_via                   object
date_sent_to_company            object
company_response_to_consumer    object
timely_response                 object
consumer_disputed?              object
complaint_id                     int64
"""
# Read in the data
consumer_complaints = pd.read_csv("../input/consumer_complaints.csv")

#Find frequencies for these items
items_of_intrest = ['product', 'issue', 'company']

for dtype in items_of_intrest:
    print(dtype)
    print(consumer_complaints[dtype].value_counts())
    print("----------------------------------")