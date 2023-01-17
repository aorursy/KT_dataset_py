# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

from google.cloud import bigquery

from bq_helper import BigQueryHelper



from IPython.display import HTML

import base64

# Any results you write to the current directory are saved as output.



bq_assistant = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")
df_names = ['accident','cevent','damage','distract','drimpair','factor','maneuver',

 'nmcrash','nmimpair','nmprior','parkwork','pbtype','person','safetyeq',

 'vehicle','vevent','vindecode','violatn','vision','vsoe']
for item in df_names:

    QUERY = """SELECT *

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.{}_2015`""".format(item)

    print(QUERY)

    df_2015 = bq_assistant.query_to_pandas(QUERY)

    df_2015.to_csv('{}_2015.csv'.format(item), index = False)

    print('{}_2015.csv'.format(item))
for item in df_names:

    QUERY = """SELECT *

            FROM `bigquery-public-data.nhtsa_traffic_fatalities.{}_2016`""".format(item)

    print(QUERY)

    df_2015 = bq_assistant.query_to_pandas(QUERY)

    df_2015.to_csv('{}_2016.csv'.format(item), index = False)

    print('{}_2016.csv'.format(item))