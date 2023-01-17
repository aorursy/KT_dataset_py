from google.cloud import bigquery

from bq_helper import BigQueryHelper

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#Set up a dataset instance

bq_assistant = BigQueryHelper("bigquery-public-data","world_bank_intl_education")

bq_assistant.list_tables()
#Table of interest -- international education

bq_assistant.table_schema('international_education')
#Table at a glance

bq_assistant.head('international_education',10)
#Explore indicators 

query0 = '''

            select distinct(indicator_name),indicator_code 

            from `bigquery-public-data.world_bank_intl_education.international_education`

            '''

indicators = bq_assistant.query_to_pandas_safe(query0)

num_indicators = len(indicators.index)

print ('There are %d indicators'%(num_indicators))

indicators.head()
#Number of indicators by category in terms of education level

for ed in ['primary','secondary','higher']:

    num_indicators = len(indicators[indicators['indicator_name'].str.contains('%s education'%(ed))].index)

    print ('%d indicators are related to %s education'%(num_indicators,ed))
#Exploratory analysis based on primary education

indicators_PE = indicators[indicators['indicator_name'].str.contains('primary education')].copy()

indicators_PE.tail(10)
#Repetition rate in primary education(all grades) by country

#Top 20 countries with highest repetition rate in primary education(all grades) after year 2000

indicator = "UIS.REPR.1.M"

query1 = '''

             select country_name,AVG(value) as percentage_of_repetition

             from `bigquery-public-data.world_bank_intl_education.international_education`

             where indicator_code = "UIS.REPR.1.M"

             and year > 2000

             group by country_name

             order by percentage_of_repetition desc

             limit 20 '''

repetition = bq_assistant.query_to_pandas_safe(query1)

repetition.head(5)
#plot

fig = plt.figure(figsize=(16,9))

pl = fig.add_subplot(111)

pl.barh(np.arange(len(repetition.index)),np.array(repetition['percentage_of_repetition']),color='burlywood')

plt.yticks(np.arange(len(repetition.index)),repetition['country_name'])

plt.title('Repetition Rate in Primary Education(all grades) after 2000')

plt.xlabel('percentage')

plt.ylabel('country')

plt.show()
#Pupil/trained teacher ratio in primary educatio

indicator ="UIS.PTRHC.1.TRAINED"

query2 = '''

             select country_name,AVG(value) as pupil_teacher_ratio

             from `bigquery-public-data.world_bank_intl_education.international_education`

             where indicator_code = "UIS.PTRHC.1.TRAINED"

             and year > 2000

             group by country_name

             order by pupil_teacher_ratio desc

             limit 20 '''

pt_ratio = bq_assistant.query_to_pandas_safe(query2)

pt_ratio.head(5)
#plot

fig = plt.figure(figsize=(16,9))

pl = fig.add_subplot(111)

pl.barh(np.arange(len(pt_ratio.index)),np.array(pt_ratio['pupil_teacher_ratio']),color='chartreuse')

plt.yticks(np.arange(len(pt_ratio.index)),pt_ratio['country_name'])

plt.title('pupil/teacher ratio in Primary Education(all grades) after 2000')

plt.xlabel('ratio')

plt.ylabel('country')

plt.show()
#Theoritical duration of primary education(in years) after 2000

query3 = '''

             select count(country_name) as number_of_countries, value as duration_of_primary_school

             from `bigquery-public-data.world_bank_intl_education.international_education`

             where indicator_code = "SE.PRM.DURS"

             and year > 2000

             group by duration_of_primary_school

             order by number_of_countries desc

             limit 20 '''

duration_PE = bq_assistant.query_to_pandas_safe(query3)

duration_PE.head(5)
#plot

fig = plt.figure(figsize=(4,4))

pl = fig.add_subplot(111)

sizes = duration_PE['number_of_countries']

labels = duration_PE['duration_of_primary_school']

pl.pie(sizes,labels = labels,autopct='%1.1f%%',

        shadow=True, startangle=90)



plt.title('Duration of Primary School')

plt.show()