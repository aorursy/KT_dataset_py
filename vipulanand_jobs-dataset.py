import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        f_name = os.path.join(dirname, filename)



import json

import pandas as pd

with open(f_name) as f:

    data_json = json.load(f)
count = 0

job_title = []

company_name = []

company_rating = []

no_review = []

job_location = []

job_description = []

date_posted = []

crawl_timestamp = []

for i in data_json:

    count += 1

    job_title.append(i.get('Job_Title'))

    company_name.append(i.get('Company_Name'))

    company_rating.append(i.get('Company_Rating'))

    no_review.append(i.get('No. of Reviews'))

    job_location.append(i.get('Job_Location'))

    job_description.append(i.get('Job_Description').replace('\n', ' '))

    date_posted.append(i.get('date_posted'))

    crawl_timestamp.append(i.get('Crawl_TimeStamp'))
len(crawl_timestamp)
pd_data = pd.DataFrame({

        'Job Title': job_title,

        'Company Name': company_name,

        'Company Rating': company_rating,

        'No. of Reviews of Company': no_review,

        'Job Location': job_location,

        'Job Description': job_description,

        'Date Posted': date_posted,

        'Crawl TimeStamp': crawl_timestamp

    })
pd_data.head()
pd_data = pd_data.dropna()
pd_data.shape
pd_data = pd_data.reset_index()
pd_data['Company Rating'].apply(lambda x: round(float(x),2))
pd_data.tail()