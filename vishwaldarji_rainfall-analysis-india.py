import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
district = pd.read_csv("../input/district wise rainfall normal.csv",sep=",")

district.info()
data = pd.read_csv("../input/rainfall in india 1901-2015.csv",sep=",")

data.info()
data.head()
data.describe()
data.hist(figsize=(12,12));
data[['SUBDIVISION','JUL','DEC'

]].groupby("SUBDIVISION").sum().plot.bar(stacked=True,figsize=(16,8));
data.columns
data[['SUBDIVISION', 'Jan-Feb', 'ANNUAL',

       'Jun-Sep', 'Oct-Dec']].groupby("SUBDIVISION").sum().plot.bar(stacked=True,figsize=(16,8));
data[['SUBDIVISION', 'Jan-Feb', 'Mar-May',

       'Jun-Sep', 'Oct-Dec']].groupby("SUBDIVISION").sum().plot.bar(stacked=True,figsize=(16,8));
data[['SUBDIVISION', 'Jan-Feb', 'Mar-May','Jun-Sep', 'Oct-Dec']].groupby("SUBDIVISION").sum().plot.bar(stacked=True,figsize=(16,8));
data[['YEAR', 'JUN','JUL']].groupby("YEAR").sum().plot.hist(figsize=(13,8));
data[['SUBDIVISION', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',

       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("SUBDIVISION").mean().plot.line(stacked=True,figsize=(13,8));
mp_data = district[district['STATE_UT_NAME'] == 'MADHYA PRADESH']

mp_data[['DISTRICT', 'ANNUAL','Jun-Sep']].groupby("DISTRICT").mean()[:40].plot.hist(stacked=True,figsize=(12,8));
mp_data[['DISTRICT', 'JUN','AUG']].groupby("DISTRICT").mean()[:40].plot.hist(stacked=True,figsize=(12,8));
mp_data[['DISTRICT','MAY','JUL','JUN','AUG']].groupby("DISTRICT").mean()[:40].plot.hist(stacked=True,figsize=(12,8));
mp_data[['DISTRICT','MAY','JUL','JUN','AUG']].groupby("DISTRICT").mean()[:40].plot.line(stacked=True,figsize=(12,8));
mp_data[['DISTRICT', 'JUN','AUG']].groupby("DISTRICT").mean()[:40].plot.line(stacked=True,figsize=(12,8));
mp_data[['DISTRICT', 'ANNUAL','Jun-Sep']].groupby("DISTRICT").mean()[:40].plot.line(stacked=True,figsize=(12,8));
mp_data[['DISTRICT', 'Jan-Feb','Jun-Sep']].groupby("DISTRICT").mean()[:40].plot.line(stacked=True,figsize=(12,8));
mp_data[['DISTRICT', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',

       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].groupby("DISTRICT").mean()[:40].plot.line(stacked=True,figsize=(13,8));
mp_data[['DISTRICT','Jun-Sep','ANNUAL']].groupby("DISTRICT").sum()[:40].plot.pie(subplots=True,figsize=(36,12),autopct='%1.1f%%');
mp_data[['DISTRICT','ANNUAL']].groupby("DISTRICT").sum()[:40].plot.pie(subplots=True,figsize=(12,12),autopct='%1.1f%%');
mp_data[['DISTRICT', 'Jun-Sep',]].groupby("DISTRICT").sum()[:40].plot.pie(subplots=True,figsize=(12,12),autopct='%1.1f%%');
mp_data[['DISTRICT', 'Jan-Feb',]].groupby("DISTRICT").sum()[:40].plot.pie(subplots=True,figsize=(12,12),autopct='%1.1f%%');
mp_data[['DISTRICT','Jun-Sep']].groupby("DISTRICT").sum()[:40].plot.pie(subplots=True,figsize=(20,12));
mp_data[['DISTRICT', 'Jan-Feb','Jun-Sep']].groupby("DISTRICT").sum()[:40].plot.pie(subplots=True,figsize=(24,12));