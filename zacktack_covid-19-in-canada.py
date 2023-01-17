# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.interactiveshell import display





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cases_df = pd.read_csv("/kaggle/input/coronaviruscovid19-canada/cases.csv",index_col='case_id')

cases_df.head()
recovered_df = pd.read_csv("../input/coronaviruscovid19-canada/recovered.csv")

study = recovered_df.loc[recovered_df['date_recovered']=='2020-04-03',['province','cumulative_recovered']]

study.index = study['province']

study.drop('province',axis=1,inplace=True)

study
death_df = pd.read_csv("../input/coronaviruscovid19-canada/mortality.csv")

death_df.head()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



case_vs_recovered = pd.concat([cases_df['province'].value_counts(),study,death_df['province'].value_counts()],axis=1,sort=False)

case_vs_recovered.index.name='province'

case_vs_recovered.columns = ['Confirmed','Recovered','Death']

case_vs_recovered.fillna(0,inplace=True)

case_vs_recovered = case_vs_recovered.astype(int)



display(case_vs_recovered)



recover_rate = pd.DataFrame([elem + "%" if elem!="nan" else "0%" for elem in map(str,round(case_vs_recovered['Recovered'] / case_vs_recovered['Confirmed'] * 100,2))],index=case_vs_recovered.index,columns=['Recover Rate(%)'])

death_rate = pd.DataFrame([elem + "%" if elem!="nan" else "0%" for elem in map(str,round(case_vs_recovered['Death'] / case_vs_recovered['Confirmed'] * 100,2))],index=case_vs_recovered.index,columns=['Death Rate(%)'])

total_rate = pd.DataFrame([round(case_vs_recovered['Recovered'].sum() / case_vs_recovered['Confirmed'].sum() * 100, 2),round(case_vs_recovered['Death'].sum() / case_vs_recovered['Confirmed'].sum() * 100 , 2)],index=['Total Recover Rate','Total Death Rate'],columns=['Percentage(%)'])

display(recover_rate,death_rate,total_rate)
ax = case_vs_recovered.plot.barh(rot=0,figsize=(35,35),width=0.8)

plt.xlabel('Province'),plt.ylabel('Cases'),plt.autoscale()



for p in ax.patches:

    ax.annotate(str(p.get_width()), (p.get_width() * 1.005, p.get_y() * 1.005))

ax