# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import os

import sys

import glob

import re

import subprocess

import time



import plotly

# import getpass

import plotly.graph_objs as go

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.offline as pyo

# Set notebook mode to work in offline

pd.options.plotting.backend = "plotly"

from plotly.offline import iplot

pyo.init_notebook_mode()



import statsmodels.api as sm



import datetime

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE



def jobClassifier(soc_code, SOC_MAP):

    soc_map = SOC_MAP

    soc = str(soc_code).split('-')[0]        

    return soc_map.get(soc,'OTHER')



SOC_MAP = {

        '11': 'MANAGERIAL, ADMIN',

        '13': 'FINANCIALS, COMPLIANCE',

        '15': 'COMPUTING, STATISTICIANS',

        '17': 'ENGINEERING EXCEPT COMPUTERS' ,

        '19': 'SCIENTISTS' ,

        '21': 'PSYCHOLOGY , COUNSELLING , SOCIAL WORKS' ,

        '23': 'LEGAL' ,

        '25': 'EDUCATORS , CURATORS' ,

        '27': 'DESIGNERS , COACHES' ,

        '29': 'MEDICALS' ,

        '31': 'HEALTHCARE ASSTS' ,

        '33': 'SECURITY' ,

        '35': 'CULINARY' ,

        '37': 'CLEANING , KEEPING' ,

        '39': 'RECEIPTIONISTS , SERVICE ATTENDANTS' ,

        '40': 'ENGINEERING EXCEPT COMPUTERS' ,

        '41': 'TRADERS , SALES REPS' ,

        '43': 'QUALITY , STATISTICAL ASSTS' ,

        '45': 'AGRICULTURAL' ,

        '47': 'ARTISANS' ,

        '49': 'SERVICE TECHNICIANS' ,

        '51': 'MACHINISTS' ,

        '53': 'TRANSPORT' ,

        '71': 'ENGINEERING EXCEPT COMPUTERS' }



us_state_abbrev = {

    'ALABAMA': 'AL',

    'ALASKA': 'AK',

    'AMERICAN SAMOA': 'AS',

    'ARIZONA': 'AZ',

    'ARKANSAS': 'AR',

    'CALIFORNIA': 'CA',

    'COLORADO': 'CO',

    'CONNECTICUT': 'CT',

    'DELAWARE': 'DE',

    'DISTRICT OF COLUMBIA': 'DC',

    'FLORIDA': 'FL',

    'GEORGIA': 'GA',

    'GUAM': 'GU',

    'HAWAII': 'HI',

    'IDAHO': 'ID',

    'ILLINOIS': 'IL',

    'INDIANA': 'IN',

    'IOWA': 'IA',

    'KANSAS': 'KS',

    'KENTUCKY': 'KY',

    'LOUISIANA': 'LA',

    'MAINE': 'ME',

    'MARYLAND': 'MD',

    'MASSACHUSETTS': 'MA',

    'MICHIGAN': 'MI',

    'MINNESOTA': 'MN',

    'MISSISSIPPI': 'MS',

    'MISSOURI': 'MO',

    'MONTANA': 'MT',

    'NEBRASKA': 'NE',

    'NEVADA': 'NV',

    'NEW HAMPSHIRE': 'NH',

    'NEW JERSEY': 'NJ',

    'NEW MEXICO': 'NM',

    'NEW YORK': 'NY',

    'NORTH CAROLINA': 'NC',

    'NORTH DAKOTA': 'ND',

    'NORTHERN MARIANA ISLANDS':'MP',

    'OHIO': 'OH',

    'OKLAHOMA': 'OK',

    'OREGON': 'OR',

    'PENNSYLVANIA': 'PA',

    'PUERTO RICO': 'PR',

    'RHODE ISLAND': 'RI',

    'SOUTH CAROLINA': 'SC',

    'SOUTH DAKOTA': 'SD',

    'TENNESSEE': 'TN',

    'TEXAS': 'TX',

    'UTAH': 'UT',

    'VERMONT': 'VT',

    'VIRGIN ISLANDS': 'VI',

    'VIRGINIA': 'VA',

    'WASHINGTON': 'WA',

    'WEST VIRGINIA': 'WV',

    'WISCONSIN': 'WI',

    'WYOMING': 'WY',

    'AL': 'AL',

    'AK': 'AK',

    'AS': 'AS',

    'AZ': 'AZ',

    'AR': 'AR',

    'CA': 'CA',

    'CO': 'CO',

    'CT': 'CT',

    'DE': 'DE',

    'DC': 'DC',

    'FL': 'FL',

    'GA': 'GA',

    'GU': 'GU',

    'HI': 'HI',

    'ID': 'ID',

    'IL': 'IL',

    'IN': 'IN',

    'IA': 'IA',

    'KS': 'KS',

    'KY': 'KY',

    'LA': 'LA',

    'ME': 'ME',

    'MD': 'MD',

    'MA': 'MA',

    'MI': 'MI',

    'MN': 'MN',

    'MS': 'MS',

    'MO': 'MO',

    'MT': 'MT',

    'NE': 'NE',

    'NV': 'NV',

    'NH': 'NH',

    'NJ': 'NJ',

    'NM': 'NM',

    'NY': 'NY',

    'NC': 'NC',

    'ND': 'ND',

    'MP':'MP',

    'OH': 'OH',

    'OK': 'OK',

    'OR': 'OR',

    'PA': 'PA',

    'PR': 'PR',

    'RI': 'RI',

    'SC': 'SC',

    'SD': 'SD',

    'TN': 'TN',

    'TX': 'TX',

    'UT': 'UT',

    'VT': 'VT',

    'VI': 'VI',

    'VA': 'VA',

    'WA': 'WA',

    'WV': 'WV',

    'WI': 'WI',

    'WY': 'WY'

}
perm = pd.read_csv('../input/h1bfy2019csv/perm2015to2019.csv', 

                        engine='python').dropna(how='all')

perm=perm.fillna("Unknown")

perm.head(10)
perm["JOB_INFO_WORK_STATE"]=perm["JOB_INFO_WORK_STATE"].map(us_state_abbrev)

perm["EMPLOYER_STATE"]=perm["EMPLOYER_STATE"].map(us_state_abbrev)

perm = perm.drop("Unnamed: 0", 1)

perm['countvar']=1

perm.head(10)
fig = px.histogram(perm, x="CASE_STATUS")

fig.update_layout(title='CASE_STATUS OVER 5 YEARS')

fig.update_layout(showlegend=False)

fig.show()
fig = make_subplots(

    rows=1, cols=1,

    specs=[[{'type':'domain'}],

          ]

)

df = perm.groupby('EMPLOYER_STATE').count()

fig.add_trace(go.Pie(

    labels=df.index,

    values=df['countvar'],

    name='EMPLOYER_STATE',

), row=1, col=1)

fig.update_layout(height=600, width=800,title='EMPLOYER_STATE_DISTRIBUTIO_OVERALL')

fig.show()
table_state1 = sm.stats.Table.from_data(perm[['CASE_STATUS', 'EMPLOYER_STATE']])

print('\nPvalue of Chi2-test is {}'.format(table_state1.test_nominal_association().pvalue))

print('\nThere is strong association of JOB_CATEGORY with EMPLOYER_STATE')

table_state1.resid_pearson
fig = make_subplots(

    rows=1, cols=1,

    specs=[[{'type':'domain'}],

          ]

)

df = perm.groupby('JOB_INFO_WORK_STATE').count()

fig.add_trace(go.Pie(

    labels=df.index,

    values=df['countvar'],

    name='EMPLOYER_STATE',

), row=1, col=1)

fig.update_layout(height=600, width=800,title='WORK_STATE_DISTRIBUTIO_OVERALL')

fig.show()
table_state2 = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JOB_INFO_WORK_STATE']])

print('\nPvalue of Chi2-test is {}'.format(table_state2.test_nominal_association().pvalue))

print('\nThere is strong association of JOB_CATEGORY with JOB_INFO_WORK_STATE')

table_state2.resid_pearson
dftop = perm.groupby('PW_LEVEL_9089',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['PW_LEVEL_9089','countvar']]

dftop1 = perm.groupby(['PW_LEVEL_9089','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['PW_LEVEL_9089'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['PW_LEVEL_9089'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Top Wage Levels and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table_wage_level = sm.stats.Table.from_data(perm[['CASE_STATUS', 'PW_LEVEL_9089']])

print('\nPvalue of Chi2-test is {}'.format(table_wage_level.test_nominal_association().pvalue))

print('\nThere is strong association of PW_LEVEL_9089 with CASE_STATUS')

table_wage_level.resid_pearson
dftop = perm.groupby('REFILE',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['REFILE','countvar']]

dftop1 = perm.groupby(['REFILE','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['REFILE'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['REFILE'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Refile status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table_refile = sm.stats.Table.from_data(perm[['CASE_STATUS', 'REFILE']])

print('\nPvalue of Chi2-test is {}'.format(table_wage_level.test_nominal_association().pvalue))

print('\nThere is strong association of REFILE with CASE_STATUS')

table_refile.resid_pearson
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'FW_OWNERSHIP_INTEREST']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('FW_OWNERSHIP_INTEREST',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['FW_OWNERSHIP_INTEREST','countvar']]

dftop1 = perm.groupby(['FW_OWNERSHIP_INTEREST','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['FW_OWNERSHIP_INTEREST'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['FW_OWNERSHIP_INTEREST'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Ownership interest status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JOB_INFO_EDUCATION']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('JOB_INFO_EDUCATION',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['JOB_INFO_EDUCATION','countvar']]

dftop1 = perm.groupby(['JOB_INFO_EDUCATION','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['JOB_INFO_EDUCATION'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['JOB_INFO_EDUCATION'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Education status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JOB_INFO_TRAINING']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('JOB_INFO_TRAINING',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['JOB_INFO_TRAINING','countvar']]

dftop1 = perm.groupby(['JOB_INFO_TRAINING','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['JOB_INFO_TRAINING'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['JOB_INFO_TRAINING'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Job training status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JOB_INFO_ALT_FIELD']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('JOB_INFO_ALT_FIELD',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['JOB_INFO_ALT_FIELD','countvar']]

dftop1 = perm.groupby(['JOB_INFO_ALT_FIELD','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['JOB_INFO_ALT_FIELD'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['JOB_INFO_ALT_FIELD'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Job alternate field status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JOB_INFO_JOB_REQ_NORMAL']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('JOB_INFO_JOB_REQ_NORMAL',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['JOB_INFO_JOB_REQ_NORMAL','countvar']]

dftop1 = perm.groupby(['JOB_INFO_JOB_REQ_NORMAL','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['JOB_INFO_JOB_REQ_NORMAL'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['JOB_INFO_JOB_REQ_NORMAL'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Job requirements status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JOB_INFO_FOREIGN_LANG_REQ']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('JOB_INFO_FOREIGN_LANG_REQ',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['JOB_INFO_FOREIGN_LANG_REQ','countvar']]

dftop1 = perm.groupby(['JOB_INFO_FOREIGN_LANG_REQ','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['JOB_INFO_FOREIGN_LANG_REQ'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['JOB_INFO_FOREIGN_LANG_REQ'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Job foreign language requirements status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JOB_INFO_COMBO_OCCUPATION']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('JOB_INFO_COMBO_OCCUPATION',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['JOB_INFO_COMBO_OCCUPATION','countvar']]

dftop1 = perm.groupby(['JOB_INFO_COMBO_OCCUPATION','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['JOB_INFO_COMBO_OCCUPATION'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['JOB_INFO_COMBO_OCCUPATION'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Job's combination of occupations status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JI_OFFERED_TO_SEC_J_FW']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('JI_OFFERED_TO_SEC_J_FW',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['JI_OFFERED_TO_SEC_J_FW','countvar']]

dftop1 = perm.groupby(['JI_OFFERED_TO_SEC_J_FW','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['JI_OFFERED_TO_SEC_J_FW'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['JI_OFFERED_TO_SEC_J_FW'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Job's offering status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JI_FW_LIVE_ON_PREMISES']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('JI_FW_LIVE_ON_PREMISES',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['JI_FW_LIVE_ON_PREMISES','countvar']]

dftop1 = perm.groupby(['JI_FW_LIVE_ON_PREMISES','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['JI_FW_LIVE_ON_PREMISES'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['JI_FW_LIVE_ON_PREMISES'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Job's premises status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'JI_LIVE_IN_DOMESTIC_SERVICE']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('JI_LIVE_IN_DOMESTIC_SERVICE',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['JI_LIVE_IN_DOMESTIC_SERVICE','countvar']]

dftop1 = perm.groupby(['JI_LIVE_IN_DOMESTIC_SERVICE','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['JI_LIVE_IN_DOMESTIC_SERVICE'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['JI_LIVE_IN_DOMESTIC_SERVICE'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Job's  live-in domestic service status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'RECR_INFO_PROFESSIONAL_OCC']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('RECR_INFO_PROFESSIONAL_OCC',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['RECR_INFO_PROFESSIONAL_OCC','countvar']]

dftop1 = perm.groupby(['RECR_INFO_PROFESSIONAL_OCC','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['RECR_INFO_PROFESSIONAL_OCC'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['RECR_INFO_PROFESSIONAL_OCC'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= " professional occupation and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'RECR_INFO_COLL_UNIV_TEACHER']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('RECR_INFO_COLL_UNIV_TEACHER',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['RECR_INFO_COLL_UNIV_TEACHER','countvar']]

dftop1 = perm.groupby(['RECR_INFO_COLL_UNIV_TEACHER','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['RECR_INFO_COLL_UNIV_TEACHER'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['RECR_INFO_COLL_UNIV_TEACHER'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= " University teacher or not and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'RECR_INFO_EMPLOYER_REC_PAYMENT']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'FW_INFO_BIRTH_COUNTRY']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('FW_INFO_BIRTH_COUNTRY',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['FW_INFO_BIRTH_COUNTRY','countvar']][0:6]

dftop1 = perm.groupby(['FW_INFO_BIRTH_COUNTRY','CASE_STATUS'],as_index=False).count()

dftop1=dftop1[dftop1.FW_INFO_BIRTH_COUNTRY.isin(dftop.FW_INFO_BIRTH_COUNTRY)]

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['FW_INFO_BIRTH_COUNTRY'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['FW_INFO_BIRTH_COUNTRY'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Top apply countries and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
perm["RESPONSE"]=perm["CASE_STATUS"].apply(lambda x:x=="Certified")

country_response19=perm.groupby("FW_INFO_BIRTH_COUNTRY").mean()[['RESPONSE']].sort_values(by='RESPONSE',ascending=False)

def addlist(df):

   

    if df == 1:

        return "LEVEL1"

   

    elif df >= 0.9:

        return "LEVEL2"

   

    elif df >= 0.8:

        return "LEVEL3"

   

    elif df > 0.7:

        return "LEVEL4"

    

    else:

        return "LEVEL5"

        

country_response19["EMPLOYER_LEVEL"] = country_response19["RESPONSE"].apply(lambda x :addlist(x))

country_response19 = country_response19.reset_index()

country_response19.head(5)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'CLASS_OF_ADMISSION']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('CLASS_OF_ADMISSION',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['CLASS_OF_ADMISSION','countvar']][0:6]

dftop1 = perm.groupby(['CLASS_OF_ADMISSION','CASE_STATUS'],as_index=False).count()

dftop1=dftop1[dftop1.CLASS_OF_ADMISSION.isin(dftop.CLASS_OF_ADMISSION)]

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['CLASS_OF_ADMISSION'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['CLASS_OF_ADMISSION'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Top former admissions and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
admission_response19=perm.groupby("CLASS_OF_ADMISSION").mean()[['RESPONSE']].sort_values(by='RESPONSE',ascending=False)

def addlist(df):

   

    if df == 1:

        return "LEVEL1"

   

    elif df >= 0.9:

        return "LEVEL2"

   

    elif df >= 0.8:

        return "LEVEL3"

   

    elif df > 0.7:

        return "LEVEL4"

    

    else:

        return "LEVEL5"

        

admission_response19["EMPLOYER_LEVEL_2"] = admission_response19["RESPONSE"].apply(lambda x :addlist(x))

admission_response19 = admission_response19.reset_index()

admission_response19.head(10)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'FOREIGN_WORKER_INFO_EDUCATION']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('FOREIGN_WORKER_INFO_EDUCATION',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['FOREIGN_WORKER_INFO_EDUCATION','countvar']]

dftop1 = perm.groupby(['FOREIGN_WORKER_INFO_EDUCATION','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['FOREIGN_WORKER_INFO_EDUCATION'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['FOREIGN_WORKER_INFO_EDUCATION'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Education level and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
table = sm.stats.Table.from_data(perm[['CASE_STATUS', 'FW_INFO_TRAINING_COMP']])

print('\nPvalue of Chi2-test is {}'.format(table.test_nominal_association().pvalue))

table.resid_pearson
dftop = perm.groupby('FW_INFO_TRAINING_COMP',as_index=False).count()

dftop = dftop.sort_values('countvar',ascending= False)[['FW_INFO_TRAINING_COMP','countvar']]

dftop1 = perm.groupby(['FW_INFO_TRAINING_COMP','CASE_STATUS'],as_index=False).count()

t1 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['FW_INFO_TRAINING_COMP'].values,y=dftop1[dftop1.CASE_STATUS == 'Certified'].sort_values('countvar',ascending= False)['countvar'].values,name='Certified')

t2 = go.Bar(x=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['FW_INFO_TRAINING_COMP'].values,y=dftop1[dftop1.CASE_STATUS == 'Denied'].sort_values('countvar',ascending= False)['countvar'].values,name='Denied')



data = [t1,t2]

layout = go.Layout(dict(title= "Training complete status and its Case status",yaxis=dict(title="Num of applications")),

    barmode='stack'

)



fig =go.Figure(data,layout)

iplot(fig)
perm20 = pd.read_csv('../input/h1bfy2019csv/PERM_2020.csv', 

                        engine='python').dropna(how='all')

perm20=perm20.fillna("Unknown")

perm20.head(10)
select_columns=[

    'CASE_STATUS',

    'REFILE',

    'WORKSITE_STATE',

    'FW_OWNERSHIP_INTEREST',

    'PW_SKILL_LEVEL',

    'MINIMUM_EDUCATION',

    'REQUIRED_TRAINING',

    'ACCEPT_ALT_FIELD_OF_STUDY',

    'JOB_OPP_REQUIREMENTS_NORMAL',

    'FOREIGN_LANGUAGE_REQUIRED',

    'PROFESSIONAL_OCCUPATION',

    'APP_FOR_COLLEGE_U_TEACHER',

    'FOREIGN_WORKER_BIRTH_COUNTRY',

    'CLASS_OF_ADMISSION',

    'FOREIGN_WORKER_TRAINING_COMP'

]
perm20 = perm20[(perm20['CASE_STATUS'].str.upper() == 'CERTIFIED') | \

                               (perm20['CASE_STATUS'].str.upper() == 'DENIED')]

perm20["WORKSITE_STATE"]=perm20["WORKSITE_STATE"].map(us_state_abbrev)

perm_sub=perm20[select_columns]

perm_sub.head(10)




cate_column_name = [   

    'REFILE',

    'WORKSITE_STATE',

    'FW_OWNERSHIP_INTEREST',

    'PW_SKILL_LEVEL',

    'MINIMUM_EDUCATION',

    'REQUIRED_TRAINING',

    'ACCEPT_ALT_FIELD_OF_STUDY',

    'JOB_OPP_REQUIREMENTS_NORMAL',

    'FOREIGN_LANGUAGE_REQUIRED',

    'PROFESSIONAL_OCCUPATION',

    'APP_FOR_COLLEGE_U_TEACHER',

    'FOREIGN_WORKER_BIRTH_COUNTRY',

    'CLASS_OF_ADMISSION',

    'FOREIGN_WORKER_TRAINING_COMP']

data = pd.get_dummies(perm_sub, columns = cate_column_name)
train, test = train_test_split(data,test_size=0.2,random_state=0)

train = train.reset_index(drop=True)

test = test.reset_index(drop=True)



# Get X_train, X_test, y_train, y_test

X_train = train.drop(['CASE_STATUS'],axis=1)

X_test = test.drop(['CASE_STATUS'],axis=1)

y_train = train['CASE_STATUS']

y_test = test['CASE_STATUS']



oversample = SMOTE()

X_train_res, y_train_res = oversample.fit_resample(X_train, y_train)

X_test_res, y_test_res = oversample.fit_resample(X_test, y_test)
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge,RidgeCV,Lasso, LassoCV

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,cross_validate

model=LogisticRegression()

model.fit(X_train_res,y_train_res)
predict=model.predict(X_test)

print(accuracy_score(y_test, predict))

confusion_matrix(y_test, predict)
from sklearn.metrics import roc_curve, auc 

predict_prob = model.predict_proba(X_test)

fpr,tpr,threshold = roc_curve(y_test,predict_prob[:,1],pos_label="Denied")

roc_auc = auc(fpr,tpr)

roc_auc
import matplotlib.pyplot as plt



plt.figure()

lw = 2

plt.figure(figsize=(10,10))

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show()
model=LogisticRegression()

model.fit(X_train,y_train)
predict2=model.predict(X_test)

print(accuracy_score(y_test, predict2))

confusion_matrix(y_test, predict2)
predict2_prob = model.predict_proba(X_test)

fpr,tpr,threshold = roc_curve(y_test,predict2_prob[:,1],pos_label="Denied")

roc_auc = auc(fpr,tpr)

roc_auc


plt.figure()

lw = 2

plt.figure(figsize=(10,10))

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show()
y_train_n = y_train_res

x_train_n = X_train_res

y_test_n = y_test

x_test_n = X_test
rfc = RandomForestClassifier(n_estimators=100,bootstrap=True,criterion='gini',oob_score=True)

rfc.fit(x_train_n,y_train_n)

pred_rfc  =rfc.predict(X_test)

print(accuracy_score(y_test,pred_rfc))

confusion_matrix(y_test, pred_rfc)
rfc = RandomForestClassifier(n_estimators=100,bootstrap=True,criterion='gini',oob_score=True)

rfc.fit(X_train,y_train)
pred_rfc  =rfc.predict(X_test)

print(accuracy_score(y_test,pred_rfc))

confusion_matrix(y_test, pred_rfc)