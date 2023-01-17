!pip install pandas_profiling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))      

bel = pd.read_csv('/kaggle/input/beliefs-about-masks-among-young-adults/MaskBeliefs.csv', parse_dates=['Timestamp'])

from pandas_profiling import ProfileReport
profile = ProfileReport(bel, title="Young Adults belief - Profiling Report")
profile.to_widgets()
profile.to_notebook_iframe()
bel['Gender'].fillna('Other',inplace=True)
bel['Gender'] = bel['Gender'].map({'Male':1,'Female':2,'Other':3})
bel['ResidentialElder'] = bel['ResidentialElder'].map({'Yes':1,'No':0})
bel['InteractedElder'] = bel['InteractedElder'].map({'Yes':1,'No':0})
bel['PreventSpread'] = bel['PreventSpread'].map({'Yes':1,'No':0})
bel['Public'] = bel['Public'].map({'Yes':1,'No':0})
bel['Boarding'] = bel['Boarding'].map({'Boarding':1,'Day':0})
bel['Reason_number'] = bel['Reason'].map({'To protect yourself AND others':1,'To protect other people':2, 'Because you are required to':3, 'To protect yourself':4, 'To protect others but also because I\'m required to':5})
bel['Restaurant'].fillna(0,inplace=True)

bel[bel['Age']<13]
bel['Reason'].value_counts()
bel.groupby(['PreventSpread','Age']).agg('PreventSpread').count()
bel.groupby(['Boarding','PreventSpread']).agg('PreventSpread').count()
bel.groupby(['PreventSpread','Gender']).agg('PreventSpread').count()
bel.groupby(['PreventSpread','Public']).agg('PreventSpread').count()
print(bel.groupby(['PreventSpread','Public','ResidentialElder']).agg('PreventSpread').count())
bel.groupby(['PreventSpread','Public','InteractedElder']).agg('PreventSpread').count()
