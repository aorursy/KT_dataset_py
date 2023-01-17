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

fire = pd.read_csv('/kaggle/input/fire-incidents/Fire_Incidents.csv')
new_fire = fire.groupby('Incident Date')

fire_count = new_fire.count()

day_count = fire_count.groupby('Incident Number')

freq = day_count.count()

freq = freq['Exposure Number']

from matplotlib import pyplot as plt
ax = freq.plot(kind="bar")

ax.set_xlabel("Number of Incidents")

ax.set_ylabel("Number of Days with Incident Count")

ax.set_xticks(ax.get_xticks()[::10])
fire
fire.head()
total_day_count = len(new_fire.count())
df = new_fire.count()
total_incident_count = sum(df['Incident Number'])
total_incident_count
lam = total_incident_count/total_day_count

lam




barf = freq.reset_index()

barf = barf.rename(columns={"Incident Number": "Number of Fire Incidents", "Exposure Number": "Number of Days"})

check = 0



from scipy.stats import poisson

barf['probability_using_poisson'] = poisson.pmf(barf['Number of Fire Incidents'], lam)
barf['probability_using_poisson'][0]

poisson.pmf(1, lam)
poisson.pmf(np.array(barf['Number of Fire Incidents']), lam)
barf
barf['Actual_probability'] = barf['Number of Days']/total_day_count

barf
ax2 = barf.plot.bar(x='Number of Fire Incidents', y='probability_using_poisson', rot=0)

ax3 = plt.subplot(111)

ax3.bar(barf['Number of Fire Incidents'], barf['Actual_probability'], label='Actual Probability')

ax3.plot(barf['Number of Fire Incidents'], barf['probability_using_poisson'], c = 'r', label = 'Expected Probability')

ax3.set_xlim([20, 160])

ax3.legend()
barf['expected_cdf'] = barf['probability_using_poisson'].cumsum()

barf['actual_cdf'] = barf['Actual_probability'].cumsum()

barf
ax4 = plt.subplot(111)

ax4.bar(barf['Number of Fire Incidents'], barf['actual_cdf'], label='Actual Cumulative Probability')

ax4.plot(barf['Number of Fire Incidents'], barf['expected_cdf'], c = 'r', label = 'Expected Cumulative Probability')

ax4.set_xlim([20, 160])

ax4.legend(loc='best')