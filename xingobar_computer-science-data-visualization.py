# http://stackoverflow.com/questions/22233488/pandas-drop-a-level-from-a-multi-level-column-index

# http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html

# http://stackoverflow.com/questions/34233347/pandas-plot-how-to-control-the-bar-width-and-the-gaps

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import colorsys

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
historical = pd.read_csv('../input/historical.csv')

pass_ = pd.read_csv('../input/pass_06_13.csv')
historical.head()
historical.describe()
pass_.head()
print(pass_.columns.tolist())
total = pass_[['year','black','hispanic','white', 'asian']].groupby(['year']).agg(['sum'])

total = total[['black','hispanic','white','asian']].reset_index()

total.columns = total.columns.droplevel()

total.columns = ['year','black','hispanic','white','asian']
total.plot.bar(x = 'year',figsize=(10,6),linewidth=0.75,width=0.7)

plt.title('Participants in the AP Computer Science Exam by Race,1999-2013')

# you can use the width parameter to set the gap
columns = ['year','black_passed', 'hispanic_passed',  'white_passed', 'asian_passed']

pass_copy = pass_[['year','black_passed', 'hispanic_passed',  'white_passed', 'asian_passed']].copy()

for col in columns:

    pass_copy[col] = pd.to_numeric(pass_copy[col],errors='coerce') ## becuase data have nonumeric



pass_copy = pass_copy[['year','black_passed', 'hispanic_passed',  'white_passed', 'asian_passed']].fillna(0)

pass_rate = pass_copy[['year','black_passed', 'hispanic_passed',  'white_passed', 'asian_passed']].groupby(['year']).agg(['sum'])

pass_rate = pass_rate[['black_passed', 'hispanic_passed',  'white_passed', 'asian_passed']].reset_index()

pass_rate.columns = pass_rate.columns.droplevel()

pass_rate.columns = ['year','black','hispanic','white','asian']
pass_rate.plot.bar(x='year',figsize=(10,6),linewidth=0.75,width=0.7)

plt.title('Pass Rate')
gender_pass = pass_[['year','black_male','black_female','white_male',

                     'white_female','hispanic_male','hispanic_female',

                     'asian_male','asian_female']].groupby(['year']).agg(['sum'])

gender_pass = gender_pass.reset_index()

gender_pass.columns = gender_pass.columns.droplevel()

gender_pass.columns = ['year','black_male','black_female','white_male',

                     'white_female','hispanic_male','hispanic_female',

                     'asian_male','asian_female']

gender_pass.plot.bar(x='year',figsize=(10,6),linewidth=0.8)

plt.title('Male vs Female')
historical.info()
historical['1999'] = pd.to_numeric(historical['1999'],errors='coerce')

historical.drop(['Pop'],axis=1,inplace=True)
hm = historical.set_index('state')

fig,ax = plt.subplots(figsize=(10,6))

sns.heatmap(hm,ax=ax)

plt.title('exam taken per state per year')