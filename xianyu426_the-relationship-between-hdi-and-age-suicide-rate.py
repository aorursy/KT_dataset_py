# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/master.csv') # read the data

df.head()
df.groupby('year')['country'].count() # the year 2016 missing a lot of data 
df = df[df['year'] != 2016]
df_suicides_total = df.groupby('country')[['suicides_no']].sum().sort_values(by = ['suicides_no']) #group by country name and sort the values

country_name = df_suicides_total.index[-10:].tolist()

suicides_num = df_suicides_total['suicides_no'][-10:].tolist()



plt.style.use('seaborn-whitegrid')

fig, ax = plt.subplots(figsize = (10, 8))

ax.barh(country_name, suicides_num) 

ax.set(xlabel = 'The total number of suicides', ylabel = 'country')

plt.title('The first ten countries having hightest suicides number')

#plt.savefig('first_ten.png')

plt.show()
hue_order = ['75+ years', '55-74 years', '35-54 years', '25-34 years', '15-24 years', '5-14 years']

sns.relplot(x = 'year', y = 'suicides/100k pop',

           hue = 'age',hue_order = hue_order, 

           kind = 'line', data = df)

#plt.savefig('age_level.png')

plt.show()
def outlier(series): #define the outlier

    ''' this function is to calculate the boundary of outlier of suicides radio;

    input: series;

    output: set, boundary of outlier.'''

    array = np.asarray(series)

    Q1 = np.quantile(array, 0.25)

    Q3 = np.quantile(array, 0.75)

    QR = Q3 - Q1

    under_min = Q1 - 1.5*QR

    above_max = Q3 + 1.5*QR

    return (under_min, above_max)
# extract the outliers in years

df_abnormal_suicides = pd.DataFrame()

for i in range(1985, 2016):

    d = df[df['year'] == i]

    abnormal = outlier(d['suicides/100k pop'])

    df_abnormal = d[(d['suicides/100k pop'] < abnormal[0]) | (d['suicides/100k pop'] > abnormal[1])]

    df_abnormal_suicides = df_abnormal_suicides.append(df_abnormal)

df_abnormal_suicides.shape
abnormal_country = list(df_abnormal_suicides['country'].value_counts().index) #extract the country names having abnormal suicides radio

df_abnormal_country = df[df['country'].isin(abnormal_country)]

df_normal_country = df[~df['country'].isin(abnormal_country)] # dataframe having normal suicides radio

#add a column to mark outliers

df_abnormal_country['outlier'] = ['y'] *df_abnormal_country.shape[0] 

df_normal_country['outlier'] = ['n']*df_normal_country.shape[0]

df_outlier = df_abnormal_country.append(df_normal_country)
sns.relplot(x = 'year', y = 'suicides/100k pop',

           hue = 'age', col = 'outlier',

           kind = 'line', data = df_outlier)

plt.show()
col_order = ['75+ years', '55-74 years', '35-54 years', '25-34 years', '15-24 years', '5-14 years']

sns.relplot(x = 'HDI for year', y = 'suicides/100k pop',

           col = 'age', col_wrap = 3,

            col_order = col_order,data = df)

#plt.savefig('age_hdi.png')

plt.show()
sns.relplot(x = 'gdp_per_capita ($)', y = 'suicides/100k pop',

           col = 'age', col_wrap = 3,

            col_order = hue_order,data = df)

plt.show()
df_old_age = df[df['age'] == '75+ years'] #select the data of old people

df_old_age = df_old_age.reset_index(drop = True)

df_old_age['index'] = df_old_age.index

above_boundary = pd.DataFrame()

for i, t in enumerate(df_old_age.country):

    if (-250*df_old_age['HDI for year'][i] + 225 <df_old_age['suicides/100k pop'][i]): # the boundary line

        above_boundary = above_boundary.append(df_old_age.iloc[i, :])

above_boundary.shape
above_boundary['country'].value_counts() # take a look of country names above the boundary
above_boundary['index'] = above_boundary.index

cond = list(above_boundary.index)

under_boundary = df_old_age[~df_old_age['index'].isin(cond)]

under_boundary.shape
x = above_boundary['HDI for year']

y = above_boundary['suicides/100k pop']

slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

line = slope*x + intercept

def boundary(x):

    return -250*x +225

f, ax = plt.subplots(figsize = (10, 8))

ax.scatter(above_boundary['HDI for year'], above_boundary['suicides/100k pop'], color = 'steelblue')

ax.scatter(under_boundary['HDI for year'], under_boundary['suicides/100k pop'], color = 'skyblue')

l = np.arange(0.5, 1, 0.1)

bound = boundary(l)

plt.plot(above_boundary['HDI for year'], above_boundary['suicides/100k pop'], 'o',above_boundary['HDI for year'], line)

plt.figtext(0.35, 0.6, 'r_value:{}'.format(r_value), fontsize = 15)

plt.plot(l, bound, 'silver', lw = 1.5)

plt.title('The relationship of old people and the HDI of country')

ax.set(xlabel = 'HDI for year', ylabel = 'suicides/100k pop')

#plt.savefig('old_hdi.png')

plt.show()