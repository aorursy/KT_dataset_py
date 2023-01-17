import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import Image

%matplotlib inline
data = pd.read_csv('../input/suicide-data/master.csv')
data[(data.country == 'Albania')&(data.year==1987)].sort_values('age')
data.age = data.age.replace('5-14 years','05-14 years')
data[(data.country == 'Albania')&(data.year==1987)].sort_values('age')
Image("../input/suicide-table/weights.png")
data.loc[data['age']=='05-14 years',"weights"] = 0.1729

data.loc[data['age']=='15-24 years',"weights"] = 0.1669

data.loc[data['age']=='25-34 years',"weights"] = 0.1554

data.loc[data['age']=='35-54 years',"weights"] = 0.2515

data.loc[data['age']=='55-74 years',"weights"] = 0.1344

data.loc[data['age']=='75+ years',"weights"] = 0.03065

data.head()
data[data['weights'].isnull()]
data['suicides/100k pop (adjusted)']=data['suicides/100k pop']*data['weights']

data.head()
data[(data['country']=='Lithuania')&(data['year']==2016)&(data['sex']=='male')]['suicides/100k pop (adjusted)'] 
data[(data['country']=='Lithuania')&(data['year']==2016)&(data['sex']=='male')]['suicides/100k pop (adjusted)'].sum()
data[(data['country']=='Lithuania')&(data['year']==2016)&(data['sex']=='female')]['suicides/100k pop (adjusted)'].sum()
data_group = data.groupby(['country', 'year', 'sex'], as_index=False)
data_group
data_group.get_group(('Argentina', 2015, 'male'))
data_agg = data_group.agg({'suicides/100k pop (adjusted)':'sum'})

data_agg.head(10)
type(data_agg)
data_group_1 = data_agg.groupby(['country', 'year'], as_index=False)
data_agg_1 = data_group_1.agg({'suicides/100k pop (adjusted)': 'sum'})

data_agg_1.head(10)
year_spec = 2015

# sex_spec = 'male' 

# sex_spec = 'female' 

sex_spec = 'both male & female'  
if sex_spec in ('male', 'female'):

    data_specific = data_agg[(data_agg.year==year_spec)&(data_agg.sex==sex_spec)].sort_values('suicides/100k pop (adjusted)', ascending=False)    

else:

    data_specific = data_agg_1[data_agg_1.year==year_spec].sort_values('suicides/100k pop (adjusted)', ascending=False)
data_specific_20 = data_specific.head(20)

data_specific_20
plt.figure(figsize=(16,11))

plt.xticks(rotation=90)

plt.yticks(np.arange(0,70,5))

plt.title('Top 20 rank of age standardized '+sex_spec+' suicide rate per 100,000 people in '+str(year_spec))

sns.barplot(x='country', y='suicides/100k pop (adjusted)', data=data_specific_20)

sns.set(font_scale=3)
data_group_2 = data.groupby(['country', 'year', 'sex', 'gdp_per_capita ($)'], as_index=False)
data_agg_2 = data_group_2.agg({'suicides/100k pop (adjusted)':'sum'})

data_agg_2.head(10)
data_group_3 = data.groupby(['country', 'year', 'gdp_per_capita ($)'], as_index=False)
data_agg_3 = data_group_3.agg({'suicides/100k pop (adjusted)':'sum'})

data_agg_3.head(10)
countries =['Lithuania', 'Republic of Korea', 'Russian Federation', 'United States', 'Japan', 

            'Finland', 'Australia','Iceland', 'Austria']

country = countries[0]
# sex_spec = 'Male' 

# sex_spec = 'Female' 

sex_spec = 'Both Male & Female' 



if sex_spec in ('Male', 'Female'):

    data_specific = data_agg_2[(data_agg_2.country==country) & (data_agg_2.sex==sex_spec.lower())]

else:

    data_specific = data_agg_3[data_agg_3.country==country]
sns.set(style='white')
sns.lmplot(x='gdp_per_capita ($)', y='suicides/100k pop (adjusted)', data=data_specific, hue='year',

           fit_reg=False, palette='icefire', height=6, aspect=9/6)

plt.title('Suicide Rate Per 100k Population as a Function of GDP per capita for '

          +sex_spec+' in '+country)
Image("../input/gdppics/Korea.png")
Image("../input/gdppics/Russia.png")
Image("../input/gdppics/Japan.png")
Image("../input/gdppics/USA.png")
Image("../input/gdppics/Finland.png")
Image("../input/gdppics/Australia.png")
Image("../input/gdppics/Iceland.png")
Image("../input/gdppics/Austria.png")