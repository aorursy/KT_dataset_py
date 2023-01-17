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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings 
warnings.simplefilter('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data_raw = pd.read_csv('/kaggle/input/obesity-among-adults-by-country-19752016/data.csv')
data_raw.head()
data_raw.shape
data_raw.columns
data_raw = data_raw.drop([0,1,2],axis=0)
data_raw.head()
data_raw=data_raw.reset_index(drop=True)
data_raw.head()
data_raw = data_raw.rename(columns={"Unnamed: 0":"Country"})
data_raw.head()
data_raw_melt=data_raw.melt('Country',var_name="Year",value_name="Obesity(%)")
data_raw_melt.head()
data_raw_melt[['Year','Sex']]=data_raw_melt['Year'].str.split('.',expand=True)
data_raw_melt.head()
data_raw_melt = data_raw_melt.sort_values(by=['Country','Year'])
data_raw_melt['Sex']=data_raw_melt['Sex'].map({None:'Both sexes','1':'Male','2':'Female'})
data_raw_melt.head()
data_splitsex = data_raw_melt[data_raw_melt['Sex']!='Both sexes']
data_splitsex.head()
data_splitsex['Avg Obesity(%)'] = data_splitsex['Obesity(%)'].str.extract(pat = '(\d*.\d*)',expand=True)
data_splitsex.head()
data_splitsex.info()
data_splitsex.drop(data_splitsex[data_splitsex['Obesity(%)']=='No data'].index, inplace=True)
data_splitsex['Avg Obesity(%)']=pd.to_numeric(data_splitsex['Avg Obesity(%)'],downcast='float')
data_splitsex.head()
data_splitsex.info()

data_splitsex.drop('Obesity(%)',axis=1,inplace=True)
data_splitsex.head()
df_male=data_splitsex[data_splitsex['Sex']=='Male']
df_male=df_male.pivot(index='Year', columns='Country', values='Avg Obesity(%)')
df_male.head()
df_female=data_splitsex[data_splitsex['Sex']=='Female']
df_female=df_female.pivot(index='Year',columns='Country',values='Avg Obesity(%)')
df_female.head()
df_SEA_Male=df_male.loc[:,['Indonesia','Singapore','Philippines','Myanmar','Thailand','Viet Nam','Malaysia','Cambodia','Brunei Darussalam']]
df_SEA_Male.head()
df_SEA_Male.plot(kind ='line',figsize=(18,5))
plt.title('Male Obesity Percentage in South East Asia 1975-2016')
plt.ylabel('% Obesity')
plt.xlabel('Years')
plt.show()
df_SEA_Female=df_female.loc[:,['Indonesia','Singapore','Philippines','Myanmar','Thailand','Viet Nam','Malaysia','Cambodia','Brunei Darussalam']]
df_SEA_Female.head()
df_SEA_Female.plot(kind ='line',figsize=(18,5))
plt.title('Female Obesity Percentage in SEA 1975-2016')
plt.ylabel('% Obesity')
plt.xlabel('Years')
plt.show()
df_male_idn=df_male.loc[:,['Indonesia']]
df_male_idn=df_male_idn.rename(columns={'Indonesia':'Idn Male'})
df_female_idn=df_female.loc[:,['Indonesia']]
df_female_idn=df_female_idn.rename(columns={'Indonesia':'Idn Female'})
df_Idn=pd.concat([df_male_idn,df_female_idn],axis=1)
df_Idn.head()
df_Idn.plot(kind ='line')
plt.title('Male vs Female Obesity Percentage in Indonesia 1975-2016')
plt.ylabel('% Obesity')
plt.xlabel('Years')
plt.show()
import folium
url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/'
sea_geo = f'{url}/public/data/southeast-asia.geojson'
sea_filter=['Indonesia','Singapore','Philippines','Myanmar','Thailand','Viet Nam','Malaysia','Cambodia','Brunei Darussalam']
data_SEA=data_splitsex[data_splitsex.Country.isin(sea_filter)]
data_SEA_male2016=data_SEA[(data_SEA['Year']=='2016')&(data_SEA['Sex']=='Male')]
data_SEA_male2016
data_SEA_male2016 = data_SEA_male2016.replace({'Viet Nam':'Vietnam'})
data_SEA_male2016
m = folium.Map(location = [-2.21797,115.66283], zoom_start = 3.5)

folium.Choropleth(
    geo_data = sea_geo,
    data = data_SEA_male2016,
    columns = ['Country','Avg Obesity(%)'],
    key_on ='feature.properties.name',
    fill_color ='YlOrRd',
    fill_opacity =0.7,
    line_opacity =0.2,
    legend_name = 'Obesity Cases South East Asia 2016'
).add_to(m)
folium.LayerControl().add_to(m)
m
