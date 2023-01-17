import pandas as pd #pandas - for data manipulation
new_data = pd.read_csv('/kaggle/input/wildfire-satellite-data/fire_nrt_M6_156000.csv') #load new data (June 2020->present)
old_data = pd.read_csv('/kaggle/input/wildfire-satellite-data/fire_archive_M6_156000.csv') #load old data (Sep 2010->June 2020)
data = pd.concat([old_data.drop('type',axis=1), new_data]) #concatenate old and new data
data['satellite'] = data['satellite'].map({'Terra':0,'Aqua':1})
data['daynight'] = data['daynight'].map({'D':0,'N':1})
data = data.sample(frac=0.1)
data = data.reset_index().drop("index", axis=1)
data.rename(columns={"acq_date":"Date"}, inplace=True)
data['month'] = data['Date'].apply(lambda x:int(x.split('-')[1]))
data.drop('instrument', axis=1, inplace=True)
wei = pd.read_excel('/kaggle/input/weekly-economic-index-wei-federal-reserve-bank/Weekly Economic Index.xlsx')
wei.drop('WEI as of 7/28/2020',axis=1,inplace=True)
wei = wei.set_index("Date")
wei.head()
from tqdm import tqdm
from dateutil import parser
for index in tqdm(range(len(data))):
    fire_date = (data["Date"][index]) 
    fire_date = parser.parse(fire_date)
    min_wei_date_value = wei.iloc[wei.index.get_loc(fire_date,method='nearest')]["WEI"]
    data.loc[index, "WEI Value"] = min_wei_date_value
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.set_style('whitegrid')
sns.scatterplot(data['longitude'],data['latitude'], alpha=0.1, color='#35c279')
plt.axis('off')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.set_style('whitegrid')
sns.scatterplot(data['longitude'],data['latitude'], alpha=0.1, hue = data['confidence'],
                palette = "magma", s=100)
plt.axis('off')
plt.show()
copy3 = data
copy3 = copy3[(copy3['latitude']<= 55) & (copy3['latitude'] >= 36)]
copy3 = copy3[(copy3['longitude']>= -135) & (copy3['longitude']<= -111)]

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import random as rand




plt.figure(figsize=(15,10))
sns.set_style('whitegrid')
sns.scatterplot(copy3['longitude'],copy3['latitude'], alpha=0.5, hue = copy3['daynight'],
                palette = "magma", size=copy3['confidence'])
plt.axis('off')
plt.show()
copy = data
copy = copy.sample(frac = 0.1)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
sns.set_style('whitegrid')
sns.scatterplot(data['frp'],data['confidence'], color='#35c279')
#plt.axis('off')
import seaborn as sns
import matplotlib.pyplot as plt
copy2 = data
copy2 = copy2.sample(frac = 0.1)
copy2 = copy2[(copy2['frp']<= 500)]
plt.figure(figsize=(20,5))
sns.set_style('whitegrid')
sns.scatterplot(copy2['frp'],copy2['confidence'], color='#35c279')
#plt.axis('off')
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,5))
sns.set_style('whitegrid')
sns.lineplot(data['acq_date'],data['confidence'], color='#35c279')
plt.axis('off')

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
sns.set_style('whitegrid')
sns.violinplot(x = data['month'],y = data['frp'], color ='#35c279',
                palette = "magma")
plt.axis('on')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

copy4 = data
copy4 = copy4[(copy4['latitude']<= 50) & (copy4['latitude'] >= 36)]
copy4 = copy4[(copy4['longitude']>= -123) & (copy4['longitude']<= -117)]
plt.figure(figsize=(15,10))
sns.set_style('whitegrid')
sns.violinplot(x = copy4['month'],y = copy4['frp'], color ='#35c279',
                palette = "magma")
plt.axis('on')
plt.show()
import matplotlib.pyplot as plt

copy4 = data
copy4 = copy4[(copy4['latitude']<= 50) & (copy4['latitude'] >= 36)]
copy4 = copy4[(copy4['longitude']>= -123) & (copy4['longitude']<= -117)]
plt.figure(figsize=(15,10))
sns.set_style('whitegrid')
sns.violinplot(x = copy4['month'],y = copy4['confidence'], color ='#35c279',
                palette = "magma")
plt.axis('on')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

copy4 = data
copy4 = copy4[(copy4['latitude']<= 42) & (copy4['latitude'] >= 32)]
copy4 = copy4[(copy4['longitude']>= -123) & (copy4['longitude']<= -113)]
plt.figure(figsize=(15,10))
sns.set_style('whitegrid')
sns.violinplot(x = copy4['month'],y = copy4['confidence'], color ='#35c279',
                palette = "magma")
plt.axis('on')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

copy5 = data
copy5 = copy5[(copy5['latitude']<= 42) & (copy5['latitude'] >= 32)]
copy5 = copy5[(copy5['longitude']>= -123) & (copy5['longitude']<= -113)]
plt.figure(figsize=(15,10))
sns.set_style('whitegrid')
sns.violinplot(x = copy5['month'],y = copy5['WEI Value'], color ='#35c279',
                palette = "magma")
plt.axis('on')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

copy5 = data
#copy5 = copy5[(copy5['latitude']<= 42) & (copy5['latitude'] >= 32)]
#copy5 = copy5[(copy5['longitude']>= -123) & (copy5['longitude']<= -113)]
plt.figure(figsize=(15,7))
sns.set_style('whitegrid')

sns.scatterplot(x = copy5['frp'],y = copy5['WEI Value'], color ='#35c279',
                palette = "magma")
plt.axis('on')
plt.show()
