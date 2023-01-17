import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd #pandas - for data manipulation
import datetime as dt
from dateutil import parser
new_data = pd.read_csv('/kaggle/input/wildfire-satellite-data/fire_nrt_M6_156000.csv') #load new data (June 2020->present)
old_data = pd.read_csv('/kaggle/input/wildfire-satellite-data/fire_archive_M6_156000.csv') #load old data (Sep 2010->June 2020)
fire_data = pd.concat([old_data.drop('type',axis=1), new_data]) #concatenate old and new data
fire_data = fire_data.reset_index().drop('index',axis=1)
fire_data = fire_data[fire_data.satellite != "Aqua"]
fire_data = fire_data.sample(frac=0.1)
fire_data = fire_data.reset_index().drop("index", axis=1)

print(f"Shape of data: {fire_data.shape}")
fire_data.rename(columns={"acq_date":"Date"}, inplace=True)
fire_data["WEI Value"] = 0
fire_data['month'] = fire_data['Date'].apply(lambda x:int(x.split('-')[1]))
fire_data.head()
wei = pd.read_excel('/kaggle/input/weekly-economic-index-wei-federal-reserve-bank/Weekly Economic Index.xlsx')
wei.drop('WEI as of 7/28/2020',axis=1,inplace=True)
wei = wei.set_index("Date")
wei.head()
from tqdm import tqdm
for index in tqdm(range(len(fire_data))):
    fire_date = (fire_data["Date"][index]) 
    fire_date = parser.parse(fire_date)
    min_wei_date_value = wei.iloc[wei.index.get_loc(fire_date,method='nearest')]["WEI"]
    fire_data.loc[index, "WEI Value"] = min_wei_date_value

fire_data['daynight'] = fire_data['daynight'].map({'D':0,'N':1})
fire_data.drop('instrument', axis=1, inplace=True)
fire_data.head()
x = fire_data[['latitude','longitude','month','brightness','scan','track',
               'acq_time','bright_t31','daynight','frp', 'confidence']]
y = fire_data['WEI Value']
