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
import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)

df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_wgen1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_wgen2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_pgen1
df_pgen1['DataFrame'] = pd.to_datetime(df_pgen1['DATE_TIME']) 
df_pgen1['date'] = pd.to_datetime(df_pgen1['DATE_TIME']).dt.date
df_pgen1['day'] = pd.to_datetime(df_pgen1['DATE_TIME']).dt.day
df_pgen1['month'] = pd.to_datetime(df_pgen1['DATE_TIME']).dt.month
df_pgen1['year'] = pd.to_datetime(df_pgen1['DATE_TIME']).dt.year
df_pgen1

# Mean DAILY_YIELD
df_pgen1["DAILY_YIELD"].mean()
df_wgen1['DataFrame'] = pd.to_datetime(df_wgen1['DATE_TIME'])
df_wgen1['date'] = pd.to_datetime(df_wgen1['DATE_TIME']).dt.date
df_wgen1['day'] = pd.to_datetime(df_wgen1['DATE_TIME']).dt.day

 #the total irradiation per day

gkki=df_wgen1.groupby(['date']).count()
gkki
gkki['IRRADIATION']
#IRRADIATION per day different approach
df_pgen1['DATE_TIME']=pd.to_datetime(df_pgen1['DATE_TIME'],format='%d-%m-%Y %H:%M')
df_pgen1['DATE'] = df_pgen1['DATE_TIME'].apply(lambda x:x.date())
df_pgen1['TIME'] = df_pgen1['DATE_TIME'].apply(lambda x:x.time())
df_pgen1.info()
df_pgen1['DATE'] = pd.to_datetime(df_pgen1['DATE'],format = '%Y-%m-%d')
df_pgen1['HOUR'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.hour
df_pgen1['MINUTES'] = pd.to_datetime(df_pgen1['TIME'],format='%H:%M:%S').dt.minute
df_pgen1.info()
df_wgen1['IRRADIATION'].sum()
#Max AMBIENT_TEMPERATURE and MODULE_TEMPERATURE
df_wgen1 [['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].max()
#Number of inverters (SOURCE_KEY) per plant
df_pgen1['SOURCE_KEY'].unique()
len(df_pgen1['SOURCE_KEY'].unique())
#Inverter(SOURCE_KEY) that produced max AC_POWER

gkk = df_pgen1.groupby(['AC_POWER']).max()
gkk
gkk1=gkk.loc[1410.950000]
gkk1
gkk1.iloc[2]
#Inverter(SOURCE_KEY) that produced max DC_POWER

gkk2 = df_pgen1.groupby(['DC_POWER']).max()
gkk2
gkk3=gkk2.loc[14471.12500]
gkk3
gkk3.iloc[2]
#Ranking based on inverters(SOURCE_KEY) based on the DC/AC power they produce
gkka=df_pgen1.groupby(['date']).max()
gkka

gkkb=df_pgen1.groupby(['SOURCE_KEY']).count()
gkkb
gkkb.iloc[:,[2,3]].rank()
#Missing data
df_pgen1.isnull() 
df_wgen1.isnull() 
import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_wgen1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_wgen2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
df_pgen2
df_pgen2['DataFrame'] = pd.to_datetime(df_pgen2['DATE_TIME']) 
df_pgen2['date'] = pd.to_datetime(df_pgen2['DATE_TIME']).dt.date
df_pgen2['day'] = pd.to_datetime(df_pgen2['DATE_TIME']).dt.day
df_pgen2['month'] = pd.to_datetime(df_pgen2['DATE_TIME']).dt.month
df_pgen2['year'] = pd.to_datetime(df_pgen2['DATE_TIME']).dt.year
df_pgen2
# Mean DAILY_YIELD
df_pgen2["DAILY_YIELD"].mean()
df_wgen2['DataFrame'] = pd.to_datetime(df_wgen2['DATE_TIME'])
df_wgen2['date'] = pd.to_datetime(df_wgen2['DATE_TIME']).dt.date
df_wgen2['day'] = pd.to_datetime(df_wgen2['DATE_TIME']).dt.day
#the total irradiation per day

gkki2=df_wgen2.groupby(['date']).count()
gkki2
gkki2['IRRADIATION']
#Max AMBIENT_TEMPERATURE and MODULE_TEMPERATURE
df_wgen2 [['AMBIENT_TEMPERATURE','MODULE_TEMPERATURE']].max()
#Number of inverters (SOURCE_KEY) per plant
df_pgen2['SOURCE_KEY'].unique()
len(df_pgen2['SOURCE_KEY'].unique())
#Inverter(SOURCE_KEY) that produced max AC_POWER

gkkx = df_pgen2.groupby(['AC_POWER']).max()
gkkx
gkky=gkkx.loc[1385.420000]
gkky
gkky.iloc[2] #Inverter tht produced max AC_POWER
#Inverter(SOURCE_KEY) that produced max DC_POWER

gkku = df_pgen2.groupby(['DC_POWER']).max()
gkku
df_pgen2['DC_POWER'].max()  #Have to do this function. Full float not given in the table hence error occurs if we copy and paste
gkkt = gkku.loc[1420.9333333333332]
gkkt
gkkt.iloc[2] #Inverter that produced max DC_POWER
#Ranking based on inverters(SOURCE_KEY) based on the DC/AC power they produce
gkkh=df_pgen2.groupby(['date']).max()
gkkh
gkkj=df_pgen2.groupby(['SOURCE_KEY']).count()
gkkj
gkkj.iloc[:,[2,3]].rank()
#Missing data
df_pgen2.isnull() 
df_wgen2.isnull() 
df_pgen1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df_wgen1=pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')    #Task 1 - Loading csv files 
df_pgen2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')
df_wgen2=pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')
import os
os.getcwd()  #For obatining exact location of data and working
type(df_pgen1)  #Exploring the types of data given
type(df_wgen1)
df_pgen1.head() #For obtaining the first few rows(By default 5 if number is not specified)
df_pgen1.head(15) #For obtaining first fifteen rows
df_pgen1.tail() #For obtaining the last few rows(By default 5 if number is not specified)
df_pgen1.tail(20) #For obtaining last twenty rows
df_pgen1['DATE_TIME'].unique() #To obtain unique values in a particular column
len((df_pgen1['DATE_TIME']).unique()) # The len command helps to specify the number of strings we want in a specific column
df_pgen1['PLANT_ID'] #To obtain one particular column in a dataframe
df_pgen1['DC_POWER'].max() #To obtain the maximum value in a particular column
df_pgen1['AC_POWER'].min() #To obtain the minimum value in a particular column
df_pgen1.info() #Provides information on a selected dataframe (Such as: dtype,memory usage,etc)
df_pgen1.describe() #Provides a full description of selected dataframe or column (Includes the mean,standard deviation,max,min,percentile)
df_pgen1.shape # Shows number of rows and columns in a dataframe
df_pgen1.dtypes #Shows all the different data types in a dataframe
categorical = df_pgen1.dtypes[df_pgen1.dtypes == "object"].index  #Shows type of index and datatype
print(categorical)
sorted(df_pgen1["DATE_TIME"])[20:25] #To help sort data in a column with a specified number of columns
df_wgen1["IRRADIATION"].sum() #To  find total or sum of a column
df_wgen1["IRRADIATION"].mean() #To find mean of data
df_pgen1 [['DATE_TIME','DAILY_YIELD','TOTAL_YIELD']]  #To select multiple columns
df_pgen1.isnull() #To find missing data in a dataframe (output-All false)
df_pgen1.notnull() #To find data in a dataframe (output-All true)
df_pgen1