# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
matplotlib.style.use('seaborn')

df_a_raw = pd.read_csv('../input/austin-weather/austin_weather.csv')
df_d_raw = pd.read_csv('../input/delhi-weather-data/testset.csv')
df_s_raw = pd.read_csv('../input/szeged-weather/weatherHistory.csv')
df_m_raw = pd.read_csv('../input/weather_madrid_lemd_1997_2015.csv/weather_madrid_LEMD_1997_2015.csv')
df_a_raw.columns = [c.lower() for c in list(df_a_raw.columns)]
df_d_raw.columns = [c.lower() for c in list(df_d_raw.columns)]
df_s_raw.columns = [c.lower() for c in list(df_s_raw.columns)]
df_m_raw.columns = [c.lower().strip() for c in list(df_m_raw.columns)]
df_a_1 = df_a_raw[['date','temphighf','tempavgf','templowf','precipitationsuminches','windhighmph','windavgmph','windgustmph','visibilityhighmiles','visibilityavgmiles','visibilitylowmiles']]

#Mean value imputing 
df_a_1.loc[df_a_1['precipitationsuminches'] == 'T', 'precipitationsuminches'] = 0
for c in list(df_a_1.columns)[1:]: 
    #print(c)
    newCol = pd.to_numeric(df_a_1[c], errors='coerce')
    df_a_1[c+'_clean']= newCol.fillna(np.mean(newCol))
# Unit conversion 
df_a_1['temphighc'] = (df_a_1['temphighf_clean']-32)*(5.0/9)
df_a_1['tempavgc'] = (df_a_1['tempavgf_clean']-32)*(5.0/9)
df_a_1['templowc'] = (df_a_1['templowf_clean']-32)*(5.0/9)
df_a_1['windhigk'] = df_a_1['windhighmph_clean']*1.609344
df_a_1['windavgk'] = df_a_1['windavgmph_clean']*1.609344
df_a_1['windgustk'] = df_a_1['windgustmph_clean']*1.609344
df_a_1['vishighk'] = df_a_1['visibilityhighmiles_clean']*1.609344
df_a_1['visavgk'] = df_a_1['visibilityavgmiles_clean']*1.609344
df_a_1['vislowk'] = df_a_1['visibilitylowmiles_clean']*1.609344
df_a_1['prcpcm'] = df_a_1['precipitationsuminches_clean']*2.54
df_a_2=df_a_1[['date','temphighc','tempavgc','templowc','windhigk','windavgk','windgustk','vishighk','visavgk','vislowk','prcpcm']]
df_a_2['location']='austin'
df_a_2.head()
df_a_2.to_csv('./out/austin_clean.csv')
df_m_1=df_m_raw[['cet', 'max temperaturec', 'mean temperaturec', 'min temperaturec', 'max visibilitykm', 'mean visibilitykm', 'min visibilitykm', 'max wind speedkm/h', 'mean wind speedkm/h', 'max gust speedkm/h', 'precipitationmm', 'winddirdegrees']]
for c in list(df_m_1.columns)[1:]: 
    #print(c)
    newCol = pd.to_numeric(df_m_1[c], errors='coerce')
    df_m_1[c+'_clean']= newCol.fillna(np.mean(newCol))
#data.rename(columns={'gdp':'log(gdp)'}, inplace=True)
df_m_1.rename(columns={
    'cet':'date',
'max temperaturec_clean': 'temphighc',
'mean temperaturec_clean': 'tempavgc',
'min temperaturec_clean': 'templowc',
'max wind speedkm/h_clean': 'windhigk',
'mean wind speedkm/h_clean': 'windavgk',
'max gust speedkm/h_clean': 'windgustk',
'max visibilitykm_clean': 'vishighk',
'mean visibilitykm_clean': 'visavgk',
'min visibilitykm_clean': 'vislowk',
'precipitationmm_clean': 'prcpcm',
'winddirdegrees_clean': 'winddir',
    },inplace=True)
            
df_m_1['date2'] = [dtm.datetime.strptime(c, "%Y-%m-%d") for c in df_m_1['date']] 
df_m_1['date'] = df_m_1['date2'].dt.strftime("%Y-%m-%d")
df_m_2=df_m_1[['date','temphighc','tempavgc','templowc','windhigk','windavgk','windgustk','vishighk','visavgk','vislowk','prcpcm','winddir']]
df_m_2['location']='madrid'
df_m_2.head()
df_m_2.to_csv('./out/madrid_clean.csv')
import datetime as dtm 
import time as tm 
df_d_1 = df_d_raw[['datetime_utc',' _tempm',' _precipm',' _vism',' _wdird',' _wgustm',' _wspdm']]
#df_d_1['date']=df_d_1['datetime_utc'].str.slice(0,4)
df_d_1['date']=df_d_1['datetime_utc'].str.slice(0,4)+'-'+df_d_1['datetime_utc'].str.slice(4,6)+'-'+df_d_1['datetime_utc'].str.slice(6,8)
grouped = df_d_1.groupby('date')
g1 = grouped[' _tempm'].agg([np.max, np.mean, np.min]).reset_index()
g1.columns=['date','temphighc','tempavgc','templowc']
g2 = grouped[' _vism'].agg([np.max, np.mean, np.min]).reset_index()
g2.columns=['date','vishighk','visavgk','vislowk']
g3 = grouped[' _wspdm'].agg([np.max, np.mean]).reset_index()
g3.columns=['date','windhigk','windavgk']
g4 = grouped[' _precipm'].agg([np.sum]).reset_index()
g4.columns=['date','prcpcm']
df_d_2 = pd.merge(g1,g2,on=['date'])
df_d_2 = pd.merge(df_d_2,g3,on=['date'])
df_d_2 = pd.merge(df_d_2,g4 ,on=['date'])
df_d_2['location']='delhi'
df_d_2.head()
df_d_2.to_csv('./out/delhi_clean.csv')
list(df_s_raw.columns)
df_s_1=df_s_raw[['formatted date',
 'temperature (c)',
 'wind speed (km/h)',
 'wind bearing (degrees)',
 'visibility (km)']]
#df_s_1['date']= df_s_1['formatted date'].str.slice(0,4)
df_s_1['date']=df_s_1['formatted date'].str.slice(0,10)
grouped = df_s_1.groupby('date')
g1 = grouped['temperature (c)'].agg([np.max, np.mean, np.min]).reset_index()
g1.columns=['date','temphighc','tempavgc','templowc']
g2 = grouped['visibility (km)'].agg([np.max, np.mean, np.min]).reset_index()
g2.columns=['date','vishighk','visavgk','vislowk']
g3 = grouped['wind speed (km/h)'].agg([np.max, np.mean]).reset_index()
g3.columns=['date','windhigk','windavgk']
df_s_2 = pd.merge(g1,g2,on=['date'])
df_s_2 = pd.merge(df_s_2,g3,on=['date'])
df_s_2['location']='szeged'
df_s_2.head()
df_s_2.to_csv('./out/szeged_clean.csv')
df_main = pd.DataFrame()
df_main = df_main.append(df_a_2)
print (df_main.shape)
df_main = df_main.append(df_m_2)
print (df_main.shape)
df_main = df_main.append(df_d_2)
print (df_main.shape)
df_main = df_main.append(df_s_2)
print (df_main.shape)
df_main.head()
df_main[df_main.location=='austin'].describe()
df_main[df_main.location=='delhi'].describe()
df_main[df_main.location=='madrid'].describe()
df_main[df_main.location=='szeged'].describe()
#Very rough outlier removal 
print(str(len(df_main)))
df_main= df_main[df_main.windavgk<50]
print(str(len(df_main)))
df_main= df_main[df_main.visavgk<50]
print(str(len(df_main)))
#df_main[['tempavgc', 'location']].hist(by='location', figsize=[16, 10])
df_main[['tempavgc', 'location']].hist(by='location', figsize=[16, 10],sharex=True)
#df_main[['windavgk', 'location']].hist(by='location', figsize=[16, 10])
df_main[['windavgk', 'location']].hist(by='location', figsize=[16, 10],sharex=True)
#df_main[['visavgk', 'location']].hist(by='location', figsize=[16, 10])
df_main[['visavgk', 'location']].hist(by='location', figsize=[16, 10],sharex=True)
df_main[['prcpcm', 'location']].hist(by='location', figsize=[16, 10])
#df_main[['prcpcm', 'location']].hist(by='location', figsize=[16, 10],sharex=True)