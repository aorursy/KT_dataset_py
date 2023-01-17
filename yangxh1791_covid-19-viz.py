%matplotlib inline

import pandas as pd
!curl -L -O https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
!curl -L -O https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv
df = pd.read_csv('/kaggle/working/time_series_covid19_confirmed_global.csv')

death = pd.read_csv('/kaggle/working/time_series_covid19_deaths_global.csv')
a = df[df['Country/Region']=='China'].reset_index(drop = True).iloc[:,4:].sum()

b = death[death['Country/Region']=='China'].reset_index(drop = True).iloc[:,4:].sum()

a.index =pd.to_datetime(a.index)

b.index =pd.to_datetime(b.index)

China = pd.DataFrame({'Confirmed':a,'Deaths':b})

China.plot(title="Covid-19, CHINA",grid=True)
a =df[df['Country/Region'].str.contains('Korea')].reset_index(drop = True).iloc[:,4:].T.reset_index()

b =death[death['Country/Region'].str.contains('Korea')].reset_index(drop = True).iloc[:,4:].T.reset_index(drop=True)

a.columns = ['date','Confirmed']

a['date']=pd.to_datetime(a['date'])

Korea = a.set_index('date')

Korea['Deaths']=b.values

Korea.plot(title="Covid-19, S.Korea",grid=True)
a =df[df['Country/Region']=='US'].reset_index(drop = True).iloc[:,4:].T.reset_index()

b =death[death['Country/Region']=='US'].reset_index(drop = True).iloc[:,4:].T.reset_index(drop=True)

a.columns = ['date','Confirmed']

a['date']=pd.to_datetime(a['date'])

US = a.set_index('date')

US['Deaths']=b.values

US.plot(title="Covid-19, US",grid=True)


a =df[df['Country/Region']=='Italy'].reset_index(drop = True).iloc[:,4:].T.reset_index()

b =death[death['Country/Region']=='Italy'].reset_index(drop = True).iloc[:,4:].T.reset_index(drop=True)

a.columns = ['date','Confirmed']

a['date']=pd.to_datetime(a['date'])

Italy = a.set_index('date')

Italy['Deaths']=b.values

Italy.plot(title="Covid-19, Italy",grid=True)

combined = pd.concat([China.Confirmed, US.Confirmed,Italy.Confirmed,Korea.Confirmed], axis=1)

combined.columns = ['China','US','Italy','Korea']

combined.plot(title="Covid-19 Confirmed cases",grid=True)

combined.plot(title="Covid-19 Confirmed cases",grid=True,logy=True)
Death_combined = pd.concat([China.Deaths, US.Deaths,Italy.Deaths,Korea.Deaths], axis=1)

Death_combined.columns = ['China','US','Italy','Korea']

Death_combined.plot(title="Covid-19 Deaths",grid=True)

Death_combined.plot(title="Covid-19 Deaths",grid=True,logy=True)
!rm /kaggle/working/*.csv