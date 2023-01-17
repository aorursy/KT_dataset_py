import pandas as pd

import matplotlib.pyplot as plt

df=pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

df
#find india's column

start_count=0

for i in df['Country/Region']:

    start_count+=1

    if(i=='India'):

        break

df['Country/Region'].loc[start_count:]        

        
end_count=0

for i in df['Country/Region'].loc[start_count:]:

    end_count+=1

    if(i!='India'):

        break

end_count=(start_count)+end_count-2

india=df['Country/Region'].loc[start_count:end_count]

india

#found indices for Indian Data
end_count-start_count
#indian confirmed cases vs fatalities

datex=df['Date'][start_count:end_count]

conf=df['ConfirmedCases'][start_count:end_count]

fatal=df['Fatalities'][start_count:end_count]



import numpy as np

datex=np.array(datex)

plt.plot(datex,conf,'r')

plt.plot(datex,fatal,'b')

plt.show()