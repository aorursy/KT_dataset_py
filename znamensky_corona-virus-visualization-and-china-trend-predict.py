import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import numpy as np

import matplotlib.markers as mmark

import matplotlib.lines as mlines



from sklearn.impute import SimpleImputer



from sklearn.svm import SVR

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error



from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

from matplotlib.ticker import MaxNLocator

from sklearn.metrics import mean_absolute_error

plt.style.use('seaborn')

data_main=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

#data_time_conf=pd.read_csv('time_series_covid_19_confirmed.csv')

#data_time_recov=pd.read_csv('time_series_covid_19_recovered.csv')

#data_time_deaths=pd.read_csv('time_series_covid_19_deaths.csv')
data_main.shape
#DateLine

print(data_main['ObservationDate'].min())

print(data_main['ObservationDate'].max())

#Countries

print('Uniq_countries',data_main['Country/Region'].nunique())

print('Uniq_prov (NAN_here)',data_main['Province/State'].nunique())
#data_time_conf.shape
#data_time_deaths[:1]
#Top 5 regions repeated in table

data_main['Country/Region'].value_counts()[:5]
#Top_20_confirmed

#Interesting despite some countries repeated often in table (like US,Australia,Canada)

# they are not in high-rate confirmed disease

#Reason could be a low rate of desease spread in this region

max_confirm_by_c=pd.DataFrame(data_main.groupby(['Country/Region'])['Confirmed'].max()).sort_values(by=['Confirmed'],ascending=False)[:20]

max_depths_by_c=pd.DataFrame(data_main.groupby(['Country/Region'])['Deaths'].max()).sort_values(by=['Deaths'],ascending=False)[:20]

max_recov_by_c=pd.DataFrame(data_main.groupby(['Country/Region'])['Recovered'].max()).sort_values(by=['Recovered'],ascending=False)[:20]



# Death_Rate

death_r=pd.concat([max_confirm_by_c,max_depths_by_c],axis=1)

death_r['Death_Rate']=((death_r['Deaths']/death_r['Confirmed'])*100)

death_r=death_r.sort_values(by=['Death_Rate'],ascending=False)



# Recov_Rate

recov_r=pd.concat([max_confirm_by_c,max_recov_by_c],axis=1)

recov_r['Recov_Rate']=((recov_r['Recovered']/recov_r['Confirmed'])*100)

recov_r=recov_r.sort_values(by=['Recov_Rate'],ascending=False)



# Plot the figure.



fig,ax=plt.subplots(1,2,figsize=(12,4))

#plt.figure(figsize=(12, 8))

max_confirm_by_c.plot(kind='bar',ax=ax[0])

max_depths_by_c.plot(kind='bar',ax=ax[1],cmap='gray')    

ax[0].set_title('Amount Confirmed')

ax[1].set_title('Amount Deaths')

#ax.set_xlabel('Country')



fig1,ax=plt.subplots(1,1,figsize=(12,4))

death_r['Death_Rate'].plot(color='tab:blue',kind='bar',alpha=1).set_title('Death_Rate,%')



fig2,ax=plt.subplots(1,1,figsize=(12,4))

recov_r['Recov_Rate'].plot(kind='bar').set_title('Recov_Rate,%')
# Top_Countries_Main_Trends (Confirmed, Deaths,Recovery)



for i in ['Mainland China','Italy','Iran','Spain','South Korea','Germany','France','US']:

    df_count=data_main.loc[(data_main['Country/Region']==i)]

    # group by date (sum by Province)

    countr_conf_gr=pd.DataFrame(df_count.groupby(['ObservationDate'])['Confirmed'].sum()).reset_index()

    countr_death_gr=pd.DataFrame(df_count.groupby(['ObservationDate'])['Deaths'].sum()).reset_index()

    countr_recov_gr=pd.DataFrame(df_count.groupby(['ObservationDate'])['Recovered'].sum()).reset_index()

    

    countr_values=countr_conf_gr.merge(countr_death_gr,left_on='ObservationDate', right_on='ObservationDate')

    countr_values=countr_recov_gr.merge(countr_values,left_on='ObservationDate', right_on='ObservationDate')

    

    

    # First_Confirmed_Case

    fst_case_conf=countr_values.loc[countr_values['Confirmed']>0][:1].index[:1]

    fst_case_death=countr_values.loc[countr_values['Deaths']>0][:1].index[:1]

#Chart



    df1=countr_values

    df2=countr_values

    df3=countr_values

    col1='Confirmed'

    col2='Deaths'

    col3='Recovered'

    

    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']      



    #figure,subplot

    fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi= 70)

    ax.set_title(str(i)+' '+'Disease Trend', fontsize=11)





    #Confirmed

    ax.fill_between(df1['ObservationDate'], y1=df1[col1],

                        label='Confirmed', alpha=0.5, color=mycolors[1],

                        y2=0, linewidth=2,zorder=1)

    plt.xticks(df1['ObservationDate'], fontsize=9, rotation=90)

    

    markers_on = fst_case_conf

    plt.plot(df1['ObservationDate'],df1['Confirmed'],markevery=markers_on,marker=('>'),markersize=18,

            markerfacecolor='orange',zorder=5,label='Confirmed')



    #plt.axvline('03/04/2020',color=mycolors[1],linewidth=1,linestyle="--")  #vertical line



    #Deaths

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis



    ax2.fill_between(df2['ObservationDate'], y1=df2[col2],

                        label='Deaths', alpha=0.5, color=mycolors[2],

                        y2=0, linewidth=2,zorder=1)

    markers_on1 = fst_case_death

    plt.plot(df2['ObservationDate'],df2['Deaths'],markevery=markers_on1,marker=('^'),markersize=18,

            markerfacecolor='red',color=mycolors[2])



    #Recovered



    ax.plot(df3['ObservationDate'],df3[col3],c='black',label='Recovered',

                alpha=2,zorder=5)



    # ask matplotlib for the plotted objects and their labels



    lines, labels = ax.get_legend_handles_labels()

    lines2, labels2 = ax2.get_legend_handles_labels()

    b=ax2.legend(lines + lines2, labels + labels2, loc='upper left')



    

    



    fig.tight_layout()  # otherwise the right y-label is slightly clipped



    #ax.grid()

    #Axes_name

    ax.set_xlabel("ObservationDate")

    ax.set_ylabel(r"Confirmed/Recovered cases")

    ax2.set_ylabel(r"Deaths")



    ax.set_ylim(0,df1[col1].max()+1000)

    #ax2.set_ylim(0,df2[col2].max()+1000)

    ax2.set_ylim(0,df1[col1].max()+1000)

    

    #Marker

    

      

    fst_conf_case = mlines.Line2D([], [], color='orange', marker='>', linestyle='None',

                          markersize=10, label='fst_conf_case')

    fst_death_case  = mlines.Line2D([], [], color='red', marker='^', linestyle='None',

                          markersize=10, label='fst_death_case')



    

    ax = plt.gca().add_artist(b)

    plt.legend(handles=[fst_conf_case, fst_death_case],loc='center left')

 

    plt.show()


#imputer = SimpleImputer(strategy='constant')



#data_main= pd.DataFrame(imputer.fit_transform(data_main), columns=data_main.columns)

data_main['Active_conf']=data_main['Confirmed']-(data_main['Deaths']+data_main['Recovered'])

#data_main['Active_conf']=data_main['Active_conf'].astype(int)

#data_main['ObservationDate']=pd.to_datetime(data_main['ObservationDate'])



country=data_main.loc[data_main['Country/Region']=='Mainland China']

country_gr=pd.DataFrame(country.groupby(['ObservationDate'])['Active_conf'].sum()).reset_index()
country_gr[:3]


#country=data_main.loc[data_main['Country/Region']=='Mainland China']



fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi= 70)

ax.set_title('China Active_Confirm (from 2020-02-18 decreasing)', fontsize=11)



plt.bar(country_gr["ObservationDate"],country_gr["Active_conf"])

plt.xticks(country["ObservationDate"], fontsize=10, rotation=90)

ax.xaxis.set_major_locator(MaxNLocator(nbins=16))

#plt.axvline('2020-02-18',color=mycolors[1],linewidth=1,linestyle="--")  #vertical line

plt.show()
#Date with max Active_Confirmed

max_act_conf=country.loc[country['Active_conf']==country['Active_conf'].max()][:1].index[:1]

b=country['ObservationDate'][max_act_conf]

b
#Active_confirmed_list

value=country_gr['Active_conf']

value[-5:]
plt.plot(country_gr['Active_conf'].values)

#plt.xticks(country_gr["ObservationDate"], fontsize=10, rotation=90)

#ax.xaxis.set_major_locator(MaxNLocator(nbins=16))

#Define number of past trend that will be used for future prediction

past=15

future=10



start=past

end=len(value)-future



pr=[]

for i in range(start,end+1):

    row_past_future=value[(i-past):(i+future)]

    pr.append(list(row_past_future))
#Name for columns

past_columns=[f'past_{i}' for i in range(past)]

future_columns=[f'future_{i}' for i in range(future)]
#New df with past trend and future

df=pd.DataFrame(pr,columns=(past_columns+future_columns))
# Train/Test df

X=df[past_columns][:-1]

y=df[future_columns][:-1]



#Validation df

X_val=df[past_columns][-1:]

y_val=df[future_columns][-1:]
from sklearn.linear_model import LinearRegression

LinReg=LinearRegression(normalize=True, fit_intercept=False)
LinReg.fit(X,y)


prediction=LinReg.predict(X_val)
prediction
np.sqrt(mean_squared_error(y_val,prediction))

plt.plot(prediction[0], label='prediction')

plt.plot(y_val.iloc[0], label='real')

plt.title('Validation_prediction_last_10_days')

plt.legend()
future_forcast = np.array([i for i in range(len(country_gr['ObservationDate'])+future)]).reshape(-1, 1)
import datetime

start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(future_forcast)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
#Last Columns for prediction

last_col=df.columns[-past:]

test=df[last_col][-1:]

#Prediction

X_train_=df[past_columns]

y_train_=df[future_columns]

X_test_=test



LinReg.fit(X_train_,y_train_)

prediction=LinReg.predict(X_test_)

prediction
#Last known data to compare with prediction

y_train_[-1:]
last_data=country_gr['Active_conf']

pred_data=prediction[0]

last_pred=list(last_data)+list(pred_data)

#dates

s=pd.DataFrame(future_forcast_dates)


#chart

fig, ax = plt.subplots(1, 1, figsize=(10,5), dpi= 70)

plt.plot(last_pred, label='prediction',color='red')

plt.plot(country_gr['Active_conf'], label='real')

plt.title('Prediction_China_Active_Confirmed_future_days')

plt.xticks(range(0,len(s.index)), s[0],fontsize=10, rotation=90)



ax.xaxis.set_major_locator(MaxNLocator(nbins=16))

plt.legend()