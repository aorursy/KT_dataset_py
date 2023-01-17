import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Read in the data

INPUT_DIR = '/kaggle/input/m5-forecasting-accuracy'

cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')

stv = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')

ss = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')

sellp = pd.read_csv(f'{INPUT_DIR}/sell_prices.csv')
#List of all events

event_list=[i for i in cal.event_name_1.fillna(0).unique() if i != 0] 



#Extract all the days an event has in the span of 1916 days

day_event_list=[cal[cal.event_name_1==i].d.tolist() for i in event_list]



#Create the Event_df dataframe which we will use throughout the notebook

event_df=pd.DataFrame({'Event Name' : event_list, 'Event day':day_event_list})

restricted_day= set(['d_'+ str(i) for i in np.arange(1916,1970)])

quantity=[]



for i in day_event_list:

    # Making sure that we exclude all the days thats are not in the training set

    clean_i=list(set(i)-restricted_day)

    temp=stv[clean_i].sum().sum() #Adding columns and then rows

    quantity.append(temp)



event_df['Quantity']=quantity

event_df
#Top 2 and bottom 2

a=event_df.sort_values('Quantity',ascending=False)[['Event Name','Quantity']].head(2)

b=event_df.sort_values('Quantity',ascending=False)[['Event Name','Quantity']].tail(3)

a.append(b)
#Sale quantity in some holidays compare to the average sale



average_quantity=stv.iloc[:,6:].sum().sum()/1913

name=['SuperBowl','Purim End','NewYear','Thanksgiving', 'All days average']

values=event_df[event_df['Event Name'].isin(name)].Quantity.tolist()

values.append(average_quantity)





plt.figure(figsize=(8,4))

plt.xlabel("Event")

plt.ylabel("Quantity")

plt.bar(name,values)

plt.title("Sale quantity in some holidays compare to the average sale")

plt.show()
#Created new columns for days before and after the event

new_cols=['3d_before','2d_before','1d_before','event','1d_after','2d_after','3d_after']

restricted_day= set(['d_'+ str(i) for i in np.arange(1916,2000)])



for index, name in enumerate(new_cols):

    quantity=[]

    for days in event_df['Event day']:

        # Get the name of the columns of the corresponding day of interest  

        corresponding_days= [int(i.split('_')[1])-3+index for i in days]

        corresponding_days_str= ['d_'+ str(i) for i in corresponding_days]

        

        #Make sure that it does not pass the 1916th

        corresponding_days_str=list(set(corresponding_days_str)-restricted_day)

        quantity_per_event=stv[corresponding_days_str].sum().sum()

        quantity.append(quantity_per_event)

    

    event_df[name]= quantity
event_df.head()
interesting_events=["SuperBowl","Mother's day","Father's day","NewYear"]

dummy=event_df[event_df['Event Name'].isin(interesting_events)].iloc[:,3:]

dummy=dummy.div(dummy.sum(axis=1), axis=0) #Normalize by row



x_axis_labels = ['3d_before','2d_before','1d_before','event','1d_after','2d_after','3d_after'] # labels for x-axis





# create seabvorn heatmap with required labels

sns.heatmap(dummy, xticklabels=x_axis_labels, yticklabels=interesting_events,cmap="YlGnBu")

plt.show()