#Starting by resetting all variables previously defined in this jupyter notebook

%reset -f
import numpy as np

import seaborn as sns

from scipy import stats, integrate

import matplotlib.pyplot as plt

import statistics

import math

import pandas as pd

pd.options.display.max_rows==1000

pd.options.display.max_columns==1000

from datetime import datetime

import pandas_datareader.data as web
%%time

#Computational time ~ 25.9 s

US_data = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')

print("The shape of US_data is:",(US_data.shape))

display(US_data.head(3))
%%time

#Computational time ~ 12.3 s

#-----------------------------

#Function to summarize details in US accidents data

def summary_fun(US_data):

    NAN_cnt=[] #Number of cells with NAN

    prcnt_NAN=[] #

    action=[]

    remove_columns=[]

    replace_columns=[]

    uniq_cnt=[]

    data_type=[]



    Nd=US_data.shape[0]

    for clmn in US_data.columns:



        NAN_c=Nd-US_data[clmn].count()

        prcnt_c=NAN_c/Nd*100

        uniq_c=US_data[clmn].nunique()

        dtype_c=US_data[clmn].dtypes



        NAN_cnt.append(NAN_c)

        prcnt_NAN.append(prcnt_c)

        uniq_cnt.append(uniq_c)

        data_type.append(str(dtype_c))



        if prcnt_c>0.0:

            act="ReplaceNANs"

            replace_columns.append(clmn)

        else:

            act="None"



        if uniq_c==1:

            act="Remove Column"

            remove_columns.append(clmn)



        action.append(act)





    data_details=pd.DataFrame({"Data Type":data_type,

                           "Unique count":uniq_cnt,

                           "NAN Count":NAN_cnt,

                           "Percent(NAN)":prcnt_NAN,

                           "Action":action},index=US_data.columns)

    

    return remove_columns,replace_columns,data_details

#-----------------------

def highlight_remove(s):

    '''

    highlight the remove in a Series with red.

    '''

    #Ref: https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html

    

    if s=="ReplaceNANs":

        color="green"

    elif s=="Remove Column":

        color="red"

    else:

        color="white"

  

    return 'background-color: %s'% color

#------------------------

def display_pandas_data(pandas_data):

    display(pandas_data.style.applymap(highlight_remove))

#----------------------------------------------------------------------------------------

remove_columns,replace_columns,data_details=summary_fun(US_data)



print("This function will summarize the details of columns of accidents dataset like data type, unique count, percentage of NAN count and shows the action that needs to be done during EDA.")

display_pandas_data(data_details)

%%time

#Computational time: 8.81 s

#-----------------------------------------------------------------------------------------

#Function to append remove_columns list

def remove_columns_append(remove_columns,columns):   

    for clmn in columns:

        remove_columns.append(clmn)

    return remove_columns

#----------------------------------------------------------------------------------------

#Function to remove columns that are in remove_columns from pandas data

def fun_remove_columns(pandas_data,remove_columns):

    for clmn in remove_columns:

        if clmn in pandas_data.columns:

            pandas_data.pop(clmn)

    return pandas_data

#----------------------------------------------------------------------------------------

#Removing some columns

remove_columns=remove_columns_append(remove_columns,["Description","End_Lat","End_Lng","ID"])



print("Removing columns:",remove_columns)

US_data=fun_remove_columns(US_data,remove_columns)



remove_columns,replace_columns,data_details=summary_fun(US_data)

%%time

#Computational time: 9.42 s

# Converting start_time, end_time to get year, month, day, hour and minute of a day and get duration 

#to clear accident from Start_time and End_time

# Ref for pd.to_datetime :https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html

US_data["Start_Time"] = pd.to_datetime(US_data["Start_Time"], errors='coerce')

US_data["End_Time"] = pd.to_datetime(US_data["End_Time"], errors='coerce')



#Finding the number of year

US_data['Year']=US_data["Start_Time"].dt.year



#Finding the month

nmonth=US_data["Start_Time"].dt.month

US_data['Month']=nmonth



#Finding the day of a year

days_each_month=np.cumsum(np.array([0,31,28,31,30,31,30,31,31,30,31,30,31]))

nday=[days_each_month[arg-1] for arg in nmonth.values]

nday=nday+US_data["Start_Time"].dt.day.values

US_data['Day']=nday



#Finding the weekday

US_data['Weekday']=US_data["Start_Time"].dt.weekday



#Finding the hour of day

US_data['Hour']=US_data["Start_Time"].dt.hour



#Finding the minute of the day

US_data['Minute']=US_data['Hour']*60.0+US_data["Start_Time"].dt.minute



# Ref for np.timedelta64: https://pandas.pydata.org/pandas-docs/stable/user_guide/timedeltas.html

US_data['Duration']=(US_data["End_Time"]-US_data["Start_Time"])/np.timedelta64(1,'m')
%%time

#Removing columns

#Computation time ~ 4 ms

remove_columns=remove_columns_append(remove_columns,["Start_Time","End_Time"])

US_data=fun_remove_columns(US_data,remove_columns)
%%time

#Computational time: 1.79 s

#Visualizing Severity

severity_vals=US_data["Severity"].unique()

print("Seveirty values:",severity_vals)



severity_count={}

severity_count[1]=US_data[US_data["Severity"]==1].shape[0]

severity_count[2]=US_data[US_data["Severity"]==2].shape[0]

severity_count[3]=US_data[US_data["Severity"]==3].shape[0]

severity_count[4]=US_data[US_data["Severity"]==4].shape[0]



frame=pd.DataFrame(severity_count,index=["Severity"])

display(frame)



fig=plt.figure(figsize=(12,4))

sns.barplot(list(severity_count.keys()),list(severity_count.values()))

plt.xlabel("Severity")

plt.ylabel("Count")

plt.show()

%%time

#Computation time ~ 43 ms

#Categorizing columns into numerical, boolian and categorical columns

num_columns_details=data_details.loc[data_details["Data Type"].isin(["int64","float64"])]

bool_columns_details=data_details[data_details["Data Type"].isin(["bool"])]

cat_columns_details=data_details[data_details["Data Type"]=="object"]



print("First visualizing categorical columns:")

display_pandas_data(cat_columns_details)
%%time

#Computation time ~ 1.19 s

remove_columns=remove_columns_append(remove_columns,["Zipcode","Airport_Code","Weather_Timestamp","Wind_Direction"])  

US_data=fun_remove_columns(US_data,remove_columns)
%%time

#Computational time ~ 3.17 s

#First visualizing source

#------------------------------

#Functiong to create a frame visualizing number of accidents with each severity for a given column name

def visualize_severity_detailed(US_data,column_name,decending_order=True):

    

    unique_types=np.sort(US_data[column_name].unique())

    unique_details={"Severity 1":[],"Severity 2":[],"Severity 3":[],"Severity 4":[],"Total":[]}



    for arg in unique_types:

        dum=US_data[US_data[column_name]==arg]



        unique_details["Total"].append(dum.shape[0])

        unique_details["Severity 1"].append(dum[dum["Severity"]==1].shape[0])

        unique_details["Severity 2"].append(dum[dum["Severity"]==2].shape[0])

        unique_details["Severity 3"].append(dum[dum["Severity"]==3].shape[0])

        unique_details["Severity 4"].append(dum[dum["Severity"]==4].shape[0])



    unique_details["Total"]=np.asarray(unique_details["Total"])

    unique_details["Severity 1"]=np.asarray(unique_details["Severity 1"])

    unique_details["Severity 2"]=np.asarray(unique_details["Severity 2"])

    unique_details["Severity 3"]=np.asarray(unique_details["Severity 3"])

    unique_details["Severity 4"]=np.asarray(unique_details["Severity 4"])

    

    if decending_order:



        ind=np.argsort(unique_details['Total'])

        ind=np.flip(ind)



        unique_types=unique_types[list(ind)]

        unique_details['Total']=unique_details['Total'][ind]

        unique_details['Severity 1']=unique_details['Severity 1'][ind]

        unique_details['Severity 2']=unique_details['Severity 2'][ind]

        unique_details['Severity 3']=unique_details['Severity 3'][ind]

        unique_details['Severity 4']=unique_details['Severity 4'][ind]    



    frame=pd.DataFrame(unique_details,index=unique_types)

    display(frame)

    

    return unique_types,unique_details

#------------------------------

#Visualizing source details

source_types,source_details=visualize_severity_detailed(US_data,column_name="Source")



#Barplot

fig=plt.figure(figsize=(15,5))

plt.subplot(121)

sns.barplot(source_types,source_details["Severity 1"],color="blue")

sns.barplot(source_types,source_details["Severity 2"],bottom=source_details["Severity 1"],color="orange")

sns.barplot(source_types,source_details["Severity 3"],bottom=source_details["Severity 2"]+source_details["Severity 1"],color="green")

sns.barplot(source_types,source_details["Severity 4"],bottom=source_details["Severity 3"]+source_details["Severity 2"]+source_details["Severity 1"],color="red")

plt.title("Severity 1:blue 2:orange 3:green 4:red")

plt.xlabel("Source")

plt.ylabel("Count")



#Pie chart

sizes = [arg/sum(source_details["Total"]) for arg in source_details["Total"]]

plt.subplot(122)

patches, texts = plt.pie(sizes)

plt.title("Total accidents reported")

plt.legend(patches, source_types, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()

%%time

#Computational time ~ 157 ms

#Source only reports accidents but not affect severity

remove_columns=remove_columns_append(remove_columns,["Source"])

US_data=fun_remove_columns(US_data,remove_columns)
%%time

#Computational time ~ 12.5

#Unique state names

state_names=list(US_data["State"].unique())



state_details={"Total accidents":[],"Severity(%) 1":[],"Severity(%) 2":[],

               "Severity(%) 3":[],"Severity(%) 4":[]}



for state in state_names:

    dum=US_data[US_data["State"]==state]

    tot_acci=dum.shape[0]

    

    state_details["Total accidents"].append(tot_acci)

    

    sev_cnt=dum[dum["Severity"]==1].shape[0]        

    state_details["Severity(%) 1"].append(sev_cnt/tot_acci*100)



    sev_cnt=dum[dum["Severity"]==2].shape[0]        

    state_details["Severity(%) 2"].append(sev_cnt/tot_acci*100)

    

    sev_cnt=dum[dum["Severity"]==3].shape[0]        

    state_details["Severity(%) 3"].append(sev_cnt/tot_acci*100)



    sev_cnt=dum[dum["Severity"]==4].shape[0]        

    state_details["Severity(%) 4"].append(sev_cnt/tot_acci*100)



print("Frame listing total number of accidents in each sate and percentage of accidents with different severity level")

frame=pd.DataFrame(state_details,index=state_names)

display(frame)

%%time

#COmputational time ~ 2.99 s

fig=plt.figure(figsize=(15,4))

sns.barplot(state_names,state_details["Total accidents"])

plt.xlabel("State")

mu=np.mean(state_details["Total accidents"])

plt.title("Total accidents, Mean:"+str(mu))

plt.show()



mu=np.mean(state_details["Severity(%) 1"])

fig=plt.figure(figsize=(15,4))

sns.barplot(state_names,state_details["Severity(%) 1"])

plt.xlabel("State")

plt.title("Severity(%) 1, Mean:"+str(mu))

plt.show()



fig=plt.figure(figsize=(15,4))

sns.barplot(state_names,state_details["Severity(%) 2"])

plt.xlabel("State")

mu=np.mean(state_details["Severity(%) 2"])

plt.title("Severity(%) 2, Mean:"+str(mu))

plt.show()



fig=plt.figure(figsize=(15,4))

sns.barplot(state_names,state_details["Severity(%) 3"])

plt.xlabel("State")

mu=np.mean(state_details["Severity(%) 3"])

plt.title("Severity(%) 3, Mean:"+str(mu))

plt.show()



fig=plt.figure(figsize=(15,4))

sns.barplot(state_names,state_details["Severity(%) 4"])

plt.xlabel("State")

mu=np.mean(state_details["Severity(%) 4"])

plt.title("Severity(%) 4, Mean:"+str(mu))

plt.show()



print("Sum of means of all sverity: 99.97 %")
%%time

#Computational time ~ 1 min 37 sec

#Plotting latitudes and longitude (Takes long time)

plt.figure(figsize=(12, 6))

# (Scatter plot is Taking long time due to color(c))

plt.scatter(US_data["Start_Lng"],US_data["Start_Lat"],s=8,c=US_data["Severity"])

plt.colorbar()

# plt.plot(US_data["Start_Lng"],US_data["Start_Lat"],'o',markersize=2)

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("US accidents map (Severity)")

plt.show()
%%time

#Computational time ~ 1 min 38 s

#Plotting Time zones (Takes long time)

TimeZone=US_data[["Start_Lng","Start_Lat","Timezone"]]

TimeZone=TimeZone.dropna()

TimeZone_vals=list(TimeZone["Timezone"].unique())



#Labeling the time zones to plot

labels={}

flag=1

for arg in TimeZone_vals:

    labels[arg]=flag

    flag=flag+1

print(labels)

colorbars=[labels[arg] for arg in TimeZone["Timezone"]]



#Plotting the Latidue and longitude with time zones

plt.figure(figsize=(12, 6))

# (Scatter plot is Taking long time due to color(c))

plt.scatter(TimeZone["Start_Lng"],TimeZone["Start_Lat"],s=8,c=colorbars)

plt.colorbar()

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("US accidents map (Time Zones)")

plt.show()

%%time

#Computation time ~ 0 ns

remove_columns=remove_columns_append(remove_columns,["Timezone"])

US_data=fun_remove_columns(US_data,remove_columns)
%%time

#Computation time ~ 771 ms

#Accidents in top7 and in "Norfolk","Virginia Beach","Hampton"

cities=US_data["City"].unique()



dum=US_data["City"].value_counts().sort_values(ascending=False)

int_cities=list(dum[:7].index)

nacci_cities=list(dum[:7].values)

city_rank=list(np.arange(1,8))



int_cities.append("Norfolk"); 

nacci_cities.append(dum["Norfolk"])

city_rank.append(list(dum.index).index("Norfolk")+1)



int_cities.append("Virginia Beach"); 

nacci_cities.append(dum["Virginia Beach"])

city_rank.append(list(dum.index).index("Virginia Beach")+1)



int_cities.append("Hampton");

nacci_cities.append(dum["Hampton"])

city_rank.append(list(dum.index).index("Hampton"))



frame=pd.DataFrame({"Number":nacci_cities,"City rank":city_rank},index=int_cities)

display(frame)



fig=plt.figure(figsize=(15,3))

sns.barplot(int_cities,nacci_cities)

plt.xlabel("Cities")

plt.ylabel("Count")

plt.show()
%%time

#Computational time ~ 286 ms

#Visualizing accidients in counties around "Norfolk","Virginia Beach","Hampton"

int_cities=["Norfolk","Virginia Beach","Hampton"]

VA_accid=US_data.loc[US_data["State"].isin(["VA"])]

local_accid=VA_accid.loc[VA_accid["City"].isin(int_cities)]

VA_county=local_accid["County"].value_counts().sort_values(ascending=False)



frame=pd.DataFrame(VA_county)

display(frame)



fig=plt.figure(figsize=(15,3))

sns.barplot(VA_county.index,VA_county.values)

plt.xlabel("County")

plt.ylabel("Count")

plt.show()

%%time

#COmputation time ~

#Accidents in Norfolk streets

int_cities=["Norfolk"]

VA_accid=US_data.loc[US_data["State"].isin(["VA"])]

local_accid=VA_accid.loc[VA_accid["City"].isin(int_cities)]



Norfolk_street=local_accid["Street"].value_counts().sort_values(ascending=False)

print("Total number of streets:",len(Norfolk_street.unique()))



print("Displaying for top 10 streets")

Norfolk_street=Norfolk_street[:10]

frame=pd.DataFrame(Norfolk_street)

display(frame)
%%time

#Computation time ~ 301 ms

remove_columns=remove_columns_append(remove_columns,["County","Street"])

US_data=fun_remove_columns(US_data,remove_columns)

%%time

#Computation time ~ 3.29 s

side_types=US_data["Side"].unique()

print("Unique side types in original data:",side_types)



US_data=US_data[~US_data['Side'].isin([' '])]

side_types=US_data["Side"].unique()

print("Side types in original data after removing null values:",side_types)



side_types,side_details=visualize_severity_detailed(US_data,column_name="Side")



fig=plt.figure(figsize=(10,6))

plt.subplot(121)

sns.barplot(side_types,side_details["Severity 1"],color="blue")

sns.barplot(side_types,side_details["Severity 2"],bottom=side_details["Severity 1"],color="orange")

sns.barplot(side_types,side_details["Severity 3"],bottom=side_details["Severity 2"]+side_details["Severity 1"],color="green")

sns.barplot(side_types,side_details["Severity 4"],bottom=side_details["Severity 3"]+side_details["Severity 2"]+side_details["Severity 1"],color="red")

plt.title("Severity 1:blue 2:orange 3:green 4:red")

plt.xlabel("Side")

plt.ylabel("Count")



sizes = [arg/sum(side_details["Total"]) for arg in side_details["Total"]]

plt.subplot(122)

patches, texts = plt.pie(sizes)

plt.legend(patches, side_types, loc="best")

plt.axis('equal')

plt.tight_layout()

plt.show()

%%time

#Computational time ~ 1.42 s

#Side might affect the severity. So it will be used for the prediction of severity

#Side will be labelled to Right:1 or Left:2

side_types=US_data["Side"].unique()

labels={}

flag=1

for arg in side_types:

    labels[arg]=flag

    flag=flag+1

print(labels)



label_vals=[labels[arg] for arg in US_data["Side"]]

US_data["Side"]=label_vals

%%time

#Coputation time ~ 163 ms

#Filling NAN values in weather_condition by clear

US_data["Weather_Condition"]=US_data["Weather_Condition"].fillna('Clear')

%%time

#Coputation time ~ 24.3 s

weather_types,weather_details=visualize_severity_detailed(US_data,column_name="Weather_Condition")

ntop=10

fig=plt.figure(figsize=(15,6))

sns.barplot(weather_types[:ntop],weather_details["Severity 1"][:ntop],color="blue")

sns.barplot(weather_types[:ntop],weather_details["Severity 2"][:ntop],bottom=weather_details["Severity 1"][:ntop],color="orange")

sns.barplot(weather_types[:ntop],weather_details["Severity 3"][:ntop],bottom=weather_details["Severity 2"][:ntop]+weather_details["Severity 1"][:ntop],color="green")

sns.barplot(weather_types[:ntop],weather_details["Severity 4"][:ntop],bottom=weather_details["Severity 3"][:ntop]+weather_details["Severity 2"][:ntop]+weather_details["Severity 1"][:ntop],color="red")

plt.title("Severity 1:blue 2:orange 3:green 4:red")

plt.xlabel("Weather Condition")

plt.ylabel("Count")

plt.show()
%%time

#COmputation time ~ 1.4 s

#Labelling weather_types

labels={}

flag=1

for arg in weather_types:

    labels[arg]=flag

    flag=flag+1



label_vals=[labels[arg] for arg in US_data["Weather_Condition"]]

US_data["Weather_Condition"]=label_vals

%%time

#Computationa time ~ 12 s

print("Sunrise_Sunset Details")

ntop=US_data["Sunrise_Sunset"].nunique()

US_data["Sunrise_Sunset"]=US_data["Sunrise_Sunset"].fillna('Day')

sunrise_types,sunrise_details=visualize_severity_detailed(US_data,column_name="Sunrise_Sunset")



#COmputational time ~ 125 ms

def barplot_customized(data_types,data_details,ntop,xlabel_val):



    sns.barplot(data_types[:ntop],data_details["Severity 1"][:ntop],

                color="blue",order=data_types[:ntop])    

    sns.barplot(data_types[:ntop],data_details["Severity 2"][:ntop],

                bottom=data_details["Severity 1"][:ntop],color="orange",order=data_types[:ntop])    

    sns.barplot(data_types[:ntop],data_details["Severity 3"][:ntop],

                bottom=data_details["Severity 2"][:ntop]+data_details["Severity 1"][:ntop],color="green",order=data_types[:ntop])

    sns.barplot(data_types[:ntop],data_details["Severity 4"][:ntop],

                bottom=data_details["Severity 3"][:ntop]+data_details["Severity 2"][:ntop]+data_details["Severity 1"][:ntop],color="red",order=data_types[:ntop])

    plt.title("Severity 1:blue 2:orange 3:green 4:red")

    plt.xlabel(xlabel_val)

    plt.ylabel("Count")    

    

    

fig=plt.figure(figsize=(10,6))

ntop=len(sunrise_types)

barplot_customized(sunrise_types,sunrise_details,ntop,"Sunrise_Sunset")

plt.show()
%%time

#COmputational time ~ 

labels={}

flag=1

for arg in sunrise_types:

    labels[arg]=flag

    flag=flag+1

print(labels)



label_vals=[labels[arg] for arg in US_data["Sunrise_Sunset"]]

US_data["Sunrise_Sunset"]=label_vals



#Appending "Civil_Twilight","Nautical_Twilight","Astronomical_Twilight" to remove columns

remove_columns=remove_columns_append(remove_columns,

                                     ["Civil_Twilight","Nautical_Twilight","Astronomical_Twilight"])

US_data=fun_remove_columns(US_data,remove_columns)
%%time

#Computational time ~ 1.35 s

unique_types=bool_columns_details.index

unique_details={"Severity 1":[],"Severity 2":[],"Severity 3":[],

                "Severity 4":[],"Total":[],"Accidents(%)":[]}



Nd=US_data.shape[0]

for arg in unique_types:

    dum=US_data[US_data[arg]]



    unique_details["Total"].append(dum.shape[0])

    unique_details["Severity 1"].append(dum[dum["Severity"]==1].shape[0])

    unique_details["Severity 2"].append(dum[dum["Severity"]==2].shape[0])

    unique_details["Severity 3"].append(dum[dum["Severity"]==3].shape[0])

    unique_details["Severity 4"].append(dum[dum["Severity"]==4].shape[0])

    unique_details["Accidents(%)"].append(unique_details["Total"][-1]/Nd*100)



unique_details["Total"]=np.asarray(unique_details["Total"])

unique_details["Severity 1"]=np.asarray(unique_details["Severity 1"])

unique_details["Severity 2"]=np.asarray(unique_details["Severity 2"])

unique_details["Severity 3"]=np.asarray(unique_details["Severity 3"])

unique_details["Severity 4"]=np.asarray(unique_details["Severity 4"])

unique_details["Accidents(%)"]=np.asarray(unique_details["Accidents(%)"])



ind=np.argsort(unique_details['Total'])

ind=np.flip(ind)



unique_types=unique_types[list(ind)]

unique_details['Total']=unique_details['Total'][ind]

unique_details['Severity 1']=unique_details['Severity 1'][ind]

unique_details['Severity 2']=unique_details['Severity 2'][ind]

unique_details['Severity 3']=unique_details['Severity 3'][ind]

unique_details['Severity 4']=unique_details['Severity 4'][ind]    

unique_details['Accidents(%)']=unique_details['Accidents(%)'][ind]



frame=pd.DataFrame(unique_details,index=unique_types)

display(frame)



ntop=len(unique_types)

fig=plt.figure(figsize=(15,6))

barplot_customized(unique_types,unique_details,ntop,"Location")

plt.show()
%%time

#Computation time ~ 6.55 s

for clmn in bool_columns_details.index:

    label_vals=[float(arg) for arg in US_data[clmn]]

    US_data[clmn]=label_vals



remove_columns=remove_columns_append(remove_columns

                                     ,["Give_Way","No_Exit","Traffic_Calming","Bump","Roundabout","Number"])

US_data=fun_remove_columns(US_data,remove_columns)

%%time

#COmputation time ~ 1.9 s

TMC_codes=[201,241,245,229,203]

TMC_names={"Traffic Message Channel (TMC) Description":["Accident(s)","Accident(s). Right lane blocked","(Q) accident(s). Two lanes blocked",

           "(Q) accident(s). Slow traffic","multi-vehicle accident"]}



frame=pd.DataFrame(TMC_names,TMC_codes)

display(frame)



TMC_types,TMC_details=visualize_severity_detailed(US_data,column_name="TMC")
%%time

#Computational time ~ 310 ms

ntop=3

fig=plt.figure(figsize=(10,4))

barplot_customized(TMC_types,TMC_details,ntop,"TMC")

plt.show()
%%time

#Computation time ~ 2.02 s

US_data["TMC"]=US_data["TMC"].fillna(201)



labels={}

flag=1

for arg in TMC_types:

    labels[arg]=flag

    flag=flag+1



label_vals=[labels[arg] for arg in US_data["TMC"]]

US_data["TMC"]=label_vals
%%time

# COmputation time ~ 2.8 s

desc_distance=US_data["Distance(mi)"].describe([.25,.50,.75,.80,.90,.95,0.975,0.99])

display(desc_distance)



fig=plt.figure(figsize=(10,4))

sns.boxplot(x="Severity", y="Distance(mi)", data=US_data)

plt.ylabel('Distance(mi)', fontsize=12)

plt.xlabel('Severity', fontsize=12)

plt.title("Box plot")

plt.show()
remove_columns=remove_columns_append(remove_columns,["Distance(mi)"])

US_data=fun_remove_columns(US_data,remove_columns)
%%time

#Computational time ~

fig=plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(x="Severity", y="Temperature(F)", data=US_data)

plt.ylabel('Temperature(F)', fontsize=12)

plt.xlabel('Severity', fontsize=12)

plt.title("Box plot")



plt.subplot(122)

sns.violinplot(x='Severity', y='Temperature(F)', data=US_data)

plt.xlabel('Severity', fontsize=12)

plt.show()

%%time

#COmputational time ~ 

remove_columns=remove_columns_append(remove_columns,["Wind_Chill(F)"])

US_data=fun_remove_columns(US_data,remove_columns)

%%time

#COmputationa time ~ 

display(US_data[["Humidity(%)","Pressure(in)","Wind_Speed(mph)","Precipitation(in)"]].describe())
%%time

#COmputational time ~ 

remove_columns=remove_columns_append(remove_columns,["Pressure(in)","Precipitation(in)"])

US_data=fun_remove_columns(US_data,remove_columns)

%%time

#Computational time ~

#Plotting humidity

display(US_data["Humidity(%)"].describe([.25,.50,.75,.80,.90,.95,0.975,0.99]))



fig=plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(x="Severity", y="Humidity(%)", data=US_data)

plt.ylabel('Humidity(%)', fontsize=12)

plt.xlabel('Severity', fontsize=12)

plt.title("Box plot")



plt.subplot(122)

sns.violinplot(x='Severity', y='Humidity(%)', data=US_data)

plt.xlabel('Severity', fontsize=12)

plt.ylabel('Humidity(%)', fontsize=12)

plt.show()
%%time

#Computational time ~ 22 s

desc=US_data["Wind_Speed(mph)"].describe(percentiles=[.25,.50,.75,.80,.90,.95,0.975,0.99])

display(desc)



fig=plt.figure(figsize=(12,4))

plt.subplot(121)

sns.boxplot(x="Severity", y="Wind_Speed(mph)", data=US_data)

plt.xlabel('Severity', fontsize=12)

plt.ylabel('Wind_Speed(mph)', fontsize=12)

plt.title("Box plot")



plt.subplot(122)

sns.violinplot(x='Severity', y='Wind_Speed(mph)', data=US_data)

plt.xlabel('Severity', fontsize=12)

plt.ylabel('Wind_Speed(mph)', fontsize=12)

plt.show()

%%time

#Computational time ~

US_data.loc[US_data["Wind_Speed(mph)"]>desc["99%"], 'Wind_Speed(mph)']=np.nan
%%time

#Computational time ~ 



fig=plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(x="Severity", y="Wind_Speed(mph)", data=US_data)

plt.xlabel('Severity', fontsize=12)

plt.ylabel('Wind_Speed(mph)', fontsize=12)

plt.title("Box plot")



plt.subplot(122)

sns.violinplot(x='Severity', y='Wind_Speed(mph)', data=US_data)

plt.xlabel('Severity', fontsize=12)

plt.ylabel('Wind_Speed(mph)', fontsize=12)

plt.show()
%%time

display(US_data["Visibility(mi)"].describe([.25,.50,.75,.80,.90,.95,0.975,0.99]))



fig=plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(x="Severity", y="Visibility(mi)", data=US_data)

plt.ylabel('Visibility(mi)', fontsize=12)

plt.xlabel('Severity', fontsize=12)

plt.title("Box plot")



plt.subplot(122)

sns.violinplot(x='Severity', y='Visibility(mi)', data=US_data)

plt.xlabel('Severity', fontsize=12)

plt.ylabel('Visibility(mi)', fontsize=12)

plt.show()
%%time

remove_columns=remove_columns_append(remove_columns,["Visibility(mi)"])

US_data=fun_remove_columns(US_data,remove_columns)
%%time

#Computational time ~ 

year_types,year_details=visualize_severity_detailed(US_data,column_name="Year",decending_order=False)



ntop=len(year_types)

fig=plt.figure(figsize=(10,4))

barplot_customized(year_types,year_details,ntop,"Year")

plt.show()
%%time

#COmputational time ~

#Removing Year

remove_columns=remove_columns_append(remove_columns,["Year"])

US_data=fun_remove_columns(US_data,remove_columns)



month_types,month_details=visualize_severity_detailed(US_data,column_name="Month",decending_order=False)

month_names=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']



ntop=len(month_names)

fig=plt.figure(figsize=(10,4))

barplot_customized(month_names,month_details,ntop,"Month")

plt.show()
%%time

day_types,day_details=visualize_severity_detailed(US_data,column_name="Day",decending_order=False)



fig=plt.figure(figsize=(15,6))

plt.plot(day_types,day_details["Severity 1"],color='b')

plt.plot(day_types,day_details["Severity 2"],color='m')

plt.plot(day_types,day_details["Severity 3"],color='g')

plt.plot(day_types,day_details["Severity 4"],color='r')

plt.plot(day_types,day_details["Total"],color='k')

plt.xlabel("Day")

plt.ylabel("Count")

plt.legend(["Severity 1","Severity 2","Severity 3","Severity 4","Total"])

plt.show()
%%time

#COmputational time ~ 

week_types,week_details=visualize_severity_detailed(US_data,column_name="Weekday",decending_order=False)

week_names=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']



ntop=len(week_types)

fig=plt.figure(figsize=(10,4))

barplot_customized(week_names,week_details,ntop,"Weekday")

plt.show()
%%time

#COmputational time ~ 

hour_types,hour_details=visualize_severity_detailed(US_data,column_name="Hour",decending_order=False)



ntop=len(hour_types)

fig=plt.figure(figsize=(15,4))

barplot_customized(hour_types,hour_details,ntop,"Hour")

plt.show()
%%time

#COmputational time ~ 



#Plotting minutes

minute_types,minute_details=visualize_severity_detailed(US_data,column_name="Minute",decending_order=False)



fig=plt.figure(figsize=(15,6))

plt.plot(minute_types,minute_details["Severity 1"],color='b')

plt.plot(minute_types,minute_details["Severity 2"],color='m')

plt.plot(minute_types,minute_details["Severity 3"],color='g')

plt.plot(minute_types,minute_details["Severity 4"],color='r')

plt.plot(minute_types,minute_details["Total"],color='k')



plt.xlabel("Minute")

plt.ylabel("Count")

plt.legend(["Severity 1","Severity 2","Severity 3","Severity 4","Total"])

plt.show()
%%time

desc=US_data["Duration"].describe([.25,.50,.75,.80,.90,.95,0.975,0.99])



print(desc)
%%time

nmeans=4

fig=plt.figure(figsize=(12,6))

plt.subplot(121)

sns.boxplot(x="Severity",y="Duration",data=US_data[US_data["Duration"]<=desc["99%"]])

plt.ylabel('Duration(min)', fontsize=12)

plt.xlabel('Severity', fontsize=12)

plt.title("Box plot")



plt.subplot(122)

sns.violinplot(x="Severity",y="Duration",data=US_data[US_data["Duration"]<=desc["99%"]])

plt.xlabel('Severity', fontsize=12)

plt.show()
%%time



US_data.loc[US_data["Duration"]>desc["99%"], 'Duration']=np.nan

US_data.loc[US_data["Duration"]<0, 'Duration']=np.nan



#Appending Year to remove columns

remove_columns=remove_columns_append(remove_columns,["Month","Year"])

US_data=fun_remove_columns(US_data,remove_columns)



num_cols=US_data.select_dtypes(include=['float64', 'int64']).columns.values 

US_data[num_cols]=US_data[num_cols].fillna(US_data.median())



print("Removing columns:",remove_columns)

US_data=fun_remove_columns(US_data,remove_columns)

print("US_data details after removing above columns")

remove_columns,replace_columns,data_details=summary_fun(US_data)

display_pandas_data(data_details)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
%%time

#Computational time ~ 5.28 s

req_data=US_data[US_data["State"]=="CA"]



# display(req_data)

dum=req_data["City"].value_counts().sort_values(ascending=False)



req_cities=['San Francisco']



req_cities=['Alameda','Albany','American','Antioch','Atherton','Belmont','San Mateo',

            'Belvedere','Benicia','Berkeley','Brentwood','Brisbane','Burlingame',

            'Calistoga','Campbell','Clayton','Cloverdale','Colma','Concord',

            'Corte Madera','Cotati','Cupertino','Daly City','Danville','Dixon	City',

            'Dublin','East Palo Alto','El Cerrito','Emeryville','Fairfax','Fairfield',

            'Foster City','Fremont','Gilroy','Half Moon Bay','Hayward','Healdsburg',

            'Hercules','Hillsborough','Lafayette','Larkspur','Livermore','Los Altos',

            'Los Altos Hills','Los Gatos','Martinez','Menlo Park','Mill Valley',

            'Millbrae','Milpitas','Monte Sereno','Moraga','Morgan Hill','Mountain View',

            'Napa','Newark','Novato','Oakland','Oakley','Orinda','Pacifica','Palo Alto',

            'Petaluma','Piedmont','Pinole','Pittsburg','Pleasant Hill','Pleasanton',

            'Portola Valley','Redwood','Richmond','Rio Vista','Rohnert Park','Ross',

            'St. Helena','San Anselmo','San Bruno','San Carlos','San Francisco',

            'San Jose','San Leandro','San Mateo','San Pablo','San Rafael','San Ramon',

            'Santa Clara','Santa Rosa','Saratoga','Sausalito','Sebastopol','Sonoma',

            'South San Francisco','Suisun City','Sunnyvale','Tiburon','Union City',

            'Vacaville','Vallejo','Walnut Creek','Windsor','Woodside','Yountville']



req_data2=req_data.loc[req_data["City"].isin(req_cities)]



severity_count={}

severity_count[1]=req_data2[req_data2["Severity"]==1].shape[0]

severity_count[2]=req_data2[req_data2["Severity"]==2].shape[0]

severity_count[3]=req_data2[req_data2["Severity"]==3].shape[0]

severity_count[4]=req_data2[req_data2["Severity"]==4].shape[0]



frame=pd.DataFrame(severity_count,index=["Severity"])

print("Severity in california bay area")

display(frame)



plt.figure(figsize=(12, 6))

plt.scatter(req_data2["Start_Lng"],req_data2["Start_Lat"],s=8,c=req_data2["Severity"])

plt.colorbar()

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.title("US accidents map (Severity)")

plt.show()



%%time

#COmputational time ~

corr=req_data2.corr()



f, ax = plt.subplots(figsize=(20, 10))

g=sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=False, ax=ax, annot=True, vmax=1.0, vmin=-1.0, cbar_kws={"shrink": 0.9})

g.set_yticklabels(g.get_yticklabels(), rotation = 0)

plt.plot()
%%time

#Computational time ~

#selecting input columns and output columns for training



x_clmns=list(US_data.columns)

x_clmns.remove("Severity")

x_clmns.remove("City")

x_clmns.remove("State")

y_clmns=["Severity"]



X=req_data2[x_clmns].values

y=req_data2[y_clmns].values



y=np.reshape(y,(-1,1))

y=y[:,]



# Split the data set into training and testing data sets

X_train, X_test, y_train, y_test = train_test_split(X,y[:,0], test_size=0.1,random_state=21)



print("Train size:",X_train.shape[0])

print("Test size:",X_test.shape[0])



severity_count_y_test={}

severity_count_y_test[1]=y_test[y_test==1].shape[0]

severity_count_y_test[2]=y_test[y_test==2].shape[0]

severity_count_y_test[3]=y_test[y_test==3].shape[0]

severity_count_y_test[4]=y_test[y_test==4].shape[0]



frame=pd.DataFrame(severity_count_y_test,index=["Severity count in test_data"])

display(frame)
%%time

#COmputationa time ~ 2 min 47 seconds

lr = LogisticRegression(random_state=0,solver='saga',max_iter=1000)

lr.fit(X_train,y_train)



y_pred=lr.predict(X_train)

acc_train_lr=accuracy_score(y_train, y_pred)

print("Accuracy of train data:",acc_train_lr)



y_pred=lr.predict(X_test)

acc_test_lr=accuracy_score(y_test, y_pred)

print("Accuracy of test data:",acc_test_lr)



mat_lr = confusion_matrix(y_pred,y_test)

sns.heatmap(mat_lr, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('true label')

plt.ylabel('predicted label')

plt.title("Confusion matrix")

plt.show()



frame=pd.DataFrame(severity_count_y_test,index=["Severity.count in test_data"])

display(frame)

%%time

# COmputational time ~

train_accuracy=[]

test_accuracy=[]

k_array=range(1,10)



for i in k_array:

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)



    y_pred = knn.predict(X_train)

    acc=accuracy_score(y_train, y_pred)

    train_accuracy.append(acc)



    y_pred = knn.predict(X_test)

    acc=accuracy_score(y_test, y_pred)

    test_accuracy.append(acc)



    print("k=",i," Train Accuracy:",train_accuracy[-1],"Test accuracy:",test_accuracy[-1])



plt.figure(figsize=(10,4))

plt.plot(k_array,train_accuracy)

plt.plot(k_array,test_accuracy)

plt.legend(["Train","Test"])

plt.title("Accuracy score")

plt.show()



frame=pd.DataFrame(severity_count_y_test,index=["Severity.count in test_data"])

display(frame)
%%time

#COmputational time ~ 

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train,y_train)



y_pred = knn.predict(X_train)

acc_train_knn=accuracy_score(y_train, y_pred)



y_pred = knn.predict(X_test)

acc_test_knn=accuracy_score(y_test, y_pred)



mat_knn = confusion_matrix(y_pred,y_test)



frame=pd.DataFrame(severity_count_y_test,index=["Severity count in test_data"])

display(frame)



frame=pd.DataFrame({"Train":[acc_train_lr,acc_train_knn],

                    "Test":[acc_test_lr,acc_test_knn]},

                   index=["Logistic","KNN"])

display(frame)



plt.figure(figsize=(8,4))



plt.subplot(121)

sns.heatmap(mat_lr, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('true label')

plt.ylabel('predicted label')

plt.title('Logitstic regression')



plt.subplot(122)

sns.heatmap(mat_knn, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('true label')

plt.title('KNN')



plt.show()



%%time

#Computational time ~

#REF https://towardsdatascience.com/decision-tree-classification-de64fc4d5aac

#REF https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

tree = DecisionTreeClassifier(max_depth=20,class_weight='balanced')

tree.fit(X_train, y_train)



y_pred = tree.predict(X_train)

acc_train_dt=accuracy_score(y_train, y_pred)



y_pred = tree.predict(X_test)

acc_test_dt=accuracy_score(y_pred,y_test)

mat_dt = confusion_matrix(y_pred,y_test)



frame=pd.DataFrame(severity_count_y_test,index=["Severity count in test_data"])

display(frame)



frame=pd.DataFrame({"Train":[acc_train_lr,acc_train_knn,acc_train_dt],

                    "Test":[acc_test_lr,acc_test_knn,acc_test_dt]},

                   index=["Logistic","KNN","Decision tree"])

display(frame)



plt.figure(figsize=(12,4))



plt.subplot(131)

sns.heatmap(mat_lr, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('True label')

plt.ylabel('predicted label')

plt.title('Logistic regression')



plt.subplot(132)

sns.heatmap(mat_knn, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('True label')

plt.title('KNN')



plt.subplot(133)

sns.heatmap(mat_dt, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('True label')

plt.title('Decision tree')



plt.show()

%%time

#COmputational time ~ 

#REF https://towardsdatascience.com/understanding-random-forest-58381e0602d2

#REF https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

RF= RandomForestClassifier(n_estimators=100,class_weight='balanced')

RF.fit(X_train, y_train)



y_pred = RF.predict(X_train)

acc_train_RF=accuracy_score(y_train, y_pred)



y_pred = RF.predict(X_test)

acc_test_RF=accuracy_score(y_pred,y_test)

mat_RF = confusion_matrix(y_pred,y_test)



frame=pd.DataFrame(severity_count_y_test,index=["Severity count in test_data"])

display(frame)



frame=pd.DataFrame({"Train":[acc_train_lr,acc_train_knn,acc_train_dt,acc_train_RF],

                    "Test":[acc_test_lr,acc_test_knn,acc_test_dt,acc_test_RF]},

                   index=["Logistic","KNN","Decision tree","Random forest"])

display(frame)



plt.figure(figsize=(16,4))



plt.subplot(141)

sns.heatmap(mat_lr, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('True label')

plt.ylabel('predicted label')

plt.title('Logistic regression')



plt.subplot(142)

sns.heatmap(mat_knn, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('True label')

plt.title('KNN')



plt.subplot(143)

sns.heatmap(mat_dt, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('True label')

plt.title('Decision tree')



plt.subplot(144)

sns.heatmap(mat_RF, square=True, annot=True, fmt='d', cbar=False,xticklabels=[1,2,3,4],yticklabels=[1,2,3,4])

plt.xlabel('True label')

plt.title('Random forest')



plt.show()



frame=pd.DataFrame(severity_count_y_test,index=["Severity.count in test_data"])

display(frame)
req_data3=req_data.loc[req_data["City"].isin(req_cities)]



severity_count={}

severity_count[1]=req_data3[req_data3["Severity"]==1].shape[0]

severity_count[2]=req_data3[req_data3["Severity"]==2].shape[0]

severity_count[3]=req_data3[req_data3["Severity"]==3].shape[0]

severity_count[4]=req_data3[req_data3["Severity"]==4].shape[0]

tot_count=severity_count[1]+severity_count[2]+severity_count[3]+severity_count[4]





frame=pd.DataFrame(severity_count,index=["Severity Count"])

display(frame)



severity_count={}

severity_count[1]=req_data3[req_data3["Severity"]==1].shape[0]/tot_count*100

severity_count[2]=req_data3[req_data3["Severity"]==2].shape[0]/tot_count*100

severity_count[3]=req_data3[req_data3["Severity"]==3].shape[0]/tot_count*100

severity_count[4]=req_data3[req_data3["Severity"]==4].shape[0]/tot_count*100



frame=pd.DataFrame(severity_count,index=["Severity percentage"])

display(frame)



print("Removing outliers")

req_data3=req_data3[~(req_data3["Severity"]==1)]

req_data3=req_data3[~(req_data3["Severity"]==4)]



severity_count={}

severity_count[1]=req_data3[req_data3["Severity"]==1].shape[0]

severity_count[2]=req_data3[req_data3["Severity"]==2].shape[0]

severity_count[3]=req_data3[req_data3["Severity"]==3].shape[0]

severity_count[4]=req_data3[req_data3["Severity"]==4].shape[0]



frame=pd.DataFrame(severity_count,index=["Severity count"])

display(frame)

%%time

#Computational time ~

#selecting input columns and output columns for training



x_clmns=list(US_data.columns)

x_clmns.remove("Severity")

x_clmns.remove("City")

x_clmns.remove("State")

y_clmns=["Severity"]



X=req_data3[x_clmns].values

y=req_data3[y_clmns].values



y=np.reshape(y,(-1,1))

y=y[:,]



# Split the data set into training and testing data sets

X_train, X_test, y_train, y_test = train_test_split(X,y[:,0], test_size=0.1,random_state=21)



print("Train size:",X_train.shape[0])

print("Test size:",X_test.shape[0])



severity_count_y_test_no_outliers={}

severity_count_y_test_no_outliers[1]=y_test[y_test==1].shape[0]

severity_count_y_test_no_outliers[2]=y_test[y_test==2].shape[0]

severity_count_y_test_no_outliers[3]=y_test[y_test==3].shape[0]

severity_count_y_test_no_outliers[4]=y_test[y_test==4].shape[0]



frame=pd.DataFrame(severity_count_y_test_no_outliers,index=["Severity count in test_data"])

display(frame)
%%time

#COmputational time ~

#Decision Tree Classifier after removing outliers

tree = DecisionTreeClassifier(max_depth=20,class_weight='balanced')

tree.fit(X_train, y_train)



y_pred = tree.predict(X_train)

acc_train_dt_no_outliers=accuracy_score(y_train, y_pred)



y_pred = tree.predict(X_test)

acc_test_dt_no_outliers=accuracy_score(y_pred,y_test)

mat_dt_no_outliers = confusion_matrix(y_pred,y_test)



frame=pd.DataFrame(severity_count_y_test_no_outliers,index=["Severity count in test_data"])

display(frame)



frame=pd.DataFrame({"Train":[acc_train_lr,acc_train_knn,acc_train_dt,acc_train_RF,acc_train_dt_no_outliers],

                    "Test":[acc_test_lr,acc_test_knn,acc_test_dt,acc_test_RF,acc_test_dt_no_outliers]},

                   index=["Logistic","KNN","Decision tree","Random forest","Decision tree:No outliers"])

display(frame)



plt.figure(figsize=(4,4))

sns.heatmap(mat_dt_no_outliers, square=True, annot=True, fmt='d', cbar=False,xticklabels=[2,3],yticklabels=[2,3])

plt.xlabel('True label')

plt.title('Decision tree (No outliers)')

plt.show()

%%time

#Computational time ~ 

RF= RandomForestClassifier(n_estimators=100,class_weight='balanced')

RF.fit(X_train, y_train)



y_pred = RF.predict(X_train)

acc_train_RF_outliers=accuracy_score(y_train, y_pred)



y_pred = RF.predict(X_test)

acc_test_RF_outliers=accuracy_score(y_pred,y_test)

mat_rf_no_outliers = confusion_matrix(y_pred,y_test)



frame=pd.DataFrame(severity_count_y_test_no_outliers,index=["Severity count in test_data"])

display(frame)



frame=pd.DataFrame({"Train":[acc_train_lr,acc_train_knn,acc_train_dt,acc_train_RF,acc_train_dt_no_outliers,acc_train_RF_outliers],

                    "Test":[acc_test_lr,acc_test_knn,acc_test_dt,acc_test_RF,acc_test_dt_no_outliers,acc_test_RF_outliers]},

                   index=["Logistic","KNN","Decision tree","Random forest","Decision tree:No outliers","Random forest:No outliers"])

display(frame)



plt.figure(figsize=(8,4))

plt.subplot(121)

sns.heatmap(mat_dt_no_outliers, square=True, annot=True, fmt='d', cbar=False,xticklabels=[2,3],yticklabels=[2,3])

plt.xlabel('True label')

plt.title('Decision tree')



plt.subplot(122)

sns.heatmap(mat_rf_no_outliers, square=True, annot=True, fmt='d', cbar=False,xticklabels=[2,3],yticklabels=[2,3])

plt.xlabel('True label')

plt.title('Random forest')



plt.show()
