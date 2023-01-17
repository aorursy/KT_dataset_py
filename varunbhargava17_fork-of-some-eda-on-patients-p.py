import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 
import os
file = pd.read_csv('../input/KaggleV2-May-2016.csv')
#NA/Missing data in dataset
null_values=file.isnull().sum()
print ("Number of null values in the data \n" , null_values)
# Lets fix the data first, Scheduled day and Appointment day can be split into Hour of the day of the appoint ment, Day of the week for appointment 
file['ScheduledDay']=file['ScheduledDay'].apply(np.datetime64)
file['AppointmentDay']=file['AppointmentDay'].apply(np.datetime64)

# Lets calcululate hour and day for the appointment set and scheduled
def calculateHour(timestamp):
    timestamp = str(timestamp)
    hour = int(timestamp[11:13])
    minute = int(timestamp[14:16])
    second = int(timestamp[17:])
    return round(hour + minute/60 + second/3600)

def seperatedate(day):
    day=str(day)
    day=str(day[:10])
    return day
file['HourOfTheDay'] = file.ScheduledDay.apply(calculateHour)
file['ScheduledDay_Date'] = file.ScheduledDay.apply(seperatedate)
file['AppointmentDay'] = file.AppointmentDay.apply(seperatedate)
file.head()
# Changing the data types and calculating the day of the week and difference of days between and scheduled day and appointment day
file['ScheduledDay_Date']=file['ScheduledDay_Date'].apply(np.datetime64)
file['AppointmentDay']=file['AppointmentDay'].apply(np.datetime64)
file['ScheduledDay_Date'] = pd.to_datetime(file['ScheduledDay_Date'])
file['difference_b/w_sch&appt']=file['AppointmentDay']-file['ScheduledDay_Date']
file['AppointmentDay'] = pd.to_datetime(file['AppointmentDay'])
file['day_week_appoitntment'] = file['AppointmentDay'].dt.weekday_name
file.head()
# I am dropping Patient ID as it has no use here and renaming few columns as they are wrongly named
file=file.drop(['PatientId','ScheduledDay'],axis=1)
file.columns=['AppointmentID', 'Gender', 'AppointmentDay', 'Age', 'Neighbourhood',
       'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap',
       'SMS_received', 'No-show', 'HourOfTheDay', 'ScheduledDay_Date',
       'difference_b/w_sch&appt', 'day_week_appoitntment']

file.head()

# Min and Max Age are odd, hence removing some impossible ages
print ("Maximum age of a patient is : ",file['Age'].max())
print ("Minimum age of a patient is : ",file['Age'].min())
#Lets ignore these ages as minimum age cannot be -1 and maximum age of 115 seems to practical impossible
#Hence we are only selecting ages greater than 0 and less than 100
file=file[(file['Age']<100) & (file['Age']>0)]
print(file.describe())
# count of show and no show
no_Show=file[['AppointmentID','No-show']].groupby('No-show').count()
no_Show_plot = file[['AppointmentID','No-show']].groupby('No-show').count().plot(kind='bar',legend=False)
print (no_Show)
# Lets see how many of Men's and Women's did not showed up for appointment
No_show_Gender=file.groupby(['Gender','No-show'])['No-show'].size()
No_show_Gender_plot=file.groupby(['Gender','No-show'])['No-show'].size().unstack().plot(kind='bar',stacked=True)
print(No_show_Gender)
# Its not clear from the graph above about the exact figure of no show between men and women, Lets talk in terms of percentage
number_Of_Men=file[(file['Gender']=='M') & (file['No-show']=='Yes')].count()
number_Of_Women=file[(file['Gender']=='F') & (file['No-show']=='Yes')].count()
total_Men=len((file['Gender']=='M'))
total_Women=len((file['Gender']=='F'))
percentage_of_Women=(number_Of_Women/total_Women)*100
percentage_of_men=(number_Of_Men/total_Men)*100
print("Percentage of women who did not show up ",np.round(percentage_of_Women['Gender'],0),"%")
print("Percentage of men who did not show up ",np.round(percentage_of_men['Gender'],0),"%")
#Perhaps this pie chart will give us more clear picture about the percentage 
labels='Female','Male'
sizes=[13,7]
plt.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)

#Lets see Which medical problem each gender is suffering from
# From the chart it can be observed that Hypertension is common in both the genders but female are more suffering from it, and in terms of alcoholisim males are more than the womens do

reason=file[['Gender','Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']].groupby(['Gender']).sum()
reason_plot=file[['Gender','Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']].groupby(['Gender']).sum().plot(kind='barh',figsize=(16,8))
print(reason)
# Scholarship (medical insurance probably), Majority of the population in data set given does not have scholarship, Hence lets analyse on the basis of no show
sns.countplot(x='Scholarship',data=file,hue='Gender')
scholarship=file.groupby(['No-show','Scholarship'])['Scholarship'].count()
print(scholarship)
#Lets observe if there is any particular day people are booking there appointment
#Now we can depict from the graph that appontments are higher on MOnday/Tuesday?wednesday and least on Saturday(Enjoying the weekend)
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for index,i in enumerate(weekdays):
    j=file[file.day_week_appoitntment==i]
    count=len(j)
    total_count=len(file)
    perc=(count/total_count)*100
    print(i,count)
    plt.bar(index,perc)
plt.xticks(range(len(weekdays)),weekdays, rotation=45)
plt.title('Day of the week for appointment')
plt.show()
#Lets see the days when people shows  no-shows
# We can depict from the graph below that people No-Shows are more on weekdays
no_Show_Yes=file[file['No-show']=='Yes']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for index,i in enumerate(weekdays):
    k=no_Show_Yes[no_Show_Yes.day_week_appoitntment==i]
    count=len(k)
    total_count=len(no_Show_Yes)
    perc=(count/total_count)*100
    print(i,count,perc)
    plt.bar(index,perc)

plt.xticks(range(len(weekdays)),weekdays, rotation=45)
plt.title('Percent of No-Show per DayOfWeek')
plt.show()

# Lets analyse the location of hospital
location=file.groupby(['Neighbourhood'],sort=False).size()
print(location.sort_values())
location_plot=file.groupby(['Neighbourhood']).size().plot(kind='bar',figsize=(20,10))
plt.xticks(rotation=90)