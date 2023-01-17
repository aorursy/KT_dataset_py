import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
%matplotlib inline 
file=pd.read_csv("../input/KaggleV2-May-2016.csv")
print(file.head())
#NA/Missing data in dataset
null_values=file.isnull().sum()
print ("Number of null values in the data \n" , null_values)
# Lets fix the data first, Scheduled day and Appointment day can be split into Hour of the day of the appointment, Day of the week for appointment 
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
# No-show between males and females
sns.set(style="darkgrid")
sns.countplot(x='No-show',hue='Gender',data=file)
plt.xlabel('Appointment show status for male and female')
# Its not clear from the graph above about the exact figure of no show between men and women, Lets talk in terms of percentage
# Can be clearly seen, females have more no-show than man do. 
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
plt.title("Percent of males and females")
#Lets see Which medical problem each gender is suffering from
# From the chart it can be observed that Hypertension is common in both the genders but female are more suffering from it, and in terms of alcoholisim males are more than the womens do

reason=file[['Gender','Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']].groupby(['Gender']).sum()
reason_plot=file[['Gender','Hypertension', 'Diabetes', 'Alcoholism', 'Handicap']].groupby(['Gender']).sum().plot(kind='barh',figsize=(16,8))
print(reason)
plt.ylabel("Gender")
plt.xlabel("Count of patients")
# Scholarship (medical insurance probably), Majority of the population in data set given does not have scholarship, Hence lets analyse on the basis of no show
sns.countplot(x='Scholarship',data=file,hue='Gender')
scholarship=file.groupby(['No-show','Scholarship'])['Scholarship'].count()
print(scholarship)
#Lets observe if there is any particular day people are booking there appointment
#Now we can depict from the graph that appontments are higher on Monday/Tuesday/Wednesday 
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
plt.xlabel("Days of week")
plt.ylabel("Percent of appointment")
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
plt.xlabel("Days of week")
plt.ylabel("Percent ")
plt.title('Percent of No-Show per DayOfWeek')
plt.show()

# Irs strange but there is no pattern of show and no show considering the day of week
no_Show_No=file[file['No-show']=='No']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for index,i in enumerate(weekdays):
    k=no_Show_No[no_Show_No.day_week_appoitntment==i]
    count=len(k)
    total_count=len(no_Show_No)
    perc=(count/total_count)*100
    print(i,count,perc)
    plt.bar(index,perc)

plt.xticks(range(len(weekdays)),weekdays, rotation=45)
plt.xlabel("Days of week")
plt.ylabel("Percent ")
plt.title('Percent of Show up per DayOfWeek')
plt.show()
# Does age plays in role of show and no show?
age_range = range(120)
age_show = np.zeros(120)
columns = file.columns
age_range = range(120)
age_show = np.zeros(120)
age_no_show = age_show.copy()

no_show_age_count = file.groupby('Age').Age.count()
print(no_show_age_count)
# Lets analyse the location of hospital
# In the figure it is observed that No show is almost in the same proportion in all neighbourhood, but it can be seen that count is exorbitant in two of the location and they are
# Jardim camburi and Maria Ortiz
location=file.groupby(['Neighbourhood'],sort=False).size()
fig, ax = plt.subplots()
fig.set_size_inches(32, 16)
sns.countplot(x='Neighbourhood',data=file,hue='No-show')
plt.xticks(rotation=90,size=20)
plt.yticks(size=20)
plt.title("All neighbourhoods and count of patients ",fontsize=40)
plt.setp(ax.get_legend().get_texts(), fontsize='22') 
plt.setp(ax.get_legend().get_title(), fontsize='32')
plt.xlabel("Name of neighbourhood ",fontsize=40)
plt.ylabel("Number of patients ",fontsize=40)