import pandas as pd

import matplotlib.pyplot as plt

import plotly

plotly.offline.init_notebook_mode()

from plotly import __version__

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import numpy as np

import seaborn as sns

import calendar

%matplotlib inline

df = pd.read_csv('../input/911.csv')

df.head()
reason = np.unique(df['title'])
reason.size
DATA = np.zeros((df.shape[0],6),dtype='O')

DATA[:,0] = df['lng'].values

DATA[:,1] = df['lat'].values

DATA[:,4] = df['title'].values

DATA[:,5] = df['twp'].values

for i in range(DATA.shape[0]):

    DATA[i,2] = df['timeStamp'].values[i][:10]

    DATA[i,3] = df['timeStamp'].values[i][10:]

    sp = DATA[i,3].split(':')

    DATA[i,3] = (int(sp[0])*3600 + int(sp[1])*60 + int(sp[2]))/3600
new_data = np.zeros(reason.size,dtype = 'O')

for i in range(reason.size):

    new_data[i] = DATA[np.where(DATA[:,4] == reason[i])]
week = np.array(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
for i in range(new_data.shape[0]):

    for j in range(new_data[i].shape[0]):

        w = np.array(new_data[i][j,2].split('-')).astype(int)

        new_data[i][j,0] = week[calendar.weekday(w[0],w[1],w[2])]
for i in range(reason.size):

    if new_data[i][:,3].size > 1700:

        sns.plt.figure(figsize=(12,4))

        sns.plt.title(new_data[i][0][-2])

        sns.plt.xlabel("Week day")

        sns.plt.ylabel(new_data[i][0][-2])

        print("Number of calls with " + new_data[i][0][-2] + " "+ str(new_data[i][:,3].size))

        sns.countplot((new_data[i][:,0]),order = week)
for i in range(reason.size):

    if new_data[i][:,3].size > 1700:

        sns.plt.figure(figsize=(12,4))

        sns.plt.title(new_data[i][0][-2])

        sns.plt.xlabel("Time(hour)")

        sns.plt.ylabel(new_data[i][0][-2])

        sns.plt.xlim(0,24)

        sns.countplot((new_data[i][:,3]).astype(int))
for i in range(DATA.shape[0]):

    DATA[i,2] = DATA[i,2][:-3]
for i in range(reason.size):

    new_data[i] = DATA[np.where(DATA[:,4] == reason[i])]
for i in range(reason.size):

    if new_data[i][:,2].size > 1700:

        sns.plt.figure(figsize=(12,4))

        sns.plt.title(new_data[i][0][-2])

        sns.plt.xlabel("month")

        sns.plt.ylabel(new_data[i][0][-2])

        sns.countplot(new_data[i][:,2])
all_ = np.zeros(df["timeStamp"].values.size,dtype='O')

for i in range(all_.size):

    all_[i] = df['timeStamp'].values[i][:7]
sns.plt.figure(figsize=(12,4))

sns.plt.title("All situations by month")

sns.countplot(all_)
all_ = np.zeros(df["timeStamp"].values.size,dtype='O')

for i in range(all_.size):

    all_[i] = df['timeStamp'].values[i][:10]
for i in range(all_.size):

    w = np.array(all_[i].split('-')).astype(int)

    all_[i] = week[calendar.weekday(w[0],w[1],w[2])]
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("Week day")

sns.plt.title("All Situations by Week day")

sns.countplot(all_,order = week)
labels = "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"

sizes = [np.sum(all_ == "Monday"),np.sum(all_ == "Tuesday"),np.sum(all_ == "Wednesday"),np.sum(all_ == "Thursday"),np.sum(all_ == "Friday"),\

         np.sum(all_ == "Saturday"),np.sum(all_ == "Sunday")]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','magenta','orange','lightgreen']

explode = (0, 0, 0, 0, 0.3, 0, 0)  # explode 1st slice

plt.figure(figsize=(8,8))

# Plot

plt.title('Week day')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()
all_ = np.zeros(df["timeStamp"].values.size,dtype='O')

for i in range(all_.size):

    h = np.array(df['timeStamp'].values[i][11:].split(":")).astype(int)

    all_[i] = (h[0] * 3600 + h[1] * 60 + h[2])/3600
all_ = all_.astype(int)
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("hour")

sns.plt.title("All Situations by time")

sns.countplot(all_)
city = list()

d = set()

for i in range(all_.size):

    city.append(df['twp'].values[i])

    d.add(city[i])

d.discard(np.nan)

for i in range(all_.size):

    if df['twp'].values[i] in d:

        city.append(df['twp'].values[i])
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("City")

sns.plt.title("ALL Situations by time")

sns.countplot(city,order = d) #alphabet order
d
TIME = np.zeros(all_.size, dtype = "O")

for i in range(all_.size):

    for j in range(len(df['desc'][i])):

        if df['desc'][i][j] == ':':

            TIME[i] = (df['desc'][i][j-2:j+6])

            break

idx = []

for i in range(TIME.size):

    try:

        TIME[i] = (int((TIME[i]).split(':')[0])*3600 + int((TIME[i]).split(':')[1])*60 + int((TIME[i]).split(':')[2]))/3600

    except:

        TIME[i] = DATA[i,3]

diff = np.zeros(all_.size)

for i in range(all_.size):

    diff[i] = min(np.abs(DATA[i,3] - TIME[i]),24 - np.abs(DATA[i,3] - TIME[i]))
plt.figure(figsize=(12,8))

plt.ylabel("difference between time in desc and timeStamp(in hours)")

plt.xlabel("number of incident")

plt.plot(diff)
number_ = np.zeros(reason.size)

for i in range(number_.size):

    number_[i] = new_data[i].shape[0]
plt.figure(figsize=(12,4))

plt.xlabel("Reason")

plt.plot(number_)
data_matrix = []

for i in range(reason.size):

    data_matrix.append(tuple([np.hstack((reason.reshape(-1,1),number_.reshape(-1,1).astype(int)))][0][i]))
dtype = [('name', 'S80'), ('number', int)]

a = np.array(data_matrix,dtype=dtype)

sorted_a = np.sort(a, order='number')  

sorted_a = sorted_a[::-1]
data_matrix = [['reason','number of incidents']]

for i in range(reason.size):

    data_matrix.append([str(sorted_a[i][0])[2:-1],int((sorted_a[i][1]))])
pd.DataFrame(data_matrix)
type_of_reason_ = np.zeros(DATA.shape[0],dtype='O')

for i in range(type_of_reason_.size):

    type_of_reason_[i] = DATA[i][4].split(' ')[0][:-1]
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("type of incident")

sns.plt.title("All Situations by time")

sns.countplot(type_of_reason_)
Traffic = DATA[type_of_reason_ == 'Traffic']

EMS = DATA[type_of_reason_ == 'EMS']

Fire = DATA[type_of_reason_ == 'Fire']
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("hour")

sns.plt.title("All Traffic incidents")

sns.countplot(Traffic[:,3].astype(int))
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("hour")

sns.plt.title("All EMS incidents")

sns.countplot(EMS[:,3].astype(int))
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("hour")

sns.plt.title("All Fire incidents")

sns.countplot(Fire[:,3].astype(int))
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("month")

sns.plt.title("All Traffic incidents")

sns.countplot(Traffic[:,2])
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("month")

sns.plt.title("All EMS incidents")

sns.countplot(EMS[:,2])
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("month")

sns.plt.title("All Fire incidents")

sns.countplot(Fire[:,2])
DATA = np.zeros((df.shape[0],6),dtype='O')

DATA[:,0] = df['lng'].values

DATA[:,1] = df['lat'].values

DATA[:,4] = df['title'].values

DATA[:,5] = df['twp'].values

for i in range(DATA.shape[0]):

    DATA[i,2] = df['timeStamp'].values[i][:10]

    DATA[i,3] = df['timeStamp'].values[i][10:]

    sp = DATA[i,3].split(':')

    DATA[i,3] = (int(sp[0])*3600 + int(sp[1])*60 + int(sp[2]))/3600

Traffic = DATA[type_of_reason_ == 'Traffic']

EMS = DATA[type_of_reason_ == 'EMS']

Fire = DATA[type_of_reason_ == 'Fire']
week_traffic = np.zeros(Traffic.shape[0],dtype = 'O')

for i in range(week_traffic.size):

    w = np.array(Traffic[i][2].split('-')).astype(int)

    week_traffic[i] = week[calendar.weekday(w[0],w[1],w[2])]
week_EMS = np.zeros(EMS.shape[0],dtype = 'O')

for i in range(week_EMS.size):

    w = np.array(EMS[i][2].split('-')).astype(int)

    week_EMS[i] = week[calendar.weekday(w[0],w[1],w[2])]
week_fire = np.zeros(Fire.shape[0],dtype = 'O')

for i in range(week_fire.size):

    w = np.array(Fire[i][2].split('-')).astype(int)

    week_fire[i] = week[calendar.weekday(w[0],w[1],w[2])]
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("week day")

sns.plt.title("All Traffic incidents")

sns.countplot(week_traffic,order=week)
all_ = week_traffic

labels = "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"

sizes = [np.sum(all_ == "Monday"),np.sum(all_ == "Tuesday"),np.sum(all_ == "Wednesday"),np.sum(all_ == "Thursday"),np.sum(all_ == "Friday"),\

         np.sum(all_ == "Saturday"),np.sum(all_ == "Sunday")]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','magenta','orange','lightgreen']

explode = (0, 0, 0.1, 0, 0.1, 0, 0)  # explode 1st slice

plt.figure(figsize=(8,8))

# Plot

plt.title('Traffic by Week day')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("week day")

sns.plt.title("All EMS incidents")

sns.countplot(week_EMS,order=week)
all_ = week_EMS

labels = "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"

sizes = [np.sum(all_ == "Monday"),np.sum(all_ == "Tuesday"),np.sum(all_ == "Wednesday"),np.sum(all_ == "Thursday"),np.sum(all_ == "Friday"),\

         np.sum(all_ == "Saturday"),np.sum(all_ == "Sunday")]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','magenta','orange','lightgreen']

explode = (0, 0, 0, 0, 0.1, 0, 0)  # explode 1st slice

plt.figure(figsize=(8,8))

# Plot

plt.title('EMS by Week day')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()
sns.plt.figure(figsize=(12,4))

sns.plt.xlabel("week day")

sns.plt.title("All Fire incidents")

sns.countplot(week_fire,order=week)
all_ = week_fire

labels = "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"

sizes = [np.sum(all_ == "Monday"),np.sum(all_ == "Tuesday"),np.sum(all_ == "Wednesday"),np.sum(all_ == "Thursday"),np.sum(all_ == "Friday"),\

         np.sum(all_ == "Saturday"),np.sum(all_ == "Sunday")]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','magenta','orange','lightgreen']

explode = (0, 0, 0.1, 0, 0., 0, 0)  # explode 1st slice

plt.figure(figsize=(8,8))

# Plot

plt.title('Fire by Week day')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()
CITY = np.unique((city))

city = np.array(city)
CITY_matrix = []

for i in range(CITY.size - 1):

    CITY_matrix.append(tuple((CITY[i],np.sum(city == CITY[i]))))
dtype = [('name', 'S80'), ('number', int)]

a = np.array(CITY_matrix,dtype=dtype)

sorted_a = np.sort(a, order='number')  

sorted_a = sorted_a[::-1]
CITY_matrix = [['Township','number of incidents']]

for i in range(CITY.size-1):

    CITY_matrix.append([str(sorted_a[i][0])[2:-1],int((sorted_a[i][1]))])
pd.DataFrame(CITY_matrix)