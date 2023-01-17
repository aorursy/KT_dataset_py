import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
OP = pd.read_csv('../input/eda-data/OperationPerformance_PerRoute-SystemwideOps_8_2-15_2020.csv',header=0)
ridership = pd.read_csv('../input/eda-data/Ridershippertrip_PerTrip-AugustWeekdays9-9.csv',header=0)
ridership.describe()
OP.describe()
ridership.head()
OP.head()
OP.shape,ridership.shape
numeric_features = ridership.select_dtypes(include=[np.number])
numeric_features.columns
numeric_features = numeric_features.dropna(axis=1, how ='all')
numeric_features = numeric_features.drop(labels=['Capacity', 'Full capacity', 
                                                 'Capacity (pract.)', 'Load factor [%]', 'Load factor (pract.)[%]', 'PM factor [%]'], axis=1)
numeric_features.head()
df = pd.DataFrame(numeric_features)
df55 = numeric_features.drop(labels=['No','Trip','Course','Vehicle'],axis=1)
corr = df55.corr()
fig, ax = plt.subplots(figsize = (9,6))
sns.heatmap(corr,annot=True)
df66 = df55.drop(labels=['Block','Min'],axis=1)
sns.pairplot(df66, kind="reg")
df['Line'] = ridership['Line']
df['Date'] = ridership['Date']
df['Sched. start'] = ridership['Sched. start']
df['Sched. end'] = ridership['Sched. end']
df
OP
df_BLUE = df[df['Line'].str.contains('BLUE')]
df_BLUE_402 = df_BLUE.loc[df_BLUE['Block']==402]

for i in range(3,32):
    data = df_BLUE_402.loc[df_BLUE_402['Date']=='8/'+str(i)+'/2020'].reset_index()
    star = data['Sched. start']
    locals()['df_BLUE_'+str(i)] = star
    
for i in range(3,32):
    locals()['x'+str(i)] = []
    locals()['y'+str(i)] = []
    for j in range(len(locals()['df_BLUE_'+str(i)])):
        x_value = locals()['df_BLUE_'+str(i)][j][:-6]
        locals()['x'+str(i)].append(x_value)
        y_value = locals()['df_BLUE_'+str(i)][j][-5:-3]
        locals()['y'+str(i)].append(y_value)


plt.figure(figsize=(12, 9.5))
for i in range(3,32):
    plt.plot(locals()['x'+str(i)],locals()['y'+str(i)],label='8/'+str(i)+'/2020',
             linewidth=3,marker='o',markerfacecolor='blue',markersize=8)

plt.legend(loc="lower right")
plt.xlabel("o'clock")
plt.ylabel("minutes")
plt.title("BLUE line start times")
plt.show()
for i in range(3,32):
    print(i,len(locals()['df_BLUE_'+str(i)]))
date2 = [3,4,5,6,7,10,11,12,13,14,17,18,19,21,24,25,26,27,28,31]
for i in date2:
    locals()['df_BLUE_'+str(i)][20] = '0:00:00'

date = [3,4,5,6,7,10,11,12,13,14,17,18,19,20,21,24,25,26,27,28,31]
aa = {}
for i in date:
    aa['df_BLUE_'+str(i)] = locals()['df_BLUE_'+str(i)]

df_a = pd.DataFrame(aa)
df_a
OP
df_b = pd.DataFrame()
df_b['Line'] = OP['Line']
OP['Sched. time'][1][:-6]
int(OP['Sched. time'][1][:-6])
st = []
for i in range(68):
    c = int(OP['Sched. time'][i][:-6])+int(OP['Sched. time'][i][-5:-3])/60
    st.append(c)

at = []
for i in range(68):
    d = int(OP['Act. time'][i][:-6])+int(OP['Act. time'][i][-5:-3])/60
    at.append(d)

x = []
for i in range(68):
    e = OP['Line'][i]
    x.append(e)
plt.figure(figsize=(18,25))

index_st = np.arange(len(x)) 
index_at = index_st + 0.4 
 
plt.barh(index_st, alpha=0.9, height=0.4, width=st, facecolor = 'lightskyblue', edgecolor = 'white', label='Scheduled time', lw=1)
plt.barh(index_at, alpha=0.9, height=0.4, width=at, facecolor = 'yellowgreen', edgecolor = 'white', label='Actual time', lw=1)
plt.legend(loc="upper left") 
plt.yticks(index_st + 0.4/2, x)
plt.ylabel('Line') 
plt.xlabel('Operation time in August (Unit: hour)') 
plt.title('Comparisons between Actual operation time and Scheduled operation time of routes')
plt.show()
time = {'line': x, 'Sched. time': st, 'Act. time': at}
df_time = pd.DataFrame(time)
df_time['difference'] = df_time['Sched. time']-df_time['Act. time']
df_time
df_time['difference'] = df_time['Sched. time']-df_time['Act. time']
df_time.describe()
df_BLUE
df_BLUE_p = pd.DataFrame(df_BLUE)
df_BLUE_p
df_BLUE_p['Line'].value_counts()
df_BLUE_p2 = df_BLUE_p.groupby(by=['Date'])['Total out'].sum()
df_BLUE_p2 = pd.DataFrame(df_BLUE_p2).reset_index()
df_BLUE_p2['Date'] = pd.to_datetime(df_BLUE_p2['Date'])
df_BLUE_p2 = df_BLUE_p2.sort_values(by = 'Date').reset_index()
df_BLUE_p2
day = []
for i in range(21):
    ss = str(df_BLUE_p2['Date'][i])[:-9]
    day.append(ss)

total_out = []
for i in range(21):
    o = df_BLUE_p2['Total out'][i]
    total_out.append(o)
plt.figure(figsize=(12, 9.5))

plt.plot(day,total_out,linewidth=3,marker='o',markerfacecolor='blue',markersize=8)

plt.xlabel("Date")
plt.ylabel("Total out")
plt.title("BLUE Line Passenger on operation days")
plt.xticks(rotation=45)
plt.show()
df3 = df.groupby(['Date','Line'])['Total out'].sum().reset_index()
df3['Date'] = pd.to_datetime(df3['Date'])
df3 = df3.sort_values(by = 'Date').reset_index()
df3
df4 = df3['Line'].unique()
route = []
for item in df4:
    route.append(item)
len(route)
i = 1
for item in route:
    locals()['p_'+str(i)] = df3.loc[df3['Line']==item].reset_index()
    i = i+1

k = 1
for i in range(1,len(route)+1):
    locals()['x_'+str(k)] = []
    locals()['y_'+str(k)] = []
    for j in range(len(locals()['p_'+str(i)])):
        x_value = locals()['p_'+str(i)]['Date'][j]
        locals()['x_'+str(k)].append(x_value)
        y_value = locals()['p_'+str(i)]['Total out'][j]
        locals()['y_'+str(k)].append(y_value)
    k = k+1
plt.figure(figsize=(18, 12))
for i in range(1,49):
    plt.plot(locals()['x_'+str(i)],locals()['y_'+str(i)],label=route[i-1],
             linewidth=3,marker='o',markerfacecolor='blue',markersize=8)

plt.legend(loc="lower right")
plt.xlabel("Date")
plt.ylabel("Total out")
plt.title("Passengers on operation days of each route")
plt.show()
df5 = df.groupby(['Line'])['Total out'].sum().reset_index()
plt.figure(figsize=(18,12))

x_sp = df5['Line']
y_sp = df5['Total out']
plt.bar(x_sp, y_sp, alpha=0.9, width=0.5, facecolor = 'lightskyblue', edgecolor = 'white', lw=1) 
plt.ylabel('Line') 
plt.xlabel('Total passengers in August') 
plt.title('Comparisons of Total passengers in August between each route')
plt.xticks(rotation=90)
plt.show()
data1 = pd.read_csv('../input/eda-data/Ridershippertrip_PerTrip-AugustWeekdays9-9.csv',header=0)
features_with_na1=[features for features in data1.columns if data1[features].isnull().sum()>1]
for feature in features_with_na1:
    print(feature, np.round(data1[feature].isnull().mean(), 4),  ' % of Missing Values')
data2 = pd.read_csv('../input/eda-data/OperationPerformance_PerRoute-SystemwideOps_8_2-15_2020.csv',header=0)
features_with_na2=[features for features in data2.columns if data2[features].isnull().sum()>1]
for feature in features_with_na2:
    print(feature, np.round(data2[feature].isnull().mean(), 4),  ' % of Missing Values')