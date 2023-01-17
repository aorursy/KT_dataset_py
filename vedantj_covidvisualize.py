import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Files Setup
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


individual  = pd.read_csv('/kaggle/input/covid19-in-india/IndividualDetails.csv')
stateWise   = pd.read_csv('/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv')
hospitalBed = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv') 
covidIndia  = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv')
icmrTest    = pd.read_csv('/kaggle/input/covid19-in-india/ICMRTestingLabs.csv')
individual = individual.dropna() # Cleaning all entries with NaN values as they skew the visualization
individual.head()
day_wise = individual["diagnosed_date"].value_counts()
day_wise = day_wise.to_frame()
day_wise = day_wise.rename(columns = {"diagnosed_date":"Cases Count"})
day_wise["Date"] = day_wise.index

g = sns.relplot(y="Cases Count", x="Date", data = day_wise, height=7, aspect=11.7/8.27)
g.set_xticklabels(rotation=90)
state_wise = individual["detected_state"].value_counts()
state_wise = state_wise.to_frame()
state_wise = state_wise.rename(columns = {"detected_state":"State Detect Count"})
state_wise["State"] = state_wise.index
plt.figure(figsize=(16, 6))
sns.barplot(x="State", y="State Detect Count", data = state_wise)
combined = pd.DataFrame(columns=['Date','State','Count'])
state  = []
date   = []
netcnt = []
for i,frame in individual.groupby(['detected_state','diagnosed_date']):
    net_count = len(frame)
    netcnt.append(net_count)
    state.append(frame['detected_state'].iloc[0])
    date.append(frame['diagnosed_date'].iloc[0])
combined['Date']  = date
combined['State'] = state
combined['Count'] = netcnt
plt.figure(figsize=(40, 40))
s = sns.catplot(x="State", y="Count", hue="Date", data=combined)
s.set_xticklabels(rotation=90)
    
stateWise
stateWise = stateWise.dropna()
combo_india = pd.DataFrame(columns=["Date","Positive","Negative","Total"])
dates = []
pos = []
neg = []
tot = []
for i, frame in stateWise.groupby("Date"):
    dates.append(frame["Date"].iloc[0])
    pos.append(frame["Positive"].sum())
    tot.append(frame["TotalSamples"].sum())
    neg.append(frame["TotalSamples"].sum() - frame["Positive"].sum())
combo_india["Date"] = dates
combo_india["Positive"] = pos
combo_india["Negative"] = neg
combo_india["Total"] = tot
plt.figure(figsize=(30, 10))
sns.lineplot(x="Date", y="Total", data=combo_india, legend='brief',label="Total")
p = sns.lineplot(x="Date", y="Positive", data=combo_india, legend='brief',label="Pos")
p.set_xticklabels(rotation=90,labels=combo_india["Date"])
ax = sns.lineplot(x="Date", y="Negative", data=combo_india, legend='brief',label="Neg")
ax.set(xlabel='Date', ylabel='Samples Distribution')
hospitalBed
hospitalBed = hospitalBed.dropna()
hospitalBed = hospitalBed.iloc[0:29]
hospitalBed["NumPrimaryHealthCenters_HMIS"] = hospitalBed["NumPrimaryHealthCenters_HMIS"].astype(int)

plt.figure(figsize=(10,5))
sns.lineplot(x="State/UT", y="NumPrimaryHealthCenters_HMIS",data=hospitalBed,label="Primary Health Care Center")
bed = sns.lineplot(x="State/UT", y="NumCommunityHealthCenters_HMIS",data=hospitalBed,label="Community Health Care Center")
bed.set_xticklabels(rotation=90,labels=hospitalBed["State/UT"])
px = sns.lineplot(x="State/UT", y="NumDistrictHospitals_HMIS",data=hospitalBed,label="District Health Care Center")
px.set(xlabel='States', ylabel='Beds Distribution')
covidIndia
ageGroup    = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
ageGroup["age"] = ageGroup["AgeGroup"]
ageGroup["Cases Distribution"] = [float(i[:-1])*0.01 for i in ageGroup["Percentage"]]
ageGroup = ageGroup.set_index("AgeGroup")
ageGroup.plot.pie(y="Cases Distribution",figsize=(10,10))
ageGroup
plt.figure(figsize=(10,10))
sns.distplot(ageGroup["Cases Distribution"])
