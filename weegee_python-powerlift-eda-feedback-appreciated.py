import seaborn as sns
sns.set(style='whitegrid')
%pylab inline
import numpy as np
import pandas as pd
import missingno as msn
df = pd.read_csv("../input/openpowerlifting.csv")
df.describe(include='all')
df.info()
msn.matrix(df, sort='ascending')
for feature in df:
    missing = df[feature].isnull().sum()
    perc_missing = round(missing/df.shape[0]*100,2)
    print("{} has {} missing entries.".format(feature, missing))
    print("That's {} % missing.".format(perc_missing))
    print('*'*44)
grouped = df.groupby("Name")

m = 0
f = 0
for i in grouped:
    if i[1]["Sex"].iloc[0] == 'M':
        m += 1
    else:
        f += 1
print("The data is composed of {} different meetings.".format(len(df["MeetID"].value_counts())))
print("Overall {} individual athletes are in the dataset.".format(len(df["Name"].value_counts())))
print("Of those {} are male, and {} female.".format(m, f))
print("The type of Equipment used:")
print(df["Equipment"].value_counts())
for i in df:
    if df[i].dtype != 'O':
        vmin = df[i].min()
        vmax = df[i].max()
        vmean = df[i].mean()
        vmedian = df[i].median()
        print(i)
        print("min: {}".format(vmin))
        print("max: {}".format(vmax))
        print("mean: {}".format(round(vmean,2)))
        print("median: {}".format(vmedian))
        print('*'*20)
plt.figure(figsize=(15,30))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Age"],20), hue=df["Sex"])

plt.subplot(6,2,2)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["BodyweightKg"],20), hue=df["Sex"])

plt.subplot(6,2,3)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Squat4Kg"],20), hue=df["Sex"])

plt.subplot(6,2,4)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["BestSquatKg"],20), hue=df["Sex"])

plt.subplot(6,2,5)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Bench4Kg"],20), hue=df["Sex"])

plt.subplot(6,2,6)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["BestBenchKg"],20), hue=df["Sex"])

plt.subplot(6,2,7)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Deadlift4Kg"],20), hue=df["Sex"])

plt.subplot(6,2,8)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["BestDeadliftKg"],20), hue=df["Sex"])

plt.subplot(6,2,9)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Wilks"],20), hue=df["Sex"])

plt.subplot(6,2,10)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["TotalKg"],20), hue=df["Sex"])

plt.subplot(6,2,11)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
sns.countplot(pd.qcut(df["Wilks"],20), hue=df["Sex"])
plt.figure(figsize=(15,15))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["Age"][df["Sex"] == 'M'], df["BodyweightKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["BodyweightKg"][df["Sex"] == 'F'], alpha=.1)

plt.figure(figsize=(15,30))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["Age"][df["Sex"] == 'M'], df["BestSquatKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestSquatKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,2)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["BestSquatKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestSquatKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,3)
plt.scatter(df["Age"][df["Sex"] == 'M'], df["BestBenchKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestBenchKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,4)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["BestBenchKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestBenchKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,5)
plt.scatter(df["Age"][df["Sex"] == 'M'], df["BestDeadliftKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestDeadliftKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,6)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["BestDeadliftKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestDeadliftKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,7)
plt.scatter(df["Age"][df["Sex"] == 'M'], df["TotalKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["TotalKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,8)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["TotalKg"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["TotalKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,9)
plt.scatter(df["Age"][df["Sex"] == 'M'], df["Wilks"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["Age"][df["Sex"] == 'F'], df["Wilks"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,10)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'M'], df["Wilks"][df["Sex"] == 'M'], alpha=.1)
plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["Wilks"][df["Sex"] == 'F'], alpha=.1)


plt.figure(figsize=(15,15))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["BodyweightKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["BodyweightKg"][df["Sex"] == 'F'], alpha=.1)

plt.figure(figsize=(15,30))
plt.subplot(6,2,1)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["BestSquatKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestSquatKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,2)
#plt.hist(df["BodyweightKg"].dropna(), bins=30);
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["BestSquatKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestSquatKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,3)
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["BestBenchKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestBenchKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,4)
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["BestBenchKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestBenchKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,5)
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["BestDeadliftKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["BestDeadliftKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,6)
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["BestDeadliftKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["BestDeadliftKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,7)
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["TotalKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["TotalKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,8)
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["TotalKg"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["TotalKg"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,9)
plt.scatter(df["Age"][df["Place"] != 'DQ'], df["Wilks"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["Age"][df["Sex"] == 'F'], df["Wilks"][df["Sex"] == 'F'], alpha=.1)

plt.subplot(6,2,10)
plt.scatter(df["BodyweightKg"][df["Place"] != 'DQ'], df["Wilks"][df["Place"] != 'DQ'], alpha=.1)
# plt.scatter(df["BodyweightKg"][df["Sex"] == 'F'], df["Wilks"][df["Sex"] == 'F'], alpha=.1)


df = df[df["Place"] != 'DQ']
cdf = df.select_dtypes(exclude='O')
corr = cdf.drop('MeetID',axis=1).corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr,annot=True, square=True, cmap='magma')
df["BestSquatKg"].fillna(df["Squat4Kg"], inplace=True)
df["BestBenchKg"].fillna(df["Bench4Kg"], inplace=True)
df["BestDeadliftKg"].fillna(df["Deadlift4Kg"], inplace=True)

df.drop(["Squat4Kg", "Bench4Kg", "Deadlift4Kg"], axis=1, inplace=True)
cdf = df.select_dtypes(exclude='O')
corr = cdf.drop('MeetID',axis=1).corr()
plt.figure(figsize=(15,15))
sns.heatmap(corr,annot=True, square=True, cmap='magma')
msn.matrix(df, sort='ascending')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


trace1 = go.Scatter3d(
    x=df["BestSquatKg"].iloc[:555],
    y=df["BestBenchKg"].iloc[:555],
    z=df["BestDeadliftKg"].iloc[:555],
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

data = [trace1]


layout = go.Layout(        
    title='Relation of single presses',
    hovermode='closest',
    scene = dict(
                    xaxis = dict(
                        title='BestSquatKg'),
                    yaxis = dict(
                        title='BestBenchKg'),
                    zaxis = dict(
                        title='BestDeadliftKg'),),
                    width=700,
                    
                  )
  

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')
df.columns
stat_min = 500
div = (df["Division"].value_counts() < stat_min)
df["Division"].fillna('Misc', inplace=True)
df["Division"] =  df["Division"].apply(lambda x: 'Misc' if div.loc[x] == True else x)
grouped = df.groupby("Equipment")
plt.figure(figsize=(15,15))
for e,i in enumerate(grouped):
    print("Stats for {}".format(i[0]))
    for j in i[1]:
        if i[1][j].dtype != 'O' and j != 'MeetID':
            vmin = i[1][j].min()
            vmax = i[1][j].max()
            vmean = i[1][j].mean()
            vmedian = i[1][j].median()
            print(j)
            print("    min: {}".format(vmin))
            print("    max: {}".format(vmax))
            print("    mean: {}".format(round(vmean,2)))
            print("    median: {}".format(vmedian))
            print('- '*10)
    print('*'*20)
    
    plt.subplot(3,2,e+1)
    plt.title(i[0])
    sns.barplot(x=i[1]["Division"].value_counts()[:10], y=i[1]["Division"].value_counts()[:10].index)
sns.violinplot(x='Equipment', y='TotalKg', data=df, hue='Sex')