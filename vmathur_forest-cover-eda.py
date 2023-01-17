# Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
filename = '../input/train.csv'
# Put the data into a dataframe

df = pd.read_csv(open(filename))
df.describe()
df.info()
# How many samples of each cover type are there?

df["Cover_Type"].value_counts().plot(kind='bar',color='gold')

plt.ylabel("Number of Occurences")

plt.xlabel("Cover Type")
# Extract column names from the dataset

col_names = df.columns.tolist()
for name in col_names:

    if name[0:4] != 'Soil' and name[0:4] != 'Wild' and name != 'Id' and name != 'Cover_Type':

        plt.figure()

        sns.distplot(df[name]);
for name in col_names:

    if name[0:4] != 'Soil' and name[0:4] != 'Wild' and name != 'Id' and name != 'Cover_Type':

        title = name + ' vs Cover Type'

        plt.figure()

        sns.stripplot(df["Cover_Type"],df[name],jitter=True)

        plt.title(title);
vars = [x for x in df.columns.tolist() if "Soil_Type" not in x]

vars = [x for x in vars if "Wilderness" not in x]

df1 = df.reindex(columns=vars)
corrmat = df1.corr()

f, ax = plt.subplots(figsize=(12, 7))

sns.heatmap(corrmat, vmax=.5, square=True);
drop_cols = ['Horizontal_Distance_To_Hydrology', 'Hillshade_3pm']

df1 = df1.drop(drop_cols, axis=1)
# So which variables are we plotting?

vars = df1.columns.tolist()

remove_cols = ['Id', 'Slope', 'Aspect', 'Cover_Type']

vars = [x for x in vars if x not in remove_cols]

vars
g = sns.pairplot(df, vars=vars, hue="Cover_Type")
col_names_wilderness = [x for x in df.columns.tolist() if "Wilderness" in x]
types_sum = df[col_names_wilderness].groupby(df['Cover_Type']).sum()
ax = types_sum.T.plot(kind='bar', figsize=(13, 7), legend=True, fontsize=12)

ax.set_xlabel("Wilderness_Type", fontsize=12)

ax.set_ylabel("Count", fontsize=12)

plt.show()
# How many of each Soil_Type are there?

A = np.array(col_names)

soil_types = [item for item in A if "Soil" in item]

for soil_type in soil_types:

    print (soil_type, df[soil_type].sum())
# Which soil_types support which cover_types?

types_sum = df[soil_types].groupby(df['Cover_Type']).sum()

types_sum.T.plot(kind='bar', stacked=True, figsize=(13,8), cmap='jet')
# Lets look at it another way.

arr = []



for i in range(1,8):

    for j in range(1,41):

        result = []

        result.append(i)

        result.append(j)

        mystr = 'Soil_Type' + str(j)

        result.append(df[df['Cover_Type'] == i].sum()[mystr])

        arr.append(result)

        

labels = ['Cover_Type', 'Soil_Type' , 'Sum']

df1 = pd.DataFrame.from_records(arr, columns=labels)
plt.figure(figsize=(15,5))

distt = df1.pivot("Cover_Type", "Soil_Type", "Sum")

ax = sns.heatmap(distt)
#Lets drop the columns with 0 samples

drop_cols = [item for item in soil_types if df[item].sum() == 0]

drop_cols
df = df.drop(drop_cols, axis=1)