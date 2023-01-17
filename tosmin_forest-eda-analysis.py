# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for plotting
import seaborn as sns #for plotting
%matplotlib inline 


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/forest/train.csv')
df.head(3)
df.tail(3)
df.shape
df.describe() # mean median sd all the basic statistic is obseved.
df.info()
df.columns
df.dtypes #datatype of each column is observed and it is seen that all are int type
df['Cover_Type'].value_counts() # value counts frelated to cover_type
df["Cover_Type"].value_counts().plot(kind='bar',color='purple') #plotting the value counts
plt.ylabel("occurences")
plt.xlabel("Cover Type")
plt.bar(df['Cover_Type'], df['Elevation']) #plottinfg with cover type and elevation in bar plot and scatter plot.
plt.scatter(df['Cover_Type'], df['Elevation'])
#box plot for cover type and elevation
x=df.groupby('Cover_Type')
x.boxplot(column='Elevation')
x=df.groupby('Cover_Type')
x.boxplot(column='Aspect') #with Aspect
x=df.groupby('Cover_Type')
x.boxplot(column='Slope') #with slope
x=df.groupby('Cover_Type')
x.boxplot(column='Horizontal_Distance_To_Hydrology')
x=df.groupby('Cover_Type')
x.boxplot(column='Horizontal_Distance_To_Roadways')
# Extract column from the dataset to do specific plotting
cl = df.columns.tolist()
for name in cl:
    if name[0:4] != 'Soil' and name[0:4] != 'Wild' and name != 'Id' and name != 'Cover_Type':
        plt.figure()
        sns.distplot(df[name])
for name in cl:
    if name[0:4] != 'Soil' and name[0:4] != 'Wild' and name != 'Id' and name != 'Cover_Type':
        title = name + ' vs Cover Type'
        plt.figure()
        sns.stripplot(df["Cover_Type"],df[name],jitter=True)
        plt.title(title);
y = [x for x in df.columns.tolist() if "Soil_Type" not in x]
y = [x for x in y if "Wilderness" not in x]
dfnew = df.reindex(columns=y)
dfnew.head(4)
cor1=dfnew.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cor1, vmax=.7, square=True);
y1 = dfnew.columns.tolist()
remove_cl = ['Id', 'Slope', 'Aspect', 'Cover_Type']
y1 = [x for x in y1 if y1 not in remove_cl]
y1
plt1=sns.pairplot(df, vars=y1, hue="Cover_Type") #A pairplot plot a pairwise relationships in a dataset.
cwild = [x for x in df.columns.tolist() if "Wilderness" in x]
t = df[cwild].groupby(df['Cover_Type']).sum()
m = t.T.plot(kind='bar', figsize=(10, 10), legend=True, fontsize=15) #here we cant put scatter plot as we now it need x and y variable
m.set_xlabel("Wilderness_Type", fontsize=15)
m.set_ylabel("Count", fontsize=15)
plt.show()
s = np.array(cl)
st = [item for item in s if "Soil" in item]
for soil_type in st:
    print (soil_type, df[soil_type].sum())
z = df[st].groupby(df['Cover_Type']).sum() #plotting the soli type w.r.t cover type
z.T.plot(kind='barh', stacked=True, figsize=(15,10))
