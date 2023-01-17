# Import required libraries

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline

import warnings

warnings.filterwarnings('ignore') 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Read datas

data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.head()
data.Nationality.unique()
data.info()
data.Club.unique()
data.Nationality.value_counts()
# Overall of each nationality

area_list = list(data['Nationality'].unique())

overall_ratio = []

for i in area_list:

    x = data[data['Nationality']==i]

    overall_rate = sum(x.Overall)/len(x)

    overall_ratio.append(overall_rate)

data = pd.DataFrame({'area_list': area_list,'overall_ratio':overall_ratio})

new_index = (data['overall_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# visualization

plt.figure(figsize=(150,100))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['overall_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('Nationality')

plt.ylabel('Overall')

plt.title('Overall Given Nationality')

plt.show()
data = pd.read_csv('/kaggle/input/fifa19/data.csv')

data.head()
#Let's Eye on Turkish Footballers

data1 = data[data['Nationality'] == "Turkey"][['Name','Overall','Potential','Position']]

data1
# Most common 25 Surname of Turkish Footballers

separate = data1.Name.str.split() 

a,b = zip(*separate)                    

name_list = b                         

name_count = Counter(name_list)         

most_common_names = name_count.most_common(25)  

x,y = zip(*most_common_names)

x,y = list(x),list(y)

# 

plt.figure(figsize=(25,10))

ax= sns.barplot(x=x, y=y,palette = "Blues_d")

plt.xlabel('Surname of Turkish Footballer')

plt.ylabel('Frequency')

plt.title('Most common 25 Surname')

plt.show()
data1.head()
# Overall of each position

area_list = list(data1['Position'].unique())

overall_ratio = []

for i in area_list:

    x = data1[data1['Position']==i]

    overall_rate = sum(x.Overall)/len(x)

    overall_ratio.append(overall_rate)

data = pd.DataFrame({'area_list': area_list,'overall_ratio':overall_ratio})

new_index = (data['overall_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# visualization

plt.figure(figsize=(30,20))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['overall_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('Position')

plt.ylabel('Overall')

plt.title('Overall Given Position')

plt.show()
# Potential of each position

area_list = list(data1['Position'].unique())

potential = []

for i in area_list:

    x = data1[data1['Position']==i]

    potential_rate = sum(x.Potential)/len(x)

    potential.append(potential_rate)

data = pd.DataFrame({'area_list': area_list,'potential': potential})

new_index = (data['potential'].sort_values(ascending=False)).index.values

sorted_data2 = data.reindex(new_index)



# visualization

plt.figure(figsize=(30,20))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['potential'])

plt.xticks(rotation= 90)

plt.xlabel('Position')

plt.ylabel('Potential')

plt.title('Potential Given Position')

plt.show()
data = pd.read_csv('/kaggle/input/fifa19/data.csv')

data.head()
# Overall vs Potential of each Position

sorted_data['overall_ratio'] = sorted_data['overall_ratio']/max(sorted_data['overall_ratio'])

sorted_data2['potential'] = sorted_data2['potential']/max(sorted_data2['potential'])

turkey_data = pd.concat([sorted_data,sorted_data2['potential']],axis=1)

turkey_data.sort_values('overall_ratio',inplace=True)



# visualize

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',y='overall_ratio',data=turkey_data,color='lime',alpha=0.8)

sns.pointplot(x='area_list',y='potential',data=turkey_data,color='red',alpha=0.8)

plt.xlabel('Position',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('Overall vs Potential',fontsize = 20,color='blue')

plt.grid()
turkey_data.head()
# Visualization of potential vs overall of each position with different style of seaborn code

# joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

g = sns.jointplot(turkey_data.overall_ratio, turkey_data.potential, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

g = sns.jointplot("overall_ratio", "potential", data=turkey_data,size=5, ratio=3, color="g")
data = pd.read_csv('/kaggle/input/fifa19/data.csv')

data["Preferred Foot"].head()
data["Preferred Foot"].dropna(inplace = True)

labels = data["Preferred Foot"].value_counts().index

colors = ['blue','yellow']

explode = [0,0]

sizes =data["Preferred Foot"].value_counts().values



# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Footballer Preferred Foot',color = 'red',fontsize = 15)

plt.show()
turkey_data.head()
# Visualization of Overall vs Potential of every position with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

sns.lmplot(x="overall_ratio", y="potential", data=turkey_data)

plt.show()
# Visualization with different style of seaborn code

# cubehelix plot

sns.kdeplot(turkey_data.overall_ratio, turkey_data.potential, shade=True, cut=3)

plt.show()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.7)

sns.violinplot(data=turkey_data, palette=pal, inner="points")

plt.show()
turkey_data.corr()
#correlation map

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(turkey_data.corr(), annot=True, linewidths=0.5,linecolor="blue", fmt= '.1f',ax=ax)

plt.show()
data.head()
sns.boxplot(x="Real Face", y="Age", hue="Preferred Foot", data=data, palette="PRGn")

plt.show()
data = data.iloc[0:500,:]
sns.swarmplot(x="Real Face", y="Age",hue="Preferred Foot", data=data)

plt.show()
# pair plot

sns.pairplot(turkey_data)

plt.show()
data = pd.read_csv('/kaggle/input/fifa19/data.csv')

data.head()
sns.countplot(data["Preferred Foot"])

plt.title("Foot",color = 'blue',fontsize=15)