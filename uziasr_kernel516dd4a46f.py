# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import matplotlib.gridspec as gridspec

from scipy import stats

import seaborn as sns

sns.set(color_codes=True)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing heart.csv as a DataFrame called, heart

heart = pd.read_csv('../input/heart.csv')
heart.info()
heart.head()
#function to make labeling visualizations easier

def set_title_x_y(axis,title,x,y,size=10):

    """Function that sets title, x, and, y"""

    axis.set_title(title,size=size)

    axis.set_xlabel(x,size=size)

    axis.set_ylabel(y,size=size)
#Creating male and female dfs

female = heart[heart.sex==0]

male = heart[heart.sex==1]

len(male), len(female)
fig,ax=plt.subplots(1,1,figsize=(12,8),constrained_layout=False)

#ax[0].hist(female.age,stacked=True)

ax.hist([female.age,male.age],stacked=False,color=['green','blue'],edgecolor='black', linewidth=1.5,bins=8)

ax.grid(axis='y')

ax.legend(['Female','Male'])

set_title_x_y(ax,'Men and Woman Heart Disease by Age','Age','Frequency',15)

print('Female'),print(female.age.describe(),'\n\nMale'), print(male.age.describe())
fig,ax = plt.subplots(1,2,figsize=(15,8))

ax[0].hist(male.chol);

ax[0].set_title('Male');

ax[1].hist(female.chol, color='green');

ax[1].set_title('Female');

for a in ax:

    a.set_xlabel('Cholesterol Level',size=14)
heart.groupby('sex').chol.mean()

len(heart[heart.chol>240])

print('Cholesterol over 240 mg/dL\n')

print('Male')

print(((male.chol>240).value_counts()).sort_index(ascending=False))

print('\nFemale')

print(((female.chol>240).value_counts()))
high_cholesterol = [57,94]

low_cholesterol = [39,113]

observed = np.array([high_cholesterol,low_cholesterol])

chi2, p, dof, expected = stats.chi2_contingency(observed, correction=False)

print('(Chi2 statistic: {}, probability: {}, degrees of freedom: {})'.format(chi2,p,dof))
plt.figure(figsize=(10,10));

heart.thalach.hist(),male.thalach.hist(),female.thalach.hist();

plt.legend(['All','Male','Female']);

plt.xlabel('Maximum Heart Rate', size=15);

plt.ylabel('Frequency',size=15);
all_ages = np.array(((heart.age)).values)
heart['age_group'] = pd.cut(all_ages,4, labels=['28 - 40','41 - 52', '53 - 64', '65 - 76']);
plt.figure(figsize=(6,7));

((heart.age_group.value_counts()).sort_index()).plot(kind='bar');

plt.xticks(rotation=0);
fig,ax = plt.subplots(1,3,figsize=(20,9))

heart_rate_age_group = ((heart.groupby('age_group').thalach.value_counts()))

blood_pressure_age_group = ((heart.groupby('age_group').trestbps.value_counts()))

cholesterol_age_group = ((heart.groupby('age_group').chol.value_counts()))

groups = [heart_rate_age_group,blood_pressure_age_group,cholesterol_age_group]

x_labels = ['Heart Rate', 'Blood Pressure', 'Cholesterol ']

for group in (heart.age_group.unique()).sort_values():

    ax[0].scatter(x=heart_rate_age_group.loc[group].index, y=heart_rate_age_group.loc[group].values,s=80)

    ax[1].scatter(x=blood_pressure_age_group.loc[group].index, y=blood_pressure_age_group.loc[group].values,s=80)

    ax[2].scatter(x=cholesterol_age_group.loc[group].index, y=cholesterol_age_group.loc[group].values,s=80,)

for axis, x_label in zip(ax, x_labels):

    axis.set_xlabel(x_label, size=14)

ax[0].set_ylabel('Frequency',size=14)

plt.legend(heart.age_group.unique().sort_values());

heart.age_group.value_counts()
gs = gridspec.GridSpec(5,5)

plt.figure(figsize=(10,10));



x = heart.chol.values

y = heart.trestbps.values



ax = plt.subplot(gs[1:7,:4]);

ax_2 = plt.subplot(gs[1:7,4:7]);

ax_3 = plt.subplot(gs[:1,:4]);



#plots

ax.scatter(x,y);

ax_2.hist(y, orientation='horizontal',bins=10);

ax_3.hist(x,bins=10);



ax_2.tick_params(axis='y',which='both',left=False,labelleft=False,labelright=True);

ax_3.tick_params(axis='x',bottom=False,which='both',labelbottom=False, labeltop=True);



ax.set_xlabel('Serum Cholesterol Level', size=14);

ax.set_ylabel('Resting Blood Pressure', size=14);



ax.grid(True)

ax_2.grid(axis='y')

ax_3.grid(axis='x')

x.mean(), y.mean()



slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

plot = [((i*slope)+intercept)for i in range(600)]

ax.plot(plot,color='green');

print('The slope of the regression is {}, The line intercepts the y-axis at {}'.format(slope,intercept))
sns.jointplot(x=x, y=y, kind="reg", height=10,);

plt.xlabel('Serum Cholesterol Level',size=15);

plt.ylabel('Resting Blood Pressure', size=15);

plt.axhline(y.mean());

plt.axvline(x.mean());
# A graph like above, but in its own way

# fig,ax = plt.subplots(1,1,figsize=(10,10))

# #plt.figure(figsize=(10,10))

# plt.axhline(y.mean())

# plt.axvline(x.mean())

# sns.regplot(x,y);

# plt.grid(True)

# set_title_x_y(ax,'','Serum Cholesterol Level','Resting Blood Pressure',15)
heart[heart.fbs==1].head()
#how many people in this data set have a fasting blood sugar over 120

len(heart[heart.fbs==1])/len(heart)
fig,ax = plt.subplots(1,1,figsize=(8,8));

ax.hist(heart.cp,);

ax.set_xlabel('Chest Pain', size=14);
print('Age')

print(heart.groupby('cp').age.mean())

print('\nMaximum Heart Rate')

print(heart.groupby('cp').thalach.mean())

print('\nResting Blood Pressre')

print(heart.groupby('cp').trestbps.mean())

print('\nCholesterol')

print(heart.groupby('cp').chol.mean())