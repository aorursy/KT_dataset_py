import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import LabelEncoder

plt.style.use('fivethirtyeight')

from subprocess import check_output

data = pd.read_csv('../input/HR_comma_sep.csv')

print(data.columns)

lbl=LabelEncoder()

data.sales=lbl.fit_transform(data.sales)
x='satisfaction_level'

y='last_evaluation'

leave=1

scatter1=plt.scatter(data[x][data.left == leave],data[y][data.left == leave],s=1.0, c='r', marker='o',alpha = 0.5)

plt.xlabel(x)

plt.ylabel(y)

leave=0

scatter2=plt.scatter(data[x][data.left == leave],data[y][data.left == leave],s=1.0, c='b', marker='o',alpha = 0.05)

legnd=plt.legend([scatter1,scatter2],['left','stayed'],fontsize=14)

plt.title('last Evalution vs Satisfaction aggregated over all jobs')

# Red means left | blue means stayed



fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

x='satisfaction_level' 

y='last_evaluation'

z='sales'

colorrole='rgbcmrbgcmrgbcmrbgcm'

ax.set_xlabel(x)

ax.set_ylabel(y)

#ax.set_zlabel(z)

ax.set_zticklabels(lbl.inverse_transform(np.arange(10)))

ax.set_zticks(np.arange(10))

ax.set_title('Last Evalution vs Satisfaction segmented over all jobs')

for role in data.sales.unique():

    roleData=data[data.sales==role]

    leave=0

    ax.scatter(roleData[x][roleData.left == leave],roleData[y][data.left == leave],roleData[z][roleData.left == leave], c='k', marker='o',alpha = 0.02)

    leave=1

    ax.scatter(roleData[x][roleData.left == leave],roleData[y][roleData.left == leave],roleData[z][roleData.left == leave], c=colorrole[role], marker='x',alpha = 0.05)

#colored points means left | grey points means stayed

fig = plt.figure(figsize=(15,10))



ax = fig.add_subplot(111, projection='3d')

x='time_spend_company'

y='average_montly_hours'

z='last_evaluation'



leave=0

scatter1=ax.scatter(data[x][data.left == leave],data[y][data.left == leave],data[z][data.left == leave], c='b', marker='o',alpha = 0.01)

ax.set_xlabel(x)

ax.set_ylabel(y)

ax.set_zlabel(z)

leave=1

scatter2=ax.scatter(data[x][data.left == leave],data[y][data.left == leave],data[z][data.left == leave], c='r', marker='o',alpha = 0.05)

ax.legend([scatter1, scatter2], ['stayed', 'left'], numpoints = 1)

#Red means left employees | blue are employees that stayed
fig = plt.figure(figsize=(15,10))



ax = fig.add_subplot(111, projection='3d')

x='satisfaction_level'

y='last_evaluation'

z='time_spend_company'



leave=0

scatter1=ax.scatter(data[x][data.left == leave],data[y][data.left == leave],data[z][data.left == leave], c='b', marker='o',alpha = 0.01)

ax.set_xlabel(x)

ax.set_ylabel(y)

ax.set_zlabel(z)

leave=1

scatter2=ax.scatter(data[x][data.left == leave],data[y][data.left == leave],data[z][data.left == leave], c='r', marker='o',alpha = 0.05)

ax.legend([scatter1, scatter2], ['stayed', 'left'], numpoints = 1)

#Red means left employees | blue are employees that stayed