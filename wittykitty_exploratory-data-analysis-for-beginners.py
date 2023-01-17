# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/xAPI-Edu-Data.csv')

data.head()
data.info()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
fig, axarr  = plt.subplots(2,2,figsize=(10,10))

sns.countplot(x='Class', data=data, ax=axarr[0,0], order=['L','M','H'])

sns.countplot(x='gender', data=data, ax=axarr[0,1], order=['M','F'])

sns.countplot(x='StageID', data=data, ax=axarr[1,0])

sns.countplot(x='Semester', data=data, ax=axarr[1,1])
fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))

sns.countplot(x='Topic', data=data, ax=axis1)

sns.countplot(x='NationalITy', data=data, ax=axis2)
fig, axarr  = plt.subplots(2,2,figsize=(10,10))

sns.countplot(x='gender', hue='Class', data=data, ax=axarr[0,0], order=['M','F'], hue_order=['L','M','H'])

sns.countplot(x='gender', hue='Relation', data=data, ax=axarr[0,1], order=['M','F'])

sns.countplot(x='gender', hue='StudentAbsenceDays', data=data, ax=axarr[1,0], order=['M','F'])

sns.countplot(x='gender', hue='ParentAnsweringSurvey', data=data, ax=axarr[1,1], order=['M','F'])
fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))

sns.countplot(x='Topic', hue='gender', data=data, ax=axis1)

sns.countplot(x='NationalITy', hue='gender', data=data, ax=axis2)
fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))

sns.countplot(x='NationalITy', hue='Relation', data=data, ax=axis1)

sns.countplot(x='NationalITy', hue='StudentAbsenceDays', data=data, ax=axis2)
fig, axarr  = plt.subplots(2,2,figsize=(10,10))

sns.barplot(x='Class', y='VisITedResources', data=data, order=['L','M','H'], ax=axarr[0,0])

sns.barplot(x='Class', y='AnnouncementsView', data=data, order=['L','M','H'], ax=axarr[0,1])

sns.barplot(x='Class', y='raisedhands', data=data, order=['L','M','H'], ax=axarr[1,0])

sns.barplot(x='Class', y='Discussion', data=data, order=['L','M','H'], ax=axarr[1,1])
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

sns.barplot(x='gender', y='raisedhands', data=data, ax=axis1)

sns.barplot(x='gender', y='Discussion', data=data, ax=axis2)
fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))

sns.swarmplot(x='gender', y='AnnouncementsView', data=data, ax=axis1)

sns.swarmplot(x='gender', y='raisedhands', data=data, ax=axis2)
fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))

sns.boxplot(x='Class', y='Discussion', data=data, order=['L','M','H'], ax=axis1)

sns.boxplot(x='Class', y='VisITedResources', data=data, order=['L','M','H'], ax=axis2)
fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))

sns.pointplot(x='Semester', y='VisITedResources', hue='gender', data=data, ax=axis1)

sns.pointplot(x='Semester', y='AnnouncementsView', hue='gender', data=data, ax=axis2)
fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))

sns.regplot(x='raisedhands', y='VisITedResources', data=data, ax=axis1)

sns.regplot(x='AnnouncementsView', y='Discussion', data=data, ax=axis2)
