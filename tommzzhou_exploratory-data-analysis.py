# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
original_data = pd.read_csv("/kaggle/input/xAPI-Edu-Data.csv")
original_data.info()
original_data.head(5)
original_data.describe()
# Change "NationallTY" to "Nationality"

print(original_data.columns)

original_data = original_data.rename(columns = {"NationalITy":"Nationality", "raisedhands":\

                               "RaisedHands", "VisITedResources":"VisitedResources"})
#Now all the columns have proper names

original_data.head()
# A few more rename operations

original_data.loc[original_data["Nationality"] == "KW", "Nationality"] = "Kuwait"

original_data.loc[original_data["PlaceofBirth"] == "KuwaIT", "PlaceofBirth"] = "Kuwait"

original_data.head(5)
# Change the "Class" attributes to numeric data

original_data.loc[original_data["Class"] == "M", "Class"] = 80

original_data.loc[original_data["Class"] == "H", "Class"] = 95

original_data.loc[original_data["Class"] == "L", "Class"] = 35

# Also change the column's name to be more indicative

original_data = original_data.rename(columns = {"Class":"Score"})
original_data.head(5)
nationality_counts = original_data["Nationality"].value_counts()

print(nationality_counts)
nationality_counts.plot.pie(figsize = (8, 8))
original_data["PlaceofBirth"].value_counts().plot.pie(figsize = (8,8))
change_nationality = original_data[original_data["Nationality"] != original_data["PlaceofBirth"]]

change_nationality["Nationality"].value_counts().plot.pie(figsize = (6,6))

print(len(list(change_nationality["Nationality"])))
original_data[["RaisedHands","Discussion", "VisitedResources", "AnnouncementsView", ]].hist(figsize=(14, 9),bins=40,linewidth='1',edgecolor='k',grid=False)
fig, ([axis1, axis2], [axis3, axis4])  = plt.subplots(2, 2,figsize=(12,8))

plt.subplots_adjust(wspace=0.25 , hspace=0.3)



sns.boxplot(x='gender', y='Discussion', data=original_data, order=['F','M'], ax=axis1)

sns.boxplot(x='gender', y='VisitedResources', data=original_data, order=['F','M'], ax=axis2)

sns.boxplot(x='gender', y='AnnouncementsView', data=original_data, order=['F','M'], ax=axis3)

sns.boxplot(x='gender', y='RaisedHands', data=original_data, order=['F','M'], ax=axis4)
fig, (axis1, axis2, axis3, axis4)  = plt.subplots(1, 4,figsize=(14,7))

plt.subplots_adjust(wspace=0.6)

sns.swarmplot(x='gender', y='AnnouncementsView', data=original_data, ax=axis1, hue = "Relation")

sns.swarmplot(x='gender', y='RaisedHands', data=original_data, ax=axis2, hue = "Relation")

sns.swarmplot(x='gender', y='Discussion', data=original_data, ax=axis3, hue = "Relation")

sns.swarmplot(x='gender', y='VisitedResources', data=original_data, ax=axis4, hue = "Relation")
fig, axes = plt.subplots(2,2, figsize = (15, 9))

sns.barplot(x="gender", y="VisitedResources", hue="Relation", data=original_data, ax = axes[0,0])

sns.barplot(x="gender", y="AnnouncementsView", hue="Relation", data=original_data, ax = axes[0,1])

sns.barplot(x="gender", y="RaisedHands", hue="Relation", data=original_data, ax = axes[1,0])

sns.barplot(x="gender", y="Discussion", hue="Relation", data=original_data, ax = axes[1,1])
sns.pairplot(original_data.loc[:,original_data.dtypes == 'int64'])
corr = original_data.loc[:,original_data.dtypes == 'int64'].corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.diverging_palette(220, 10, as_cmap=True))
fig, (axis1, axis2, axis3, axis4)  = plt.subplots(1, 4,figsize=(20,4))

plt.subplots_adjust(wspace=0.6)



sns.regplot(x = "AnnouncementsView", y = "VisitedResources", data = original_data, ax = axis1)

sns.regplot(x = "AnnouncementsView", y = "Discussion", data = original_data, ax = axis2)

sns.regplot(x = "RaisedHands", y = "VisitedResources", data = original_data, ax = axis3)

sns.regplot(x = "AnnouncementsView", y = "RaisedHands", data = original_data, ax = axis4)

print(original_data.columns)

parents_data = original_data[["Relation", "ParentAnsweringSurvey", "ParentschoolSatisfaction"]]

parents_data.head(5)
parents_data.describe()
parents_data["Relation"].value_counts().plot.pie()
#Plot relation between ParentAnsweringSurvey with School's satisfaction with the parent

# 不会画 555~