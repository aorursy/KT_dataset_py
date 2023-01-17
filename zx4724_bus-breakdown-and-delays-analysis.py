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

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()





df = pd.read_csv("../input/Bus_Breakdown_and_Delays.csv")

print(df.head())



run_type = df.Run_Type.value_counts()

plt.figure(figsize=(12,8))

sns.barplot(run_type.index, run_type.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Run type', fontsize=12)

plt.title('Count of rows in each dataset (run_type)', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()



reason = df.Reason.value_counts()

plt.figure(figsize=(12,8))

sns.barplot(reason.index, reason.values, alpha=0.8, color=color[0])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Reason type', fontsize=12)

plt.title('Count of rows in each dataset (reason_type)', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()



boro = df.Boro.value_counts()

plt.figure(figsize=(12,8))

sns.barplot(boro.index, boro.values, alpha=0.8, color=color[2])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('boro type', fontsize=12)

plt.title('Count of rows in each dataset (boro)', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()



notify1 = df.Has_Contractor_Notified_Schools.value_counts()

plt.figure(figsize=(12,8))

sns.barplot(notify1.index, notify1.values, alpha=0.8, color=color[3])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('notify type', fontsize=12)

plt.title('Count of rows in each dataset (Has_Contractor_Notified_Schools)', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()



notify2 = df.Has_Contractor_Notified_Parents.value_counts()

plt.figure(figsize=(12,8))

sns.barplot(notify2.index, notify2.values, alpha=0.8, color=color[4])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('notify type', fontsize=12)

plt.title('Count of rows in each dataset (Has_Contractor_Notified_Parents)', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()



notify3 = df.Have_You_Alerted_OPT.value_counts()

plt.figure(figsize=(12,8))

sns.barplot(notify3.index, notify3.values, alpha=0.8, color=color[5])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('notify type', fontsize=12)

plt.title('Count of rows in each dataset (Have_You_Alerted_OPT)', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()





grouped_df = df.groupby(["Run_Type", "Reason"])["Number_Of_Students_On_The_Bus"].aggregate(np.sum).reset_index()

grouped_df = grouped_df.pivot('Run_Type', 'Reason', 'Number_Of_Students_On_The_Bus')

plt.figure(figsize=(12,6))

sns.heatmap(grouped_df)

plt.title("Students of delay of Run_Type Vs Reason")

plt.show()



grouped_df = df.groupby(["Have_You_Alerted_OPT", "School_Age_or_PreK"])["Number_Of_Students_On_The_Bus"].aggregate(np.sum).reset_index()

grouped_df = grouped_df.pivot('Have_You_Alerted_OPT', 'School_Age_or_PreK', 'Number_Of_Students_On_The_Bus')

plt.figure(figsize=(12,6))

sns.heatmap(grouped_df)

plt.title("Students of delay of Altered_OPT Vs School_Age")

plt.show()



grouped_df = df.groupby(["Reason", "Boro"])["Number_Of_Students_On_The_Bus"].aggregate(np.sum).reset_index()

grouped_df = grouped_df.pivot('Reason', 'Boro', 'Number_Of_Students_On_The_Bus')

plt.figure(figsize=(12,6))

sns.heatmap(grouped_df)

plt.title("Students of delay of Boro Vs Reason")

plt.show()



grouped_df = df.groupby('School_Year')["Number_Of_Students_On_The_Bus"].aggregate(np.sum).reset_index()

print(grouped_df.head())

sns.barplot(grouped_df.School_Year, grouped_df.Number_Of_Students_On_The_Bus, alpha=0.8, color=color[1])

plt.ylabel('Number_Of_Students_On_The_Bus', fontsize=12)

plt.xlabel('School_Year', fontsize=12)

plt.title('Number of students delayed in recent years', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()




