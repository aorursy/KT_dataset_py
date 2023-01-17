# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





plt.style.use('seaborn-dark-palette')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/globalterrorismdb.csv', encoding="latin1")

print(df.head())
#Extracting rows where the region of attack was Western Europe (i.e where region = 8) AND where the attack type (field

#is called attacktype1) is of type 'Assasination (or value 1) and then plotting it by year



dfWEurope = df[(df['region']==8) & (df['attacktype1'] == 1)]

assasinationcounts = pd.DataFrame(dfWEurope.groupby(['eventid.1'], as_index=False).size().rename('count')).reset_index()



plt.title('Assasination attacks in Western Europe')

plt.bar(assasinationcounts['eventid.1'], assasinationcounts['count'])

plt.xlabel('Year')

plt.ylabel('Count')
#Extracting rows where the region of attack was Western Europe (i.e where region = 8) AND where the attack type (field

#is called attacktype1) is of type 'Armed Assault' (or value 2) and then plotting it by year



dfWEurope = df[(df['region']==8) & (df['attacktype1'] == 2)]

armedassaultcounts = pd.DataFrame(dfWEurope.groupby(['eventid.1'], as_index=False).size().rename('count')).reset_index()



plt.title('Armed assaults in Western Europe')

plt.bar(armedassaultcounts['eventid.1'], armedassaultcounts['count'])

plt.xlabel('Year')

plt.ylabel('Count')
#Extracting rows where the region of attack was Western Europe (i.e where region = 8) AND where the attack type (field

#is called attacktype1) is of type 'Bombing' (or value 3) and then plotting it by year



dfWEurope = df[(df['region']==8) & (df['attacktype1'] == 3)]

bombingattackcounts = pd.DataFrame(dfWEurope.groupby(['eventid.1'], as_index=False).size().rename('count')).reset_index()



plt.title('Bombing attacks in Western Europe')

plt.bar(bombingattackcounts['eventid.1'], bombingattackcounts['count'])

plt.xlabel('Year')

plt.ylabel('Count')
#Extracting rows where the region of attack was Western Europe (i.e where region = 8) AND where the attack type (field

#is called attacktype1) is of type 'Hijacking' (or value 4) and then plotting it by year



dfWEurope = df[(df['region']==8) & (df['attacktype1'] == 4)]

hijackingcounts = pd.DataFrame(dfWEurope.groupby(['eventid.1'], as_index=False).size().rename('count')).reset_index()



plt.title('Hijacking attacks in Western Europe')

plt.bar(hijackingcounts['eventid.1'], hijackingcounts['count'])

plt.xlabel('Year')

plt.ylabel('Count')
#Extracting rows where the region of attack was Western Europe (i.e where region = 8) AND where the attack type (field

#is called attacktype1) is of type 'Hostage Barricades' (or value 5) and then plotting it by year



dfWEurope = df[(df['region']==8) & (df['attacktype1'] == 5)]

hostagebarricadecounts = pd.DataFrame(dfWEurope.groupby(['eventid.1'], as_index=False).size().rename('count')).reset_index()



plt.title('Hostage Barricades in Western Europe')

plt.bar(hostagebarricadecounts['eventid.1'], hostagebarricadecounts['count'])

plt.xlabel('Year')

plt.ylabel('Count')
#Extracting rows where the region of attack was Western Europe (i.e where region = 8) AND where the attack type (field

#is called attacktype1) is of type 'Hostage Kidnapping' (or value 6) and then plotting it by year



dfWEurope = df[(df['region']==8) & (df['attacktype1'] == 6)]

hostagekidnappingcounts = pd.DataFrame(dfWEurope.groupby(['eventid.1'], as_index=False).size().rename('count')).reset_index()



plt.title('Hostage Kidnappings in Western Europe')

plt.bar(hostagekidnappingcounts['eventid.1'], hostagekidnappingcounts['count'])

plt.xlabel('Year')

plt.ylabel('Count')
#Extracting rows where the region of attack was Western Europe (i.e where region = 8) AND where the attack type (field

#is called attacktype1) is of type 'Infrastructure' (or value 7) and then plotting it by year



dfWEurope = df[(df['region']==8) & (df['attacktype1'] == 7)]

facilityattackcounts = pd.DataFrame(dfWEurope.groupby(['eventid.1'], as_index=False).size().rename('count')).reset_index()



plt.title('Infrastructure attacks in Western Europe')

plt.bar(facilityattackcounts['eventid.1'], facilityattackcounts['count'])

plt.xlabel('Year')

plt.ylabel('Count')
#Extracting rows where the region of attack was Western Europe (i.e where region = 8) AND where the attack type (field

#is called attacktype1) is of type 'Unarmed Assaults' (or value 8) and then plotting it by year



dfWEurope = df[(df['region']==8) & (df['attacktype1'] == 8)]

unarmedcounts = pd.DataFrame(dfWEurope.groupby(['eventid.1'], as_index=False).size().rename('count')).reset_index()



plt.title('Unarmed assaults in Western Europe')

plt.bar(unarmedcounts['eventid.1'], unarmedcounts['count'])

plt.xlabel('Year')

plt.ylabel('Count')
#Extracting rows where the region of attack was Western Europe (i.e where region = 8) AND where the attack type (field

#is called attacktype1) is of type 'Unknown' (or value 9) and then plotting it by year



dfWEurope = df[(df['region']==8) & (df['attacktype1'] == 9)]

unknownattackcounts = pd.DataFrame(dfWEurope.groupby(['eventid.1'], as_index=False).size().rename('count')).reset_index()



plt.title('Unknown attacks in Western Europe')

plt.bar(unknownattackcounts['eventid.1'], unknownattackcounts['count'])

plt.xlabel('Year')

plt.ylabel('Count')