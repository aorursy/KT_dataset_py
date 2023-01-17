# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/indian-rainfall-dataset/rainfall_in_india_1901-2015.csv')
df.head()
df.info()
df['SUBDIVISION'].value_counts()
# Had taken input in the above markdown but then copied it for ease of use.

region = {'ANDAMAN & NICOBAR ISLANDS': 'w',

 'ARUNACHAL PRADESH': 'ne',

 'ASSAM & MEGHALAYA': 'ne',

 'NAGA MANI MIZO TRIPURA': 'ne',

 'SUB HIMALAYAN WEST BENGAL & SIKKIM': 'ne',

 'GANGETIC WEST BENGAL': 'ne',

 'ORISSA': 'e',

 'JHARKHAND': 'e',

 'BIHAR': 'n',

 'EAST UTTAR PRADESH': 'n',

 'WEST UTTAR PRADESH': 'n',

 'UTTARAKHAND': 'n',

 'HARYANA DELHI & CHANDIGARH': 'n',

 'PUNJAB': 'n',

 'HIMACHAL PRADESH': 'n',

 'JAMMU & KASHMIR': 'n',

 'WEST RAJASTHAN': 'w',

 'EAST RAJASTHAN': 'w',

 'WEST MADHYA PRADESH': 'c',

 'EAST MADHYA PRADESH': 'c',

 'GUJARAT REGION': 'w',

 'SAURASHTRA & KUTCH': 'w',

 'KONKAN & GOA': 'w',

 'MADHYA MAHARASHTRA': 'w',

 'MATATHWADA': 'w',

 'VIDARBHA': 'w',

 'CHHATTISGARH': 'c',

 'COASTAL ANDHRA PRADESH': 'e',

 'TELANGANA': 'e',

 'RAYALSEEMA': 'c',

 'TAMIL NADU': 's',

 'COASTAL KARNATAKA': 's',

 'NORTH INTERIOR KARNATAKA': 's',

 'SOUTH INTERIOR KARNATAKA': 's',

 'KERALA': 's',

 'LAKSHADWEEP': 'e'}
df['Region'] = df.SUBDIVISION.apply(lambda x: 'South' if region[x] == 's' else 'North' if region[x] == 'n' else 'East' if region[x] == 'e'

                                   else 'West' if region[x] == 'w' else 'Central' if region[x] == 'c' else 'NorthEast' if region[x] == 'ne'

                                   else 'No region')
df.Region.value_counts()
df_region_year = df.groupby(['Region','YEAR']).mean()
df_region = df.groupby(['Region']).mean()
df_region
import seaborn as sb
plot = sb.lineplot(x = 'YEAR', y = 'ANNUAL', hue = 'Region' , data = df, err_style = None)
df_region_subdiv = df.groupby(['SUBDIVISION','Region']).mean()
df_region_subdiv.head()
import matplotlib.pyplot as plt

g = sb.stripplot(x = 'SUBDIVISION' , y = 'ANNUAL' , data = df[df['Region'] == 'West'])

g.set_xticklabels(g.get_xticklabels(), rotation=30, ha = 'right')

plt.title('Rainfall in western India', fontsize = 20)

g
import matplotlib.pyplot as plt

g = sb.stripplot(x = 'SUBDIVISION' , y = 'ANNUAL' , data = df[df['Region'] == 'South'])

plt.ylim(0,4000)

g.set_xticklabels(g.get_xticklabels(), rotation=30, ha = 'right')

plt.title('Rainfall in Southern India', fontsize = 20)

g
import matplotlib.pyplot as plt

g = sb.stripplot(x = 'SUBDIVISION' , y = 'ANNUAL' , data = df[df['Region'] == 'East'])

g.set_xticklabels(g.get_xticklabels(), rotation=30, ha = 'right')

plt.ylim(0,4000)

plt.title('Rainfall in Eastern India', fontsize = 20)

g
import matplotlib.pyplot as plt

g = sb.stripplot(x = 'SUBDIVISION' , y = 'ANNUAL' , data = df[df['Region'] == 'Central'])

g.set_xticklabels(g.get_xticklabels(), rotation=30, ha = 'right')

plt.ylim(0,4000)

plt.title('Rainfall in Central India', fontsize = 20)

g
import matplotlib.pyplot as plt

g = sb.stripplot(x = 'SUBDIVISION' , y = 'ANNUAL' , data = df[df['Region'] == 'North'])

g.set_xticklabels(g.get_xticklabels(), rotation=30, ha = 'right')

plt.ylim(0,4000)

plt.title('Rainfall in Nothern India', fontsize = 20)

g
import matplotlib.pyplot as plt

g = sb.stripplot(x = 'SUBDIVISION' , y = 'ANNUAL' , data = df[df['Region'] == 'NorthEast'])

g.set_xticklabels(g.get_xticklabels(), rotation=30, ha = 'right')

plt.ylim(0,4000)

plt.title('Rainfall in Northeastern India', fontsize = 20)

g
plt.figure(figsize=(20,5))

plot = sb.lineplot(x = 'YEAR', y = 'ANNUAL', hue = 'SUBDIVISION' , data = df[df['Region'] == 'NorthEast'], err_style = None)
