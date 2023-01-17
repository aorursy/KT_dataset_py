# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import gender_guesser.detector as gender
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
gd = gender.Detector()
df = pd.read_csv(r"/kaggle/input/golden-globe-awards/golden_globe_awards.csv")
df.info()
genders = []
for i in df.category:
    if "actor" in i.lower():
        genders.append("male")
    elif "actress" in i.lower():
        genders.append("female")
    else:
        genders.append("null")
df['genders'] = genders
df.head()
df.genders.value_counts()
nm = df.nominee.str.split()
firstname = []
for i in range(len(nm)):
    firstname.append(nm[i][0])
guess = []
for i in range(len(firstname)):
    guess.append(gd.get_gender(firstname[i]))    # assign gender from all the first name into guess's list.
    
for i in range(len(df)):
    if "null" in df.genders[i]:
        df.genders[i] = df.genders[i].replace("null", guess[i])    # replace the nulls from df.genders to the assigned guess's list.
        
df.genders.value_counts()
df.genders = df.genders.replace("mostly_female", "female")
df.genders = df.genders.replace("mostly_male", "male")
df.genders.value_counts().plot.pie(autopct='%1.1f%%')
year = year = df[(df.genders == 'male')| (df.genders == 'female')]
year = year.groupby('year_film').genders.value_counts()
year.unstack().plot.line()
win = df[(df.genders == 'male')| (df.genders == 'female')]
win = win[win.win == True]
win = win.groupby('year_film').genders.value_counts()
win.unstack().plot.line()
