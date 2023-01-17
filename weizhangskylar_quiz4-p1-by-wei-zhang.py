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
crimes = pd.read_csv("/kaggle/input/crimes-in-boston/crime.csv",header=0,encoding = 'unicode_escape')

crimes.head(5)

crimes.shape[0]
streets = crimes.groupby("STREET").size()

streets
streets.sort_values(ascending = False).head(3)
streets.idxmax()
#Washington = crimes[crimes.STREET == "WASHINGTON ST"]

Washington = crimes.query('STREET =="WASHINGTON ST"')

Washington.head(5)
reasons = Washington.loc[:,"OFFENSE_CODE_GROUP"]

reasons
reasons.value_counts().sort_values(ascending = False).head(3)
years = crimes.groupby("YEAR").size()

years.plot.bar()
years.mean()