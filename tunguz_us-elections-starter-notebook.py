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
usa_2016_presidential_election_by_county = pd.read_csv('/kaggle/input/us-elections-dataset/usa-2016-presidential-election-by-county.csv', sep=';')

usa_2016_presidential_election_by_county.head()
usa_2016_presidential_election_by_county.shape
usa_2016_presidential_election_by_county.describe()
us_2016_primary_results = pd.read_csv('/kaggle/input/us-elections-dataset/us-2016-primary-results.csv', sep=';')

us_2016_primary_results.head()
us_2016_primary_results.shape
us_2016_primary_results.describe()
US_elect_county_2012 = pd.read_csv('/kaggle/input/us-elections-dataset/2012_US_elect_county.csv')

US_elect_county_2012.head()
US_elect_county_2012.shape
US_elect_county_2012.describe()
senate_1976_2018 = pd.read_csv('/kaggle/input/us-elections-dataset/1976-2018-senate.csv', encoding= 'unicode_escape')

senate_1976_2018.head()
senate_1976_2018.shape
senate_1976_2018.describe()