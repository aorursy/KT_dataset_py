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
import pandas as pd
print(pd.__version__)
df = pd.read_json("https://data.smcgov.org/resource/mb6a-xn89.json")

df.head(5)
df.shape
df.describe()

df.dtypes
df.bachelor_s_degree_or_higher.mean()
df.geography.count()
df.geography_type.unique()
df.less_than_high_school_graduate.value_counts()
def mapGeography(x):
    if x == "City":
        return 1
    else:
        return 0


df['geography_mapped_value'] = df.geography_type.apply(mapGeography)

df.geography_mapped_value.value_counts()
df['geography_mapped_value_lambda'] = df.geography_type.apply(lambda y: 1 if y == "City" else 0)

df.geography_mapped_value_lambda.value_counts()