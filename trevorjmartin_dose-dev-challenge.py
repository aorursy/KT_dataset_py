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
        fname = os.path.join(dirname, filename)
        print(f'reading {fname}')
        df = pd.read_csv(fname, sep="\t")
# df.head()
# df.tail()
# df.hist()
# df.hist(column="rating")  # Alexa general ratings
# df.hist(column="feedback")  # Alexa general feedback

print("Charcoal Fabric")
df[df['variation'].str.contains("Charcoal Fabric")].hist(column="rating")
print("Walnut Finish")
df[df['variation'].str.contains("Walnut Finish")].hist(column="rating")
print("Heather Gray Fabric")
df[df['variation'].str.contains("Heather Gray Fabric")].hist(column="rating")
print("Sandstone Fabric")
df[df['variation'].str.contains("Sandstone Fabric")].hist(column="rating")
print("Charcoal")
df[df['variation'].str.contains("Charcoal")].hist(column="rating")

print('general')
df.hist(column='rating')

print('sample data')
df.head()
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

