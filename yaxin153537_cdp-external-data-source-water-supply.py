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
water_abstract=pd.read_csv('/kaggle/input/water-supply/WATER_ABSTRACT_23102020142414424.csv')

print(water_abstract.shape)
water_abstract.head(1)
water_abstract['Source'].value_counts()
water_abstract['Variable'].value_counts()
water_abstract['Country'].nunique()
water_abstract['Year'].value_counts().sort_index()
water_treatment=pd.read_csv('/kaggle/input/water-treatment/WATER_TREAT_23102020144712652.csv')

print(water_treatment.shape)

water_treatment.head(1)
water_treatment['Variable'].value_counts()
water_treatment['Country'].nunique()
water_treatment['Year'].value_counts()