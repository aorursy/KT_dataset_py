# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
male = (pd.read_csv("../input/Indian-Male-Names.csv"))

female = (pd.read_csv("../input/Indian-Female-Names.csv"))
male.head()
male = (male.assign( firstname = lambda x : x.name.str.split(' ').str[0],

    tri_last = lambda x : x.firstname.str[-3:],

    bi_last = lambda x : x.firstname.str[-2:]))

female = (female.assign( firstname = lambda x : x.name.str.split(' ').str[0],

    tri_last = lambda x : x.firstname.str[-3:],

    bi_last = lambda x : x.firstname.str[-2:]))

female.head()
female['tri_last'].value_counts().head(10).plot('bar')
female['bi_last'].value_counts().head(10).plot('bar')
male['tri_last'].value_counts().head(10).plot('bar')
male['bi_last'].value_counts().head(10).plot('bar')