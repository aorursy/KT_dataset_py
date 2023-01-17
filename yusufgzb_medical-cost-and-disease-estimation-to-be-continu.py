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
df=pd.read_csv("/kaggle/input/insurance/insurance.csv")
df.columns
df.head()
df.info()
df.describe()
df.corr()
filter1=df[(df["age"]>30) & (df["smoker"]=="yes")&(df["children"]>0) ]
filter1

disease=[]
for i in filter1.bmi:
    if  i > 30:
        disease.append(1)
    else :
        disease.append(0)


filter1["dangerous risk"]=disease

filter1["sex"]=[1 if each == "male" else 0 for each in filter1.sex]
filter1["smoker"]=[1 if each == "yes" else 0 for each in filter1.smoker]
filter1.corr()










