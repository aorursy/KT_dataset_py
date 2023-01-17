# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
zom=pd.read_csv('../input/zomato.csv')
zom
zom['rate']=zom['rate'].str.rstrip('/5')
zom['dish_liked'].fillna('Normal',inplace=True)

zom.loc[(zom['rate']=='NEW')|(zom['rate']=='-'),'rate']=0

zom['rate'].fillna(0,inplace=True)
zom['rate']=zom['rate'].astype(float)
zom.dropna(how='any',inplace=True)
zom.shape
zom['location'].nunique()
(((zom['location'].value_counts()/50279)*100).round()[:20]).reset_index()
