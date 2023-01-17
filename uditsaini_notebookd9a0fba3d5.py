# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import re

%matplotlib inline

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

from subprocess import check_output

import matplotlib.pylab as pylab

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

sns.set_style("darkgrid")

params = {'legend.fontsize': 20,

          'figure.figsize': [13,6],

          #'legend.linewidth': 4,

          'xtick.labelsize':15,

          'ytick.labelsize':15,

          'lines.linewidth':4,

          'font.size': 22,

         }

import warnings

warnings.filterwarnings('ignore')

plt.rcParams.update(params)

from subprocess import check_output



GSAF = "../input/attacks.csv"

AllData = pd.read_csv(GSAF, encoding = 'ISO-8859-1')

AllData['Age']=AllData['Age'].str.extract('(\d+)').astype(float)



# Any results you write to the current directory are saved as output.
AllData.head()
# 30 most deadly rears

AllData.Year.value_counts().head(30).plot(kind='bar')
# 30 most deadly Country

AllData.Country.value_counts().head(30).plot(kind='bar')
#age distribution of attaked persons

sns.kdeplot(AllData.Age.dropna())
#age distribution of attaked persons by gender

sns.kdeplot(AllData.Age[AllData['Sex ']=="M"], label="Male")

sns.kdeplot(AllData.Age[AllData['Sex ']=="F"],  label="Female")

plt.legend();
# 30 most common injuriy

AllData.Injury.value_counts().head(30).plot(kind='bar')
#hourly distribution of attack

time=AllData.Time.dropna().str.extract('(\d+)').astype(str).apply(lambda x: x[:2])

time=time[time!='na'].astype(int)

sns.kdeplot(time[(time>=0) & (time <= 24)])
#most deadly Species

AllData['Species '].dropna().value_counts().head(40).plot(kind='bar')
#attack treands

from dateutil.parser import parse

def validate(date_text):

    try:

        parse(date_text)

        return True

    except ValueError:

        return False

idx=AllData.Date.apply(lambda x:validate(x))

AllData=AllData[idx]

AllData.Date=AllData.Date.apply(lambda x:parse(x))

AllData['Date']=pd.to_datetime(AllData['Date'], errors='coerce')

#df.Date=df.Date.apply(lambda x: x.date())

df=AllData[AllData.Date.notnull()]
