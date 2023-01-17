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
df=pd.read_csv("/kaggle/input/us-counties-covid-19-dataset/us-counties.csv")
data=df[df['state']=='New York']

data=data.groupby("date")['cases','deaths'].sum().reset_index()
data['suspectible']=19450000

data['infected']=0

data['recovered']=data['deaths']*1.5

data['recovered']=data['recovered'].astype(int)

data['currently_infected']=data['cases']-data['deaths']-data['recovered']

data['removed']=data['deaths']+data['recovered']

data['suspectible']=data['suspectible']-data['currently_infected']-data['removed']
dataframe=data[['date','suspectible','currently_infected','removed']]

dataframe
dataframe.to_csv("SIR.csv")