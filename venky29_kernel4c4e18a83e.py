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
import pandas as pd

CAvideos = pd.read_csv("../input/youtube-new/CAvideos.csv")

DEvideos = pd.read_csv("../input/youtube-new/DEvideos.csv")

FRvideos = pd.read_csv("../input/youtube-new/FRvideos.csv")

GBvideos = pd.read_csv("../input/youtube-new/GBvideos.csv")

INvideos = pd.read_csv("../input/youtube-new/INvideos.csv")

JPvideos = pd.read_csv("../input/youtube-new/JPvideos.csv")

KRvideos = pd.read_csv("../input/youtube-new/KRvideos.csv")

MXvideos = pd.read_csv("../input/youtube-new/MXvideos.csv")

RUvideos = pd.read_csv("../input/youtube-new/RUvideos.csv")

USvideos = pd.read_csv("../input/youtube-new/USvideos.csv")

import csv

import pandas as pd

import numpy as np

import seaborn as sns

CAvideos = pd.read_csv("../input/youtube-new/CAvideos.csv")

df=CAvideos[['views', 'likes']]

corr = df.corr()

print(corr)

sns.heatmap(corr)

sns.pairplot(df)




