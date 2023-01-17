# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.book import FreqDist

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
biology = pd.read_csv('../input/biology.csv')
biology.head()
tags_list = []

for tag in biology['tags']:

    for t in tag.split():

        tags_list.append(t)

freq = FreqDist(tags_list)

freq.plot(50,cumulative=True)
cooking = pd.read_csv('../input/cooking.csv')
cooking.head()
tags_list = []

for tag in cooking['tags']:

    for t in tag.split():

        tags_list.append(t)

freq= FreqDist(tags_list)

freq.plot(50,cumulative=True)