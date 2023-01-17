# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Load CSV using Pandas

from pandas import read_csv

filename = 'pima-indians-diabetes.data.csv'

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = read_csv("../input/"+filename, names=names)

print(data.shape)
class_counts = data.groupby('class').size()

print(class_counts)
types = data.dtypes

print(types)
from pandas import set_option

set_option('display.width', 100)

set_option('precision', 3)

description = data.describe()

print(description)
shape = data.shape

print(shape)
peek = data.head(20)

print(peek)
set_option('display.width', 100)

set_option('precision', 3)

correlations = data.corr(method='pearson')

print(correlations)
skew = data.skew()

print(skew)
# Correction Matrix Plot

from matplotlib import pyplot

import numpy

correlations = data.corr()

# plot correlation matrix

fig = pyplot.figure(figsize=(10,8))

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = numpy.arange(0,9,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

pyplot.show()
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False,figsize=(10,8))

pyplot.show()
correlations = data.corr()

# plot correlation matrix

fig = pyplot.figure(figsize=(10,8))

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

pyplot.show()
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False,figsize=(10,8))

pyplot.show()
data.hist(figsize=(10,8))

pyplot.show()
from pandas.plotting import scatter_matrix

scatter_matrix(data,figsize=(15,13))

pyplot.show()