# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib import pyplot

import pylab

from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/HR_comma_sep.csv")

sales = dataset#.loc[dataset['sales'] == "sales"]

dataset = sales.sample(frac=0.1)

dataset.head(10)
#dataset[["average_montly_hours","satisfaction_level","salary"]].head(10)
colors = {'low':'red', 'medium':'blue', 'high':'green'}

fig = pylab.figure()

ax = Axes3D(fig)

ax.scatter(dataset["average_montly_hours"],dataset["satisfaction_level"].apply(lambda x: x*100),dataset["number_project"].apply(lambda x: x),c = dataset["salary"].apply(lambda x: colors[x]))

#,c = dataset["salary"].apply(lambda x: colors[x])

#figure.set_xlabel('average_montly_hours')

#figure.set_ylabel('satisfaction_level')

plt.show()
sb.set()

sb.pairplot(dataset[["last_evaluation","number_project","average_montly_hours","time_spend_company","sales","salary"]],

             hue="salary", diag_kind="kde")
feature = ["last_evaluation","number_project","average_montly_hours","time_spend_company"]

dt = DecisionTreeClassifier()

dt.fit(dataset[feature], dataset["salary"])