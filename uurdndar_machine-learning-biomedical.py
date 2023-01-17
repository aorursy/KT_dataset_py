import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings

import warnings

# ignore warnings

warnings.filterwarnings("ignore")

from subprocess import check_output

print(check_output(["ls", "../input/biomechanical-features-of-orthopedic-patients"]).decode("utf8"))
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

print(plt.style.available) # look at available plot styles

plt.style.use('ggplot')
data.head()
data.info()

data.describe()

color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]

pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],

                                       c=color_list,

                                       figsize= [15,15],

                                       diagonal='hist',

                                       alpha=0.5,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")

plt.show()
sns.countplot(x="class", data=data)

data.loc[:,'class'].value_counts()
data1 = data[data['class'] =='Abnormal']

x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)

y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)

# Scatter

plt.figure(figsize=[10,10])

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

# Predict space

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

# Fit

reg.fit(x,y)

# Predict

predicted = reg.predict(predict_space)

# R^2 

print('R^2 score: ',reg.score(x, y))

# Plot regression line and scatter

plt.plot(predict_space, predicted, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('pelvic_incidence')

plt.ylabel('sacral_slope')

plt.show()