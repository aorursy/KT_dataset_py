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

#print(check_output(["ls", "../input"]).decode("utf8"))

print(check_output(["ls", "../input/mri-and-alzheimers"]).decode("utf8"))
#data = pd.read_csv('../input/gender-classification/Transformed Data Set - Sheet1.csv')

data = pd.read_csv('../input/mri-and-alzheimers/oasis_cross-sectional.csv')

data.info()

data.head()
# we should drop some columns. because we do not need them and these columns are not important for our solution.

data.drop(["ID","Delay","Hand"],axis=1, inplace = True)

data.head()
# we have NaN values. we should initialize to 0.

data.fillna(0,inplace=True)

data.head(10)
# we want to use male / female rate for x axis.

data['M/F'] = [1 if each == "M" else 0 for each in data['M/F']]

# male   = 1

# female = 0

y = data['M/F'].values

x_data = data.drop(['M/F'], axis = 1)
# now we should normalize our data.

# (x - min(x)) / (max(x) - min(x))

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values 

# our dataset only has between 0 and 1 values

print(x)
# we separete data for test and train

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
# logistic regression

from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()

logReg.fit(x_train, y_train)

print("test accurency : {}".format(logReg.score(x_test, y_test)))