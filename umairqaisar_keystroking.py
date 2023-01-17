# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data=pd.read_csv("../input/keystroke.csv")
data=data[:1000]
# Any results you write to the current directory are saved as output.
to_drop="UD.l.Return"
y= data[to_drop]
data=data.drop("subject",axis=1)
x=data.drop(to_drop,axis=1)

x.columns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
X_train, X_test, y_train,y_test=train_test_split(x,y,test_size=0.70, random_state=22, shuffle=True)
clf=XGBRegressor()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
import matplotlib.pyplot as plt
import numpy as np
dataa=pd.read_csv("../input/keystroke.csv")
import seaborn as sns
sns.countplot(dataa['H.period'])
#tol="subject"
#y_data=dataa[tol]
#x_data=dataa.drop("subject",axis=1)

#def scatterplot(x_data, y_data, x_label="keystrokes elements", y_label="users", title="keystrokes", color = "r", yscale_log=False):

    # Create the plot object
 #   _, ax = plt.subplots()

    # Plot the data, set the size (s), color and transparency (alpha)
    # of the points
  #  ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)

   # if yscale_log == True:
    #    ax.set_yscale('log')

    # Label the axes and provide a title
    #ax.set_title(title)
    #ax.set_xlabel(x_label)
    #ax.set_ylabel(y_label)
    #ax.plot.line()
    
sns.countplot(dataa['DD.period.t'])
sns.countplot(dataa['UD.period.t'])
import matplotlib.pyplot as plt
import random
tol="subject"
lol="DD.l.Return"
colors=["blue","black","grey"]
y_data=dataa[lol]
x_data=dataa[tol]
plt.bar(x_data, y_data, width=0.5, color=colors[random.randint(0,len(colors)-1)])
plt.show()
import matplotlib.pyplot as plt
import random
tol="subject"
lol="UD.l.Return"
colors=["blue","black","grey"]
y_data=dataa[lol]
x_data=dataa[tol]
plt.bar(x_data, y_data, width=0.5, color=colors[random.randint(0,len(colors)-1)])
plt.show()
import matplotlib.pyplot as plt
import random
tol="subject"
lol="H.Return"
colors=["blue","black","grey"]
y_data=dataa[lol]
x_data=dataa[tol]
plt.bar(x_data, y_data, width=0.5, color=colors[random.randint(0,len(colors)-1)])
plt.show()