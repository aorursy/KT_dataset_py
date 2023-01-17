# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", ".."]).decode("utf8"))



# Any results you write to the current directory are saved as output.
print(check_output(["ls", "../config"]).decode("utf8"))
print(check_output(["ls", "../lib"]).decode("utf8"))
print(check_output(["ls", "../working"]).decode("utf8"))
from sklearn.datasets import load_boston

boston = load_boston()

import pandas as pd

data=pd.DataFrame(boston.data, columns=boston.feature_names)

import matplotlib.pyplot as plt

boston.target # housing price in $1000 units
data.head(2)
plt.scatter(data['CRIM'], boston.target)

plt.show()
plt.scatter(data['CRIM'], boston.target,c=boston.target)

plt.show()
plt.scatter(data['CRIM'], boston.target,c=data['CRIM'])

plt.show()
plt.scatter(data['CRIM'], boston.target,c=(boston.target>30))

plt.show()
plt.scatter(data['CRIM'], boston.target,c=(data['CRIM']>40))

plt.show()
# https://www.kaggle.com/c/boston-housing

plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=5)

plt.colorbar()

plt.show()
plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=100,alpha=0.3)

plt.colorbar()

plt.show()
plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=100,alpha=0.3,cmap="plasma")

plt.colorbar()

plt.show()
plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=100,alpha=0.3,cmap="plasma",vmin=10,vmax=20)

plt.colorbar()

plt.show()
plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=100,alpha=0.3,cmap="plasma",vmin=10,vmax=20,marker="D")

plt.colorbar()

plt.show()
# https://matplotlib.org/api/markers_api.html

plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=100,alpha=0.3,cmap="plasma",vmin=10,vmax=20,marker="X")

plt.colorbar()

plt.show()
plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=100,alpha=0.3,cmap="plasma",vmin=10,vmax=20,marker=",",edgecolors="black")

plt.colorbar()

plt.title("Price Against Crime")

plt.xlabel("Crime")

plt.ylabel("Price")

plt.grid(True)

plt.show()
plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=100,alpha=0.3,cmap="plasma",vmin=10,vmax=20,marker=",",edgecolors="black",lw=3)

plt.colorbar()

plt.title("Price Against Crime")

plt.xlabel("Crime")

plt.ylabel("Price")

plt.grid(True)

plt.show()
plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=100,alpha=0.3,cmap="plasma",edgecolors="black")

plt.colorbar()

plt.title("Price Against Crime")

plt.xlabel("Crime")

plt.ylabel("Price")

plt.axvline(x=30,c="black",ls="--",lw=0.5)

plt.show()
plt.scatter(data['CRIM'], boston.target,c=data['INDUS'],s=100,alpha=0.3,cmap="plasma",edgecolors="black")

plt.colorbar()

plt.title("Price Against Crime")

plt.xlabel("Crime")

plt.ylabel("Price")

plt.axhline(y=25,c="purple",lw=0.5)

plt.show()
plt.scatter(data['CRIM'][data['INDUS']>15], boston.target[data['INDUS']>15],c="red",s=100,alpha=0.3,edgecolors="black",label="high")

plt.scatter(data['CRIM'][data['INDUS']<=15], boston.target[data['INDUS']<=15],c="green",s=100,alpha=0.3,edgecolors="black",label="low")

plt.title("Price Against Crime")

plt.xlabel("Crime")

plt.ylabel("Price")

plt.show()
plt.scatter(data['CRIM'][data['INDUS']>15], boston.target[data['INDUS']>15],c="red",s=100,alpha=0.3,edgecolors="black",label="high")

plt.scatter(data['CRIM'][data['INDUS']<=15], boston.target[data['INDUS']<=15],c="green",s=100,alpha=0.3,edgecolors="black",label="low")

plt.title("Price Against Crime")

plt.xlabel("Crime")

plt.ylabel("Price")

plt.legend() # plt.legend(loc="upper left")

plt.show()
import statsmodels.api as sm

lowess = sm.nonparametric.lowess(data['CRIM'], boston.target, frac=.3)

lowess_x = list(zip(*lowess))[0]

lowess_y = list(zip(*lowess))[1]

plt.plot(lowess_x, lowess_y)

plt.show()
lowess.shape
lowess
len(list(zip(*lowess)))
len(list(zip(*lowess))[1])
a1=[1,2,3,4,5,6]

a2=[6,5,4,3,2,1]

for (i, j) in zip(a1, a2):

    print(i,j)
x=[1,2,3]

def add(x1,x2,x3):

    return x1+x2+x3

add(*x) # add(x) would give error
# https://matplotlib.org/examples/color/named_colors.html

plt.scatter(data['CRIM'][data['INDUS']>15], boston.target[data['INDUS']>15],c="red",s=100,alpha=0.3,edgecolors="black",label="high")

plt.scatter(data['CRIM'][data['INDUS']<=15], boston.target[data['INDUS']<=15],c="green",s=100,alpha=0.3,edgecolors="black",label="low")

plt.title("Price Against Crime")

plt.xlabel("Crime")

plt.ylabel("Price")

plt.plot(lowess_x, lowess_y,lw=5,alpha=0.6,c="magenta")

plt.legend() # plt.legend(loc="upper left")

plt.show()
from sklearn.datasets import load_iris

iris = load_iris()

data=pd.DataFrame(iris.data, columns=iris.feature_names)

data.head()
iris.target
iris.target_names
plt.boxplot(data['sepal width (cm)'])

plt.show()
plt.boxplot(data['sepal width (cm)'],vert=False)

plt.show()
plt.boxplot(data['sepal width (cm)'],vert=False,whis='range')

plt.title('sepal width of iris')

plt.xlabel('sepal width (cm)')

plt.yticks([])

plt.show()
setosa=data["sepal width (cm)"][iris.target==0]

versicolor=data["sepal width (cm)"][iris.target==1]

virginica=data["sepal width (cm)"][iris.target==2]

plt.boxplot(x=(setosa, versicolor, virginica),vert=True)

plt.title('sepal width of iris')

plt.ylabel('sepal width (cm)')

plt.xlabel('species')

plt.xticks([1, 2, 3], ['setosa', 'versicolor', 'vriginica'])

plt.show()