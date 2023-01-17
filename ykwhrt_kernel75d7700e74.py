# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(52*125)

print(2 % 3)
#1
import numpy as np

stor = np.array([1,2,3,4,5])

stor = stor + 1
print(stor)
#2
w = np.array([1,2,3,4])
x = np.array([6,7,8,9])
print(w.dot(x) + 4)
#3
arr = np.array([[1,2,3],[6,50,400],[5,10,100]])
print(np.average(arr,axis = 0))
#4
import pandas as pd

index = ["Taro","Jiro","Saburo","Hanako","Yoshiko"]
data = [90,100,70,80,100]
series = pd.Series(data,index = index)
# print(series)
series = series[series != 100]
print(series)
# 5
import pandas as pd
pan_data = pd.read_csv("../input/bostonhoustingmlnd/housing.csv",delimiter = ",")
pan_data.head(5)
#6
import pandas as pd
pan2_data = pd.read_csv("../input/pandas/pandasdata2.csv",delimiter = ",")
pan2_data = pan2_data.drop(['Unnamed: 0','id','name'],axis = 1)
pan2_data[10:20].to_csv("pandasdata3.csv",index=False)


import pandas as pd
pan3_data = pd.read_csv("./pandasdata3.csv",delimiter = ",")
print(pan3_data)
# 7
import pandas as pd
titanic_data = pd.read_csv("../input/titanic-machine-learning-from-disaster/train.csv",delimiter = ",")
titanic_data = titanic_data.drop(['Name','Ticket','Cabin'],axis = 1)
print(titanic_data)
titanic_data.to_csv("train2.csv")
# 8
import pandas as pd
titanic2_data = pd.read_csv("./train2.csv",delimiter = ",")
titanic2_data['male'] = (titanic2_data["Sex"] == "male").astype(int)
titanic2_data['female'] = (titanic2_data["Sex"] == "female").astype(int)
titanic2_data['embarked c'] = (titanic2_data["Embarked"] == "C").astype(int)
titanic2_data['embarked q'] = (titanic2_data["Embarked"] == "Q").astype(int)
titanic2_data['embarked s'] = (titanic2_data["Embarked"] == "S").astype(int)
titanic2_data = titanic2_data.drop(['Sex',"Embarked"],axis = 1)
print(titanic2_data)
titanic2_data.to_csv("train3.csv")
import pandas as pd
import numpy as np
import matplotlib as mt
from statsmodels.graphics.mosaicplot import mosaic

train_data = pd.read_csv("../input/titanic/train.csv",delimiter = ",")

# train_data.head(5)
# print(np.c_[train_data['Sex'],train_data['Survived']])
engineer_data = np.c_[train_data['Sex'],train_data['Survived']]
my_dataframe = pd.DataFrame(engineer_data,columns = ['Sex','Survived'])
mosaic(data = my_dataframe,index = ['Sex','Survived'],title = 'Mosaic Plot')
mt.pyplot.show()
import pandas as pd
import numpy as np
import matplotlib as mt
from statsmodels.graphics.mosaicplot import mosaic

train_data = pd.read_csv("../input/titanic/train.csv",delimiter = ",")

# train_data.head(5)
# print(np.c_[train_data['Sex'],train_data['Survived']])
engineer_data = np.c_[train_data['Pclass'],train_data['Survived']]
my_dataframe = pd.DataFrame(engineer_data,columns = ['Pclass','Survived'])
mosaic(data = my_dataframe,index = ['Pclass','Survived'],title = 'Mosaic Plot')
mt.pyplot.show()
import pandas as pd
import numpy as np
import matplotlib as mt
from statsmodels.graphics.mosaicplot import mosaic

train_data = pd.read_csv("../input/titanic/train.csv",delimiter = ",")

eng_data = np.c_[train_data['Embarked'],train_data['Survived']]
myDataframe = pd.DataFrame(eng_data,columns=['Embaraked','Survived'])
mosaic(data=myDataframe,index=['Embaraked','Survived'],title='MosaicPlot')
mt.pyplot.show()
import pandas as pd
import numpy as np
import matplotlib as mt
from statsmodels.graphics.mosaicplot import mosaic

survived_train_data1 = train_data[train_data.Survived == 1]['Fare']
survived_train_data2 = train_data[train_data.Survived == 0]['Fare']

data1 = survived_train_data1.dropna(how = 'all').values.tolist()
data2 = survived_train_data2.dropna(how = 'all').values.tolist()

data1 = list(filter(lambda x: x != 0,data1))
data2 = list(filter(lambda x: x != 0,data2))

data1 = np.log(data1)
data2 = np.log(data2)

beard = (data1,data2)
fig = mt.pyplot.figure()
ax = fig.subplots()
bp = ax.boxplot(beard)
ax.set_xticklabels(['Survived','Drad'])
mt.pyplot.title('Box plot')
mt.pyplot.grid()
mt.pyplot.xlabel('Survived')
mt.pyplot.ylabel('Fare')
mt.pyplot.ylim([0,8])

mt.pyplot.show()
import pandas as pd
import numpy as np
import matplotlib as mt
from statsmodels.graphics.mosaicplot import mosaic
import seaborn as sb

train_data = pd.read_csv("../input/titanic/train.csv",delimiter = ",")

print("【データ総数】 \n", train_data.shape[0])
print("【欠損データ有無】 \n",train_data.isnull().any())
print("【欠損データ数】 \n",train_data.isnull().sum())

print("【年齢:平均値】",train_data['Age'].dropna().mean())
print("【年齢:中央値】",train_data['Age'].dropna().median())
print("【年齢:標準偏差】",train_data['Age'].dropna().std())
print("【年齢:範囲】",train_data['Age'].dropna().min(),'~',
     train_data['Age'].dropna().max())
train_data['Age'].plot(kind='hist',bins=50,subplots=True);
mt.pyplot.show()
import matplotlib.pyplot as plt
import matplotlib.patches as patch

fig = plt.figure()
ax = plt.axes()

c = patch.Circle(xy=(0,0), radius=0.5, fc='w', ec='r')
ax.add_patch(c)

plt.axis('scaled')
ax.set_aspect('equal')

%matplotlib nbagg
from matplotlib.animation import PilowWritter,FuncAnimation

fig = plt.figure()
ax1 = fig.add_subplot(111)



A=np.array([-3,2])
B=np.array([-3,-2])
C=np.array([3,-2])
D=np.array([3,2])

scale=1.1
scale_P = 1.2

p = patch.Polygon(xy = [A,B,C,D],
                 edgecolor = 'black',
                 facecolor = 'white',
                 linewidth = 1.6)

def initialize():
    ax1.set_xlim(-4,4)
    ax1.set_ylim(-3,3)

    ax1.text(-3.5,0.0,"4cm",horizontalalignment='center',verticalalignment='center')
    ax1.text(0.0,-2.5,"6cm",horizontalalignment='center',verticalalignment='center')

    ax1.add_patch(p)


    ax1.text(A[0]*scale,A[1]*scale,"A",fontsize=15,horizontalalignment='center',verticalalignment='center')
    ax1.text(B[0]*scale,B[1]*scale,"B",fontsize=15,horizontalalignment='center',verticalalignment='center')
    ax1.text(C[0]*scale,C[1]*scale,"C",fontsize=15,horizontalalignment='center',verticalalignment='center')
    ax1.text(D[0]*scale,D[1]*scale,"D",fontsize=15,horizontalalignment='center',verticalalignment='center')

P = np.array([-3.0,1.0])


ax1.plot(P[0],P[1],marker = 'o', color = 'black')
ax1.text(P[0]*scale_P,P[1]*scale_P,"P",fontsize = 15,horizontalalignment = 'center',verticalalignment = 'center')

S = patch.Polygon(xy = [A,P,D],
                    edgecolor='black',
                    facecolor='lightgray',
                    linewidth=1.6)

ax1.add_patch(S)

#枠を消す
plt.axis('off')

plt.show()