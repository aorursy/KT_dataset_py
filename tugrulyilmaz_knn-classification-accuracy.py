# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")
df.head()
df.info()
st=df["class"].value_counts().index
sy=df["class"].value_counts().values
explode=[0.2,0.2]


plt.pie(sy,explode=explode,labels=st,autopct="%1.1f%%")
plt.title("abnormal-normal rate")
plt.show()
sns.countplot(df["class"])
plt.show()
sayici=0
for i in df["class"]:
    if(i=="Abnormal"):
        df["class"][sayici]=1
    else:
            df["class"][sayici]=0
    sayici+=1            

df["class"]=df["class"].astype("int64")
y=df["class"].values
x_data=df.drop(["class"],axis=1)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#knn model
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

score=knn.score(x_test,y_test)

print("accuracy:",score)