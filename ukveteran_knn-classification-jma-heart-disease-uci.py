import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df=pd.read_csv("../input/heart-disease-uci/heart.csv")

df



#d=diseased, not_d =not diseased

d =df[df.target==1]

not_d=df[df.target==0]



plt.scatter(d.trestbps,d.thalach,color="red",label="diseased")

plt.scatter(not_d.trestbps,not_d.thalach,color="green",label="not diseased")

plt.xlabel("trestbps")

plt.ylabel("thalach")

plt.legend()

plt.show()





y=df.target.values

x_data=df.drop(["target"],axis=1)



#normalization

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))



#train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=1)



#knn model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=50)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)



print(" {} nn  score: {}".format(50,knn.score(x_test,y_test)))



#find k value

score_list=[]

for each in range(1,150):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,150),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df=pd.read_csv("../input/heart-disease-uci/heart.csv")

df



d =df[df.exang==1]

not_d=df[df.exang==0]



plt.scatter(d.trestbps,d.thalach,color="red",label="diseased")

plt.scatter(not_d.trestbps,not_d.thalach,color="green",label="not diseased")

plt.xlabel("trestbps")

plt.ylabel("thalach")

plt.legend()

plt.show()





y=df.exang.values

x_data=df.drop(["exang"],axis=1)



#normalization

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))



#train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=1)



#knn model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=50)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)



print(" {} nn  score: {}".format(50,knn.score(x_test,y_test)))



#find k value

score_list=[]

for each in range(1,150):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,150),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df=pd.read_csv("../input/heart-disease-uci/heart.csv")

df



d =df[df.restecg==1]

not_d=df[df.restecg==0]



plt.scatter(d.trestbps,d.thalach,color="red",label="diseased")

plt.scatter(not_d.trestbps,not_d.thalach,color="green",label="not diseased")

plt.xlabel("trestbps")

plt.ylabel("thalach")

plt.legend()

plt.show()





y=df.restecg.values

x_data=df.drop(["restecg"],axis=1)



#normalization

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))



#train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=1)



#knn model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=50)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)



print(" {} nn  score: {}".format(50,knn.score(x_test,y_test)))



#find k value

score_list=[]

for each in range(1,150):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,150),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df=pd.read_csv("../input/heart-disease-uci/heart.csv")

df



d =df[df.fbs==1]

not_d=df[df.fbs==0]



plt.scatter(d.trestbps,d.thalach,color="red",label="diseased")

plt.scatter(not_d.trestbps,not_d.thalach,color="green",label="not diseased")

plt.xlabel("trestbps")

plt.ylabel("thalach")

plt.legend()

plt.show()





y=df.fbs.values

x_data=df.drop(["fbs"],axis=1)



#normalization

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))



#train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.3,random_state=1)



#knn model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=50)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)



print(" {} nn  score: {}".format(50,knn.score(x_test,y_test)))



#find k value

score_list=[]

for each in range(1,150):

    knn2=KNeighborsClassifier(n_neighbors=each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

plt.plot(range(1,150),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()