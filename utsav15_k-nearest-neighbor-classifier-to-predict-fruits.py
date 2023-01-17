import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for plotting and visualozing data
#our dataset
fruits=pd.read_table('../input/fruit_data_with_colors.txt')
#checking first five rows of our dataset
fruits.head()
# create a mapping from fruit label value to fruit name to make results easier to interpret
predct = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))   
predct
#checking how many unique fruit names are present in the dataset
fruits['fruit_name'].value_counts()
apple_data=fruits[fruits['fruit_name']=='apple']
orange_data=fruits[fruits['fruit_name']=='orange']
lemon_data=fruits[fruits['fruit_name']=='lemon']
mandarin_data=fruits[fruits['fruit_name']=='mandarin']
apple_data.head()
mandarin_data.head()
orange_data.head()
lemon_data.head()
plt.scatter(fruits['width'],fruits['height'])
plt.scatter(fruits['mass'],fruits['color_score'])
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X=fruits[['mass','width','height']]
Y=fruits['fruit_label']
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=0)
X_train.describe()
X_test.describe()
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_test,y_test)
#parameters of following function are mass,width and height
#example1
prediction1=knn.predict([['100','6.3','8']])
predct[prediction1[0]]
#example2
prediction2=knn.predict([['300','7','10']])
predct[prediction2[0]]
