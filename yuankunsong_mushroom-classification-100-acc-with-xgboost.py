import numpy as np 
import pandas as pd 

# graphics
import seaborn as sns
import matplotlib.pyplot as plt

# modeling
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
mushroom = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
mushroom.describe()
mushroom.info()
mushroom.isnull().sum()
# count classes

plt.figure(figsize=(6,6))
sns.countplot(data=mushroom, x='class')
plt.title('classes', size=19)
plt.show()
col = list(mushroom.columns)

for var in col:
    print(mushroom[var].value_counts())
    print('\n')
mush = mushroom.copy()

# base X and y
y = mush.iloc[:,0]
X = mush.iloc[:,1:]

# encode y
le = LabelEncoder()
le = le.fit(y)
le_y = le.transform(y)

# print encoded class for Edible and Poisonous
a = le.classes_
b = le.transform(le.classes_)

print('encoded class:', dict(zip(a,b)))
print('\n')

# encode X
features = []

for i in range(0, X.shape[1]):
    le = LabelEncoder()
    feature = le.fit_transform(X.iloc[:,i])
    features.append(feature)

le_X = np.array(features)
le_X = pd.DataFrame(le_X.T)

x_train, x_test, y_train, y_test = train_test_split(le_X, le_y, test_size = 0.3, random_state=10)

print('x_train:', x_train.shape)
print('x_test:', x_test.shape)
print('y_train:', y_train.shape)
print('y_test:', y_test.shape)
model = XGBClassifier()
model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy_score(y_test, pred)