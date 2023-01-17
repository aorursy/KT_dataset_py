



import pandas as pd

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/into-the-future/train.csv")

df.head()
x = df.iloc[:,2:3].values

y = df.iloc[:,3].values
clf = PolynomialFeatures(degree = 24)

x_poly = clf.fit_transform(x)

clf2 = LinearRegression()

clf2.fit(x_poly,y)

accu = clf2.score(clf.fit_transform(x),y)

print(accu)

plt.scatter(x,y,color = 'red')

plt.plot(x,clf2.predict(clf.fit_transform(x)),color = 'blue')

plt.show()
test = pd.read_csv("../input/into-the-future/test.csv")

test.head()
x2 = test.iloc[:,2:3].values
pred_test_data = clf2.predict(clf.fit_transform(x2))
solution = pd.DataFrame({

    'id' : test.id,

    'feature_2' : pred_test_data 

})
solution
solution.to_csv('solution.csv',index = False)