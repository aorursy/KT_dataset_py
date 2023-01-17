import pandas as pd



# data. I'll use the hard_drives dataset & the cameras dataset is for you 

# to use as part of your exerise

hard_drives = pd.read_csv("../input/hard-drive-test-data/harddrive.csv")

cameras = pd.read_csv("../input/1000-cameras-dataset/camera_dataset.csv")
hard_drives.head()
from sklearn import linear_model

import numpy as np

import matplotlib.pyplot as plt

logreg = linear_model.LogisticRegression()

x = hard_drives['smart_1_normalized'][:, np.newaxis]

y = hard_drives['failure']

logreg.fit(x, y)



# and plot the result

plt.scatter(x.ravel(), y, color='black', zorder=20)

plt.plot(x, logreg.predict_proba(x)[:,1], color='blue', linewidth=3)

plt.xlabel('smart_1_normalized')

plt.ylabel('failure')
print('coefficient = ' + str(logreg.coef_))

print('intercept = ' + str(logreg.intercept_))
cameras.head()
from sklearn import linear_model

import numpy as np

import matplotlib.pyplot as plt

logreg = linear_model.LogisticRegression()

x = cameras['Max resolution'][:, np.newaxis]

y = cameras['Price']

logreg.fit(x, y)



# and plot the result

plt.scatter(x.ravel(), y, color='black', zorder=20)

plt.plot(x, logreg.predict_proba(x)[:,1], color='blue', linewidth=3)

plt.xlabel('Max resolution')

plt.ylabel('Price')
print('coefficient = ' + str(logreg.coef_))

print('intercept = ' + str(logreg.intercept_))