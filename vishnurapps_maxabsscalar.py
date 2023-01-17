import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for plotting figures



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from sklearn.preprocessing import MaxAbsScaler #We are studying the normalization here
heart_data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv', sep = ',')
heart_data.head()
len(heart_data)
data_to_work = heart_data[["age", "trestbps", "chol", "thalach", "oldpeak"]]
data_to_work.head()
data_to_work.describe()
scalar = MaxAbsScaler().fit(data_to_work)
scalar
scalar.max_abs_
maxabs_x = scalar.transform(data_to_work)
maxabs_x
df = pd.DataFrame(data = maxabs_x)
column_names = ["age", "trestbps", "chol", "thalach", "oldpeak"]
df.columns = column_names
df.head()
df.describe()
plt.figure(figsize=(8,6))

plt.hist(heart_data['trestbps'], facecolor = 'tomato', label = "Resting Blood Pressure")

plt.title('Distribution before normalization')

plt.xlabel('Blood Pressure')

plt.ylabel('Counts')

plt.legend()

plt.grid()

plt.show()
plt.figure(figsize=(8,6))

plt.hist(df['trestbps'], facecolor = 'tomato', label = "Resting Blood Pressure")

plt.title('Distribution before normalization')

plt.xlabel('Blood Pressure')

plt.ylabel('Counts')

plt.legend()

plt.grid()

plt.show()
plt.scatter(data_to_work['trestbps'], df['trestbps'])