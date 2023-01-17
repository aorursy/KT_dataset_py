import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split # for letting sklearn do the optimal splitting

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn import svm



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/train.csv')
print(df.describe())
# taking ~10% of the samples for

images = df.iloc[0:5000,1:]

labels = df.iloc[0:5000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
i = 0

plt.hist(train_images.iloc[i])
clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test_images[test_images>0]=1

train_images[train_images>0]=1
plt.hist(train_images.iloc[i])
clf = svm.SVC()

clf.fit(train_images, train_labels.values.ravel())

clf.score(test_images,test_labels)
test_data=pd.read_csv('../input/test.csv')

test_data[test_data>0]=1

results=clf.predict(test_data[0:5000])
df = pd.DataFrame(results)

df.index.name='ImageId'

df.index+=1

df.columns=['Label']

df.to_csv('results.csv', header=True)