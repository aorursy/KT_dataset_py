import pandas as pd

import matplotlib.pyplot as plt, matplotlib.image as mpimg

from sklearn.model_selection import train_test_split

from sklearn import svm

%matplotlib inline



pd.options.mode.chained_assignment = None
train_data = pd.read_csv('../input/train.csv')

test_data=pd.read_csv('../input/test.csv')



images = train_data.iloc[0:,1:]

labels = train_data.iloc[0:,:1]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.9, random_state=0)



test_images[test_images>0]=1

train_images[train_images>0]=1

test_data[test_data>0]=1
clf = svm.SVC()



clf.fit(train_images, train_labels.values.ravel())

print(clf.score(test_images,test_labels))
df = pd.DataFrame(

    clf.predict(test_data)

)

df.index.name='ImageId'

df.index += 1

df.columns = ['Label']

df.to_csv('results.csv', header=True)