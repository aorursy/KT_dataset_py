import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split, GridSearchCV
from mlxtend.plotting import plot_decision_regions
from sklearn import svm, datasets, cross_validation
import numpy as np 
from scipy.ndimage.filters import gaussian_filter
%matplotlib inline
labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[0:5000,1:]
labels = labeled_images.iloc[0:5000,:1]

for i in range(images.shape[0]):
    img=images.iloc[i].as_matrix()
    img=img.reshape((28,28))
    img=gaussian_filter(img, sigma=1)
    img=img.reshape(784,1)
    images.iloc[i].as_matrix=img
    
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

i=5
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
img=gaussian_filter(img, sigma=1)
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
test_images[test_images<1]=0
test_images[test_images>=1]=1
train_images[train_images<1]=0
train_images[train_images>=1]=1
#test_images /=255 
#train_images /=255 

i=5
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])
clf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma=0.009, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(train_images, train_labels.values.ravel())
print ('Test score:')
clf.score(test_images,test_labels)

test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=clf.predict(test_data[0:42000])
results
df = pd.DataFrame(results)
df.index += 1
df.index.name = 'ImageId'
df.columns=['Label']
df.to_csv('results.csv', header=True)