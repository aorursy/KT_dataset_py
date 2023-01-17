%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# let's limit our dataset
train_start=1
train_end=10000

print('Running MNIST, stats:')
print('... train_start:',train_start)
print('... train_end:',train_end)

print('Loading csv...')
train_file='../input/train.csv'
temp_train=np.array(pd.read_csv(train_file).ix[train_start-1:train_end-1].as_matrix(),dtype='uint8')

# remove label column
#print('Deleting labels...')
#train=np.delete(temp_train,0,1)

# display the first 100 digits
fig=plt.figure(figsize=(10,10))
for i in range(100):
    ax=fig.add_subplot(10,10,i+1)
    ax.set_axis_off()
    #a=np.copy(train[i])
    a=np.copy(temp_train[i,1:])
    #a=np.reshape(a,(28,28))
    #ax.imshow(a, cmap='gray', interpolation='nearest')
    ax.imshow(a.reshape((28,28)), cmap='gray', interpolation='nearest')
    plt.show()
# display 'average' digits
fig=plt.figure(figsize=(10,3))
for i in range(10):
    #train_ave=np.mean(train[np.where(temp_train[:,0]==i)],axis=0)
    #train_ave=np.mean(temp_train[:,1:][np.where(temp_train[:,0]==i)],axis=0)
    train_ave=np.mean(temp_train[:,1:][temp_train[:,0]==i],axis=0)
    ax=fig.add_subplot(2,5,i+1)
    ax.set_axis_off()
    a=np.copy(train_ave)
    #a=np.reshape(a,(28,28))
    ax.imshow(a.reshape((28,28)), cmap='gray', interpolation='nearest', clim=(0,255))
    plt.show()
