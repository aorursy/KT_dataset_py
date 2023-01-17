import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.applications import InceptionV3
import keras.backend as K
train = pd.read_csv('../input/sign_mnist_train.csv')
print(train.shape)
