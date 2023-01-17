# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation

from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical



np.random.seed(0)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
testdf = pd.read_csv('../input/test.csv')
testdf.head()
traindf = pd.read_csv('../input/train.csv')
traindf.head()
traindf.describe()
y_train = traindf.pop('label').values

x_train = traindf.values



x_subtrain, x_subtest, y_subtrain, y_subtest = train_test_split(x_train, y_train, test_size=0.2, random_state=0)



y_trainbinary = to_categorical(y_subtrain)

y_testbinary = to_categorical(y_subtest)
x_subtrain_norm = x_subtrain / 255

x_subtest_norm = x_subtest / 255

x_test = testdf.values

x_test_norm = x_test/255
y_subtest.shape
model = Sequential()

#Normalize the data?





#first number should be number of features

model.add(Dense(250, activation='relu', kernel_initializer='normal', input_dim=x_subtrain.shape[1] ))

model.add(Dropout(0.3))

#second dense num should be y.shape[1]

model.add(Dense(10, kernel_initializer='normal', activation='softmax'))



#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



model.fit(x_subtrain_norm, y_trainbinary, epochs=100, batch_size=150)



score = model.evaluate(x_subtest_norm, y_testbinary, batch_size=150)


print(score)

predictions = model.predict(x_test_norm, batch_size=150)
p_s = []

for p in predictions:

    p_s.append(np.argmax(p))

    
p_df = pd.DataFrame(p_s)
p_df['ImageId'] = p_df.index

p_df.head()
del p_df.index.name



p_df = p_df[['ImageId', 'Label']]
p_df.head()
p_df.to_csv('../predictions.csv')
%ls
print(check_output(["ls", "./"]).decode("utf8"))
