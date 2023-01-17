from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import TensorBoard

from keras.layers import *

import numpy



from sklearn.model_selection import train_test_split



#ignoring the first row (header) 

# and the first column (unique experiment id, which I'm not using here)

dataset = numpy.loadtxt("../input/grasping-dataset/shadow_robot_dataset.csv", skiprows=1, usecols=range(1,30), delimiter=",")
header = ""



with open('../input/grasping-dataset/shadow_robot_dataset.csv', 'r') as f:

    header = f.readline()

    

header = header.strip("\n").split(',')

header = [i.strip(" ") for i in header]



saved_cols = []

for index,col in enumerate(header[1:]):

    if ("vel" in col) or ("eff" in col):

        saved_cols.append(index)
# only use velocity and effort, not position

new_X = []

for x in dataset:

    new_X.append([x[i] for i in saved_cols])

   

X = numpy.array(new_X)
Y = dataset[:,0]
# fix random seed for reproducibility

# and splitting the dataset

seed = 7

numpy.random.seed(seed)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)



# this is a sensible grasp threshold for stability

GOOD_GRASP_THRESHOLD = 100



# we're also storing the best and worst grasps of the test set to do some sanity checks on them

itemindex = numpy.where(Y_test>1.05*GOOD_GRASP_THRESHOLD)

best_grasps = X_test[itemindex[0]]

itemindex = numpy.where(Y_test<=0.95*GOOD_GRASP_THRESHOLD)

bad_grasps = X_test[itemindex[0]]



# discretizing the grasp quality for stable or unstable grasps

Y_train = numpy.array([int(i>GOOD_GRASP_THRESHOLD) for i in Y_train])

Y_train = numpy.reshape(Y_train, (Y_train.shape[0],))



Y_test = numpy.array([int(i>GOOD_GRASP_THRESHOLD) for i in Y_test])

Y_test = numpy.reshape(Y_test, (Y_test.shape[0],))
# create model

model = Sequential()



model.add(Dense(20*len(X[0]), use_bias=True, input_dim=len(X[0]), activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(1, activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_split=0.20, epochs=50,

          batch_size=500000)
import h5py

model.save("./model.h5")
scores = model.evaluate(X_test, Y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(best_grasps)



%matplotlib inline

import matplotlib.pyplot as plt



plt.hist(predictions,

         color='#77D651',

         alpha=0.5,

         label='Good Grasps')



plt.title('Histogram of grasp prediction')

plt.ylabel('Number of grasps')

plt.xlabel('Grasp quality prediction')

plt.legend(loc='upper right')



plt.show()
predictions_bad_grasp = model.predict(bad_grasps)





# Plot a histogram of defender size

plt.hist(predictions_bad_grasp,

         color='#D66751',

         alpha=0.3,

         label='Bad Grasps')



plt.title('Histogram of grasp prediction')

plt.ylabel('Number of grasps')

plt.xlabel('Grasp quality prediction')

plt.legend(loc='upper right')



plt.show()