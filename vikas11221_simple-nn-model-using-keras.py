from keras.models import Sequential

from keras.layers import Dense

from keras import optimizers

from keras.utils import plot_model

import numpy
# fix random seed for reproducibility

numpy.random.seed(7)



# load pima indians dataset

dataset = numpy.loadtxt("../input/diabetes.csv", delimiter=",", skiprows=1)

# split into input (X) and output (Y) variables

X = dataset[:,0:8]

Y = dataset[:,8]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
# create model

model = Sequential()

model.add(Dense(12, input_dim=8,kernel_initializer='uniform', activation='relu'))

model.add(Dense(8,kernel_initializer='uniform', activation='relu'))

model.add(Dense(1,kernel_initializer='uniform', activation='sigmoid'))
# Compile model

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# Fit the model

model.fit(X_train, y_train, epochs=200, batch_size=10)
# evaluate the model

scores = model.evaluate(X_test, y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))