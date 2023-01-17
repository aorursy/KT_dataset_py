import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
# Load training and test data
train = pd.read_csv("../input/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashion-mnist_test.csv")
train = train.values
test = test.values
np.random.shuffle(train)
np.random.shuffle(test)
X = train[:,1:].reshape(-1, 28, 28, 1) / 255.0
y = train[:,0].astype(np.int32)
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import matplotlib.pyplot as plt
# Perform one-hot encoding of label values
num_classes=10

y = to_categorical(y, num_classes=num_classes)
# Creating the CNN as a sequence
model = Sequential()

model.add(Conv2D(input_shape=(28,28,1), filters=32, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=128, kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=num_classes))
model.add(Activation('softmax'))
# Compiling the model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
fitted_model = model.fit(X, y, validation_split=0.33, epochs=15, batch_size=32)
print(fitted_model)
# Plot training and validation loss
plt.plot(fitted_model.history['loss'], label='loss')
plt.plot(fitted_model.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
# Plot training and validation accuracy
plt.plot(fitted_model.history['acc'], label='acc')
plt.plot(fitted_model.history['val_acc'], label='val_acc')
plt.legend()
plt.show()
X_test = test[:,1:].reshape(-1, 28, 28, 1) / 255.0
y_test = test[:,0].astype(np.int32)
y_test = to_categorical(y_test, num_classes=num_classes)
test_model = model.evaluate(X_test, y_test, verbose=1)
print("On Test Data:")
print(model.metrics_names[0] + " = {}".format(test_model[0]))
print(model.metrics_names[1] + " = {}".format(test_model[1]))
mapping = {
    "0" : "T-shirt/top",
    "1" : "Trouser",
    "2" : "Pullover",
    "3" : "Dress",
    "4" : "Coat",
    "5" : "Sandal",
    "6" : "Shirt",
    "7" : "Sneaker",
    "8" : "Bag",
    "9" : "Ankle boot"
}
# Visualizing results
plt.imshow(X_test[89].reshape(28,28))
plt.axis('off')
plt.figure(figsize=(28,28))
class_val = np.argmax(model.predict(X_test[89].reshape(-1,28,28,1)))
print("Predicted as {}".format(mapping[str(class_val)]))
