import pandas as pd

from keras.utils import to_categorical
from keras import models
from keras import layers
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print('Train data shape : %s' % str(train.shape))
print('Test data shape : %s' % str(test.shape))
train_images = train.iloc[:,1:]
train_images = train_images.astype('float32') / 255

test_images = test.astype('float32') / 255
train_labels = train['label']
train_labels = to_categorical(train_labels)
print('Train labels shape : %s' % str(train_labels.shape))
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128)
predictions = network.predict_classes(test_images, verbose=0)
submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                            "Label": predictions})
submissions.to_csv("keras_first_dnn.csv", index=False, header=True)