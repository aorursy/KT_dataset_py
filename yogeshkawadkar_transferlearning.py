from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train=x_train/255

x_test=x_test/255
import matplotlib.pyplot as plt
plt.imshow(x_train[2])
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout

from keras.models import Sequential
model = Sequential([

                  Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(32, 32, 3)),

                  Dropout(0.2),

                  Conv2D(16, kernel_size = (3, 3)),

                  MaxPooling2D(pool_size=(2,2)),

                  Flatten(),

                  Dense(10,activation='softmax')

])
model.summary()
model.compile(

            optimizer='adam',

            loss='sparse_categorical_crossentropy',

            metrics=['sparse_categorical_accuracy'])

history=model.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.3)
import keras

pretrained_model = keras.applications.vgg16.VGG16(weights ='imagenet',

                  include_top=False,

                  input_shape=(32,32, 3))
pretrained_model.summary()
from keras.models import Model





intermediate_layer_model = Model(inputs=pretrained_model.input,

                                 outputs=pretrained_model.layers[6].output)

intermediate_layer_model.trainable = False
intermediate_layer_model.summary()
model1 = keras.Sequential([

    intermediate_layer_model,

    Conv2D(32, kernel_size = (3, 3)),

    Dropout(0.2),

    Conv2D(16, kernel_size = (3, 3)),

    MaxPooling2D(pool_size=(2,2)),

    Flatten(),

    Dense(10, activation='softmax')

])
model1.summary()
model1.compile(

            optimizer='adam',

            loss='sparse_categorical_crossentropy',

            metrics=['sparse_categorical_accuracy'])

history1=model1.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.3)
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(15,15))



plt.subplot(2, 2, 1)

plt.plot(history.history['sparse_categorical_accuracy'])

plt.plot(history.history['val_sparse_categorical_accuracy'])

plt.title('model accuracy for normal model')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')





plt.subplot(2, 2, 2)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss for normal model')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')



plt.subplot(2, 2, 3)

plt.plot(history1.history['sparse_categorical_accuracy'])

plt.plot(history1.history['val_sparse_categorical_accuracy'])

plt.title('model accuracy for Transfer model')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')



plt.subplot(2, 2, 4)

plt.plot(history1.history['loss'])

plt.plot(history1.history['val_loss'])

plt.title('model loss for Transfer model')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')



plt.plot(history.history['val_loss'])

plt.plot(history1.history['val_loss'])

plt.title('model val-loss for models')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['normal', 'transfer'], loc='upper left')
plt.plot(history.history['val_sparse_categorical_accuracy'])

plt.plot(history1.history['val_sparse_categorical_accuracy'])

plt.title('model accuracy for models')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['normal', 'transfer'], loc='upper left')
results = model.evaluate(x_test, y_test, batch_size=128)

results1 = model1.evaluate(x_test, y_test, batch_size=128)
print(results[1])

print(results1[1])