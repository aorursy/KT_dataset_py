#Sign Language Digits Dataset using convolutional neural networks
import matplotlib.pyplot as plt
import numpy as np

x_l = np.load('../input/Sign-language-digits-dataset/X.npy')
y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')

plt.subplot(1,2,1)
plt.imshow(x_l[0].reshape((64,64)))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(x_l[300].reshape((64,64)))
plt.axis('off')         
print(x_l.shape)

y_l = np.argmax(y_l,axis = 1)

x_l = x_l.reshape(x_l.shape[0],64,64,1)
print(x_l.shape)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x_l,y_l,random_state = 40,test_size = 0.2)
no_of_train = X_train.shape[0]
no_of_test = X_test.shape[0]

print('{}\n{}\n'.format(no_of_train,no_of_test))


from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Input, BatchNormalization
from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D,Dropout
from keras.layers import Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D
def build_classifier(x):
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(512, activation = 'tanh', input_dim = x*x))
    classifier.add(Dropout(0.4))
    classifier.add(Dense(512,activation = 'tanh'))
    classifier.add(Dropout(0.4))
    classifier.add(Dense(10,activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier
    
    
# model = build_classifier(X_train.shape[1])

# model.summary()

# model.fit(x = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]),y = keras.utils.to_categorical(Y_train, num_classes=10)
# ,batch_size=16,epochs=30,validation_split=0.2, verbose=1)


#model.evaluate(x=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]), y=keras.utils.to_categorical(Y_test, num_classes=10), verbose=1)


# y_test_predict = np.argmax(model.predict(x=X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]), verbose=1),axis = 1)

# print('\n',y_test_predict)

# print(Y_test)

# print("Test accuracy : {}%".format(np.sum(y_test_predict == Y_test)/(no_of_test)))

def conv_classifier(a):    
    model_input = Input(shape=(a, a,1))
    
    # Define a model architecture
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(model_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)       
    x = MaxPooling2D(pool_size=(2, 2))(x)    
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)    
    x = Dropout(0.25)(x)
    
    y1 = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=model_input, outputs= y1)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model = conv_classifier(X_train.shape[1])

model.summary()
print(X_train.shape)

history = model.fit(x = X_train,y = keras.utils.to_categorical(Y_train, num_classes=10),batch_size=16,epochs=30,validation_split=0.2, verbose=1)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

print(model.evaluate(x=X_test, y=keras.utils.to_categorical(Y_test, num_classes=10), verbose=1))

y_test_predict = np.argmax(model.predict(x=X_test),axis = 1)

print('\n',y_test_predict)

print(Y_test)

print("Test accuracy : {}%".format(np.sum(y_test_predict == Y_test)/(no_of_test)))
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(64, 64, 3))
conv_base.trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True
    if layer.name == 'block4_conv1':
        layer.trainable = True    
    else:
        layer.trainable = False

conv_base.summary()

def VGG16_classifier():    
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten()) 
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
model1 = VGG16_classifier()
history_vgg = model1.fit(x = np.stack((X_train,)*3, -1),y = keras.utils.to_categorical(Y_train, num_classes=10),batch_size=32,epochs=10,validation_split=0.2, verbose=1)

print(model1.evaluate(x=X_test, y=keras.utils.to_categorical(Y_test, num_classes=10), verbose=1))

y_test_predict = np.argmax(model1.predict(x=X_test),axis = 1)

print('\n',y_test_predict)

print(Y_test)

print("Test accuracy : {}%".format(np.sum(y_test_predict == Y_test)/(no_of_test)))