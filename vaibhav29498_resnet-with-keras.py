import numpy as np
import keras
import csv
def identityBlock(X, f, filters, stage, block):
    conv_name = 'conv' + str(stage) + block + '_'
    bn_name = 'bn' + str(stage) + block + '_'
    
    X_prev = X
    
    X = keras.layers.Conv2D(filters[0], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2a', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    
    X = keras.layers.Conv2D(filters[1], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name + '2b', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    
    X = keras.layers.Conv2D(filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2c', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2c')(X)
    X = keras.layers.Add()([X, X_prev])
    X = keras.layers.Activation('relu')(X)
    
    return X
def convolutionalBlock(X, f, s, filters, stage, block):
    conv_name = 'conv' + str(stage) + block + '_'
    bn_name = 'bn' + str(stage) + block + '_'
    
    X_prev = keras.layers.Conv2D(filters[2], kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name + '1a', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X_prev = keras.layers.BatchNormalization(axis=3, name=bn_name + '1a')(X_prev)
    
    X = keras.layers.Conv2D(filters[0], kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name + '2a', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    
    X = keras.layers.Conv2D(filters[1], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name + '2b', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    
    X = keras.layers.Conv2D(filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2c', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2c')(X)
    X = keras.layers.Add()([X, X_prev])
    X = keras.layers.Activation('relu')(X)
    
    return X
def ResNet(input_shape=(28, 28, 1), classes=10):
    X_input = keras.Input(input_shape)
    X = keras.layers.ZeroPadding2D((3, 3))(X_input)
    
    X = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv1', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.BatchNormalization(axis=3, name='bn1')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)
    
    X = convolutionalBlock(X, 3, 1, [16, 16, 64], stage=2, block='a')
    X = identityBlock(X, 3, [16, 16, 64], stage=2, block='b')
    
    X = convolutionalBlock(X, 3, 2, [32, 32, 128], stage=3, block='a')
    X = identityBlock(X, 3, [32, 32, 128], stage=3, block='b')
    X = identityBlock(X, 3, [32, 32, 128], stage=3, block='c')
    
    X = keras.layers.AveragePooling2D((2, 2), name='avg_pool')(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(200, activation='relu', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.Dense(classes, activation='softmax', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(0.001))(X)
    
    model = keras.models.Model(inputs=X_input, outputs=X, name='ResNet')
    
    return model
model = ResNet()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
X_train = []
Y_train = []
X_test = []

with open('../input/train.csv', 'r') as train_file:
    reader = csv.reader(train_file)
    next(reader)
    for row in reader:
        Y_train.append(int(row[0]))
        X_train.append(list(map(int, row[1:])))

with open('../input/test.csv', 'r') as test_file:
    reader = csv.reader(test_file)
    next(reader)
    for row in reader:
        X_test.append(list(map(int, row)))

X_train = np.array(X_train)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
Y_train = np.eye(10)[Y_train]

print('Shape of X_train:', X_train.shape)
print('Shape of Y_train:', Y_train.shape)
print('Shape of X_test:', X_test.shape)
model.fit(X_train, Y_train, epochs = 40, batch_size = 512)
Y_test = np.argmax(model.predict(X_test), axis=1)
Y_test = np.column_stack((np.arange(1, 28001), Y_test)).astype(int)
np.savetxt('submission.csv', Y_test, fmt='%i', header='ImageId,Label', comments='', delimiter=',')