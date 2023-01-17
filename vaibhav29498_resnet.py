import keras
import numpy as np
import pickle
def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='latin1')
    return d

!tar xzvf ../input/cifar-10-python.tar.gz
X_train = []
Y_train = []
for i in range(1, 6):
    d = unpickle('cifar-10-batches-py/data_batch_' + str(i))
    for j in range(10000):
        img_R = d['data'][j][0:1024].reshape((32, 32))
        img_G = d['data'][j][1024:2048].reshape((32, 32))
        img_B = d['data'][j][2048:3072].reshape((32, 32))
        img = np.dstack((img_R, img_G, img_B))
        X_train.append(img)
        Y_train.append(d['labels'][j])

X_train = np.array(X_train)
Y_train = np.eye(10)[Y_train]

print('Shape of X_train:', X_train.shape)
print('Shape of Y_train:', Y_train.shape)
def identityBlock(X, f, filters, stage, block, lambd=0.0):
    conv_name = 'conv' + str(stage) + block + '_'
    bn_name = 'bn' + str(stage) + block + '_'
    
    X_prev = X
    
    X = keras.layers.Conv2D(filters[0], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2a', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    
    X = keras.layers.Conv2D(filters[1], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name + '2b', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    
    X = keras.layers.Conv2D(filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2c', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2c')(X)
    X = keras.layers.Add()([X, X_prev])
    X = keras.layers.Activation('relu')(X)
    
    return X
def convolutionalBlock(X, f, s, filters, stage, block, lambd=0.0):
    conv_name = 'conv' + str(stage) + block + '_'
    bn_name = 'bn' + str(stage) + block + '_'
    
    X_prev = keras.layers.Conv2D(filters[2], kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name + '1a', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    X_prev = keras.layers.BatchNormalization(axis=3, name=bn_name + '1a')(X_prev)
    
    X = keras.layers.Conv2D(filters[0], kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name + '2a', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2a')(X)
    X = keras.layers.Activation('relu')(X)
    
    X = keras.layers.Conv2D(filters[1], kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name + '2b', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2b')(X)
    X = keras.layers.Activation('relu')(X)
    
    X = keras.layers.Conv2D(filters[2], kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name + '2c', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    X = keras.layers.BatchNormalization(axis=3, name=bn_name + '2c')(X)
    X = keras.layers.Add()([X, X_prev])
    X = keras.layers.Activation('relu')(X)
    
    return X
def ResNet(input_shape=(32, 32, 3), classes=10, lambd=0.0):
    X_input = keras.Input(input_shape)
    X = keras.layers.ZeroPadding2D((3, 3))(X_input)
    
    X = keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv1', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    X = keras.layers.BatchNormalization(axis=3, name='bn1')(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)
    
    X = convolutionalBlock(X, 3, 1, [16, 16, 64], stage=2, block='a', lambd=lambd)
    X = identityBlock(X, 3, [16, 16, 64], stage=2, block='b', lambd=lambd)
    
    X = convolutionalBlock(X, 3, 2, [32, 32, 128], stage=3, block='a', lambd=lambd)
    X = identityBlock(X, 3, [32, 32, 128], stage=3, block='b', lambd=lambd)
    X = identityBlock(X, 3, [32, 32, 128], stage=3, block='c', lambd=lambd)
    
    X = convolutionalBlock(X, 3, 2, [64, 64, 256], stage=4, block='a', lambd=lambd)
    X = identityBlock(X, 3, [64, 64, 256], stage=4, block='b', lambd=lambd)
    X = identityBlock(X, 3, [64, 64, 256], stage=4, block='c', lambd=lambd)
    X = identityBlock(X, 3, [64, 64, 256], stage=4, block='d', lambd=lambd)
    X = identityBlock(X, 3, [64, 64, 256], stage=4, block='e', lambd=lambd)
    
    X = convolutionalBlock(X, 3, 2, [128, 128, 512], stage=5, block='a', lambd=lambd)
    X = identityBlock(X, 3, [128, 128, 512], stage=5, block='b', lambd=lambd)
    X = identityBlock(X, 3, [128, 128, 512], stage=5, block='c', lambd=lambd)
    
    X = keras.layers.AveragePooling2D((2, 2), name='avg_pool')(X)
    X = keras.layers.Flatten()(X)
    # X = keras.layers.Dense(100, activation='relu', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    X = keras.layers.Dense(classes, activation='softmax', kernel_initializer=keras.initializers.glorot_normal(), kernel_regularizer=keras.regularizers.l2(lambd))(X)
    
    model = keras.models.Model(inputs=X_input, outputs=X, name='ResNet')
    
    return model
model = ResNet(lambd=0.05)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train, epochs = 100, batch_size = 256)
d = unpickle('cifar-10-batches-py/test_batch')
X_test = []
Y_test = []
for j in range(10000):
    img_R = d['data'][j][0:1024].reshape((32, 32))
    img_G = d['data'][j][1024:2048].reshape((32, 32))
    img_B = d['data'][j][2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    X_test.append(img)
    Y_test.append(d['labels'][j])

X_test = np.array(X_test)
Y_test = np.eye(10)[Y_test]

print('Shape of X_test:', X_test.shape)
print('Shape of Y_test:', Y_test.shape)

model.evaluate(x=X_test, y=Y_test, batch_size=512)