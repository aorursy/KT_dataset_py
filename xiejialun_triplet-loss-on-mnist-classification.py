import numpy as np 

import pandas as pd 

import os

import warnings

warnings.filterwarnings('ignore')



np.random.seed(2019)



main_dir = '../input/digit-recognizer/'

os.listdir(main_dir)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

import keras

import keras.layers as klayers

import tensorflow as tf

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
#Some hyper-parameters



embedding_dims = 32

batch_size = 128

epochs = 30


train_df = pd.read_csv(main_dir + 'train.csv')

test_df = pd.read_csv(main_dir + 'test.csv')



print("Shape of train dataframe : ", train_df.shape)

print("Shape of test dataframe : ", test_df.shape)



#Check the data

train_df.head(5)
train_label = train_df['label']

train_df = train_df.drop(['label'], axis = 1)
#Check some images in training dataset

imgs = np.asarray( train_df.iloc[:10], dtype = np.uint8)

labels = np.asarray( train_label.iloc[:10], dtype = int)

ncols = 5

nrows = 2



plt.suptitle("Digits in training dataset")

plt.figure(figsize = (12,8))

for i,(img,label) in enumerate(zip(imgs,labels)):

    plt.subplot(nrows, ncols, i+1)

    plt.imshow(img.reshape(28,28), cmap = 'binary')

    plt.title("label : " + str(label))

    plt.axis("off")

plt.show()
#Split the training and validation dataset



train_x, valid_x, train_y, valid_y = train_test_split(train_df, train_label, test_size = 0.25, random_state = 2019)

print("train_x shape : ", train_x.shape)

print("train_y shape : ", train_y.shape)

print("valid_x shape : ", valid_x.shape)

print("valid_y shape : ", valid_y.shape)
train_x = np.asarray(train_x).reshape(-1,28,28,1)

valid_x = np.asarray(valid_x).reshape(-1,28,28,1)

train_y = np.asarray(train_y)

valid_y = np.asarray(valid_y)

num_classes = len(np.unique(train_y))



print("train_x shape : ", train_x.shape)

print("valid_x shape : ", valid_x.shape)
def input_generator(x,y,batch_size):

    

    out_imgs = []

    out_labels = []

    

    data_len = len(x)

    

    while True:

        

        idx = np.random.choice(range(data_len))

        out_imgs.append(x[idx])

        out_labels.append(y[idx])

        

        if len(out_imgs) >= batch_size:

            yield np.stack(out_imgs), np.stack(out_labels)

            out_imgs = []

            out_labels = []

            

def augmentation_generator(input_gen, data_gen, batch_size, embedding_dims):

    

    

    dummy_y = np.zeros((batch_size, embedding_dims + 1))

    for data, label in input_gen:    

        x = data_gen.flow(data, batch_size = batch_size, shuffle = False)

        

        yield [next(x),label], dummy_y
train_data_generator = ImageDataGenerator(

                                    rescale = 1.0 / 255.0,

                                    rotation_range = 40,

                                    width_shift_range = 0.15,

                                    height_shift_range = 0.15,

                                    shear_range = 0.1,

                                    zoom_range = 0.1,

                                    )

valid_data_generator = ImageDataGenerator(rescale = 1.0/255.0)

train_input_gen = input_generator(train_x,train_y,batch_size)

valid_input_gen = input_generator(valid_x,valid_y,batch_size)



train_aug = augmentation_generator(train_input_gen, train_data_generator, batch_size, embedding_dims)

valid_aug = augmentation_generator(valid_input_gen, valid_data_generator, batch_size, embedding_dims)
#Check the images after augmentation



(imgs, labels),dummy = next(train_aug)

ncols = 5

nrows = 2



plt.suptitle("Digits in training dataset")

plt.figure(figsize = (12,8))

for i,(img,label) in enumerate(zip(imgs,labels)):

    if i > 9 :

        break

    plt.subplot(nrows, ncols, i+1)

    plt.imshow(img.reshape(28,28), cmap = 'binary')

    plt.title("label : " + str(label))

    plt.axis("off")

plt.show()
def _cn_bn_relu(filters = 64, kernel_size = (3,3), strides = (1,1), padding = "same"):

    

    def f(input_x):

        

        x = input_x

        x = klayers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = padding,

                          kernel_initializer = "he_normal")(x)

        x = klayers.BatchNormalization()(x)

        x = klayers.Activation("relu")(x)

        

        return x

    return f



def _dn_bn_relu(units = 256):

    def f(input_x):

        

        x = input_x

        x = klayers.Dense(units = units)(x)

        x = klayers.BatchNormalization()(x)

        x = klayers.Activation("relu")(x)

        

        return x

    return f

    

def build_model(image_input, embedding_dims):

    

    x = _cn_bn_relu(filters = 64, kernel_size = (3,3))(image_input)

    x = klayers.MaxPooling2D(pool_size = (2,2))(x)

    x = _cn_bn_relu(filters = 64, kernel_size = (3,3))(x)

    x = klayers.MaxPooling2D(pool_size = (2,2))(x)

    x = klayers.Dropout(0.35)(x)

    x = klayers.Flatten()(x)

    x = _dn_bn_relu(units = 256)(x)

    x = _dn_bn_relu(units = 128)(x)

    x = klayers.Dropout(0.25)(x)

    x = _dn_bn_relu(units = 64)(x)

    x = klayers.Dense(units = embedding_dims, name = "embedding_layer")(x)

    

    return x



image_input = klayers.Input(shape = train_x.shape[1:], name = "image_input")

label_input = klayers.Input(shape = (1,), name = "label_input")



base_model = build_model(image_input, embedding_dims)

output = klayers.concatenate([label_input, base_model])



model = keras.models.Model(inputs = [image_input, label_input], outputs = [output])

model.summary()
def triplet_loss(y_true, y_pred, margin = 1.2):

    

    del y_true

    

    labels = y_pred[:,:1]

    labels = tf.dtypes.cast(labels, tf.int32)

    labels = tf.reshape(labels, (tf.shape(labels)[0],))

    

    embeddings = y_pred[:,1:]

    return tf.contrib.losses.metric_learning.triplet_semihard_loss(labels = labels,

                                                                 embeddings = embeddings,

                                                                 margin = margin)
optimizer = keras.optimizers.Adam(lr = 1e-3, decay = 1e-6)

model.compile(optimizer = optimizer, loss = triplet_loss)



reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience = 3, factor = 0.5, mode = 'min', verbose = 1, min_lr = 1e-6)

es = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15, mode = 'min')



steps_per_epoch = len(train_x) // batch_size

validation_steps = len(valid_x) // batch_size
history = model.fit_generator(train_aug, steps_per_epoch = steps_per_epoch,

                              epochs = epochs, verbose = 1,

                              validation_data = valid_aug, validation_steps = validation_steps,

                              shuffle = True, callbacks = [reduce_lr,es])
plt.figure(figsize = (12,8))

plt.plot(history.history['loss'], '-', label = 'train_loss', color = 'g')

plt.plot(history.history['val_loss'], '-', label = 'valid_loss', color = 'r')

plt.xlabel('epochs')

plt.ylabel('loss')

plt.title('triple loss of embedding model')

plt.legend()

plt.show()
#Check the layers

model.layers
#Transfer the weights from original model to embedding model



image_input = klayers.Input(shape = train_x.shape[1:])

embedding_output = build_model(image_input, embedding_dims = embedding_dims)

embedding_model = keras.models.Model(inputs = [image_input], outputs = [embedding_output])



for idx in range(1,18):

    target_layer = embedding_model.layers[idx]

    source_layer = model.layers[idx]

    target_layer.set_weights(source_layer.get_weights())

    

embedding_model.layers[-1].set_weights(model.layers[-2].get_weights())



embedding_model.summary()
#Use PCA to check the effect of embedding model



def plot_2d_distribution(embedding_data, num_classes, y, steps = 10, title = ""):

    pca = PCA(n_components = 2)

    decomposed_data = pca.fit_transform(embedding_data)

    plt.figure(figsize = (8,8))

    for label in range(num_classes):

        

        decomposed_class = decomposed_data[label == y]

        plt.scatter(decomposed_class[::steps, 1], decomposed_class[::steps,0], label = str(label))

    plt.legend()

    plt.title(title)

    plt.show()

    



train_x_embeddings = embedding_model.predict(train_x/255.0)

valid_x_embeddings = embedding_model.predict(valid_x/255.0)



plot_2d_distribution(train_x_embeddings, num_classes, train_y, title = 'training embeddings distribution in 2 dimensions')

plot_2d_distribution(valid_x_embeddings, num_classes, valid_y, title = 'validation embeddings distribution in 2 dimensions')
#Use SVC to predict the final label of embedding data



svc = SVC()

svc.fit(train_x_embeddings, train_y)

valid_prediction = svc.predict(valid_x_embeddings)

print("validation accuracy : ", accuracy_score(valid_y, valid_prediction))
test_x = np.asarray(test_df.values)

test_x = test_x.reshape(-1,28,28,1) / 255.0



test_embeddings = embedding_model.predict([test_x])

test_prediction = svc.predict(test_embeddings)
submission = pd.read_csv(main_dir + 'sample_submission.csv')

submission['Label'] = test_prediction



submission.to_csv('v3.csv', index = False)

submission.head(10)