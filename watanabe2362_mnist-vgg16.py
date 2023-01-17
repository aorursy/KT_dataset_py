# -*- coding: utf-8 -*-



#　parameters



pixel_H = 28

pixel_W = 28



batch_size = 32

epochs = 30



split_rate = 0.1



cut_block = "block5_conv3"      # last layer to take

untrainable = "block5_conv2"    # layers over untrainable are trainable 
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#from keras.datasets import mnist



#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()



#print('train_images.shape:', train_images.shape, 'train_labels.shape', train_labels.shape)

#print('test_images.shape:', test_images.shape, 'test_labels.shape', test_labels.shape)



# Load the data

import pandas as pd



train = pd.read_csv("../input/train.csv",dtype='uint')

test = pd.read_csv("../input/test.csv",dtype='uint')



train_labels = train["label"]

train_labels = train_labels.values

test_labels = None



# Drop 'label' column

train_images = train.drop(labels = ["label"],axis = 1)

del train



test_images = test

del test



# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)

train_images = train_images.values.reshape(-1,28,28,1)

test_images = test_images.values.reshape(-1,28,28,1)



print(type(train_images),train_images.shape,type(train_labels),train_labels.shape,test_images.shape)
def display_subplots(imges, labels=None, title=None, ncols=5, nrows_max=50):

    assert labels is None or len(imges) == len(labels)

    n = 0

    l = len(imges)

    nrows = (len(imges)+ncols-1)//ncols

    if nrows > nrows_max: nrows = nrows_max

    fig, ax = plt.subplots(nrows,ncols,figsize=(8,3*nrows),

                           squeeze=False,sharex=True,sharey=True)

    if title is not None: fig.suptitle(title)

    for row in range(nrows):

        for col in range(ncols):

            img = imges[n]

            ax[row,col].imshow(img.reshape((28,28)))

            if labels is not None:

                label = labels[n]  

                ax[row,col].set_title("{}".format(label))

            n += 1

            if n >= l: break

    plt.show()



display_subplots(train_images[0:10], train_labels[0:10], title="Train")

display_subplots(test_images[0:10], test_labels[:10] if test_labels is not None else None,title='Test')
from sklearn.model_selection import train_test_split



trains, valids, train_split_labels, valid_labels = train_test_split(train_images, train_labels, train_size=(1-split_rate))



print("trains:",trains.shape, train_split_labels.shape)

print("valids:", valids.shape,valid_labels.shape)

print("tests:", test_images.shape)



display_subplots(trains[0:10],train_split_labels[0:10],title="trains")

display_subplots(valids[0:10],valid_labels[0:10],title="valids")

display_subplots(test_images[0:10],title="tests")
# augmentation



from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.15, # Randomly zoom image 

        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(trains)



for images, labels in datagen.flow(trains, train_split_labels, batch_size=10):

    display_subplots(images,labels,title="Augmentation")

    break
# Model construction



#from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

from keras.models import Model, load_model

from keras.layers.core import Dense

from keras.optimizers import Adam, RMSprop, SGD

from keras.callbacks import EarlyStopping, ReduceLROnPlateau



# trained model(ex.VGG16)

base_model = VGG16()#include_top=False)

#base_model = VGG16()

base_model.summary()



#print("{}".format(len(base_model.layers)))

from keras import layers



# Modify input layer

# RGB input -> mono input

# from preprocess_input averages are here,

avR = 103.939   # R[0,255] - avR

avG = 116.779  # G - avG

avB = 123.68   # B - avB



# mono(gray) means R=G=B=x(gray)

# By preprocess_input,

#  r = R - avR = x - avR

#  g = G - avG = x - avG

#  b = B - avB = x - avB

#

# Output channel is

#  Wr*r+Wg*g+Wb*b+B

#  =Wr*(x-avR)+Wg*(x-avG)*Wb*(x-avB)+B      

#  =(Wr+Wg+Wb)*x+B-(avR*sum(Wr)+avG*sum(Wg)+avB*sum(Wb))



layer = base_model.get_layer(index=1)



weights=layer.get_weights() # (filter_w,filter_h,input_channel,output_channel)



w = weights[0]

w = w.sum(axis=2)

w = np.expand_dims(w,axis=2)



w1 = weights[0]

b = weights[1]

print('weights[0]',w1.shape)

for i in range(w1.shape[3]):

#    print(w1[:,:,0,i].shape)

    s = avR*np.sum(w1[:,:,0,i])+avG*np.sum(w1[:,:,1,i])+avB*np.sum(w1[:,:,2,i])

    b[i] -= s
#

# New Model by Functional API

#

from keras.models import Sequential, clone_model

from keras import regularizers

from keras.layers import MaxPooling2D, AveragePooling2D

from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.layers import Dense, Dropout, Flatten, Input

from keras.layers import Conv2D, LeakyReLU, Activation, BatchNormalization



def NewModel(params=None):

# New input layer and 1-st Convolution for new model M

    inputs = Input(name='NewInput',shape=(pixel_H,pixel_W,1))

    l = Conv2D(64, kernel_size=(3, 3),padding='same',

                 activation='relu', name='FirstLayer')

    x = l(inputs)          # layerのshapeを決定

    l.set_weights([w,b])

    l.trainable = False



# Construct upper layers from VGG16:only config without weights

# Insert BatchNormalization and Dropout



    for base_layer in base_model.layers[2:]:

        if 'pool' in base_layer.name: continue     # delete pooling

        config = base_layer.get_config()

        layer = layers.deserialize({

                    'class_name': base_layer.__class__.__name__,

                    'config': config})

        layer.trainable = False

        x = layer(x)

        if layer.name == cut_block: break



# Add last layers

#    x = GlobalAveragePooling2D()(x)

    x = AveragePooling2D(pool_size=(28 if params is None else params['size'],

                                28 if params is None else params['size']),

                     name='Add_pool_1')(x)

    x = Flatten(name='Add_flatten_1')(x)

    x = BatchNormalization(name='Add_batch_1')(x)

    x = Dropout(0.0 if params is None else params['Dropout_0'],name='Add_drop_1')(x)

    x = Dense(256 if params is None else params['Dense'],

              kernel_regularizer=regularizers.l2(

                  0.0 if params is None else params['Regularize_0']),name='Add_dense_0')(x) 

    x = Activation('relu',name='Add_activation_0')(x)

#    x = BatchNormalization(name='Add_batch_0')(x)

    x = Dropout(0.0 if params is None else params['Dropout_0'],

                name='Add_drop_0')(x)

    predictions = Dense(10,

                        name='Add_dense_10',

                        kernel_regularizer=regularizers.l2(

                            0.0 if params is None else params['Regularize_0']),

                        activation = 'softmax')(x)

# New model

    M = Model(inputs=inputs,outputs=predictions)



# Set weights

    for layer in M.layers[2:]:

        try:

            lb = base_model.get_layer(layer.name)

            layer.set_weights(lb.get_weights())

        except:

            pass

           

#    M.summary()



#    print("modified {}".format(len(M.layers)))

    return M

#M=NewModel()
def Transfer(params=None,model=None):

# after setting layer.trainable

    model.compile(

        optimizer = Adam(clipnorm=0.5),

        loss = 'sparse_categorical_crossentropy',

        metrics = ["accuracy"]

    )

# Initial Training

#history = M.fit(trains, train_labels[:train_size], epochs=epochs,batch_size=batch_size,

# #                       validation_data=(tests, test_labels[:test_size]),

#                validation_split = split_rate,

#               callbacks = [EarlyStopping()])

#print(batch_size,trains.shape[0])

    history = model.fit_generator(

            datagen.flow(trains, train_split_labels, batch_size=batch_size),

            verbose=2, steps_per_epoch=trains.shape[0]//batch_size,

            epochs=epochs,

            validation_data=(valids,valid_labels),

            callbacks = [EarlyStopping(verbose=1, patience=2)])

    return history
def plot_history(histories, key='binary_crossentropy'):

    plt.figure()



    for name, history in histories:

        val = plt.plot(history.epoch, history.history['val_'+key],

                   '--', label=name.title()+' Val')

        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),

             label=name.title()+' Train')



        plt.xlabel('Epochs')

        plt.ylabel(key.replace('_',' ').title())

        plt.legend()



        plt.xlim([0,max(history.epoch)])



#plot_history([('',history)],key='acc')

#plot_history([('',history)],key='loss')
def FineTuning(params=None,model=None):

# fine tuning

    for layer in model.layers:

        if 'Add' not in layer.name:          

            layer.trainable = False

    histories = []

    for layer in model.layers[::-1]:

        if 'Add' in layer.name: continue

        if layer.name == untrainable: break

        layer.trainable = True

#        print( layer.name)

    

        model.compile(

            optimizer = SGD(lr=0.001, momentum=0.9,nesterov=True,clipnorm=0.5),

            loss = 'sparse_categorical_crossentropy',

            metrics = ["accuracy"]

        )

#        model.summary()

#fine_history = M.fit(trains, train_labels[:train_size], epochs=epochs,batch_size=batch_size,

##                        validation_data=(tests, test_labels[:test_size]),

#                validation_split = split_rate,

#                callbacks = [EarlyStopping(patience=3)])



        fine_history = model.fit_generator(

            datagen.flow(trains, train_split_labels, batch_size=batch_size),

            verbose=2, steps_per_epoch=trains.shape[0]//batch_size,

            epochs=epochs,

            validation_data=(valids,valid_labels),

            callbacks = [EarlyStopping(verbose=1, patience=3),

                        ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)])

        histories.append(fine_history)

        

    return histories

#M=NewModel()

#hist=Transfer(model=M)

#hists=FineTuning(model=M)
def plot_histories(histories, key='binary_crossentropy'):

    i = 0

    epochs = []

    trains = []

    vals   = []

    for history in histories:

      for j in range(len(history.epoch)):

        epochs.append(i)

        i += 1

      trains = trains + history.history[key]

      vals = vals + history.history['val_'+key]

      

    plt.figure()



    val = plt.plot(epochs, vals, '--', label=' Val')

    plt.plot(epochs, trains, color=val[0].get_color(), label=' Train' )

    plt.xlabel('Epochs')

    plt.ylabel(key.replace('_',' ').title())

    plt.legend()

    plt.xlim([0,max(epochs)])

    

#plot_histories( [history, fine_history], key = 'acc')

#plot_histories( [history, fine_history], key = 'loss')

import hyperopt

from hyperopt import hp, fmin, rand, tpe, Trials, space_eval, STATUS_OK



params = {

    'Dropout_0':0.35,# hp.uniform('Dropout_0', 0.25, 0.35),

#    'Dropout_1': hp.uniform('Dropout_1', 0.2, 0.5),

    'Regularize_0':hp.uniform('Regularize_0', 0.0, 0.0001),

#    'Regularize_1':hp.uniform('Regularize_1', 0.0, 0.0001),

    'Dense':256,#hp.choice('Dense', [128,256,512]),

    'size':14,#hp.choice('size',[14,7,4])

}



def TransferFineModel(params):

    print(params)

    M = NewModel(params)

    hist = Transfer(params,M)

    hists = FineTuning(params,M)

    acc = hists[-1].history['val_acc'][-1]

    return {'loss':-acc, 

            'status':STATUS_OK, 

            'model':M, 

            'hist':hist, 

            'fine':hists}



trials = Trials()

best = fmin(fn=TransferFineModel, 

            space=params, 

            algo=tpe.suggest, 

            max_evals=8,

            trials=trials,

            rstate=np.random.RandomState(1234),

            verbose=1)
print(space_eval(params, best))

best_model = trials.best_trial['result']['model']

best_model.save('hyperopt_mnist_VGG16.hdf5')



score = best_model.evaluate(valids, valid_labels)

print("Test loss:{:.4f} Test acc:{:.4f}".format(score[0], score[1]))



hist = trials.best_trial['result']['hist']

hists = trials.best_trial['result']['fine']

hists.insert(0,hist)

plot_histories(hists, key = 'acc')

plot_histories(hists, key = 'loss')
# Error check

# Predict the values from the validation dataset

Y_true = valid_labels

Y_pred = best_model.predict(valids)

Y_pred_classes = Y_pred.argsort()[:,-3:][:,::-1]



# Errors are difference between predicted labels and true labels

errors = (Y_pred_classes[:,0] != Y_true)



Y_pred_classes_errors = Y_pred_classes[errors]

Y_pred_errors = Y_pred[errors]

Y_true_errors = Y_true[errors]

X_val_errors = valids[errors]



def display_errors( title, classes, preds, trues, images ):

    titles = []

    for i in range(preds.shape[0]):

        index0 = classes[i,0]

        index1 = classes[i,1]

        index2 = classes[i,2]

        titles.append("True:{}\nPredict:{} {:.3}\n{} {:.3}\n{} {:.3}"

                  .format(trues[i],

                        index0, preds[i,index0],

                        index1, preds[i,index1],

                        index2, preds[i,index2]))

    

    display_subplots(images,titles,title=title )

    

title = "Errors {}/{} {:.3}%".format( Y_pred_errors.shape[0], Y_pred.shape[0], 

                                     Y_pred_errors.shape[0]/Y_pred.shape[0])

display_errors( title, Y_pred_classes_errors, Y_pred_errors, Y_true_errors, X_val_errors[:,:,:,0])
errorss = (Y_pred_classes_errors[:,1] != Y_true_errors)

for i in range(len(errorss)):

    errorss[i] = errorss[i] and (Y_pred_errors[i,1] < 0.001)

Y_pred_classes_errors = Y_pred_classes_errors[errorss]

Y_pred_errors = Y_pred_errors[errorss]

Y_true_errors = Y_true_errors[errorss]

X_val_errors = X_val_errors [errorss]



title = "Errors {} {:.3}%".format( Y_pred_errors.shape[0], Y_pred_errors.shape[0]/Y_pred.shape[0])

display_errors( title, Y_pred_classes_errors, Y_pred_errors, Y_true_errors, X_val_errors[:,:,:,0])

# predict

preds = best_model.predict(test_images)



labels = []

errors = []

for i, scores in enumerate(preds):

    top3 = scores.argsort()[-3:][::-1]

    if scores[top3[0]] < 0.75:

        labels.append('{}:{:.3}\n{}:{:.3}\n{}:{:.3}'.format(top3[0], scores[top3[0]],

                                            top3[1], scores[top3[1]],

                                            top3[2], scores[top3[2]]))

        errors.append(i)

display_subplots(test_images[errors[:50]],labels[:50],

            title="Prediction errors? {}/{}".format(len(errors),len(test_images)))
# select the indix with the maximum probability

results = np.argmax(preds,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)