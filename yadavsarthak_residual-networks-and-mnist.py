import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(2142)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from keras.models import Model, Sequential

from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input,Activation, ZeroPadding2D, MaxPooling2D, Flatten, merge

from keras.optimizers import SGD

from keras.objectives import sparse_categorical_crossentropy as scc
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# let's separate stuff to make it more manageable
y_train = train['label']

train.drop(['label'], axis=1, inplace=True)
x_train = train.values.astype('float32') / 255

x_test = test.values.astype('float32') / 255
# below is a custom code for per image normalization. 

# It is faster than looping



# the constant term is as Advised by Andrew Ng in his UFLDL Tutorials



def per_image_normalization(X, constant=10.0, copy=True):

    if copy:

        X_res = X.copy()

    else:

        X_res = X



    means = np.mean(X, axis=1)

    variances = np.var(X, axis=1) + constant

    X_res = (X_res.T - means).T

    X_res = (X_res.T / np.sqrt(variances)).T

    return X_res
x_train = per_image_normalization(x_train)

x_test = per_image_normalization(x_test)
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)

x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
# lets get to it and define the function that will make up the network
def get_resnet():

    # In order to make things less confusing, all layers have been declared first, and then used

    

    # declaration of layers

    input_img = Input((1, 28, 28), name='input_layer')

    zeroPad1 = ZeroPadding2D((1,1), name='zeroPad1', dim_ordering='th')

    zeroPad1_2 = ZeroPadding2D((1,1), name='zeroPad1_2', dim_ordering='th')

    layer1 = Convolution2D(6, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv', dim_ordering='th')

    layer1_2 = Convolution2D(16, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv2', dim_ordering='th')

    zeroPad2 = ZeroPadding2D((1,1), name='zeroPad2', dim_ordering='th')

    zeroPad2_2 = ZeroPadding2D((1,1), name='zeroPad2_2', dim_ordering='th')

    layer2 = Convolution2D(6, 3, 3, subsample=(1,1), init='he_uniform', name='l1_conv', dim_ordering='th')

    layer2_2 = Convolution2D(16, 3, 3, subsample=(1,1), init='he_uniform', name='l1_conv2', dim_ordering='th')





    zeroPad3 = ZeroPadding2D((1,1), name='zeroPad3', dim_ordering='th')

    zeroPad3_2 = ZeroPadding2D((1,1), name='zeroPad3_2', dim_ordering='th')

    layer3 = Convolution2D(6, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv', dim_ordering='th')

    layer3_2 = Convolution2D(16, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv2', dim_ordering='th')



    layer4 = Dense(64, activation='relu', init='he_uniform', name='dense1')

    layer5 = Dense(16, activation='relu', init='he_uniform', name='dense2')



    final = Dense(10, activation='softmax', init='he_uniform', name='classifier')

    

    # declaration completed

    

    first = zeroPad1(input_img)

    second = layer1(first)

    second = BatchNormalization(0, axis=1, name='major_bn')(second)

    second = Activation('relu', name='major_act')(second)



    third = zeroPad2(second)

    third = layer2(third)

    third = BatchNormalization(0, axis=1, name='l1_bn')(third)

    third = Activation('relu', name='l1_act')(third)



    third = zeroPad3(third)

    third = layer3(third)

    third = BatchNormalization(0, axis=1, name='l1_bn2')(third)

    third = Activation('relu', name='l1_act2')(third)





    res = merge([third, second], mode='sum', name='res')





    first2 = zeroPad1_2(res)

    second2 = layer1_2(first2)

    second2 = BatchNormalization(0, axis=1, name='major_bn2')(second2)

    second2 = Activation('relu', name='major_act2')(second2)





    third2 = zeroPad2_2(second2)

    third2 = layer2_2(third2)

    third2 = BatchNormalization(0, axis=1, name='l2_bn')(third2)

    third2 = Activation('relu', name='l2_act')(third2)



    third2 = zeroPad3_2(third2)

    third2 = layer3_2(third2)

    third2 = BatchNormalization(0, axis=1, name='l2_bn2')(third2)

    third2 = Activation('relu', name='l2_act2')(third2)



    res2 = merge([third2, second2], mode='sum', name='res2')



    res2 = Flatten()(res2)



    res2 = layer4(res2)

    res2 = Dropout(0.4, name='dropout1')(res2)

    res2 = layer5(res2)

    res2 = Dropout(0.4, name='dropout2')(res2)

    res2 = final(res2)

    model = Model(input=input_img, output=res2)

    

    

    sgd = SGD(decay=0., lr=0.01, momentum=0.9, nesterov=True)

    model.compile(loss=scc, optimizer=sgd, metrics=['accuracy'])

    return model
res = get_resnet()
res.summary()
#from IPython.display import SVG

#from keras.utils.visualize_util import model_to_dot



#SVG(model_to_dot(res).create(prog='dot', format='svg'))
# we'll use a simple cross validation split of 5%, because any other cross validation scheme doesn't make sense
history = res.fit(x_train, y_train, validation_split=0.05, verbose=2, nb_epoch=1, batch_size=32)
import matplotlib.pyplot as plt
#plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

#plt.legend(['train', 'val'])

#plt.show()
# SUMMON sklearn's LabelBinarizer

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer().fit(y_train)
# lets get predictions now
preds = res.predict(x_test)
classes = (preds > 0.5).astype('int32')





# for those that dont know what happened, the above statement gave us binarized labels for each class

# this will give us labels as we need for submission

p = lb.inverse_transform(classes)
# sub = pd.DataFrame()

# ids = [x for x in range(1, 28001)]

# sub['ImageId'] = ids

# sub['Label'] = p

# sub.to_csv('resnet.csv', index=False)