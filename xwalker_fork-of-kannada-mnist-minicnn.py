! pip install -U keras-tuner
import numpy as np 

import pandas as pd 

from glob import glob

import matplotlib.pyplot as plt

from skimage.io import imread

from skimage.transform import resize

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (  Dense, 
                            Conv2D,
                            MaxPooling2D,
                            AveragePooling2D,
                            Dropout,
                            Flatten,
                            Input
                         )

from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
def metrics(pred_tag, y_true_tag):
    
    print(classification_report(pred_tag, y_true_tag))
    print('- ' * 20)
    print("PREC: ", precision_score(pred_tag, y_true_tag, average='micro'))
    print("REC: ", recall_score(pred_tag, y_true_tag, average='micro'))
    print("F1: ", f1_score(pred_tag, y_true_tag, average='micro'))
train_id = pd.read_csv('../input/Kannada-MNIST/train.csv')
test_id = pd.read_csv('../input/Kannada-MNIST/test.csv')
dig_id = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
x, y, z = 28, 28, 1
qtd_classes = 10

images = []
images_labels = []

valid = []
valid_labels = []

for count, (index, row) in enumerate(train_id.iterrows()):
    images.append(row.values[1:].reshape(x, y, z))
    images_labels.append(row.values[:1])
    
for count, (index, row) in enumerate(dig_id.iterrows()):
    valid.append(row.values[1:].reshape(x, y, z))
    valid_labels.append(row.values[:1])
TESTER = []
for count, (index, row) in enumerate(test_id.iterrows()):
    TESTER.append(row.values[1:].reshape(x, y, z))
SUBMIT =  np.asarray(TESTER) / 255.
TEST   =  np.asarray(valid) / 255.
TRAIN  =  np.asarray(images) / 255.

TRAIN_labels = to_categorical(np.asarray(images_labels), qtd_classes)
TEST_labels  = to_categorical(np.asarray(valid_labels), qtd_classes)
train = np.concatenate([TEST, TRAIN])
labels = np.concatenate([TRAIN_labels, TEST_labels])
plt.figure(figsize = (15, 5))
for i in range(0,10):
    plt.subplot(2,5,i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(np.argmax(labels[i])))
    plt.imshow(train[i,:,:,0], cmap="inferno")
x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size = 0.3)
NUM_CLASSES = 10
EPOCHS = 3
BATCH_SIZE = 64
from kerastuner import RandomSearch

def model_cnn(hp):
    
   
    inputShape = Input(shape=(x, y, z))
    tmp_model = inputShape
    
    num_layers = hp.Int('num_layers', 2, 8, default=6)
    lr = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5, 1e-8])
    
    for idx in range(num_layers):
        idx = str(idx)
        filters = hp.Int('filters_' + idx, 32, 256, step=32, default=64)
        tmp_model = Conv2D(filters=filters, kernel_size = 3, padding="same", activation='relu')(tmp_model)
        
        if tmp_model.shape[1] >= 8:
            pool_type = hp.Choice('pool_' + idx, values=['max', 'avg'])
            if pool_type == 'max':
                tmp_model = MaxPooling2D(2)(tmp_model)
            elif pool_type == 'avg':
                tmp_model = AveragePooling2D(2)(tmp_model)

        tmp_model = Dropout(0.25)(tmp_model)

    tmp_model = Flatten()(tmp_model)
    tmp_model = Dense(units=hp.Int('units',
                    min_value=32,
                    max_value=512,
                    step=32), activation='relu')(tmp_model)
    
    
    tmp_model = Dropout(0.5)(tmp_model)
    model = keras.Model(inputShape, Dense(10, activation='softmax')(tmp_model))
    model.compile(optimizer=Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
TRIALS = 5  # Quantidade de Modelos
EPOCHS = 15  # Épocas

tuner = RandomSearch(
    model_cnn, 
    objective='val_loss', 
    max_trials=TRIALS,
    project_name='helloworld'
)

# Display search space overview
tuner.search_space_summary()


tuner.search(
    x_train, 
    y_train, 
    batch_size=128, 
    epochs=EPOCHS,        
    validation_data=(x_test, y_test)
)

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
loss, accu = best_model.evaluate(x_test, y_test, verbose = 0)
print("%s: %.2f%%" % ('Accuracy...', accu))
print("%s: %.2f" % ('Loss...', loss))

print('-'*35)

preds =  np.argmax(best_model.predict(x_test), axis = 1)
y_true =  np.argmax(y_test,axis = 1)
metrics(preds, y_true)
results = best_model.predict(SUBMIT)
results = np.argmax(results,axis = 1)
data_out = pd.DataFrame({'id': range(len(SUBMIT)), 'label': results})
plt.figure(figsize = (15, 5))
for i in range(0,10):
    plt.subplot(2,5,i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(str(results[i]))
    plt.imshow(TESTER[i][:,:,0], cmap="inferno")
data_out.head()
data_out.to_csv('submission.csv', index = None)
