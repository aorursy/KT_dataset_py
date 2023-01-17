# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import datetime

import matplotlib

matplotlib.use('Agg') # enable matplotlib to save .png to disk

import matplotlib.pyplot as plt

from functools import partial

import os

from sklearn.metrics import confusion_matrix, classification_report

%matplotlib inline
batch_size = 64

epochs = 5

regularizer = 1e-3

total_train_samples = 60000

total_test_samples = 10000

lr_decay_epochs = 1

output_folder = './model_output'



if not os.path.exists(output_folder):

    os.makedirs(output_folder)

    

save_format="hdf5" # or saved_model



if save_format == "hdf5":

    save_path_models = os.path.join(output_folder,"hdf5_models")

    if not os.path.exists(save_path_models):

        os.makedirs(save_path_models)

    save_path = os.path.join(save_path_models,"ckpt_epoch{epoch:02d}_val_acc{val_acc:.2f}.hdf5")

elif save_format == "saved_model":

    save_path_models = os.path.join(output_folder,"saved_models")

    if not os.path.exists(save_path_models):

        os.makedirs(save_path_models)

    save_path = os.path.join(save_path_models,"ckpt_epoch{epoch:02d}_val_acc{val_acc:.2f}.ckpt")

    

# To save logs

log_dir = os.path.join(output_folder,'logs_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

if not os.path.exists(log_dir):

    os.makedirs(log_dir)
physical_devices = tf.config.experimental.list_physical_devices('GPU') # List all the visable GPU

print("All the available GPUs:\n",physical_devices)

if physical_devices:

    gpu=physical_devices[0] # show the first GPU

    tf.config.experimental.set_memory_growth(gpu, True) # Increase the memory automatically if needed

    tf.config.experimental.set_visible_devices(gpu, 'GPU') # Only choose the first GPU
train = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

test = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_test.csv")
train_x = train.drop(['label'], axis=1).values.reshape([-1, 28, 28])

train_y = train.label

test_x = test.drop(['label'], axis=1).values.reshape([-1, 28, 28])

test_y = test.label



train_x,test_x = train_x[...,np.newaxis]/255.0,test_x[...,np.newaxis]/255.0

total_train_sample = train_x.shape[0]

total_test_sample = test_x.shape[0]

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_ds = tf.data.Dataset.from_tensor_slices((train_x,train_y))

test_ds = tf.data.Dataset.from_tensor_slices((test_x,test_y))

 

train_ds=train_ds.shuffle(buffer_size=batch_size*10).batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE).repeat()

test_ds = test_ds.batch(batch_size).prefetch(buffer_size = tf.data.experimental.AUTOTUNE) # without repeat，only execute once
l2 = tf.keras.regularizers.l2(regularizer) # define the regularizer

ini = tf.keras.initializers.he_normal() # define the initializer of params

conv2d = partial(tf.keras.layers.Conv2D,activation='relu',padding='same',kernel_regularizer=l2,bias_regularizer=l2)

fc = partial(tf.keras.layers.Dense,activation='relu',kernel_regularizer=l2,bias_regularizer=l2)

maxpool = tf.keras.layers.MaxPooling2D

dropout = tf.keras.layers.Dropout
x_input = tf.keras.layers.Input(shape=(28,28,1),name='input_node')

x = conv2d(128,(5,5))(x_input)

x = maxpool((2,2))(x)

x = conv2d(256,(5,5))(x)

x = maxpool((2,2))(x)

x = tf.keras.layers.Flatten()(x)

x = fc(128)(x)

x_output = fc(10,activation=None,name='output_node')(x)

model = tf.keras.models.Model(inputs=x_input,outputs=x_output)                
print("The model architure:\n")

print(model.summary())
tf.keras.utils.plot_model(model, to_file=os.path.join(log_dir,'model.png'), show_shapes=True, show_layer_names=True)
# set the learning decay rate, use the exponential decay

train_steps_per_epoch = int(total_train_samples // batch_size)

initial_learning_rate = 0.01



# optimizer

optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=0.95)



# loss func

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)



# Evaluation factors

metrics = ['acc', 'sparse_categorical_crossentropy']
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
# default save type: saved_model

ckpt = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_acc', verbose=1,

                                          save_best_only=False, save_weight_only=False,

                                          save_frequency=1)
# Stop training if the imporvement of acc is less than 0.01% for 5 epochs

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=True)
class LearningRateExponentialDecay:

    def __init__(self, initial_learning_rate, decay_epochs, decay_rate):

        self.initial_learning_rate = initial_learning_rate

        self.decay_epochs = decay_epochs

        self.decay_rate = decay_rate

    def __call__(self, epoch):

        dtype = type(self.initial_learning_rate)

        decay_epochs = np.array(self.decay_epochs).astype(dtype)

        decay_rate = np.array(self.decay_rate).astype(dtype)

        epoch = np.array(epoch).astype(dtype)

        

        p = epoch / decay_epochs

        lr = self.initial_learning_rate * np.power(decay_rate, p)

        return lr



lr_schedule = LearningRateExponentialDecay(initial_learning_rate, lr_decay_epochs, 0.96)

lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
# Add the use of tensorboard

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
terminate = tf.keras.callbacks.TerminateOnNaN()
# reduce the lr, that needs to get greater change and longer monitoring period than auto lr decay

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_delta=0.0001, min_lr=0)
csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(log_dir, 'logs.log'), separator=',')
""" Please refer to the documents about how to use these params"""

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,

                                             histogram_freq=1, # plot histogram of params and activations, must has test dataset

                                             write_graph=True, # diagram of model architecture

                                             write_images=True, # save model params by images

                                             update_freq='epoch', # epoch/batch/integer higher value will lead to low speed

                                             profile_batch=2, # record the performance of the model

                                             embeddings_freq=1,

                                             embeddings_metadata=None # not very clear about how to use this

                                            )
file_writer_cm = tf.summary.create_file_writer(log_dir, filename_suffix='cm')

def plot_to_image(figure, log_dir, epoch):

    """Converts the matplotlib plot specified by 'figure' to a PNG image and

    returns it. The supplied figure is closed and inaccessible after this call."""

    

    # Save the plot to a PNG in memory

    buf = io.BytesIO()

    plt.savefig(buf, format='png')

    fig = figure

    fig.savefig(os.path.join(log_dir, 'during_train_confusion_epoch_{}.png'.format(epoch)))

    # Closing the figure prevents it from being displayed directly inside the notebook

    plt.close(figure)

    buf.seek(0)

    # Convert PNG buffer to TF image

    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension

    image = tf.expend_dims(image, 0)

    return image



def plot_confusion_matrix(cm, class_names):

    """

    Returns a matplotlib figure containing the plotted confusion matrix.



    Args:

    cm (array, shape = [n, n]): a confusion matrix of integer classes

    class_names (array, shape = [n]): String names of the integer classes

    """

    figure = plt.figure(figsize=(8,8))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title('Confusion matrix')

    plt.colorbar

    tick_marks = np.array(len(class_names))

    plt.xticks(tick_marks, class_names, rotation=45)

    plt.yticks(tick_marks, class_names)

    

    # Normalize the confusion matrix

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    

    # Use white text if squares are dark, otherwise black

    threshold = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        color = 'white' if cm[i, j] > threshold else 'black'

        plt.text(j, i, cm[i, j], horizontalalignment='center', color=color)

    

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return figure



def log_confusion_matrix(epoch, logs):

    # Use the model to predict the values fro the validation dataset.

    test_pred_raw = model.predict(test_x)

    test_pred = mp.argmax(test_pred_raw, axis=1)

    

    # Calculate the confusion matrix.

    cm = confusion_matrix(test_y, test_pred)

    # Log the confusion matrix as an image summary.

    figure = plot_confusion_matrix(cm, class_names=class_names)

    cm_image = plot_to_image(figure, log_dir, epoch)

    

    # Log the confusion matrix as an image summary

    with file_writer_cm.as_default():

        tf.summary.image('Confusion Matrix', cm_image, step=epoch)

        

# Define the per-epoch callback

cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)
callbacks = [ckpt, earlystop, lr, tensorboard, terminate, reduce_lr, csv_logger]
train_steps_per_epoch = np.floor(total_train_sample / batch_size).astype(np.int32)

test_steps_per_epoch = np.ceil(total_test_sample / batch_size).astype(np.int32)

history = model.fit(train_ds, epochs=epochs,

                    steps_per_epoch=train_steps_per_epoch,

                    validation_data=test_ds,

                    validation_steps=test_steps_per_epoch,

                    callbacks=callbacks, verbose=1)
def plot(lrs, title='learing rate schedule'):

    # calculate the lr according to epoch

    epochs = np.arange(len(lrs))

    plt.figure()

    plt.plot(epochs, lrs)

    plt.xticks(epochs)

    plt.scatter(epochs, lrs)

    plt.title(title)

    plt.xlabel('Epoch #')

    plt.ylabel('Learning Rate')



plot(history.history['lr'])

plt.savefig(os.path.join(log_dir, 'learning_rate.png'))
N=np.arange(epochs)

plt.figure()

plt.plot(N, history.history['loss'], label='train_loss')

plt.scatter(N, history.history['loss'])

plt.plot(N, history.history['val_loss'], label='val_loss')

plt.scatter(N, history.history['val_loss'])

plt.plot(N, history.history['acc'], label='train_acc')

plt.scatter(N, history.history['acc'])

plt.plot(N, history.history['val_acc'], label='val_acc')

plt.scatter(N, history.history['val_acc'])

plt.title('Training Loss and Accuracy on Our_dataset')

plt.xlabel('Epoch #')

plt.ylabel('Loss/Accuracy')

plt.legend()

plt.savefig(os.path.join(log_dir,'training.png'))
model_json = model.to_json()

with open(os.path.join(log_dir, 'model_json.json'), 'w') as json_file:

    json_file.write(model_json)
metrics = model.evaluate(test_ds, verbose=1)

print('val_loss:', metrics[0], 'val_acc:', metrics[1])
predictions = model.predict(test_ds,verbose=1)
def print_metrics(labels, predictions, target_names, save=False, save_path=None):

    # Calculate confusion result

    preds = np.argmax(predictions, axis=1)

    confusion_result = confusion_matrix(labels, preds)

    pd.set_option('display.max_rows', 500)

    pd.set_option('display.max_columns', 500)

    pd.set_option('display.width', 1500)

    confusion_result = pd.DataFrame(confusion_result, index=target_names, columns=target_names)

    # classification results

    report = classification_report(labels, preds, target_names=target_names, digits=4)

    result_report = 'Confise_matrix:\n{}\n\nClassification_report:\n{}\n'.format(confusion_result, report)

    print(result_report)

    if save:

        savepath = os.path.join(save_path, 'predited_result.txt')

        print('The result saved in %s' % savepath)

        

        with open(savepath, 'w') as f:

            f.write(result_report)
print_metrics(test_y, predictions, class_names, True, log_dir)
for dirname, _, filenames in os.walk('/kaggle/working/model_output/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# %%bash

# tree model_output