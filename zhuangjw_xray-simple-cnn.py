import numpy as np

import pandas as pd

import xarray as xr

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve, auc



import tensorflow as tf

from tensorflow import keras

tf.__version__, keras.__version__
tf.config.experimental.list_physical_devices('GPU')
ds = xr.open_dataset('../input/chest-xray-cleaned/chest_xray.nc')

ds
ds['image'].isel(sample=slice(0, 12)).plot(col='sample', col_wrap=4, cmap='gray')
ds['label'].mean(dim='sample').to_pandas().plot.barh()  # proportion
all_labels = ds['feature'].values.astype(str)

all_labels
X_all = ds['image'].values[..., np.newaxis]

y_all = ds['label'].values



X_train, X_test, y_train, y_test = train_test_split(

    X_all, y_all, test_size = 0.2, random_state = 42

)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
def make_model(filters=16, input_shape=(128, 128, 1), num_output=14):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(filters, (3, 3), input_shape=input_shape, activation='relu'),

        tf.keras.layers.Conv2D(filters, (3, 3), activation='relu'),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(filters * 2, (3, 3), activation='relu'),

        tf.keras.layers.Conv2D(filters * 2, (3, 3), activation='relu'),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(num_output, activation='sigmoid')  

        # not softmax, as here is independent binary classification, not multi-label 

        ])

    return model



model = make_model()
%%time

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC', 'binary_accuracy'])



history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
pd.DataFrame(history.history).plot(marker='o')
%time y_train_pred = model.predict(X_train)

%time y_test_pred = model.predict(X_test)
def plot_ruc(y_true, y_pred):

    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    

    for (idx, c_label) in enumerate(all_labels):

        fpr, tpr, thresholds = roc_curve(y_true[:,idx].astype(int), y_pred[:,idx])

        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))



    c_ax.legend()

    c_ax.set_xlabel('False Positive Rate')

    c_ax.set_ylabel('True Positive Rate')
plot_ruc(y_train, y_train_pred)

plt.title('ROC training set')
plot_ruc(y_test, y_test_pred)

plt.title('ROC test set')