#!usr/bin/env/python3
# util.py

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, BatchNormalization, Flatten, Conv2D, Reshape
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image

def load_proc_img(path_list, size=224, ch=1):
    """
    Load & Preprocess image
    :reference: https://www.kaggle.com/anmour/convolutional-autoencoder-with-keras
    :reference: https://www.kaggle.com/nxrprime/siim-d3-eda-augmentations-resnext-and-grad-cam#ca
    """
    image_list = np.zeros((len(path_list), size, size, ch));
    for key, val in enumerate(path_list):
        img = image.load_img(val); #target size can be set
        img = image.img_to_array(img).astype("float32");
        if ch == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
        img /= 255.0;
        # img = cv2.addWeighted(img,4,cv2.GaussianBlur(img,(0,0),10),-4,128);
        image_list[key] = np.reshape(img, [size,size,ch]);
    return image_list;

def divide_in_batches(seq, num):
    """
    Split an array into multiple chunks
    :author M Shawabkheh at StackOverflow
    """
    avg = len(seq) / float(num);
    out = [];
    last = 0.0;
    while last < len(seq):
        out.append(seq[int(last): int(last+avg)]);
        last += avg;
    return out;


class ConvolutionalAutoencoder:
    def __init__(self, size=224, channel=1, batch_size=8, epochs=3, patience=3,
                 encoder_dim=2, loss="binary_crossentropy", metrics=["mse"], optimizer=Adam(lr=0.0001)):
        """ Constructor """
        self.size = size;
        self.channels = channel;
        self.batch_size = batch_size;
        self.epochs = epochs;
        self.patience = patience;
        self.size_lower = encoder_dim;
        self.loss = loss;
        self.metrics = metrics;
        self.optimzer = optimizer;
        self.history = None;

        self.img_shape = (self.size, self.size, self.channels);
        self.model, self.encoder = self.setup_arch();
        print(self.model.summary())

    def setup_arch(self):
        """ Sets up architecture """
        ## Encoder network:
        input_layer = Input(shape=self.img_shape);

        h = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer);
        h = MaxPooling2D((2, 2), padding='same')(h);
        h = BatchNormalization()(h);

        encoded = Flatten()(h);
        encoded = Dense(self.size_lower, activation="linear")(encoded);
        encoder = Model(inputs=input_layer, outputs=encoded);

        ## Decoder network:
        h = Dense(256, activation="relu")(encoded);
        h = Reshape((16, 16, 1))(h);
        h = BatchNormalization()(h);
        h = UpSampling2D((7, 7))(h);

        h = Conv2D(32, (3, 3), activation='relu', padding='same')(h);
        h = BatchNormalization()(h);
        h = UpSampling2D((2, 2))(h);
        
        decoded = Conv2D(1, (3, 3), activation="sigmoid", padding='same')(h);

        model = Model(input_layer, decoded);
        model.compile(optimizer=self.optimzer, loss=self.loss, metrics=self.metrics);
        return model, encoder;

    def train(self, x_train, x_val=None):
        es = EarlyStopping(monitor="val_loss", patience=self.patience, verbose=1);
        validation_data = None if x_val is None else (x_val, x_val);
        history = self.model.fit(
            x_train, x_train,
            batch_size = self.batch_size,
            epochs = self.epochs,
            validation_data = validation_data,
            callbacks = [es]
        );
        self.history = history.history;

    def sketch_loss(self):
        plt.figure(figsize=(6, 6), dpi=100);
        plt.plot(self.history["loss"]);
        plt.plot(self.history["val_loss"]);
        plt.xlabel("Epoch");
        plt.ylabel("Loss");
        plt.title("Loss over Epoch");
        plt.legend(["Training","Validation"], loc="upper left");
        plt.show();

    def predict(self, newX, newIdx=None, batch_size=1):
        """ Helper to make prediction on new data """
        predictions = pd.DataFrame(self.encoder.predict(newX, batch_size=batch_size));
        if newIdx is not None:
            predictions.set_index(newIdx, inplace=True);
        return predictions;
import glob
import socket
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

PRODUCTION = True;

PREFIX = "../input/siic-isic-224x224-images/";
OUT_PATH = "/kaggle/working/learned_output(2).csv";
NUM_SPLITS = 5;
TARGET_DIM = 2;
EPOCHS = 20 if PRODUCTION else 2;
BATCH_SIZE = 1 if PRODUCTION else 32;
PROP_VAL = 0.10;

print("Host Name: %s" % socket.gethostname())
print(device_lib.list_local_devices())
IMAGE_LIST = glob.glob(PREFIX + "train/*.png") + glob.glob(PREFIX + "test/*.png");
IMAGE_LIST = divide_in_batches(IMAGE_LIST, NUM_SPLITS);
all_predictions = pd.DataFrame();

for fold_num in range(NUM_SPLITS):
    image_batch, val_images = train_test_split(IMAGE_LIST[fold_num], test_size=PROP_VAL);
    
    if PRODUCTION:
        print("In production mode! All images used in feature extraction...")
    else:
        print("Subsetting data for development purposes...")
        image_batch, val_images = np.random.choice(image_batch, 80), np.random.choice(val_images, 20);

    Xtrain = load_proc_img(image_batch);
    Xvalid = load_proc_img(val_images);
    
    cae = ConvolutionalAutoencoder(epochs=EPOCHS, encoder_dim=TARGET_DIM, batch_size=BATCH_SIZE);
    cae.train(Xtrain, Xvalid);
    cae.sketch_loss();
    predictions = cae.predict(
        np.concatenate((Xtrain, Xvalid)),
        np.concatenate((image_batch, val_images))
    );
    all_predictions = pd.concat([all_predictions, predictions], axis=0);
    del cae;
all_predictions.to_csv(OUT_PATH);
all_predictions.head()
import pandas as pd
import numpy as np
import copy as cp
import re
import matplotlib.pyplot as plt
import seaborn as sns

def zscore_norm(x):
    """ Helper for Z score standardization of structured dataframe """
    return (x - x.mean()) / x.std();

def load_cae(path, normalize=True, aggregate=False):
    """ Wrapper to load & optionally preprocess the data"""
    all_predictions = pd.read_csv(path, index_col=0);
    if normalize:
        all_predictions = zscore_norm(all_predictions);
    if aggregate:
        all_predictions = pd.DataFrame(all_predictions.mean(axis=1), columns=["meanLower"]);
    return all_predictions;

def separate_cae(cae_data, pattern="train/"):
    """ Separates preprocessed training & test CAE data """
    dat = cp.deepcopy(cae_data);
    dat = dat.loc[[pattern in x for x in dat.index], :];
    pattern = "../input/siic-isic-224x224-images/" + pattern; 
    pattern += "|.png";
    dat[PRIMARY_KEY] = [re.sub(pattern, "", x) for x in dat.index];
    return dat;

PRIMARY_KEY = "image_name";
train_data = load_cae(LEARNED_DATA_PATH);
train_data.head()
## Load annotations
train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv"); 
train_df.head()
## Join training data
train_data = separate_cae(train_data);
train_data = pd.merge(train_df, train_data, on=PRIMARY_KEY); 
train_data.set_index(PRIMARY_KEY, inplace=True, drop=True);
train_data.head()
x, y = train_data["0"], train_data["1"]; 
train_data["ageOver50"] = 1 * (train_data["age_approx"] > 50);

plt.figure(dpi=100, figsize=(10, 10));

plt.subplot(221);
sns.scatterplot(x, y, hue=train_data.benign_malignant, legend="full");

plt.subplot(222);
sns.scatterplot(x, y, hue=train_data.sex, legend="full");

plt.subplot(223);
sns.scatterplot(x, y, hue=train_data.anatom_site_general_challenge, legend="full");

plt.subplot(224);
sns.scatterplot(x, y, hue=train_data.ageOver50, legend="full");

plt.show();
