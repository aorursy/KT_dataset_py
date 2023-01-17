import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
import matplotlib.animation as animation
import re
import matplotlib.animation as animation
from IPython.display import HTML
import os
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tqdm import tqdm
from PIL import Image
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, add, concatenate, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model 
import tensorflow as tf
import random
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import  ReduceLROnPlateau
import warnings
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
IMG_DIR_TRAIN = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
IMG_DIR_TEST = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/"
FILE_DIR = "/kaggle/input/osic-pulmonary-fibrosis-progression/"
IMAGE_SIZE = 224
train_data = pd.read_csv(FILE_DIR + "train.csv")
test_data = pd.read_csv(FILE_DIR + "test.csv")
train_data.head()
PATIENT_ID = train_data["Patient"][0]    
print("Patient : ", PATIENT_ID)
print("Number of FVC observations : ", len(train_data[train_data["Patient"] == PATIENT_ID]))
print("Age : ", (train_data[train_data["Patient"] == PATIENT_ID]["Age"].values[0]))
print("Sex : ", (train_data[train_data["Patient"] == PATIENT_ID]["Sex"].values[0]))
print("SmokingStatus : ", (train_data[train_data["Patient"] == PATIENT_ID]["SmokingStatus"].values[0]))
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

fig = plt.figure()

img_names = []
for dirictory,_,img in os.walk(IMG_DIR_TRAIN + train_data["Patient"][0]):
    img_names.append(img)

img_names = natural_sort(img_names[0])

images = []
k = 0
for i in img_names:
    images.append([plt.imshow(pydicom.dcmread(IMG_DIR_TRAIN + PATIENT_ID + "/" + i).pixel_array, cmap=plt.cm.bone)])
    k += 1
    
ani = animation.ArtistAnimation(fig, images)
plt.close()

HTML('<center>' + ani.to_html5_video() + '</center>')
FVC = train_data[train_data["Patient"] == PATIENT_ID]["FVC"]
Week = train_data[train_data["Patient"] == PATIENT_ID]["Weeks"]

fig = px.line(x=Week, y=FVC, title='FVC over time of patient with id ' + PATIENT_ID)

fig.update_layout(
    xaxis=dict(title = "Week"),
    yaxis=dict(title = "FVC"),
    plot_bgcolor='white'
)

fig.add_shape(
            type="line",
            x0=0,
            y0=min(FVC),
            x1=0,
            y1=max(FVC),
            line=dict(
                color="Red",
                width=2,
                dash="dashdot",
            ),
    )
py.offline.iplot(fig)
print("Number of patients: ", len(train_data["Patient"].unique()))
l1 = list(train_data["SmokingStatus"].unique())
smokers = ""
for i in l1:
    smokers = smokers + i + ", "
print("Among them: ", smokers[:-2])
min_Age = min(train_data["Age"].unique())
max_Age = max(train_data["Age"].unique())
print("Ages vary from ", min_Age," to ",max_Age)
train_data["dummy"] = 1
fig = px.bar(train_data.drop_duplicates(subset=["Patient"])[["Sex","Age","dummy"]].groupby(["Sex","Age"]).sum().reset_index().rename(columns={"dummy":"Count"}), x="Age", y="Count",color = "Sex")
fig.update_layout(
    plot_bgcolor='white'
)
py.offline.iplot(fig)
fig = px.bar(train_data.drop_duplicates(subset=["Patient"])[["Sex","SmokingStatus","dummy"]].groupby(["Sex","SmokingStatus"]).sum().reset_index().rename(columns={"dummy":"Count"}), x="SmokingStatus", y="Count",color = "Sex")
fig.update_layout(
    plot_bgcolor='white'
)
py.offline.iplot(fig)
fig = go.Figure()
Pat_Ids = train_data["Patient"].unique()
for i in Pat_Ids[:15]:
    fig.add_trace(go.Scatter(
            x=train_data[train_data["Patient"]==i]["Weeks"],
            y=train_data[train_data["Patient"]==i]["FVC"],
            name = i
        ))
fig.update_layout(
    plot_bgcolor='white'
)
fig.add_shape(
            type="line",
            x0=0,
            y0=1000,
            x1=0,
            y1=5000,
            line=dict(
                color="Red",
                width=2,
                dash="dashdot",
            ),
    )
py.offline.iplot(fig)
fig = px.scatter_matrix(train_data, dimensions=["Weeks", "FVC", "Percent", "Age"], color="Sex")
fig.update_layout(
    plot_bgcolor='white'
)
py.offline.iplot(fig)
num_files = []
for i in Pat_Ids:
    num_files.append(len([name for name in os.listdir(IMG_DIR_TRAIN + i + '/') if os.path.isfile(os.path.join(IMG_DIR_TRAIN + i + '/', name))]))
fig = go.Figure(go.Bar(name='SF Zoo',x=Pat_Ids,y=num_files))
fig.update_layout(
    plot_bgcolor='white'
)
py.offline.iplot(fig)
train_data = train_data.drop(columns =["dummy"])
class DataGenCT(Sequence):
    
    def __init__(self, patients, dataset, cols, batch_size=32, train = 1):
        
        self.patients = [i for i in patients if i not in ['ID00011637202177653955184', 'ID00052637202186188008618']]
        self.dataset = dataset
        self.batch_size = batch_size
        self.cols = cols
        self.patient_scans = {}
        self.train = train
        IMG_DIR_TRAIN = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/"
        IMG_DIR_TEST = "/kaggle/input/osic-pulmonary-fibrosis-progression/test/"
        if train:
            self.IMG_DIR = IMG_DIR_TRAIN
        else:
            self.IMG_DIR = IMG_DIR_TEST
        
        for patient in patients:
            self.patient_scans[patient] = natural_sort([i for i in os.listdir(self.IMG_DIR + patient + "/")])
    
    def __len__(self):
        return 1100

    def __getitem__(self,idx):
        CT_Scan = []
        Answer, Table = [], [] 
        
        keys = np.random.choice(self.patients, size = self.batch_size)
        for key in keys:
            try:
                idx = np.random.choice(self.patient_scans[key], size=1)[0]
                dataset_copy = self.dataset[self.dataset["Patient"] == key]
                rand_week = random.choice(list(dataset_copy["Weeks"]))

                img = pydicom.dcmread(self.IMG_DIR + key + "/" + idx).pixel_array
                img_min = img.min()
                img_max = img.max()
                img = cv2.resize((img - img_min) / (img_max - img_min), (IMAGE_SIZE, IMAGE_SIZE))
                CT_Scan.append(img)
                Answer.append(dataset_copy[dataset_copy["Weeks"] == rand_week]["FVC"].values[0])
                Table.append(dataset_copy[dataset_copy["Weeks"] == rand_week][self.cols].values[0])
            except Exception as e:
                continue

        CT_Scan = np.expand_dims(np.array(CT_Scan), axis=-1)
        return [CT_Scan, np.array(Table)] , np.array(Answer)
le_sex = LabelEncoder()
le_smoke = LabelEncoder()

le_sex = le_sex.fit(train_data["Sex"])
train_data["Sex"] = le_sex.transform(train_data["Sex"])
le_smoke = le_smoke.fit(train_data["SmokingStatus"])
train_data["SmokingStatus"] = le_smoke.transform(train_data["SmokingStatus"])
test_data["Sex"] = le_sex.transform(test_data["Sex"])
test_data["SmokingStatus"] = le_smoke.transform(test_data["SmokingStatus"])

fig = ff.create_distplot([train_data["Weeks"].values], ['Weeks distribution'], show_rug=False)
fig.update_layout(
    plot_bgcolor='white'
)
py.offline.iplot(fig)
fig = ff.create_distplot([train_data["Percent"].values], ['Percent distribution'], show_rug=False)
fig.update_layout(
    plot_bgcolor='white'
)
py.offline.iplot(fig)
fig = ff.create_distplot([train_data["Age"].values], ['Age distribution'], show_rug=False)
fig.update_layout(
    plot_bgcolor='white'
)
py.offline.iplot(fig)
transformer_weeks = RobustScaler().fit(np.array(train_data["Weeks"]).reshape(-1, 1))
train_data["Weeks"] = transformer_weeks.transform(np.array(train_data["Weeks"]).reshape(-1, 1)).reshape(1,-1)[0]
transformer_perc = RobustScaler().fit(np.array(train_data["Percent"]).reshape(-1, 1))
train_data["Percent"] = transformer_perc.transform(np.array(train_data["Percent"]).reshape(-1, 1)).reshape(1,-1)[0]
transformer_age = RobustScaler().fit(np.array(train_data["Age"]).reshape(-1, 1))
train_data["Age"] = transformer_age.transform(np.array(train_data["Age"]).reshape(-1, 1)).reshape(1,-1)[0]
test_data["Weeks"] = transformer_weeks.transform(np.array(test_data["Weeks"]).reshape(-1, 1)).reshape(1,-1)[0]
test_data["Percent"] = transformer_perc.transform(np.array(test_data["Percent"]).reshape(-1, 1)).reshape(1,-1)[0]
test_data["Age"] = transformer_age.transform(np.array(test_data["Age"]).reshape(-1, 1)).reshape(1,-1)[0]
train_data.head()
test_data.head()
C1 = tf.constant(70, dtype='float32')
C2 = tf.constant(1000, dtype='float32')
quantiles = [.15, .50, .85]

def metric(y_true, y_pred, Sigma):
    Sigma_clipped = np.clip(Sigma, 70, 9e9)  
    Delta = np.clip(np.abs(y_true - y_pred), 0 , 1000)  
    return np.mean(-1 * (np.sqrt(2) * Delta / Sigma_clipped) - np.log(np.sqrt(2) * Sigma_clipped))

def FVC_score(y_true, y_pred):
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)
    sigma = y_pred[:, 2] - y_pred[:, 1]
    fvc_pred = y_pred[:, 1]
    Sigma_clipped = tf.maximum(sigma, C1)
    Delta = tf.abs(y_true[:, 0] - fvc_pred)
    Delta = tf.minimum(Delta, C2)
    sq2 = tf.sqrt(tf.dtypes.cast(2, dtype=tf.float32))
    metric = sq2 * (Delta / Sigma_clipped) * sq2 + tf.math.log(Sigma_clipped * sq2)
    return K.mean(metric)

def Quantile_loss(y_true, y_pred):
    q = tf.constant(np.array([quantiles]), dtype=tf.float32)
    y_true = tf.dtypes.cast(y_true, tf.float32)
    y_pred = tf.dtypes.cast(y_pred, tf.float32)
    e = y_true - y_pred
    v = tf.maximum(q * e, (q - 1) * e)
    return K.mean(v)

def model_loss():
    def loss(y_true, y_pred):
        lambd = 0.8
        return lambd * Quantile_loss(y_true, y_pred) + (1 - lambd) * FVC_score(y_true, y_pred)
    return loss
def identity_block(input_tensor, filters):
  
    filters1, filters2, filters3 = filters


    x = Conv2D(filters1, (1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, filters):
   
    filters1, filters2, filters3 = filters


    x = Conv2D(filters1, (1, 1), strides=(2, 2))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters3, (1, 1), strides=(2, 2))(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x
start1 = Input(shape=(5,),name = "Tab_input")
start2 = Input(shape=(IMAGE_SIZE, IMAGE_SIZE,1), name = "Image_input")

x = ZeroPadding2D((3, 3))(start2)
x = Conv2D(64, (7, 7), strides=(2, 2))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv_block(x, [64, 64, 256])
x = identity_block(x, [64, 64, 256])
x = identity_block(x, [64, 64, 256])

x = conv_block(x, [128, 128, 512])
x = identity_block(x, [128, 128, 512])
x = identity_block(x, [128, 128, 512])
x = identity_block(x, [128, 128, 512])

x = AveragePooling2D((7, 7))(x)
x = Flatten()(x)
x1 = Dense(100, activation="relu")(start1)
x1 = Dense(100, activation="relu")(x1)
x = concatenate([x, x1])
x = Dense(50, activation="relu")(x)

out = Dense(3, activation='relu',)(x)
model = Model([start2, start1], out)
model.compile(loss=model_loss(), 
              optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, 
                                                 epsilon=None, decay=0.01, amsgrad=False), 
              metrics=[FVC_score])
Lr_decr = ReduceLROnPlateau(
    monitor='val_loss',
    factor= 0.9,
    patience=3,
    min_lr=1e-5,
    mode='min',
    verbose = 1
)
Test_generator = DataGenCT(patients=Pat_Ids[150:len(Pat_Ids)] ,
                            dataset = train_data,
                            cols= ["Weeks","Percent","Age","Sex","SmokingStatus"],
                          )


Train_generator = DataGenCT(patients=Pat_Ids[0:150],
                            dataset = train_data,
                            cols= ["Weeks","Percent","Age","Sex","SmokingStatus"])



history = model.fit_generator(Train_generator , 
                    steps_per_epoch = 100,
                    epochs = 10,
                    validation_data = Test_generator,
                    use_multiprocessing = False,
                    workers = 1,
                    callbacks = [Lr_decr],
                    validation_steps = 20,
                    verbose=1
                             )
fig = go.Figure()

fig.add_trace(go.Scatter(
        x=np.r_[1:11],
        y=history.history["loss"],
        name = "training loss"
    ))
fig.add_trace(go.Scatter(
        x=np.r_[1:11],
        y=history.history["val_loss"],
        name = "validation loss"
    ))
fig.update_layout(
    xaxis=dict(title = "Epoch"),
    yaxis=dict(title = "Loss"),
    plot_bgcolor='white',
    title = "Loss over epoch"
)

py.offline.iplot(fig)
