import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.applications.resnet50 import ResNet50
from keras import layers as KL
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
TRAIN_PATH = '/kaggle/input/histopathologic-cancer-detection/train/'
TRAIN_LABELS = '/kaggle/input/histopathologic-cancer-detection/train_labels.csv'
SIZE_IMG = 96
EPOCHS = 10

model_path = '../input/resnet-cancer-detection/cancer_detection_resnet.h5'
saved_model = os.path.isfile(model_path)
df = pd.read_csv(TRAIN_LABELS)

#remove unwanted data detected by other kaggle users
df = df[df['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']
df = df[df['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']

print(df['label'].value_counts(), 
      '\n\n', df.describe(), 
      '\n\n', df.head())
def display_random_data(dataframe, path, rows):

    imgs = dataframe.sample(rows *2)
    fig, axarr = plt.subplots(2, rows, figsize=(rows*10, rows*4))

    for i in range(1,rows*2+1):
        img_path = path + imgs.iloc[i-1]['id'] + '.tif'
        img = image.load_img(img_path, target_size=(96, 96))
        img = image.img_to_array(img)/255
        axarr[i//(rows+1),i%rows].imshow(img)
        axarr[i//(rows+1),i%rows].set_title(imgs.iloc[i-1]['label'], fontsize=35)
        axarr[i//(rows+1),i%rows].axis('off')
        
display_random_data(df,TRAIN_PATH, 5)
#add .tif to ids in the dataframe to use flow_from_dataframe
df["id"]=df["id"].apply(lambda x : x +".tif")
df.head()
if saved_model:
    val = 0
else:
    val = 0.15
    
datagen= ImageDataGenerator(
            rescale=1./255,
            samplewise_std_normalization= True,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=90,
            zoom_range=0.2, 
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.05,
            channel_shift_range=0.1,
            validation_split=val)

train_generator=datagen.flow_from_dataframe(
    dataframe=df,
    directory=TRAIN_PATH,
    x_col="id",
    y_col="label",
    subset="training",
    batch_size=64,
    shuffle=True,
    class_mode="binary",
    target_size=(96,96))

valid_generator=datagen.flow_from_dataframe(
    dataframe=df,
    directory=TRAIN_PATH,
    x_col="id",
    y_col="label",
    subset="validation",
    batch_size=64,
    shuffle=True,
    class_mode="binary",
    target_size=(96,96))
def build_model():
    input_shape = (SIZE_IMG, SIZE_IMG, 3)
    inputs = KL.Input(input_shape)
    resnet = ResNet50(include_top=False, input_shape=input_shape) 
    x  = KL.GlobalAveragePooling2D()(resnet(inputs))
    x = KL.Dropout(0.5)(x)
    outputs = KL.Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs)
def first_training():
    '''
    train the model and save it if the val_acc test is better than the precedent epoch
    '''
    model = build_model()
    
    model.compile(optimizer=Adam(lr=0.0001, decay=0.00001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, 
                                       verbose=1, mode='max', min_lr=0.000001)
    
    checkpoint = ModelCheckpoint("resnet_cancer_detection.h5", monitor='val_acc', verbose=1, 
                              save_best_only=True, mode='max')

    history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.n//train_generator.batch_size, 
                              validation_data=valid_generator,
                              validation_steps=valid_generator.n//valid_generator.batch_size,
                              epochs=EPOCHS,
                              callbacks=[checkpoint,reduce_lr])
    
    return history, model
def second_training():
    '''
    Tune the model using all available data and a small learning rate
    '''
    model = load_model(model_path)
    
    model.compile(optimizer=Adam(lr=0.000001, decay=0.00001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.n//train_generator.batch_size, 
                              epochs=10)
    
    return history, model
if saved_model:
    history, model = second_training()
else:
    history, model = first_training()
def analyse_results(epochs):
    metrics = ['loss', "acc", 'val_loss','val_acc']
        
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(1, 4, figsize=(30, 5))
    fig.subplots_adjust(hspace=0.1, wspace=0.3)

    for (i, l) in enumerate(metrics):
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel(l.split('_')[-1])
        ax[i].plot(np.arange(0, epochs), history.history[l], label=l)
        ax[i].legend() 

if EPOCHS > 1 and saved_model == False:        
    analyse_results(EPOCHS)
test_path = '/kaggle/input/histopathologic-cancer-detection/test/'
df_test = pd.read_csv('../input/histopathologic-cancer-detection/sample_submission.csv')
df_test["id"]=df_test["id"].apply(lambda x : x +".tif")
test_datagen = ImageDataGenerator(rescale=1./255,
                                 samplewise_std_normalization= True)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=test_path,
    x_col="id",
    y_col=None,
    target_size=(96, 96),
    color_mode="rgb",
    batch_size=64,
    class_mode=None,
    shuffle=False,
)  
test_generator.reset()
pred=model.predict_generator(test_generator,verbose=1).ravel()
results = dict(zip(test_generator.filenames, pred))

label = []
for i in range(len(df_test["id"])):
    label.append(results[df_test["id"][i]])
    
df_test["id"]=df_test["id"].apply(lambda x : x[:-4])
submission=pd.DataFrame({"id":df_test["id"],
                      "label":label})
submission.to_csv("submission.csv",index=False)
submission.head()