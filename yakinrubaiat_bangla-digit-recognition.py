import tensorflow as tf
import numpy as np
import pandas as pd 
import h5py
import os
import glob
import cv2 # for porcess the image
from keras.utils import to_categorical  # to transfrom integer value to catagorial value
import matplotlib.pyplot as plt # For creating plot
from scipy.misc.pilutil import Image #Convert Image to grayscale
from skimage.feature import canny #Transfrom The image in black and white
import scipy.misc   # Convert it to the numpy array
import keras 
from keras.models import Model, load_model
from keras.layers import Dense, Input, Conv2D, Flatten,Activation,ZeroPadding2D,MaxPool2D,Dropout
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import optimizers as opt #ReSet the adam optimizer
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras import applications
from IPython.display import HTML #For Download CSV File 
import base64 # For Download CSV File
data_dir = os.path.join('..','input')
paths_train_a=glob.glob(os.path.join(data_dir,'training-a','*.png'))
paths_train_b=glob.glob(os.path.join(data_dir,'training-b','*.png'))
paths_train_e=glob.glob(os.path.join(data_dir,'training-e','*.png'))
paths_train_c=glob.glob(os.path.join(data_dir,'training-c','*.png'))
paths_train_d=glob.glob(os.path.join(data_dir,'training-d','*.png'))
print(os.listdir("../input"))
path_label_train_a=os.path.join(data_dir,'training-a.csv')
path_label_train_b=os.path.join(data_dir,'training-b.csv')
path_label_train_e=os.path.join(data_dir,'training-e.csv')
path_label_train_c=os.path.join(data_dir,'training-c.csv')
path_label_train_d=os.path.join(data_dir,'training-d.csv')
paths_test_a=glob.glob(os.path.join(data_dir,'testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(data_dir,'testing-b','*.png'))
paths_test_e=glob.glob(os.path.join(data_dir,'testing-e','*.png'))
paths_test_c=glob.glob(os.path.join(data_dir,'testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(data_dir,'testing-d','*.png'))
paths_test_f=glob.glob(os.path.join(data_dir,'testing-f','*.png'))+glob.glob(os.path.join(data_dir,'testing-f','*.JPG'))
paths_test_auga=glob.glob(os.path.join(data_dir,'testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join(data_dir,'testing-augc','*.png'))

def get_key(path):
    # seperates the key of an image from the filepath
    key=path.split(sep=os.sep)[-1]
    return key
def get_data(paths_img,path_label=None,resize_dim=None):
    '''reads images from the filepaths, resizes them (if given), and returns them in a numpy array
    Args:
        paths_img: image filepaths
        path_label: pass image label filepaths while processing training data, defaults to None while processing testing data
        resize_dim: if given, the image is resized to resize_dim x resize_dim (optional)
    Returns:
        X: group of images
        y: categorical true labels
    '''
    X=[]
    for i,path in enumerate(paths_img):
        img = Image.open(path).convert('L') # Convert to the grayscale 
        img = np.asarray(img) # Convert to the numpy
        if resize_dim is not None:
            img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) #Create 32*32 image
        X.append(img) # expand image to 32x32 and append to the list
        if i==len(paths_img)-1:
            end='\n'
        else: end='\r'
        print('processed {}/{}'.format(i+1,len(paths_img)),end=end)
        
    #X=np.array(X)
    X = np.array(X).astype('float32') # tranform list to numpy array
    if  path_label is None:
        return X
    else:
        df = pd.read_csv(path_label) # read labels
        df=df.set_index('filename') 
        y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img] # get the labels corresponding to the images
        y=to_categorical(y_label,10) # transfrom integer value to categorical variable
        return X, y

PIC_SIZE = 28
X_train_a,y_train_a=get_data(paths_train_a,path_label_train_a,resize_dim=PIC_SIZE)
X_train_b,y_train_b=get_data(paths_train_b,path_label_train_b,resize_dim=PIC_SIZE)
X_train_c,y_train_c=get_data(paths_train_c,path_label_train_c,resize_dim=PIC_SIZE)
X_train_d,y_train_d=get_data(paths_train_d,path_label_train_d,resize_dim=PIC_SIZE)
X_train_e,y_train_e=get_data(paths_train_e,path_label_train_e,resize_dim=PIC_SIZE)
X_train_all=np.concatenate((X_train_a,X_train_b,X_train_c,X_train_d,X_train_e),axis=0)
y_train_all=np.concatenate((y_train_a,y_train_b,y_train_c,y_train_d,y_train_e),axis=0)
X_train = X_train_all[:,:,:,np.newaxis]
y_train= y_train_all
print(X_train.shape)
print(y_train.shape)
Image_Size = 28
X_test_a=get_data(paths_test_a,resize_dim=Image_Size)
X_test_b=get_data(paths_test_b,resize_dim=Image_Size)
X_test_c=get_data(paths_test_c,resize_dim=Image_Size)
X_test_d=get_data(paths_test_d,resize_dim=Image_Size)
X_test_e=get_data(paths_test_e,resize_dim=Image_Size)
X_test_f=get_data(paths_test_f,resize_dim=Image_Size)
X_test_auga=get_data(paths_test_auga,resize_dim=Image_Size)
X_test_augc=get_data(paths_test_augc,resize_dim=Image_Size)
X_test_all=np.concatenate((X_test_a,X_test_b,X_test_c,X_test_d,X_test_e,X_test_f,X_test_auga,X_test_augc))
X_test = X_test_all[:,:,:,np.newaxis]
plt.figure(figsize = (10, 8))
a, b = 9, 3
for i in range(27):
    plt.subplot(b, a, i+1)
    plt.imshow(X_train_all[i])
plt.show()
X_train = X_train/255.0
X_Test = X_test/255.0
#model.summary()
nets = 1
model = [0] *nets
for j in range(nets):
    model[j] = Sequential()

    model[j].add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',kernel_initializer='he_normal', 
                 activation ='relu', input_shape = (28,28,1)))
    model[j].add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', kernel_initializer='he_normal', 
                 activation ='relu'))
    model[j].add(MaxPool2D(pool_size=(2,2)))
    model[j].add(Dropout(0.20))


    model[j].add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', kernel_initializer='he_normal', 
                 activation ='relu'))
    model[j].add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', kernel_initializer='he_normal',
                 activation ='relu'))
    model[j].add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model[j].add(Dropout(0.25))
    
    
    model[j].add(Conv2D(128, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
    model[j].add(Dropout(0.25))


    model[j].add(Flatten())
    model[j].add(Dense(128, activation = "relu"))
    model[j].add(BatchNormalization())
    model[j].add(Dense(10, activation = "softmax"))
    
    # COMPILE WITH RMSprop OPTIMIZER AND CROSS ENTROPY COST
    opt = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model[j].compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    
    
datagen = keras.preprocessing.image.ImageDataGenerator(
                              featurewise_center=False,  # set input mean to 0 over the dataset
                              samplewise_center=False,  # set each sample mean to 0
                              featurewise_std_normalization=False,  # divide inputs by std of the dataset
                              samplewise_std_normalization=False,  # divide each input by its std
                              zca_whitening=False,  # apply ZCA whitening
                              rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                              zoom_range = 0.1, # Randomly zoom image 
                              width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                              height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                              horizontal_flip=False,  # randomly flip images
                              vertical_flip=False)  # randomly flip images
                              
annealer = keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)
# TRAIN NETWORKS
history = [0] * nets
epochs = 45
for j in range(nets):
    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, y_train, test_size = 0.1)
    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),
        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  
        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)
    print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        j+1,epochs,max(history[j].history['acc']),max(history[j].history['val_acc']) ))
predictions_prob = np.zeros( (X_test.shape[0],10) ) 
for j in range(nets):
    predictions_prob = predictions_prob + model[j].predict(X_test)
#results = np.argmax(results,axis = 1)
predictions_prob=model.predict(X_test)
type(predictions_prob)
n_sample=200
np.random.seed(42)
ind=np.random.randint(0,len(X_test_all), size=n_sample)
FIG_WIDTH=20 # Width of figure
ROW_HEIGHT=3 
def imshow_group(X,y=None,y_pred=None,n_per_row=10):
    '''helper function to visualize a group of images along with their categorical true labels (y) and prediction probabilities.
    Args:
        X: images
        y: categorical true labels
        y_pred: predicted class probabilities
        n_per_row: number of images per row to be plotted
    '''
    n_sample=len(X)
    img_dim=X.shape[1]
    j=np.ceil(n_sample/n_per_row)
    fig=plt.figure(figsize=(FIG_WIDTH,ROW_HEIGHT*j))
    for i,img in enumerate(X):
        plt.subplot(j,n_per_row,i+1)
        plt.imshow(img)
        if y is not None:
                plt.title('true label: {}'.format(np.argmax(y[i])))
        if y_pred is not None:
            top_n=3 # top 3 predictions with highest probabilities
            ind_sorted=np.argsort(y_pred[i])[::-1]
            h=img_dim+4
            for k in range(top_n):
                string='pred: {} ({:.0f}%)\n'.format(ind_sorted[k],y_pred[i,ind_sorted[k]]*100)
                plt.text(img_dim/2, h, string, horizontalalignment='center',verticalalignment='center')
                h+=4
        plt.axis('off')
    plt.show()
def create_submission(predictions,keys):
    result = pd.DataFrame(
        predictions,
        columns=['label'],
        index=keys
        )
    result.index.name='key'
    return result
    #result.to_csv(path, index=True)
imshow_group(X=X_test_all[ind],y=None,y_pred=predictions_prob[ind])
labels=[np.argmax(pred) for pred in predictions_prob]
paths_test_all=paths_test_a+paths_test_b+paths_test_c+paths_test_d+paths_test_e+paths_test_f+paths_test_auga+paths_test_augc
keys=[get_key(path) for path in paths_test_all]
result = create_submission(predictions=labels,keys=keys)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(result)