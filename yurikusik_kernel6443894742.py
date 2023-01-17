from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

#defines function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

#loads train, test, and validation datasets
train_files, train_targets = load_dataset('../input/dogimages/dogImages/train')
valid_files, valid_targets = load_dataset('../input/dogimages/dogImages/valid')
test_files, test_targets = load_dataset('../input/dogimages/dogImages/test')

#loads list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../input/dogimages/dogImages/train/*/"))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))
import random
random.seed(8675309)

#loads filenames in shuffled human dataset
human_files = np.array(glob("../input/humanfaces/lfw/*/*"))
random.shuffle(human_files)

#prints statistics about the dataset
print('There are %d total human images.' % len(human_files))
import cv2                
import matplotlib.pyplot as plt                        
%matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('../input/haarcascade/haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[5])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
human_files_short = human_files[:100]
dog_files_short = train_files[:100]

human_faces = [face_detector(human_file) for human_file in human_files_short]
percentage_human = 100 * (np.sum(human_faces)/len(human_faces))

dog_faces = [face_detector(dog_file) for dog_file in dog_files_short]
percentage_dog = 100 * (np.sum(dog_faces)/len(dog_faces))
        
print('There are %.1f%% images of the first 100 human_files that have a detected human face.' % percentage_human)
print('There are %.1f%% images of the first 100 dog_files that have a detected human face.' % percentage_dog)
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))            
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))
human_faces = [dog_detector(human_file) for human_file in human_files_short]
percentage_human = 100 * (np.sum(human_faces)/len(human_faces))

dog_faces = [dog_detector(dog_file) for dog_file in dog_files_short]
percentage_dog = 100 * (np.sum(dog_faces)/len(dog_faces))
        
print('There are %.1f%% images of the first 100 human_files that have a detected dog face.' % percentage_human)
print('There are %.1f%% images of the first 100 dog_files that have a detected dog face.' % percentage_dog)
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()  
#The model type that we will be using is Sequential. 
#Sequential is the easiest way to build a model in Keras. 
#It allows you to build a model layer by layer.


model.add(Conv2D(filters=16, kernel_size=2, padding='valid', activation='relu', input_shape=(224, 224, 3)))
#Our first layers is Conv2D layers.
#convolution layers that will deal with our input images, which are seen as 2-dimensional matrices.
#16 in the first layer is the number of nodes, We want 16 feature maps
#Kernel size is the size of the filter matrix for our convolution. 
#So a kernel size of 2 means we will have a 2x2 filter matrix or feature detector .
#Padding is used on the convolutional layers to ensure the height and width of the output feature maps matches the inputs.
#The activation function we will be using for our layers is the ReLU Rectifier Linear Unit which helps with non linearity in the neural network.
#Our first layer also takes in an input shape. 
#This is the shape of each input image, 224, 224, 3 

model.add(MaxPooling2D(pool_size=2))

#we apply max pooling for translational invariance. Translational invariance is when we change the input by a small amount the outputs do not change. 
#Max pooling reduces the number of cells.
#Pooling helps detect features like colors, edges etc.
#For max pooling, we use the pool_size of 2 by 2 matrix for all 32 feature maps.

model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#We can add one more convolutional layer.
#This time we will have 32 feature maps with the kernel of (2,2).
#We then apply the max pooling to the convolutional layers.

model.add(Conv2D(filters=64, kernel_size=2, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#We can add one more convolutional layer.
#This time we will have 64 feature maps with the kernel of (2,2).
#We then apply the max pooling to the convolutional layers.

model.add(Flatten())

#there is a ‘Flatten’ layer. Flatten serves as a connection between the convolution and dense layers.
#step is to flatten all the inputs. The flattened data will be the input to the fully connected neural network.

model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(133, activation='softmax'))

#Dense’ is the layer type we will use in for our output layer. Dense is a standard layer type that is used in many cases for neural networks.
#we use Dropout rate of 20% to prevent overfitting.
#The activation is ‘softmax’. Softmax makes the output sum up to 1 so the output can be interpreted as probabilities. The model will then make its prediction based on which option has the highest probability.



model.summary()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 5

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, verbose=1)
# get index of predicted dog breed for each image in test set
dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
bottleneck_features = np.load('../input/dogtrainedmodels/bottleneck_features/bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']
VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.summary()
VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
VGG16_model.fit(train_VGG16, train_targets, 
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, verbose=1)
# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

#Obtain bottleneck features from another pre-trained CNN.

bottleneck_features = np.load('../input/dogtrainedmodels/bottleneck_features/bottleneck_features/DogVGG16Data.npz')
train_Resnet50 = bottleneck_features['train']
valid_Resnet50 = bottleneck_features['valid']
test_Resnet50 = bottleneck_features['test']
#Our architecture.
Resnet50_model = Sequential()
Resnet50_model.add(GlobalAveragePooling2D(input_shape=train_Resnet50.shape[1:]))
Resnet50_model.add(Dense(133, activation='softmax'))

Resnet50_model.summary()
#Compile the model.
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#Train the model.

Resnet50_model.fit(train_Resnet50, train_targets, 
          validation_data=(valid_Resnet50, valid_targets),
          epochs=20, batch_size=20, verbose=1)
#Calculate classification accuracy on the test dataset.
Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) for feature in test_Resnet50]

# report test accuracy
test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)


def human_or_dog(img_path):
    if dog_detector(img_path):
        print("you are a dog!")
    else: 
        face_detector(img_path)
        print("you are a human!")
def show_img(img_path):
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
img_path = "../input/imageskat2/imageskat/image4.jpg"
print(img_path)
show_img(img_path)
human_or_dog(img_path)
img_path = "../input/imageskat2/imageskat/image5.jpg"
print(img_path)
show_img(img_path)
human_or_dog(img_path)
img_path = "../input/imageskat/imageskat/image1.jpg"
print(img_path)
show_img(img_path)
human_or_dog(img_path)
img_path = "../input/imageskat/imageskat/image3.jpg"
print(img_path)
show_img(img_path)
human_or_dog(img_path)
