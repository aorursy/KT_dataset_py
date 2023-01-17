import numpy as np

import pandas as pd

import os

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

%matplotlib inline 

import cv2 as cv
DATA_FOLDER = '../input/deepfake-detection-challenge'

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'

TEST_FOLDER = 'test_videos'



print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")

print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")
FACE_DETECTION_FOLDER = '../input/haar-cascades-for-face-detection'

print(f"Face detection resources: {os.listdir(FACE_DETECTION_FOLDER)}")
train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))

ext_dict = []

for file in train_list:

    file_ext = file.split('.')[1]

    if (file_ext not in ext_dict):

        ext_dict.append(file_ext)

print(f"Extensions: {ext_dict}")      
for file_ext in ext_dict:

    print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")
test_list = list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))

ext_dict = []

for file in test_list:

    file_ext = file.split('.')[1]

    if (file_ext not in ext_dict):

        ext_dict.append(file_ext)

print(f"Extensions: {ext_dict}")

for file_ext in ext_dict:

    print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")
json_file = [file for file in train_list if  file.endswith('json')][0]

print(f"JSON file: {json_file}")
def get_meta_from_json(path):

    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))

    df = df.T

    return df



meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)

meta_train_df.head()
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
missing_data(meta_train_df)
missing_data(meta_train_df.loc[meta_train_df.label=='REAL'])
def unique_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    uniques = []

    for col in data.columns:

        unique = data[col].nunique()

        uniques.append(unique)

    tt['Uniques'] = uniques

    return(np.transpose(tt))
unique_values(meta_train_df)
def most_frequent_values(data):

    total = data.count()

    tt = pd.DataFrame(total)

    tt.columns = ['Total']

    items = []

    vals = []

    for col in data.columns:

        itm = data[col].value_counts().index[0]

        val = data[col].value_counts().values[0]

        items.append(itm)

        vals.append(val)

    tt['Most frequent item'] = items

    tt['Frequence'] = vals

    tt['Percent from total'] = np.round(vals / total * 100, 3)

    return(np.transpose(tt))
most_frequent_values(meta_train_df)
def plot_count(feature, title, df, size=1):

    '''

    Plot count of classes / feature

    param: feature - the feature to analyze

    param: title - title to add to the graph

    param: df - dataframe from which we plot feature's classes distribution 

    param: size - default 1.

    '''

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()    
plot_count('split', 'split (train)', meta_train_df)
plot_count('label', 'label (train)', meta_train_df)
meta = np.array(list(meta_train_df.index))

storage = np.array([file for file in train_list if  file.endswith('mp4')])

print(f"Metadata: {meta.shape[0]}, Folder: {storage.shape[0]}")

print(f"Files in metadata and not in folder: {np.setdiff1d(meta,storage,assume_unique=False).shape[0]}")

print(f"Files in folder and not in metadata: {np.setdiff1d(storage,meta,assume_unique=False).shape[0]}")
fake_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='FAKE'].sample(3).index)

fake_train_sample_video
def display_image_from_video(video_path):

    '''

    input: video_path - path for video

    process:

    1. perform a video capture from the video

    2. read the image

    3. display the image

    '''

    capture_image = cv.VideoCapture(video_path) 

    ret, frame = capture_image.read()

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    ax.imshow(frame)
for video_file in fake_train_sample_video:

    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
real_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='REAL'].sample(3).index)

real_train_sample_video
for video_file in real_train_sample_video:

    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
meta_train_df['original'].value_counts()[0:5]
def display_image_from_video_list(video_path_list, video_folder=TRAIN_SAMPLE_FOLDER):

    '''

    input: video_path_list - path for video

    process:

    0. for each video in the video path list

        1. perform a video capture from the video

        2. read the image

        3. display the image

    '''

    plt.figure()

    fig, ax = plt.subplots(2,3,figsize=(16,8))

    # we only show images extracted from the first 6 videos

    for i, video_file in enumerate(video_path_list[0:6]):

        video_path = os.path.join(DATA_FOLDER, video_folder,video_file)

        capture_image = cv.VideoCapture(video_path) 

        ret, frame = capture_image.read()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        ax[i//3, i%3].imshow(frame)

        ax[i//3, i%3].set_title(f"Video: {video_file}")

        ax[i//3, i%3].axis('on')
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='meawmsgiti.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='atvmxvwyns.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='qeumxirsme.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='kgbkktcjxf.mp4'].index)

display_image_from_video_list(same_original_fake_train_sample_video)
test_videos = pd.DataFrame(list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER))), columns=['video'])
test_videos.head()
display_image_from_video(os.path.join(DATA_FOLDER, TEST_FOLDER, test_videos.iloc[0].video))
display_image_from_video_list(test_videos.sample(6).video, TEST_FOLDER)
class ObjectDetector():

    '''

    Class for Object Detection

    '''

    def __init__(self,object_cascade_path):

        '''

        param: object_cascade_path - path for the *.xml defining the parameters for {face, eye, smile, profile}

        detection algorithm

        source of the haarcascade resource is: https://github.com/opencv/opencv/tree/master/data/haarcascades

        '''



        self.objectCascade=cv.CascadeClassifier(object_cascade_path)





    def detect(self, image, scale_factor=1.3,

               min_neighbors=5,

               min_size=(20,20)):

        '''

        Function return rectangle coordinates of object for given image

        param: image - image to process

        param: scale_factor - scale factor used for object detection

        param: min_neighbors - minimum number of parameters considered during object detection

        param: min_size - minimum size of bounding box for object detected

        '''

        rects=self.objectCascade.detectMultiScale(image,

                                                scaleFactor=scale_factor,

                                                minNeighbors=min_neighbors,

                                                minSize=min_size)

        return rects
#Frontal face, profile, eye and smile  haar cascade loaded

frontal_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_frontalface_default.xml')

eye_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_eye.xml')

profile_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_profileface.xml')

smile_cascade_path= os.path.join(FACE_DETECTION_FOLDER,'haarcascade_smile.xml')



#Detector object created

# frontal face

fd=ObjectDetector(frontal_cascade_path)

# eye

ed=ObjectDetector(eye_cascade_path)

# profile face

pd=ObjectDetector(profile_cascade_path)

# smile

sd=ObjectDetector(smile_cascade_path)
def detect_objects(image, scale_factor, min_neighbors, min_size):

    '''

    Objects detection function

    Identify frontal face, eyes, smile and profile face and display the detected objects over the image

    param: image - the image extracted from the video

    param: scale_factor - scale factor parameter for `detect` function of ObjectDetector object

    param: min_neighbors - min neighbors parameter for `detect` function of ObjectDetector object

    param: min_size - minimum size parameter for f`detect` function of ObjectDetector object

    '''

    

    image_gray=cv.cvtColor(image, cv.COLOR_BGR2GRAY)





    eyes=ed.detect(image_gray,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=(int(min_size[0]/2), int(min_size[1]/2)))



    for x, y, w, h in eyes:

        #detected eyes shown in color image

        cv.circle(image,(int(x+w/2),int(y+h/2)),(int((w + h)/4)),(0, 0,255),3)

 

    # deactivated due to many false positive

    #smiles=sd.detect(image_gray,

    #               scale_factor=scale_factor,

    #               min_neighbors=min_neighbors,

    #               min_size=(int(min_size[0]/2), int(min_size[1]/2)))



    #for x, y, w, h in smiles:

    #    #detected smiles shown in color image

    #    cv.rectangle(image,(x,y),(x+w, y+h),(0, 0,255),3)





    profiles=pd.detect(image_gray,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=min_size)



    for x, y, w, h in profiles:

        #detected profiles shown in color image

        cv.rectangle(image,(x,y),(x+w, y+h),(255, 0,0),3)



    faces=fd.detect(image_gray,

                   scale_factor=scale_factor,

                   min_neighbors=min_neighbors,

                   min_size=min_size)



    for x, y, w, h in faces:

        #detected faces shown in color image

        cv.rectangle(image,(x,y),(x+w, y+h),(0, 255,0),3)



    # image

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    ax.imshow(image)
def extract_image_objects(video_file, video_set_folder=TRAIN_SAMPLE_FOLDER):

    '''

    Extract one image from the video and then perform face/eyes/smile/profile detection on the image

    param: video_file - the video from which to extract the image from which we extract the face

    '''

    video_path = os.path.join(DATA_FOLDER, video_set_folder,video_file)

    capture_image = cv.VideoCapture(video_path) 

    ret, frame = capture_image.read()

    #frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    detect_objects(image=frame, 

            scale_factor=1.3, 

            min_neighbors=5, 

            min_size=(50, 50))  

  
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='kgbkktcjxf.mp4'].index)

for video_file in same_original_fake_train_sample_video[1:4]:

    print(video_file)

    extract_image_objects(video_file)
train_subsample_video = list(meta_train_df.sample(3).index)

for video_file in train_subsample_video:

    print(video_file)

    extract_image_objects(video_file)
subsample_test_videos = list(test_videos.sample(3).video)

for video_file in subsample_test_videos:

    print(video_file)

    extract_image_objects(video_file, TEST_FOLDER)
fake_videos = list(meta_train_df.loc[meta_train_df.label=='FAKE'].index)
from IPython.display import HTML

from base64 import b64encode



def play_video(video_file, subset=TRAIN_SAMPLE_FOLDER):

    '''

    Display video

    param: video_file - the name of the video file to display

    param: subset - the folder where the video file is located (can be TRAIN_SAMPLE_FOLDER or TEST_Folder)

    '''

    video_url = open(os.path.join(DATA_FOLDER, subset,video_file),'rb').read()

    data_url = "data:video/mp4;base64," + b64encode(video_url).decode()

    return HTML("""<video width=500 controls><source src="%s" type="video/mp4"></video>""" % data_url)
play_video(fake_videos[0])
play_video(fake_videos[1])
play_video(fake_videos[18])