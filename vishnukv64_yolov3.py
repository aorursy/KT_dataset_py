''' Uncomment the lines if GPU or else run for cpu '''
# This cell can be commented once you checked the current CUDA version
# CUDA: Let's check that Nvidia CUDA is already pre-installed and which version is it. In some time from now maybe you 

# !nvidia-smi (gpu)

# !nvcc -V (gpu)
#upload the cudnn appropriate version and follow the below process to successfully install darknet with GPU

# !tar -xzvf ../input/cuda-driver/cudnn-10.1-linux-x64-v7.6.5.32.tgz -C /usr/local/ (gpu)
# We're unzipping the cuDNN files from your Drive folder directly to the VM CUDA folders

# !chmod a+r /usr/local/cuda/include/cudnn.h (gpu)

# Now we check the version we already installed. Can comment this line on future runs

# !cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2 (gpu)
# Leave this code uncommented on the very first run of your notebook or if you ever need to recompile darknet again.
# Comment this code on the future runs.
!git clone https://github.com/AlexeyAB/darknet
    
''' We then locally copy all the files needed inside the darknet folder for ease of access and we need to write certain text files into the images folder'''

!cp -r "../input/global-wheat-detection/train" darknet/train

!cp -r "../input/global-wheat-detection/test" darknet/test

!cp -r "../input/global-wheat-detection/train.csv" darknet/train.csv

%cd darknet

# Check the folder
!ls

#uncomment these lines if youre using GPU for training 

# !sed -i 's/GPU=0/GPU=1/' Makefile 
# !sed -i 's/CUDNN=0/CUDNN=1/' Makefile
# !sed -i 's/OPENCV=0/OPENCV=1/' Makefile
# !sed -i 's!/usr/local/cudnn/!/usr/local/cuda/!' Makefile

#Compile Darknet
!make

#Copies the Darknet compiled version to Google drive
# !cp ./darknet /content/drive/My\ Drive/darknet/

#check whether changes have beem made or not
# !cat Makefile

# Set execution permissions to Darknet
!chmod +x ./darknet
#downloading the pre-trained weights for YOLO
# !wget https://pjreddie.com/media/files/darknet53.conv.74
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
sns.set()
import cv2
# loading our required data

train_wheat= pd.read_csv('train.csv') # we are loading the csv file into a dataframe for us to start extracting all the data needed to create a model

print(train_wheat.head()) #just printing out the first 5 values in our table
print(train_wheat.tail()) #just printing out the last 5 values in our table

train_wheat= train_wheat.drop(['width', 'height'], axis=1)

source = train_wheat['source'].unique().tolist()
print(source)
no_of_classes = len(source)

train_images_name = train_wheat['image_id'].tolist()
train_images_source = train_wheat['source'].tolist()
normalised_values = pd.factorize(train_images_source)[0]

print(normalised_values)

train_images_bbox = train_wheat['bbox'].tolist()

source_list = train_wheat['source'].value_counts()


fig = go.Figure(data=[
    go.Pie(labels=source_list.index, values=source_list.values)
])

fig.update_layout(title='Source distribution')
fig.show()

'''making a dictionary to have the image name as key and the bbox as values'''

dc= {}
for i, (img,bbox ,name) in enumerate(zip(train_images_name,train_images_bbox, normalised_values)):
    # print(img, bbox, name)
    key = img
    if key not in dc:
        dc[key] = []
    dc[key].append(bbox.strip("[]")+"/"+"{}".format(name))

''' the given format in the csv file is [x,y,w,h] as we need to convert into yolo formats [centerx, centery, width , height] '''

width = 1024
height = 1024

for key, values in dc.items():
  # print(key,values)
  for i in values:
    bbox, obj_id = i.split('/')
    # print(bbox)
    x_center = (float(bbox.split(",")[0])+(float(bbox.split(",")[2])/2))/width
    y_center = (float(bbox.split(",")[1])+(float(bbox.split(",")[3])/2))/height
    wd = float(bbox.split(",")[2])/width
    ht = float(bbox.split(",")[3])/height
#     print(x_center,y_center,wd,ht)

    with open("train/{}.txt".format(key), "a+") as f:
      f.write(f"{obj_id} {x_center} {y_center} {wd} {ht}")
      f.write('\n')

#preprocessing and data preparation needed for yolo

def check_images_labels(images, labels):
    image =[]
    label=[]

    image_list = glob.glob(images)
    labels_list = glob.glob(labels)

    for i in image_list:
        split_img = i.split("/")[-1].split(".")[0]
        image.append(split_img)
    for i in labels_list:
        split_label = i.split("/")[-1].split(".")[0]
        label.append(split_label)

    def Diff(li1, li2):
        return (list(set(li1) - set(li2))) #this is to find out the missing labels for the images that has not been annotated

    diff_list = Diff(image, label)

    print(diff_list)
    """ We found out that the images and labels are not equal, by cross checking it, those images are just noise and doesnt need to be included so
    either we make a empty text file or delete those images having no labels_list or all those images can be used as negative images"""

    for i in diff_list:
        with open ("train/{}.txt".format(i),"a+") as ff: # makes txt files for those negative images
            ff.close()
    print("made txt files for missing annotated images")

    # for i in diff_list:
    #     os.remove("train/{}.jpg".format(i))
    # print("deleted image files for missing annotated images")


'''Generating the train and text files'''

def generate_txt():
    try:


        dataset_path =  "train"

        # Percentage of images to be used for the test set
        percentage_test = 20;
        # Create and/or truncate train.txt and test.txt
        file_train = open('train.txt', 'w')
        file_test = open('test.txt', 'w')

        # Populate train.txt and test.txt
        counter = 1
        index_test = round(100 / percentage_test)
        for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
            title, ext = os.path.splitext(os.path.basename(pathAndFilename))

            if counter == index_test+1:
                counter = 1
                file_test.write(dataset_path + "/" + title + '.jpg' + "\n")
            else:
                file_train.write(dataset_path + "/" + title + '.jpg' + "\n")
                counter = counter + 1
        print('train and text generated')
    except Exception as e:
        print(e)

'''Function to generate trainer.data'''

def gen_trainer(no_of_classes, config):

    try:
        with open("trainer.data", "a+") as f:
            f.writelines('classes = {}\n'.format(no_of_classes))
            f.writelines('train = {}\n'.format('train.txt'))
            f.writelines('valid = {}\n'.format('test.txt'))
            f.writelines('names = {}\n'.format('Object' + '/' + 'object.names'))
            f.writelines('backup = {}'.format('backup/'))

        print('trainer.data generated')

    except Exception as e:
        print('error while generating trainer.data')
        print(e)


'''Function to generate object.names'''

def objectnames(class_names):
    if not os.path.exists('Object' + '/'):
        os.mkdir('Object' + '/')
    try:
        for i in class_names:
            with open ('Object' + '/'+'object.names', 'a+') as f:
                f.writelines(i)
                f.writelines('\n')
        print("object.names is created")
    except Exception as e:
        print('error while generating Object.names')
        print(e)

''' creating the new config file needed for yolo to train '''

def config_change(No_of_Classes, batch, subdivision):

    config_path='cfg/yolov3.cfg'

    classes=int(No_of_Classes)
    filters=(classes + 5)*3



    with open(config_path, 'r+') as f:
        cfg=f.readlines()

    '''Batch'''
    cfg[2]='#batch=1\n'
    cfg[3]='#subdivisions=1\n'
    cfg[4]='Training\n'
    cfg[5]='batch={}\n'.format(batch)
    cfg[6]='subdivisions={}\n'.format(subdivision)

    '''filters and classes'''
    cfg[602]='filters={}\n'.format(filters)
    cfg[609]='classes={}\n'.format(classes)
    cfg[688]='filters={}\n'.format(filters)
    cfg[695]='classes={}\n'.format(classes)
    cfg[775]='filters={}\n'.format(filters)
    cfg[782]='classes={}\n'.format(classes)


    with open('yolov3.cfg', 'w+') as f:
        f.writelines(cfg)
    print('full config file changed and saved')
    return config_path

batch_size=64
subdivisions=16

check_images_labels("train/*.jpg", "train/*.txt")

objectnames(source)

generate_txt()

cfg_use = config_change(no_of_classes, batch_size, subdivisions)

gen_trainer(no_of_classes, cfg_use)

# !cat trainer.data

# !ls train
'''uncomment to start start training'''
#start the training
# !./darknet detector train trainer.data yolov3.cfg darknet53.conv.74 -dont_show
'''Uncomment this cell if you have trained the model '''
# Start training at the point where the last runtime finished

# !./darknet detector train trainer.data yolov3.cfg backup/yolov3_last.weights -dont_show 
'''Uncomment this cell if you have trained the model '''
# in order to check your Model's mAP 

# !./darknet detector map trainer.data yolov3.cfg yolov3_last.weights
'''Uncomment this cell if you have trained the model '''

#in order to check your model inference after training with the test images using darknet



# def imShow(path):
#   image = cv2.imread(path)
#   height, width = image.shape[:2]
#   resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

#   fig = plt.gcf()
#   fig.set_size_inches(18, 10)
#   plt.axis("off")
#   plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
#   plt.show()
    
# import PIL
# import PIL.Image as Image

# d = 0

# image_list = glob.glob("test/*")
# for images in image_list:
#   print(images , d)
#   commands = './darknet detector test trainer.data yolov3.cfg backup/yolov3_last.weights {} -dont_show'.format(images)
#   os.system(commands)
#   predicted_image = Image.open("predictions.jpg")
#   output = ("predicted_image{}.jpg".format(d))
#   print(output)
#   predicted_image.save(output)
#   d+=1
#   imShow(output)
# check the Mean average precision of our model
%cd ..

!cp -r "../input/yolo-weights/yolov3_wheat.weights" darknet/yolov3.weights #downloading my locally trained weights

%cd darknet
#over here we need to change the yolo config file for inference type and not training

def config_change_inference(No_of_Classes, batch, subdivision):

    config_path='cfg/yolov3.cfg'

    classes=int(No_of_Classes)
    filters=(classes + 5)*3

    filters = str(filters)
    classes= str(No_of_Classes)



    with open(config_path, 'r') as f:
        cfg=f.readlines()


    '''Batch'''
    cfg[2]='# Testing\n'
    cfg[2]='# batch=1\n'
    cfg[3]='# subdivisions=1\n'
    cfg[4]='# Training\n'
    cfg[5]='batch={}\n'.format(batch)
    cfg[6]='subdivisions={}\n'.format(subdivision)

    '''filters and classes'''
    cfg[602]='filters={}\n'.format(filters)
    cfg[609]='classes={}\n'.format(classes)
    cfg[688]='filters={}\n'.format(filters)
    cfg[695]='classes={}\n'.format(classes)
    cfg[775]='filters={}\n'.format(filters)
    cfg[782]='classes={}\n'.format(classes)


    with open('yolov3_inference.cfg', 'w+') as f:
        f.writelines(cfg)
    print('full config file changed and saved')
    return config_path
#make a custom inference using opencv readnet

config_change_inference(no_of_classes, batch_size, subdivisions)
print('created the yolo cfg file for inference')

cfg = "yolov3_inference.cfg"
weights = "yolov3.weights"
net = cv2.dnn.readNet(weights,cfg)

conf_threshold = 0.3
nms_threshold = 0.4

classes = None
f10=open("Object/object.names", 'r')
classes = [line.strip() for line in f10.readlines()]
f10.close()
print(classes)
COLORS =  np.random.uniform(0,255, size=(len(classes),3))
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
  

def imShow(image):

  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
    
def yolo_detect(image):
  image = cv2.imread(image)
  obj_result=[]
  if image is not None:
    Width = image.shape[1]
    Height = image.shape[0]
    blob = cv2.dnn.blobFromImage(image,0.00392,(608,608), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
      for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
          center_x = int(detection[0]*Width)
          center_y = int(detection[1]*Height)
          w =int(detection[2]*Width)
          h =int(detection[3] * Height)
          x = center_x - w // 2
          y = center_y - h // 2
          class_ids.append(class_id)
          confidences.append(float(confidence))
          boxes.append([x, y, w, h])
          
          
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
      i = i[0]
      box=boxes[i]
      x = box[0]
      y = box[1]
      w = box[2]
      h = box[3]
      cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
      imShow(image)
      obj_result.append("{} {} {} {} {} {}".format(class_ids[i], round(confidences[i],2), round(x), round(y), round(w), round(h)))
      
  return " ".join(obj_result)
#saving into sample submission format

finaldc={}

for i in glob.glob("test/*"):
  image_id = i.split("/")[-1].split(".")[0]
  results = yolo_detect(i)

  key = image_id
  if key not in finaldc:
    finaldc[key] = []
    finaldc[key].append(results)

df = pd.DataFrame(finaldc,columns = ['image_id','Prediction_string'])


# df.to_csv("wheat_submission.csv")

%cd ..
df.to_csv("submission.csv")
print(df)