import numpy as np

import os

import cv2

import glob

import time

from datetime import datetime

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import seaborn as sns

sns.set()

import pandas as pd

yolo_weights = '../input/yolo-trained-files/yolov3_wheat.weights'



yolo_cfg='../input/yolo-trained-files/yolov3.cfg'



# read class names from text file

classes = None

with open("../input/yolo-trained-files/classes.txt", 'r') as f:

    classes = [line.strip() for line in f.readlines()]



COLORS = np.random.uniform(0, 255, size=(len(classes), 3))



net = cv2.dnn.readNet(yolo_cfg,yolo_weights)
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





def draw_bounding_box(img,x, y, x_plus_w, y_plus_h):



#     label = str(classes[class_id])



#     color = COLORS[class_id]



    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,0,255), 2)



#     cv2.putText(img, label, (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
def yolo_detect(path_of_images):

    image = cv2.imread(path_of_images)

    try:

        image.size == 0

    except:

        print('excepted image {}'.format(image))

        return ["No object detected and color detected "]





    Width = image.shape[1]

    Height = image.shape[0]

    scale = 0.00392

    start1 = time.time()



    blob = cv2.dnn.blobFromImage(image, scale, (608,608), (0,0,0), True, crop=False)



    net.setInput(blob)







    start2 = time.time()



# run inference through the network

# and gather predictions from output layers

    outs = net.forward(get_output_layers(net))



# initialization

    class_ids = []

    confidences = []

    boxes = []

    conf_threshold = 0.1

    nms_threshold = 0.3

# for each detetion from each output layer

# get the confidence, class id, bounding box params

# and ignore weak detections (confidence < 0.5)

    for out in outs:

        for detection in out:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]

            if confidence > 0.5:

                center_x = int(detection[0] * Width)

                center_y = int(detection[1] * Height)

                w = int(detection[2] * Width)

                h = int(detection[3] * Height)

                x = center_x - w / 2

                y = center_y - h / 2

                class_ids.append(class_id)

                confidences.append(float(confidence))

                boxes.append([x, y, w, h])

    #arr=np.array([[0,0,0,0,0]])

    #arr=np.ndarray.astype(arr,dtype='str',casting='unsafe')

    obj_result = []



# apply non-max suppression

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)



# go through the detections remaining

# after nms and draw bounding box

    for i in indices:

        i = i[0]

        box = boxes[i]

        x = box[0]

        y = box[1]

        w = box[2]

        h = box[3]



        draw_bounding_box(image,round(x), round(y), round(x+w), round(y+h))







        obj_result.append("{} {} {} {} {}".format(round(confidences[i],2), round(x), round(y), round(w), round(h)))



    imShow(image)



    return " ".join(obj_result)


results=[]



for i in glob.glob("../input/global-wheat-detection/test/*"):

    image_id = i.split("/")[-1].split(".")[0]



    key = image_id



    res = yolo_detect(i)



    result={'image_id': key, "PredictionString": res}



    results.append(result)









df = pd.DataFrame(results,columns = ['image_id','PredictionString'])



df.to_csv("submission.csv",index=False)

print(df)