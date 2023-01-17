# inspired by https://github.com/OlafenwaMoses/ImageAI/ , https://www.kaggle.com/shivamb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

#importing required libraries
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import math, os
from keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import GeneratorEnqueuer
image_path ="../input/google-ai-open-images-object-detection-track/test/"
batch_size = 100
img_generator = ImageDataGenerator().flow_from_directory(image_path, shuffle=False, batch_size = batch_size)                                                                                      
#calculating size of epoch                                                                                         
n_rounds = math.ceil(img_generator.samples / img_generator.batch_size)  # size of an epoch

filenames = img_generator.filenames
img_generator = GeneratorEnqueuer(img_generator)
img_generator.start()
img_generator = img_generator.get()

##Using image Object Detection, using yolo-tiny which is optimized for speed 
from imageai.Detection import ObjectDetection
import os
weights_path = "../input/yolotiny/yolo-tiny.h5"

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath( weights_path)
detector.loadModel()
for i in range(n_rounds):
    batch = next(img_generator)
    for j, prediction in enumerate(batch):
        image = filenames[i * batch_size + j]
        detections = detector.detectObjectsFromImage(input_image=image_path+image, output_image_path="image_with_box.png", minimum_percentage_probability = 80)        
        for eachObject in detections:
            print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
            plt.figure(figsize=(12,12))
            plt.imshow(plt.imread("image_with_box.png"))
            plt.show()
    if i==10:
        break     
##ADDED VIDEO DETECTION 
from imageai.Detection import VideoObjectDetection

detector_video = VideoObjectDetection()
detector_video.setModelTypeAsTinyYOLOv3()
detector_video.setModelPath('../input/yolotiny/yolo-tiny.h5')
detector_video.loadModel()

print(detector_video)
video_path = detector_video.detectObjectsFromVideo(input_file_path='../input/traffic/traffic.mp4',
                                            output_file_path='../input/traffic_detected',
                                            frames_per_second=20, log_progress=True)
print(video_path)