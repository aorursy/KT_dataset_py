import cv2, os

haar_path = '/opt/conda/lib/python3.6/site-packages/cv2/data'

!ls {haar_path}
xml_name = 'haarcascade_frontalface_alt2.xml'

xml_path = os.path.join(haar_path, xml_name)
clf = cv2.CascadeClassifier(xml_path)
import cv2 as cv

import os

import matplotlib.pylab as plt



fig, ax = plt.subplots(1,1, figsize=(15, 15))

video_file = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/akxoopqjqz.mp4'

cap = cv.VideoCapture(video_file)

success, image = cap.read()



# Convert image into gray scale and classify images

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_locations = clf.detectMultiScale(gray)



# Continue with the original code

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

cap.release()   

ax.imshow(image)

ax.xaxis.set_visible(False)

ax.yaxis.set_visible(False)

ax.title.set_text(f"FRAME 0: {video_file.split('/')[-1]}")

plt.grid(False)
from PIL import Image



print("I found {} face(s) in this photograph.".format(len(face_locations)))



for face_location in face_locations:



    # Print the location of each face in this image

    x, y, w, h = face_location

    print("A face is located at pixel location X: {}, Y: {}, Width: {}, Height: {}".format(x, y, w, h))



    # You can access the actual face itself like this:

    face_image = image[y:y+h, x:x+w]

    fig, ax = plt.subplots(1,1, figsize=(5, 5))

    plt.grid(False)

    ax.xaxis.set_visible(False)

    ax.yaxis.set_visible(False)

    ax.imshow(face_image)
import glob

all_xmls = glob.glob(haar_path + '/*.xml')
print("Option of XML features:")

print([xml.split('/')[-1] for xml in all_xmls])
for xml in all_xmls:

    if xml.split("/")[-1]=='haarcascade_licence_plate_rus_16stages.xml': # Skipping. This attribute is throwing an error

        print(f"Skipping {xml}")

        continue 

        

    clf = cv2.CascadeClassifier(xml)

    locations = clf.detectMultiScale(gray)



    name_xml = xml.split("/")[-1].split(".")[0].replace("haarcascade_", "")

    print('='*80)

    print(f'Feature to be extracted: {name_xml}')

    print(f"I found {len(locations)} {name_xml} in this photograph.")



    for location in locations:



        # Print the location of each face in this image

        x, y, w, h = location

        print(f"A {name_xml} is located at pixel location X: {x}, Y: {y}, Width: {w}, Height: {h}")



        # You can access the actual face itself like this:

        attribute_image = image[y:y+h, x:x+w]

        fig, ax = plt.subplots(1,1, figsize=(5, 5))

        plt.grid(False)

        ax.xaxis.set_visible(False)

        ax.yaxis.set_visible(False)

        ax.imshow(attribute_image)

        plt.show()
!pip install ../input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl
from mtcnn import MTCNN

detector = MTCNN()

result = detector.detect_faces(image); result
x, y, w, h = result[0]['box']

right_eye = result[0]['keypoints']['right_eye']

nose = result[0]['keypoints']['nose']

mouth_left = result[0]['keypoints']['mouth_left']

mouth_right = result[0]['keypoints']['mouth_right']
video_file = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/akxoopqjqz.mp4'

cap = cv.VideoCapture(video_file)

success, image = cap.read()

image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

cap.release()



cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), thickness=5)

fig, ax = plt.subplots(1,1, figsize=(15, 15))

plt.grid(False)

ax.xaxis.set_visible(False)

ax.yaxis.set_visible(False)

ax.imshow(image)

ax.plot(right_eye[0], right_eye[1], 'go') # Right eye in green

ax.plot(nose[0], nose[1], 'yo') # Nose in yellow

ax.plot(mouth_left[0], mouth_left[1], 'ro') # Left and right mouth in red

ax.plot(mouth_right[0], mouth_right[1], 'ro')