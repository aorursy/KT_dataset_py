# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install numpy opencv-python dlib imutils

!wget -nd https://github.com/JeffTrain/selfie/raw/master/shape_predictor_68_face_landmarks.dat
import numpy as np
def curve(points):
	x = points[:,0]
	y = points[:,1]

	z = np.polyfit(x, y, 2)
	f = np.poly1d(z)

	x_n = np.linspace(x[0], x[-1], 100)
	y_n = f(x_n)
	return list(zip(x_n, y_n))
#from curve_fitting import curve
from imutils import face_utils
import numpy as np
import dlib
import cv2


path='/kaggle/working/'
img_path='/kaggle/input/image-utils/faces_featureimg.jpg'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/kaggle/working/shape_predictor_68_face_landmarks.dat")




img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray)



for (j,face) in enumerate(faces):
        points = predictor(gray, face)

        print('\nface #',j+1)
        l=[]
        print('\nface boundary coordinate\n')
        for i in range(0, 27):  # loop for displaying face boundary coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)
        print('\nnose coordinate\n')
        for i in range(27, 36):  # loop for displaying nose  coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)
        print('\nleft eye coordinate\n')
        for i in range(36, 42):  # loop for displaying left eye coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)
        print('\nright eye coordinate\n')
        for i in range(42, 48):  # loop for displaying right eye coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)
        print('\nlips coordiante\n')
        for i in range(48, 68):  # loop for displaying lips coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)

        for i in range(5, 12):                          #loop for storing jaw coordinates
            curr_c=(points.part(i).x, points.part(i).y)
            l.append(curr_c)

        cur=np.array(curve(np.array(l)), np.int32)      # calling function to trace proper fitting curve


        for i in range(len(cur)-1):                          #loop for tracing jaw line
            curr_c=(cur[i][0], cur[i][1])
            next_cordi=(cur[i+1][0], cur[i+1][1])
            cv2.line(img, curr_c, next_cordi, (0, 0, 0), 3)
        for n in range(0, 68):                          #loop for detecting feature points on face
        	x = points.part(n).x
        	y = points.part(n).y
        	cv2.circle(img, (x, y), 3, (0, 0, 255), 2)

        #points = face_utils.shape_to_np(points)

        # to  convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # to display the face number
        cv2.putText(img, "Face #{}".format(j + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#cv2.imwrite(path, img) 


cv2.imwrite(path+"/faces_featureimg.jpg", img) #writes image with landmark points