import cv2                
import matplotlib.pyplot as plt 
def face_detection(img):
    face_cascade = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_alt.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    print('Number of faces detected:', len(faces))
        
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    plt.show()
    
img=cv2.imread("../input/mun-fb/timg.jpg")
face_detection(img)
