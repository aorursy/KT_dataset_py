#import libraries

from os import listdir

from os.path import isfile,join

import cv2
#Load Neural Transfer Models

model_file_path='../input/fast-neural-style-models/'

model_file_paths=[f for f in listdir(model_file_path) if isfile(join(model_file_path,f))]
#Load Test Image

img=cv2.imread('../input/images/london.jpg')

#Load Transfer Style Model

model=('la_muse.t7')
#Loop and apply each model style on our test image

for i in model_file_paths:

    style=cv2.imread('../styles/'+str(model)[:-3]+'.jpg')

    

    #Load the Neural Transfer Style Model. (dnn is deep neural network)

    neural_style_model=cv2.dnn.readNetFromTorch(model_file_path+model)

    

    #resize it to fix height

    height,width=int(img.shape[0]),int(img.shape[1])

    new_width=int((640/height)*width)

    resized_image=cv2.resize(img,(new_width,640),interpolation=cv2.INTER_AREA)

    

    #Create our blob from the image

    #Then perform a forward run pass of the network

    #The mean values for the Imagenet Training set are R=103.93, G=116.77 ,B=123.68

    inp_blob=cv2.dnn.blobFromImage(resized_image,1.0,(new_width,640),(103.93,116.77,123.68),swapRB=False,crop=False)

    neural_style_model.setInput(inp_blob)

    output=neural_style_model.forward()

    

    #Reshape the output Tensor,

    #add back the mean subtraction(de-process the thing)

    #re-order the channels

    output=output.reshape(3,output.shape[2],output.shape[3])

    output[0]+=103.93

    output[1]+=116.77

    output[2]+=123.68

    output/=255

    output=output.transpose(1,2,0)

    

    #Display:

    #1. Original/Test Image

    #2. The Style of the Neural Transfer

    #3. Our result from them

    

    #cv2.imwrite('Original.jpg',img)

    #cv2.imwrite('Style.jpg',style)

    cv2.imwrite('Neural Transfer Style_1.jpg',output)

    cv2.waitKey(0)

    

    #Close everything

    if cv2.waitKey(0) & 0xFF==27: #This is the escape key

        break

        

#Destroy all Windows

#cv2.destroyAllWindows()

    