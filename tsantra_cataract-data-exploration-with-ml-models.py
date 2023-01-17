# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

import os

import cv2

import ntpath

from sklearn.linear_model import LogisticRegression

#from sklearn import svm, datasets

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier

from scipy import interp

from sklearn import svm

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier

from scipy import interp

from itertools import cycle

from sklearn.decomposition import PCA



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

files=[] #store the filenames here

labels1=[] #store the labels here

dirname='/kaggle/input'

for dirname, _, filenames in os.walk(dirname):

    for filename in filenames:

        if filename.endswith('.png'):

            files.append(os.path.join(dirname, filename))            

            if filename.startswith('NL'):                

                labels1.append('NL')     #Normal retina      

                

            elif filename.startswith('ca'):  #Cataract              

                labels1.append('ca')

            elif filename.startswith('Gl'):  #Glaucoma              

                labels1.append('Gl')

            elif filename.startswith('Re'):  #Retina Disease              

                labels1.append('Re')

            





# Shuffle the data

combined = list(zip(files,labels1)) # combine the lists

np.random.shuffle(combined) # shuffle two lists together to keep order

files[:],labels1[:] = zip(*combined) #unzip the shuffled lists
########################################################################

#########The following function normalizes the image histogram##########

#######################################################################



def normalize_histograms(im): #normalizes the histogram of images

    im1=im.copy()

    for i in range(3):

        imi=im[:,:,i]

        #print(imi.shape)

        minval=np.min(imi)

        maxval=np.max(imi)

        #print(minval,maxval)

        imrange=maxval-minval

        im1[:,:,i]=(255/(imrange+0.0001)*(imi-minval)) # imi-minval will turn the color range between 0-imrange, and the scaleing will stretch the range between 0-255

    return im1



######################################################################

# This following function reads the images from file,

#auto crops the image to its relevant content, then normalizes

#the histograms of the cropped images

######################################################################



def read_and_process_image(filename):

        im=cv2.imread(filename) #read image from file 

        # The following steps re needed for auto cropping the black paddings in the images

        

        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) # convert 2 grayscale

        _,thresh = cv2.threshold(gray,10,255,cv2.THRESH_BINARY) # turn it into a binary image

        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # find contours

        if len(contours) != 0:

            #find the biggest area

            cnt = max(contours, key = cv2.contourArea)

                      

            #find the bounding rect

            x,y,w,h = cv2.boundingRect(cnt)                  



            crop = im[y:y+h,x:x+w]# crop image

            #crop1=cv2.resize(crop,(im_size,im_size)) # resize to im_size X im_size size

            crop=normalize_histograms(crop)

            return crop

        else:

            return( normalize_histograms(im))





##################################################################################

#### The following functions are for extracting features from the images #########

##################################################################################

        

# histogram statistics (mean, standard deviations, energy, entropy, log-kurtosis)





def histogram_statistics(hist):

    #hist= cv2.calcHist([gr],[0],None,[256],[0,256])

    hist=hist/np.sum(hist)#probabilities

    hist=hist.reshape(-1)

    hist[hist==0]=10**-20 # replace zeros with a small number

    mn=np.sum([i*hist[i] for i in range(len(hist))]) # mean

    std_dev=np.sqrt(np.sum([((i-mn)**2)*hist[i] for i in range(len(hist))])) # standard deviation

    energy=np.sum([hist[i]**2 for i in range(len(hist))]) #energy

    entropy=np.sum([hist[i]*np.log(hist[i]) for i in range(len(hist))]) #entropy

    kurtosis=np.log(np.sum([(std_dev**-4)*((i-mn)**-4)*hist[i] for i in range(len(hist))])) # kurtosis

    return[mn,std_dev,energy,entropy,kurtosis]



#################################################################

# create thresholding based features, the idea is to hand engineer some features based on adaptive thresholding.

#After looking at the images it appeared  that adaptive thresholding may

#leave different artifacts in the processed images, we can extract several features from these artifacts            

##################################################################



def thresholding_based_features(im,imsize,quartiles):

    im=cv2.resize(im,(imsize,imsize))

    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    w=11 #window

    t=5#threshold

    counts=[]

    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,w,t) # adaptive gaussian threshold the image

    th=cv2.bitwise_not(th)    #invert the image (the black pixels will turn white and the white pixels will turn black)

    contours,hierarchy = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #find cntours in the image

    #print(len(contours))

    

    q=np.zeros(len(quartiles)) # quartiles of contours will be stored here

    



    for cnt in contours:

        area=cv2.contourArea(cnt) # calculate the area of the contours

        if area<40000: #Exclude contours that are too big, generally these are the image outlines

            counts.append(area) 

    if len(counts)>1:

        q=np.quantile(np.array(counts),quartiles) # contour quartiles

    

    return (q,len(counts),np.sum(th)/(255*th.shape[0]*th.shape[1]))# return the contour quartiles, number of contours, proportion of white pixels in the thresholded images

    #counts.append(np.sum(th)/(normalizing_factor*(th.shape[0]*th.shape[1])))



##########################################################################

############ The following code creates the various features #############

##########################################################################

    

# color averages

B=[] 

G=[]

R=[]



#mini 16 bin histograms

hist_B=[]

hist_G=[]

hist_R=[]



#statistics fom full 256 bin shitogram

hist_feat_B=[]

hist_feat_G=[]

hist_feat_R=[]

hist_feat_GS=[]



#thresholding based features

mean_pixels=[] #proportion of white pixels

contour_quartiles=[] # contour area quartiles

no_of_contours=[] #total number of contours





quartiles=np.arange(0.1,1,0.1) # contour area quartiles

bins=16 # mini histogram bins



for f in files:

    im=read_and_process_image(f)

    #im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)



    # equalize the histogram of the Y channel

    #im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])



    # convert the YUV image back to RGB format

    #im = cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)

    

    #median color

    B.append(np.median(im[:,:,0]))

    G.append(np.median(im[:,:,1]))

    R.append(np.median(im[:,:,2]))

    

    #histograms

    hist_B.append(cv2.calcHist([im],[0],None,[bins],[0,256])/(im.size/3))

    hist_G.append(cv2.calcHist([im],[1],None,[bins],[0,256])/(im.size/3))

    hist_R.append(cv2.calcHist([im],[2],None,[bins],[0,256])/(im.size/3))

    

    

    #more histogram features

    

    hist_feat_B.append(histogram_statistics(cv2.calcHist([im],[0],None,[256],[0,256])))

    hist_feat_G.append(histogram_statistics(cv2.calcHist([im],[1],None,[256],[0,256])))

    hist_feat_R.append(histogram_statistics(cv2.calcHist([im],[2],None,[256],[0,256])))

    

    gr=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    gr=cv2.equalizeHist(gr)

    hist_feat_GS.append(histogram_statistics(cv2.calcHist([gr],[0],None,[256],[0,256])))

    

    #threshold featues

    q,nc,m=thresholding_based_features(im,256,quartiles)

    mean_pixels.append(m)

    contour_quartiles.append(q)

    no_of_contours.append(nc)



#create feature vectors

width_of_features=3*bins+len(quartiles)+2+20 #20 features are histogram statistics



X=np.zeros((len(files),width_of_features)) # this is where all features will be stored



for i in range(len(files)):

    X[i,0:bins]=hist_B[i].reshape(-1)

    X[i,bins:2*bins]=hist_G[i].reshape(-1)

    X[i,2*bins:3*bins]=hist_R[i].reshape(-1)

    X[i,3*bins:3*bins+len(quartiles)]=contour_quartiles[i].reshape(-1)

    X[i,3*bins+len(quartiles)]=mean_pixels[i]

    X[i,3*bins+len(quartiles)+1]=no_of_contours[i]

    start=3*bins+len(quartiles)+2

    X[i,start:start+5]=hist_feat_B[i]

    X[i,start+5:start+10]=hist_feat_G[i]

    X[i,start+10:start+15]=hist_feat_R[i]

    X[i,start+15:start+20]=hist_feat_B[i]



    

#######################################################################

########### Divide the dataset into 70%train and 30% test data########

######################################################################



#train test devide (70:30)

index=int(len(labels1)*0.7)



X_train=X[:index,:]

Y_train1=labels1[:index]



X_test=X[index:,:]

Y_test1=labels1[index:]

#pca = PCA(n_components=30, svd_solver='arpack')

#pca.fit(X_train)

#X_train=pca.transform(X_train)

#X_test=pca.transform(X_test)


#########################################################################

################### train a multiclass logistic regression ##############

#########################################################################



lr = LogisticRegression(multi_class='multinomial', solver='lbfgs',C=1, penalty='l2', fit_intercept=True, max_iter=5000, random_state=42)

lr.fit(X_train,Y_train1)



Y_testp_proba=lr.predict_proba(X_test)

Y_testp=lr.predict(X_test)

Y_trainp=lr.predict(X_train)

Y_trainp_proba=lr.predict_proba(X_train)

#percent accuracy

test_accuracy=np.sum(np.array(Y_test1)==Y_testp)/len(Y_testp)

train_accuracy=np.sum(np.array(Y_train1)==Y_trainp)/len(Y_trainp)

print('LR ',train_accuracy,test_accuracy)











#########################################################################

################### train a multiclass random forest ####################

#########################################################################





forest=RandomForestClassifier(n_estimators=100, max_depth=3,random_state=42)

forest.fit(X_train,Y_train1)



Y_testp_forest=forest.predict(X_test)

Y_trainp_forest=forest.predict(X_train)

Y_testp_proba_forest=forest.predict_proba(X_test)

Y_trainp_proba_forest=forest.predict_proba(X_train)



test_accuracy=np.sum(np.array(Y_test1)==Y_testp_forest)/len(Y_testp)

train_accuracy=np.sum(np.array(Y_train1)==Y_trainp_forest)/len(Y_trainp)

print('RF: ',train_accuracy,test_accuracy)



#########################################################################

################### train a multiclass gradient boosting classifier #####

#########################################################################





gb=GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, subsample=1, max_depth=2,  random_state=42 )

gb.fit(X_train,Y_train1)



Y_testp_gb=gb.predict(X_test)

Y_trainp_gb=gb.predict(X_train)

Y_testp_proba_gb=gb.predict_proba(X_test)

Y_trainp_proba_gb=gb.predict_proba(X_train)



test_accuracy=np.sum(np.array(Y_test1)==Y_testp_gb)/len(Y_testp)

train_accuracy=np.sum(np.array(Y_train1)==Y_trainp_gb)/len(Y_trainp)

print('GB ',train_accuracy,test_accuracy)







#########################################################################

################### train a multiclass gSVM classifier ###################

#########################################################################





svm1 = svm.SVC(gamma='scale',probability=True)

svm1.fit(X_train, Y_train1)

Y_testp_svm=svm1.predict(X_test)

Y_trainp_svm=svm1.predict(X_train)

Y_testp_proba_svm=svm1.predict_proba(X_test)

Y_trainp_proba_svm=svm1.predict_proba(X_train)



test_accuracy=np.sum(np.array(Y_test1)==Y_testp_svm)/len(Y_testp)

train_accuracy=np.sum(np.array(Y_train1)==Y_trainp_svm)/len(Y_trainp)

print('SVM ',train_accuracy,test_accuracy)



#########################################################

############### One hot encoder#########################

##########################################################



def one_hot_encode(Y,classes=None):

    if classes is None:

        classes=np.unique(Y)

    output=np.zeros((len(Y),len(classes)))

    for i,c in enumerate(classes):

        #print(i,c)

        output[Y==c,i]=1

        #print(np.sum(output))

    return output



############################################################################

########### the following function plots multiclass ROCs###################

############################################################################



#multiclass classification micro vs macro rocs

def multiclass_roc(y_test,y_score,n_classes,lw,classes):

    # Compute ROC curve and ROC area for each class

    fpr = dict()

    tpr = dict()

    roc_auc = dict()

    for i in range(n_classes):

        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])

    

    # Compute micro-average ROC curve and ROC area

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())

    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    

    # First aggregate all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    

    # Then interpolate all ROC curves at this points

    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):

        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    

    # Finally average it and compute AUC

    mean_tpr /= n_classes

    

    fpr["macro"] = all_fpr

    tpr["macro"] = mean_tpr

    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    

    plt.figure()

    plt.plot(fpr["micro"], tpr["micro"],

             label='micro-average ROC curve (area = {0:0.2f})'

                   ''.format(roc_auc["micro"]),

             color='deeppink', linestyle=':', linewidth=4)

    

    plt.plot(fpr["macro"], tpr["macro"],

             label='macro-average ROC curve (area = {0:0.2f})'

                   ''.format(roc_auc["macro"]),

             color='navy', linestyle=':', linewidth=4)

    

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    for i, color in zip(range(n_classes), colors):

        plt.plot(fpr[i], tpr[i], color=color, lw=lw,

                 label='ROC curve of class '+classes[i]+' (area = {1:0.2f})'

                 ''.format(i, roc_auc[i]))

    

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Extension of Receiver operating characteristic to multi-class')

    plt.legend(loc="lower right")

    plt.show()











#Logistic Regression

multiclass_roc(one_hot_encode(np.array(Y_test1),lr.classes_),Y_testp_proba,4,2,lr.classes_)

multiclass_roc(one_hot_encode(np.array(Y_train1),lr.classes_),Y_trainp_proba,4,2,lr.classes_)



#Random Forest

multiclass_roc(one_hot_encode(np.array(Y_test1),lr.classes_),Y_testp_proba_forest,4,2,forest.classes_)

multiclass_roc(one_hot_encode(np.array(Y_train1),lr.classes_),Y_trainp_proba_forest,4,2,forest.classes_)



#Gradient Boosting

multiclass_roc(one_hot_encode(np.array(Y_test1),lr.classes_),Y_testp_proba_gb,4,2,gb.classes_)

multiclass_roc(one_hot_encode(np.array(Y_train1),lr.classes_),Y_trainp_proba_gb,4,2,gb.classes_)



#SVM

multiclass_roc(one_hot_encode(np.array(Y_test1),lr.classes_),Y_testp_proba_svm,4,2,gb.classes_)

multiclass_roc(one_hot_encode(np.array(Y_train1),lr.classes_),Y_trainp_proba_svm,4,2,gb.classes_)



#all diseases

diseases=np.unique(labels1)  



for d in diseases:

    if d!='NL': # not for normal

        

        Y_train2=np.where(np.array(Y_train1)==d,d,'ALL') # relabel training data

        Y_test2=np.where(np.array(Y_test1)==d,d,'ALL')   #relabel test data

        

        #logistic regression

        

        lr = LogisticRegression(multi_class='multinomial', solver='lbfgs',C=1, penalty='l2', fit_intercept=True, max_iter=5000, random_state=42)

        lr.fit(X_train,Y_train2)

        Y_testp=lr.predict(X_test)

        Y_trainp=lr.predict(X_train)

        #percent accuracy

        test_accuracy=np.sum(np.array(Y_test2)==Y_testp)/len(Y_testp)

        print('LR '+d,': ',test_accuracy)



        #random forest

        

        forest=RandomForestClassifier(n_estimators=100, max_depth=3,random_state=42)

        forest.fit(X_train,Y_train2)        

        Y_testp_forest=forest.predict(X_test)        

        test_accuracy=np.sum(np.array(Y_test2)==Y_testp_forest)/len(Y_testp)

        print('RF '+d,': ',test_accuracy)

        



        #gradient boosting

        gb=GradientBoostingClassifier(learning_rate=0.015, n_estimators=100, subsample=1, max_depth=2,  random_state=42 )

        gb.fit(X_train,Y_train2)

        #models[d]=gb

        Y_testp=gb.predict(X_test)

        test_accuracy=np.sum(np.array(Y_test2)==Y_testp)/len(Y_testp)

        print('GB '+d,': ',test_accuracy)

        

       

        #SVM

        svm1 = svm.SVC(gamma='scale',probability=True)

        svm1.fit(X_train, Y_train2)

        Y_testp_svm=svm1.predict(X_test)

        

        test_accuracy=np.sum(np.array(Y_test2)==Y_testp_svm)/len(Y_testp)

        print('SVM '+d,': ',test_accuracy)

        

        print('.................')
