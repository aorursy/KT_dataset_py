import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt

from PIL import Image
F_TRAN='../input/plant-pathology-2020-fgvc7/train.csv'
F_TEST='../input/plant-pathology-2020-fgvc7/test.csv'

CAH='healthy'
CAM='multiple_diseases'
CAR='rust'
CAS='scab'

CATEGORY_LIST=['healthy','multiple_diseases','rust','scab']

NROW=5
NCOL=4
NSIZE=14
# load data from csv file of training and test 
# for training
data_tran_pd=pd.read_csv(F_TRAN)
print(data_tran_pd.shape)

data_tran_pd.head()
# for test 
data_test_pd=pd.read_csv(F_TEST)
print(data_test_pd.shape)

data_test_pd.head()
# transform dummy into categorycal variable and add them to training data
for category in CATEGORY_LIST: 
   data_tran_pd.loc[data_tran_pd[category]==1,'category']=category

data_tran_pd.head()
# show histgram for category
plt.figure()
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.grid(linestyle='dashed')
data_tran_pd['category'].hist(alpha=0.7,color='green')
# sort randomly data of training and test 
data_tran_pd=data_tran_pd.sample(frac=1).reset_index(drop=True)
data_test_pd=data_test_pd.sample(frac=1).reset_index(drop=True)
data_cah_pd=data_tran_pd[data_tran_pd['category']==CAH]
data_cam_pd=data_tran_pd[data_tran_pd['category']==CAM]
data_car_pd=data_tran_pd[data_tran_pd['category']==CAR]
data_cas_pd=data_tran_pd[data_tran_pd['category']==CAS]

print(data_cah_pd.shape)
print(data_cam_pd.shape)
print(data_car_pd.shape)
print(data_cas_pd.shape)
# view randomly 20 pictures of trainng and test 
# for training
num=1

plt.figure(figsize=(NSIZE,NSIZE))
for id,category in zip(data_tran_pd['image_id'],data_tran_pd['category']): 
   if num <= NROW*NCOL:
      fjpg='../input/plant-pathology-2020-fgvc7/images/'+id+'.jpg'    
      plt.subplot(NROW,NCOL,num)
      img=Image.open(fjpg,"r")
      plt.imshow(np.array(img))
      plt.xticks([])
      plt.yticks([])
      plt.title(category)
      plt.xlabel(id+'.jpg')        
   else: 
      break 
   num+=1
# for test
num=1

plt.figure(figsize=(NSIZE,NSIZE))
for id in data_test_pd['image_id']: 
   if num <= NROW*NCOL:
      fjpg='../input/plant-pathology-2020-fgvc7/images/'+id+'.jpg'    
      plt.subplot(NROW,NCOL,num)
      img=Image.open(fjpg,"r")
      plt.imshow(np.array(img))
      plt.xticks([])
      plt.yticks([])
      plt.xlabel(id+'.jpg')        
   else: 
      break 
   num+=1