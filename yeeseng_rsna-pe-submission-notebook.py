import numpy as np, pandas as pd, os

import matplotlib.pyplot as plt

import glob

import datetime

import torch

import torchvision.transforms as transforms

import pydicom

from pydicom import dcmread

from tqdm import tqdm



startTime = datetime.datetime.now()
# Codes from this cell are adopted from Quadcore/Richard Epstein public notebook

# This notebook loads GDCM without Internet access.

# GDCM is needed to read some DICOM compressed images.

# Once you run a notebook and get the GDCM error, you must restart that Kernel to read the files, even if you load the GDCM software.

# Note that you do not "import GDCM". You just "import pydicom".

# The Dataset (gdcm-conda-install) was provided by Ronaldo S.A. Batista. Definitely deserves an upvote!



!cp ../input/gdcm-conda-install/gdcm.tar .

!tar -xvzf gdcm.tar

!conda install --offline ./gdcm/gdcm-2.8.9-py37h71b2a6d_0.tar.bz2



print("GDCM installed.")



import pydicom
testDataDF = pd.read_csv('../input/rsna-str-pulmonary-embolism-detection/test.csv', dtype={'StudyInstanceUID':'string', 'SeriesInstanceUID':'string', 'SOPInstanceUID':'string'})

testDataDF = testDataDF.set_index('SOPInstanceUID')
testDataDF.head()
listOfStudyID = testDataDF['StudyInstanceUID'].unique()

print(len(listOfStudyID))
# Sanity Check

#thisStudyDF.head()

#print(len(thisStudyDF))



#thisImageIDlist = thisStudyDF.index.to_list()

#for eachItem in thisStudyDF.index:

#    print(type(eachItem))
def window(img, WL=50, WW=350):

    upper, lower = WL+WW//2, WL-WW//2

    X = np.clip(img.copy(), lower, upper)

    X = X - np.min(X)

    X = X / np.max(X)

    X = (X*255.0).astype('uint8')

    return X



data_transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(mean=[0.485, 0.456, 0.406],

                             std=[0.229, 0.224, 0.225])

    ])
'''

model_Path = '../input/firstbaselinemodel/baseMod4.pth' 

baseModel = torch.load(model_Path) 

baseModel.eval();

'''
#scoreDF = pd.DataFrame(columns=['id','label'])

#scoreDF = scoreDF.set_index('id')



f = open('submission.csv', 'w')

f.write('id,label\n')



with torch.no_grad():



    for eachStudyID in tqdm(listOfStudyID):

        

        thisStudyDF = testDataDF[testDataDF['StudyInstanceUID']==eachStudyID]

        

        for eachImageID in thisStudyDF.index:

            

            '''

            try:

                eachImagePath = '../input/rsna-str-pulmonary-embolism-detection/test/'+testDataDF.loc[eachImageID, 'StudyInstanceUID']+'/'+testDataDF.loc[eachImageID, 'SeriesInstanceUID']+'/'+eachImageID+'.dcm'

                dcm_data = dcmread(eachImagePath)

                image = dcm_data.pixel_array * int(dcm_data.RescaleSlope) + int(dcm_data.RescaleIntercept)

                image = np.stack([window(image, WL=-600, WW=1500),

                                  window(image, WL=40, WW=400),

                                  window(image, WL=100, WW=700)], 2)



                image = image.astype(np.float32)

                image = data_transform(image)

                toPred = image.unsqueeze(0).cuda()

                z = baseModel(toPred)

                pred = torch.sigmoid(z)

                pred = pred.cpu().detach().numpy().astype('float32')[0,0]

            except:

                pred = defaultScore['_pe_present_on_image']

            '''

            

            #scoreDF.loc[imageID, 'label'] = 0.5

            f.write(eachImageID+',0.5\n')

            

        # Study level labels

        listOfMetricLabels = ['_negative_exam_for_pe', '_rv_lv_ratio_gte_1', '_rv_lv_ratio_lt_1', '_leftsided_pe', '_chronic_pe', '_rightsided_pe', '_acute_and_chronic_pe', '_central_pe', '_indeterminate']



        for eachMetric in listOfMetricLabels:

            #scoreDF.loc[studyID+eachMetric, 'label'] = 0.5

            f.write(eachStudyID+eachMetric+',0.5\n')

            

f.close()



#print("totalEntries",len(scoreDF))

#scoreDF.to_csv('submission.csv', index=True)



print('finish')