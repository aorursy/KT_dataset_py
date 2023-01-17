import os
import numpy as np
import pandas as pd
import pydicom as dicom
import matplotlib.pylab as plt
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
traindata = pd.read_csv("../input/melanoma-merged-external-data-512x512-jpeg/marking.csv")
traindata.rename({'image_id':'image_name'}, axis = 1, inplace = True)
testdata = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")
print(traindata.shape, testdata.shape)
traindata
testdata
print(traindata['sex'][traindata['source'] == 'SLATMD'].unique())
print(traindata['age_approx'][traindata['source'] == 'SLATMD'].unique())
####To extract info from .dcm format files
#for dirname, _, filenames in os.walk('/kaggle/input/siim-isic-melanoma-classification/train/'):
#    for filename in filenames:
#        imgpath = os.path.join(dirname, filename)
#        ds = dicom.dcmread(imgpath)
#        print(ds)
#        ds.pixel_array ##This converts the image in DICOM file into array of integers
####To generate own datasets.
#def preprocess_image(image_path, pxlsize):
#    im_array = Image.open(image_path)
#    im_array = im_array.resize((pxlsize, )*2, resample = Image.LANCZOS)
#    return im_array


#pxl = 512
#n = traindata.shape[0]
#X_train = np.empty((n, pxl, pxl, 3), dtype = np.uint8)
#fpath = '../input/siim-isic-melanoma-classification/jpeg/train/'
#
#for i, image_id in enumerate(tqdm(traindata['image_name'])):
#    X_train[i, :, :, :] = preprocess_image(fpath + image_id + '.jpg', pxl)
path128 = '../input/siimisic-melanoma-resized-images/x_train_128.npy'
X_train = np.load(path128)

Y_train = np.array(traindata.target)
Y_train = Y_train.reshape((Y_train.shape[0], 1))

path128 = '../input/siimisic-melanoma-resized-images/x_test_128.npy'
X_test = np.load(path128)
np.equal(Y_train[:, 0], traindata.target).all()
plt.imshow(X_train[0,:,:,:]);
traindata.info()
def printf(col):
    print(col, traindata[col].unique(), "\nlength: ", len(traindata[col].unique()), "\n")
printf("image_name")
printf("patient_id")
printf("sex")
printf("age_approx")
printf("anatom_site_general_challenge")
printf("target")
t1 = sum(traindata['target'] == 1)
t0 = sum(traindata['target'] == 0)
print("Number of examples for:", "\nTarget = 1: ", t1, "\nTarget = 0: ", t0, "\nRatio", round(t1/(t1 + t0), 3))
figure, (subp1, subp2, subp3) = plt.subplots(1, 3, figsize = (10,3));
subp1.hist(traindata.age_approx, bins = 50);
subp2.hist(traindata['age_approx'][traindata['target'] == 1], bins = 50);
subp3.hist(traindata['age_approx'][traindata['target'] == 0], bins = 50);
figure.tight_layout(pad = 3);

subp1.set_title("Overall age distribution");
subp2.set_title("Age distribution for target=1");
subp3.set_title("Age distribution for target=0");
sum(traindata['age_approx'] == 0)
traindata['age_group'] = pd.cut(traindata['age_approx'], 4)
agetarget = traindata[['age_group', 'target']].groupby('age_group',as_index = False).sum().sort_values(
                                                                        by = 'target', ascending = True)

print(agetarget)
agetarget = np.array(agetarget)
agecount = traindata.age_group.value_counts().sort_values(ascending = True)
print(agecount)

agecount = np.array(agecount)
traindata = traindata.drop('age_group', axis = 1)
sitecount = traindata.anatom_site_general_challenge.value_counts().sort_values(ascending = True)
print(sitecount)
sitecount = np.array(sitecount)
sitetarget = traindata[['anatom_site_general_challenge', 'target']].groupby(['anatom_site_general_challenge'],
                                    as_index = False).sum().sort_values(by = 'target', ascending = True)

print(sitetarget)
sitetarget = np.array(sitetarget.target)
print(np.around(np.divide(sitetarget, sitecount)*100, 3))
traindata.sex.value_counts()
gender = traindata[['sex', 'target']].groupby(['sex'], as_index = False).sum()
print(gender, '\nTotal sum: ', gender.target.sum())
corr_with_id = traindata[['patient_id', 'target']].groupby(['patient_id'], as_index = False)
print(corr_with_id.sum().target.unique())
corr_with_id_sum = corr_with_id.sum()
idlist_multiple = []
for i in range(len(corr_with_id)):
    if corr_with_id_sum.target[i] > 1:
        idlist_multiple.append(corr_with_id_sum.patient_id[i])
len(idlist_multiple)
idlist_mean = []
corr_with_id_mean = corr_with_id.mean()

for i in range(len(corr_with_id_mean)):
    if ((corr_with_id_mean.target[i] > 0) & (corr_with_id_mean.target[i] < 1)):
        idlist_mean.append(corr_with_id_mean.patient_id[i])
len(idlist_mean)
train_df = traindata.copy()
test_df = testdata.copy()
train_df.isna().sum()
testdata.isna().sum()
age_median = train_df['age_approx'].median()
train_df['age_approx'].fillna(age_median, inplace = True)

train_df.fillna('unknown', inplace = True)
test_df.fillna('unknown', inplace = True)
train_df['agesquare'] = train_df['age_approx']**2
test_df['agesquare'] = test_df['age_approx']**2
scaler = StandardScaler(copy = False)
scaler.fit(train_df[['age_approx', 'agesquare']])

trainage = scaler.transform(train_df[['age_approx', 'agesquare']])
testage = scaler.transform(test_df[['age_approx', 'agesquare']])

train_df.rename({'age_approx':'age'}, axis = 1, inplace = True)
test_df.rename({'age_approx':'age'}, axis = 1, inplace = True)

train_df['age'] = trainage[:, 0]
train_df['agesquare'] = trainage[:, 1]
test_df['age'] = testage[:, 0]
test_df['agesquare']= testage[:, 1]
trainage[:, 0].shape
train_df
"""
data = [train_df]
for dataset in data:
    dataset.loc[train_df['age_approx'] <= 20, 'age_approx'] = 0
    dataset.loc[(train_df['age_approx'] > 20) & (train_df['age_approx'] <= 40), 'age_approx'] = 1
    dataset.loc[(train_df['age_approx'] > 40) & (train_df['age_approx'] <= 60), 'age_approx'] = 2
    dataset.loc[train_df['age_approx'] > 60, 'age_approx'] = 3
    
data = [test_df]
for dataset in data:
    dataset.loc[test_df['age_approx'] <= 20, 'age_approx'] = 0
    dataset.loc[(test_df['age_approx'] > 20) & (test_df['age_approx'] <= 40), 'age_approx'] = 1
    dataset.loc[(test_df['age_approx'] > 40) & (test_df['age_approx'] <= 60), 'age_approx'] = 2
    dataset.loc[test_df['age_approx'] > 60, 'age_approx'] = 3
"""
train_df = pd.concat([train_df, pd.get_dummies(train_df['sex'], prefix = 'sex')], axis = 1)
#train_df = pd.concat([train_df, pd.get_dummies(train_df['age_approx'], prefix = 'age')], axis = 1)
#train_df = pd.concat([train_df, pd.get_dummies(train_df['anatom_site_general_challenge'], prefix = 'site')], axis = 1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['sex'], prefix = 'sex')], axis = 1)
#test_df = pd.concat([test_df, pd.get_dummies(test_df['age_approx'], prefix = 'age')], axis = 1)
#test_df = pd.concat([test_df, pd.get_dummies(test_df['anatom_site_general_challenge'], prefix = 'site')], axis = 1)
train_df = train_df.drop(['sex', 'anatom_site_general_challenge', 'patient_id', 'sex_unknown'], axis = 1)
test_df = test_df.drop(['sex', 'anatom_site_general_challenge', 'patient_id'], axis = 1)
train_df
test_df
X_train_df = train_df.drop('target', axis = 1)
Y_train_df = train_df.target
print('Training Data:\nTabular (X_train_df), (Y_train_df): ', X_train_df.shape, Y_train_df.shape, '\nImages (X_train),(Y_train): ', X_train.shape, Y_train.shape)
print('\n\nTest Data:\nTabular (test_df): ', test_df.shape, '\nImages (X_test): ', X_test.shape)
train_df.to_csv('trainingwithexternaldata-melanoma-2020.csv', index = False)
test_df.to_csv('test-melanoma-2020.csv', index = False)