import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import cv2
import time
## Remove All of Files in Directory
!rm -rf  /kaggle/working/*
## import thai-mnist .csv
th_mnist = pd.read_csv('../input/thai-mnist-classification/mnist.train.map.csv')
cats = th_mnist['category'].unique()
## Do Oversampling to fix imbalance in class
def ovsamp_by_class(data, class_col, concat = True):
    maxcat = data[class_col].value_counts().max()
    nclass = data[class_col].nunique()
    n_elements = data[class_col].value_counts()
    sampsize_class = maxcat - data[class_col].value_counts()
    sampsize = sampsize_class.to_dict()
    cats = data[class_col].unique()
    df_store = []
    
    for cat in cats:
        df = data[data[class_col] == cat].sample(sampsize[cat])
        df_store.append(df)

    data_sampled = pd.concat([data, pd.concat(df_store)])
    return data_sampled
    
    if concat == False:
        return df_store
th_mnist_ovsmp = ovsamp_by_class(th_mnist, 'category')
t0 = time.time()

from skimage.morphology import convex_hull_image
from skimage.util import invert

## Change Files Directory for use Tensorflow Image Generator.
path_load = "../input/thai-mnist-classification/train/"
path_save = "./"
n = 0
for cat in cats:
    img_fnames = th_mnist_ovsmp[th_mnist_ovsmp['category'] == cat]
    path_fn = path_load + img_fnames['id']
    path_fn = np.array(path_fn)
    folder = cat
    os.makedirs(os.path.join(path_save,"train",str(folder)))
    img_n = 0 
    
    ## Load Images
    for imagepath in path_fn: 
        img_n = img_n + 1 
        img = cv2.imread(str(imagepath))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        hull = convex_hull_image(invert(gray))
        hull_converted = hull.astype('uint8')*255
        x,y,w,h = cv2.boundingRect(hull_converted)
        cropped = gray[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (64,64))/255.
        blur = cv2.blur(resized, (3, 3))
        thresh = cv2.threshold(blur, 0.999, 1, cv2.THRESH_BINARY)[1]
        cv2.imwrite(os.path.join(path_save,"train",str(folder), f"{int(cat)}_{int(img_n):04d}.png"), thresh)
    n = n + 1
    print("category: ",n, "/", len(cats))
    
print("Times:", (time.time()-t0)/60, "minutes")
## Flow From Directory
import tensorflow as tf

### Image Loading Parameters
img_path = "./train"
batch_size = 32
img_height = 64
img_width = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(img_path,
                                                                validation_split = 0.2,
                                                                subset = "training",
                                                                seed = 123,
                                                                image_size = (img_height, img_width),
                                                                batch_size = batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(img_path,
                                                                validation_split = 0.2,
                                                                subset = "validation",
                                                                seed = 123,
                                                                image_size = (img_height, img_width),
                                                                batch_size = batch_size)
class_names = train_ds.class_names
class_names
## ResNet101V2 - epoch = 35, lr = 0.00001
resnet = tf.keras.applications.ResNet101V2(include_top = False, weights = 'imagenet', input_shape = (64, 64, 3),
                                                     pooling = 'max')
# fit output
x = tf.keras.layers.Flatten()(resnet.output)
x = tf.keras.layers.Dropout(rate = 0.8)(x)
x = tf.keras.layers.Dense(270, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.6)(x)
x = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(resnet.input, x)
model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.00001),metrics=['accuracy'])
model.summary()
## Fit Model
history = model.fit(train_ds, epochs = 35, validation_data = val_ds)
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# save the model to disk
filename = 'ResNet101V2_35ep_round1.h5'
model.save(filename)
## Import Library
from tensorflow.keras.models import load_model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.morphology import convex_hull_image
from skimage.util import invert
import os
import cv2
## Load Model
#model = load_model('../input/resnet101-50/ResNet101_50ep_round1.h5')
## Import Dataset2
trn_rules = pd.read_csv("../input/thai-mnist-classification/train.rules.csv")
tst_rules = pd.read_csv("../input/thai-mnist-classification/test.rules.csv")
## Keep file names
f1 = np.array(trn_rules[trn_rules['feature1'].notna()].feature1.to_list())
f2 = np.array(trn_rules[trn_rules['feature2'].notna()].feature2.to_list())
f3 = np.array(trn_rules[trn_rules['feature3'].notna()].feature3.to_list())
predict = np.array(trn_rules['predict'].to_list())

## Keep Index
f1_ind = trn_rules[trn_rules['feature1'].notna()].index
f2_ind = trn_rules[trn_rules['feature2'].notna()].index
f3_ind = trn_rules[trn_rules['feature3'].notna()].index
## Import Images to input in Model
from skimage.transform import resize
import time

t0 = time.time()
path = '../input/thai-mnist-classification/train'
num_predict = []

# Set Feature to Read and Change Images to Number
for f in [f1,f2,f3]: 
    n = 0
    f_img = []
    print('START!!')
    ## Read Images from each Feature
    for fn in f:
        f_path = os.path.join(path, fn)
        img = cv2.imread(str(f_path))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        hull = convex_hull_image(invert(gray))
        hull_converted = hull.astype('uint8')*255
        x,y,w,h = cv2.boundingRect(hull_converted)
        cropped = gray[y:y+h, x:x+w]
        resized = (cv2.resize(cropped, (64,64))/255.).astype('float32')
        blur = cv2.blur(resized, (3, 3))
        thresh = cv2.threshold(blur, 0.999, 1, cv2.THRESH_BINARY)[1]
        img = cv2.cvtColor(thresh, cv2.COLOR_BGRA2RGB)
        
        f_img.append(img)
        n = n +1
        if n%100 == 0:
            print('Image : ',n,'/',len(f))
    
    ## Convert image to number by ResNet101 Model
    f_img = np.array(f_img)
    f_number = model.predict(f_img)
    num_predict.append(f_number)
    print(len(f_number))
    print('feature: ',len(num_predict),'/3')
print("Times:", (time.time()-t0)/60, "minutes")
## Data Wrangling
### Encode predicting result
f1_num = num_predict[0].argmax(axis=1)
f2_num = num_predict[1].argmax(axis=1)
f3_num = num_predict[2].argmax(axis=1)

### Create DataFrame of number from Classification Model to Join together
f1_df = pd.DataFrame({'f1_index' : f1_ind,'f1_num' : f1_num}).set_index('f1_index')
f2_df = pd.DataFrame({'f2_index' : f2_ind,'f2_num' : f2_num}).set_index('f2_index')
f3_df = pd.DataFrame({'f3_index' : f3_ind,'f3_num' : f3_num}).set_index('f3_index')

### Join number from Classification Model together
trn_rules_num = trn_rules.join(f1_df).join(f2_df).join(f3_df)

### Select column to use
trn_rules_num  = trn_rules_num[['id','f1_num','f2_num','f3_num','predict']]

### Do Onehot Encode
trn_rules_num_clf = pd.get_dummies(trn_rules_num, columns = ['predict'])
# Random Forest Classification
## Import Library
t0 = time.time()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_absolute_error

## Set X2 and y2 for split
X2 = trn_rules_num_clf[['f1_num','f2_num','f3_num']].fillna(-99)
y2 = trn_rules_num_clf.drop(['id','f1_num','f2_num','f3_num'], axis = 1)

## Split train test
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.33, random_state=42)

## K-Folds Cross Validation
kf = KFold(n_splits=5, shuffle = True)

ntrees_set = [100,200,300,400,500]
mean_mae_bag = []

for ntrees in ntrees_set:
    mae = []
    n = 0
    print('ntrees: ',ntrees)
    for train_index, test_index in kf.split(X_train2):
        n = n + 1
        X_train_kf, X_val = X_train2.iloc[train_index,], X_train2.iloc[test_index,]
        y_train_kf, y_val = y_train2.iloc[train_index,], y_train2.iloc[test_index,]

        clf = RandomForestClassifier(n_estimators = ntrees)
        model2 = clf.fit(X_train_kf, y_train_kf)
        y_hat_kf = model2.predict(X_val)

        # Measure MAE
        mae.append(mean_absolute_error(y_val, y_hat_kf))
        print("Folds:",n,"/",kf.get_n_splits(X_train2),"ntrees:",ntrees,"MAE:",mean_absolute_error(y_val, y_hat_kf))
    mean_mae = np.array(mae).mean()
    mean_mae_bag.append(mean_mae)
    print("Average 5 Folds / ntrees:",ntrees,"MAE is", mean_mae)
    print("")

summary_df = pd.DataFrame({'ntrees':ntrees_set,'MAE':mean_mae_bag})
ntrees_use = summary_df[summary_df['MAE'] == summary_df['MAE'].min()]['ntrees']
print("Summary")
print(summary_df)

## Build Model
clf = RandomForestClassifier(n_estimators = ntrees_use.values[0])
model2 = clf.fit(X_train2, y_train2)
print("Times:", (time.time()-t0)/60, "minutes")
## Test Model2
y_hat2 = model2.predict(X_test2)
print('MAE:',mean_absolute_error(y_test2, y_hat2))
column_names = np.array(y_test2.columns)
## Set Columns Name
y_hat2_df = pd.DataFrame(y_hat2)
y_hat2_df.columns = column_names
y_hat2_df.columns = y_hat2_df.columns.str.replace('^predict_', '')
y_hat2_decoded = np.array(y_hat2_df.idxmax(1))

## y_test2 decode
y_test2_df = pd.DataFrame(y_test2)
y_test2_df.columns = column_names
y_test2_df.columns = y_test2_df.columns.str.replace('^predict_', '')
y_test2_decoded = np.array(y_test2_df.idxmax(1))

## Look!!
pd.DataFrame({'test': y_test2_decoded, 'y_hat2_decoded': y_hat2_decoded})
f1_tst = np.array(tst_rules[tst_rules['feature1'].notna()].feature1.to_list())
f2_tst = np.array(tst_rules[tst_rules['feature2'].notna()].feature2.to_list())
f3_tst = np.array(tst_rules[tst_rules['feature3'].notna()].feature3.to_list())
f1_tst_ind = tst_rules[tst_rules['feature1'].notna()].index
f2_tst_ind = tst_rules[tst_rules['feature2'].notna()].index
f3_tst_ind = tst_rules[tst_rules['feature3'].notna()].index
t0 = time.time()
path = '../input/thai-mnist-classification/test'
num_predict_tst = []

# Set Feature to Read and Change Images to Number
for f in [f1_tst,f2_tst,f3_tst]: 
    n = 0
    f_img = []
    print('START!!')
    ## Read Images from each Feature
    for fn in f:
        f_path = os.path.join(path, fn)
        img = cv2.imread(str(f_path))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        hull = convex_hull_image(invert(gray))
        hull_converted = hull.astype('uint8')*255
        x,y,w,h = cv2.boundingRect(hull_converted)
        cropped = gray[y:y+h, x:x+w]
        resized = (cv2.resize(cropped, (64,64))/255.).astype('float32')
        blur = cv2.blur(resized, (3, 3))
        thresh = cv2.threshold(blur, 0.999, 1, cv2.THRESH_BINARY)[1]
        img = cv2.cvtColor(thresh, cv2.COLOR_BGRA2RGB)
        
        f_img.append(img)
        n = n +1
        if n%100 == 0:
            print('Image : ',n,'/',len(f))
    
    ## Convert image to number by ResNet101 Model
    f_img = np.array(f_img)
    f_number = model.predict(f_img)
    num_predict_tst.append(f_number)
    print(len(f_number))
    print('feature: ',len(num_predict_tst),'/3')
print("Times:", (time.time()-t0)/60, "minutes")
## Map values to index
f1_tst_num = num_predict_tst[0].argmax(axis=1)
f2_tst_num = num_predict_tst[1].argmax(axis=1)
f3_tst_num = num_predict_tst[2].argmax(axis=1)
f1_tst_df = pd.DataFrame({'f1_index' : f1_tst_ind,'f1_num' : f1_tst_num}).set_index('f1_index')
f2_tst_df = pd.DataFrame({'f2_index' : f2_tst_ind,'f2_num' : f2_tst_num}).set_index('f2_index')
f3_tst_df = pd.DataFrame({'f3_index' : f3_tst_ind,'f3_num' : f3_tst_num}).set_index('f3_index')
tst_rules_num = tst_rules.join(f1_tst_df).join(f2_tst_df).join(f3_tst_df)
X_test_md2 = tst_rules_num[['f1_num','f2_num','f3_num']].fillna(-99)
predict_arr = model2.predict(X_test_md2)
## Set Columns Name
predict_df = pd.DataFrame(predict_arr)
predict_df.columns = column_names
predict_df.columns = predict_df.columns.str.replace('^predict_', '')
predict = np.array(predict_df.idxmax(1))
## Create Submission DataFrame
submission = pd.DataFrame({'id': tst_rules_num['id'],
                              'predict':predict})

## Export .csv file
submission.to_csv('submit_resnet101v2_35ep.csv', index = False)
import os
os.chdir(r'../working')
from IPython.display import FileLink
FileLink(r'submit_resnet101v2_35ep.csv')
## ResNet101 - epoch = 35, lr = 0.00001 - submit score = 2.64190
resnet = tf.keras.applications.ResNet101(include_top = False, weights = 'imagenet', input_shape = (64, 64, 3),
                                                     pooling = 'max')
# fit output
x = tf.keras.layers.Flatten()(resnet.output)
x = tf.keras.layers.Dropout(rate = 0.8)(x)
x = tf.keras.layers.Dense(270, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.6)(x)
x = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(resnet.input, x)
model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.00001),metrics=['accuracy'])
model.summary()