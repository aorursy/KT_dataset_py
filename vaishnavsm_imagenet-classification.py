import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.metrics import accuracy_score
import os
import scipy.io
import math

from sklearn.utils import shuffle

from PIL import Image
import requests
from io import BytesIO

from tensorflow.keras.applications import resnet50
from keras.preprocessing import image

import matplotlib.pyplot as plt

import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions

import tensorflow as tf
from keras.preprocessing import image

from sklearn.model_selection import train_test_split

from scipy import spatial
from tqdm import tqdm

import gc
resnet50_model = resnet50.ResNet50(weights='imagenet')
styles = pd.read_csv('/kaggle/input/fashion-product-images-dataset/fashion-dataset/styles.csv', error_bad_lines=False)

shirts = styles[styles['articleType'].isin(['Shirts'])]
tshirts = styles[styles['articleType'].isin(['Tshirts'])]
pants =  styles[styles['articleType'].isin(['Track Pants','Shorts', 'Trunk', 'Trousers', 'Track Pants', 'Tights', 'Lounge Pants', 'Lounge Shorts', 'Leggings', 'Jeans', 'Jeggings'])]
# np.unique(styles['articleType'])
shirts, tshirts, pants = shirts['id'].to_numpy(), tshirts['id'].to_numpy(), pants['id'].to_numpy()
shirts.shape, tshirts.shape, pants.shape
image_path = '/kaggle/input/fashion-product-images-dataset/fashion-dataset/images/'

IMG_SIZE = 224
LIMIT_IMAGES = 2000
NUM_OUTPUTS = 3
def load_imgs(names):
    imgs = []
    for i, image_name in enumerate(tqdm(names)):
#         if i% 50 == 0 :
#             print(f"Loading Image {i}")
        try:
            img = image.load_img(f'{image_path}{image_name}.jpg', target_size=(IMG_SIZE, IMG_SIZE))
        except:
            img = None
        if img is None:
            continue
        img = np.array(img)
        imgs.append(img)
    return np.array(imgs)
print("Loading Images...")
print("Shirts")
shirt_images = load_imgs(shirts[:LIMIT_IMAGES])
gc.collect()
print("TShirts")
tshirt_images = load_imgs(tshirts[:LIMIT_IMAGES])
gc.collect()
print("Pants")
pant_images = load_imgs(pants[:LIMIT_IMAGES])
gc.collect()
print("Done")
shirt_images.shape, tshirt_images.shape, pant_images.shape
train_shirt_images, test_shirt_images, _, _ = train_test_split(shirt_images, np.repeat(0, shirt_images.shape[0]), test_size = 0.2)
train_shirt_images.shape, test_shirt_images.shape
train_tshirt_images, test_tshirt_images, _, _ = train_test_split(tshirt_images, np.repeat(0, tshirt_images.shape[0]), test_size = 0.2)
train_tshirt_images.shape, test_tshirt_images.shape
train_pant_images, test_pant_images, _, _ = train_test_split(pant_images, np.repeat(0, pant_images.shape[0]), test_size = 0.2)
train_pant_images.shape, test_pant_images.shape
gc.collect()
def get_vectors(imgs):
    processed_batch = preprocess_input(imgs, mode="caffe")
    return resnet50_model.predict(processed_batch)

def get_average_vector(imgs):
    vectors = get_vectors(imgs)
    print(vectors.shape)
    return np.mean(vectors, axis=0)

def closeness(a, b):
#     print(a.shape)
#     print(b.shape)
    return 1 - spatial.distance.cosine(a, b)

def closest(vector, compared_to):
    best = -5
    best_idx = -1
#     print(compared_to.shape)
    for i, cmp in enumerate(compared_to):
        c = closeness(vector, cmp)
        if c > best:
            best_idx = i
            best = c
    return best_idx, best

def b_closest(vectors, compared_to):
    return np.array([closest(vector, compared_to)[0] for vector in vectors])
# Get Test And Train Sets
train_X = np.concatenate((train_shirt_images, train_tshirt_images, train_pant_images), axis = 0)
train_Y = np.repeat((0, 1, 2), (train_shirt_images.shape[0], train_tshirt_images.shape[0], train_pant_images.shape[0]), axis = 0)

train_vecs = get_vectors(train_X)
gc.collect()
test_X = np.concatenate((test_shirt_images, test_tshirt_images, test_pant_images), axis = 0)
test_Y = np.repeat((0, 1, 2), (test_shirt_images.shape[0], test_tshirt_images.shape[0], test_pant_images.shape[0]), axis = 0)

test_vecs = get_vectors(test_X)
gc.collect()
from sklearn.metrics import confusion_matrix, classification_report
shirt_vector = get_average_vector(train_shirt_images)
tshirt_vector = get_average_vector(train_tshirt_images)
pant_vector = get_average_vector(train_pant_images)
gc.collect()
pred_classes = decode_predictions(np.expand_dims(shirt_vector, axis=0), top=3)
pred_classes
pred_classes = decode_predictions(np.expand_dims(tshirt_vector, axis=0), top=3)
pred_classes
pred_classes = decode_predictions(np.expand_dims(pant_vector, axis=0), top=3)
pred_classes
test_vector = np.array([shirt_vector, tshirt_vector, pant_vector])

train_predictions = b_closest(train_vecs, test_vector)

train_accuracy = accuracy_score(train_Y, train_predictions)
print(f"In Sample Accuracy: {train_accuracy}")

test_predictions = b_closest(test_vecs, test_vector)

test_accuracy = accuracy_score(test_Y, test_predictions)
print(f"Out Of Sample Accuracy: {test_accuracy}")
gc.collect()
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

trainx, trainy = shuffle(train_vecs, train_Y)
# trainx = np.expand_dims(trainx, axis=1)
trainx.shape, trainy.shape
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim=1000, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)
classify_model = estimator.fit(trainx, trainy)
preds = estimator.predict(train_vecs)
acc = accuracy_score(train_Y, preds)
print(f"In sample Accuracy: {acc}")
preds = estimator.predict(test_vecs)
acc = accuracy_score(test_Y, preds)
print(f"Out of sample Accuracy: {acc}")
estimator.model.save('/kaggle/working/keras-vlarge-963.h5')
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
abc = AdaBoostClassifier(n_estimators=400, learning_rate=1)
abcm = abc.fit(trainx, trainy)
preds = abc.predict(train_vecs)
acc = accuracy_score(train_Y, preds)
print(f"In sample Accuracy: {acc}")

print(classification_report(train_Y, preds))

preds = abc.predict(test_vecs)
acc = accuracy_score(test_Y, preds)
print(f"Out of sample Accuracy: {acc}")

print(classification_report(test_Y, preds))
import pickle
pickle.dump(abc, open('/kaggle/working/ada-vvlarge-964.pickle', 'wb'))
gc.collect()
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
trainx, trainy = shuffle(train_vecs, train_Y)
trainx.shape, trainy.shape
tx, vx, ty, vy = train_test_split(trainx, trainy, test_size = 0.2)
testx, testy = test_vecs.copy(), test_Y.copy()
tx.shape, ty.shape, vx.shape, vy.shape, testx.shape, testy.shape
scaler = MinMaxScaler()
tx = scaler.fit_transform(tx)
vx = scaler.transform(vx)
testx = scaler.transform(testx)

tx.shape, ty.shape, vx.shape, vy.shape, testx.shape, testy.shape
EPOCHS = 300
BATCH_SIZE = 100
LEARNING_RATE = 0.00007
NUM_FEATURES = 1000
NUM_CLASSES = NUM_OUTPUTS
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
gc.collect()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch_model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
torch_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(torch_model.parameters(), lr=LEARNING_RATE)
print(torch_model)
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc) * 100
    
    return acc
accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}
class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_dataset = ClassifierDataset(torch.from_numpy(tx).float(), torch.from_numpy(ty).long())
val_dataset = ClassifierDataset(torch.from_numpy(vx).float(), torch.from_numpy(vy).long())
test_dataset = ClassifierDataset(torch.from_numpy(testx).float(), torch.from_numpy(testy).long())
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)
print("Begin training.")
for e in tqdm(range(1, EPOCHS+1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    torch_model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = torch_model(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        torch_model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = torch_model(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
            loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
                              
    
    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
y_pred_list = []
with torch.no_grad():
    torch_model.eval()
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = torch_model(X_batch)
        y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
print(classification_report(testy, y_pred_list))
pickle.dump(scaler, open('/kaggle/working/torch-vvlarge.scaler.pickle', 'wb'))
torch.save(torch_model.state_dict(), '/kaggle/working/torch-vvlarge.dict')
gc.collect()
trainx2, trainy2 = shuffle(train_vecs, train_Y)
trainx2.shape, trainy2.shape
tx2, vx2, ty2, vy2 = train_test_split(trainx, trainy, test_size = 0.2)
testx2, testy2 = test_vecs.copy(), test_Y.copy()
tx2.shape, ty2.shape, vx2.shape, vy2.shape, testx2.shape, testy2.shape
scaler = MinMaxScaler()
tx2 = scaler.fit_transform(tx2)
vx2 = scaler.transform(vx2)
testx2 = scaler.transform(testx2)

tx2.shape, ty2.shape, vx2.shape, vy2.shape, testx2.shape, testy2.shape
EPOCHS = 300
BATCH_SIZE = 100
LEARNING_RATE = 0.00007
NUM_FEATURES = 1000
NUM_CLASSES = NUM_OUTPUTS
gc.collect()
class MulticlassClassification2(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification2, self).__init__()
        
        self.layer_1 = nn.Linear(num_feature, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 256)
        self.layer_4 = nn.Linear(256, 64)
        self.layer_out = nn.Linear(64, num_class) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)
        self.batchnorm4 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch_model2 = MulticlassClassification2(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
torch_model2.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(torch_model2.parameters(), lr=LEARNING_RATE)
print(torch_model2)



def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc) * 100
    
    return acc

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}
class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_dataset2 = ClassifierDataset(torch.from_numpy(tx2).float(), torch.from_numpy(ty2).long())
val_dataset2 = ClassifierDataset(torch.from_numpy(vx2).float(), torch.from_numpy(vy2).long())
test_dataset2 = ClassifierDataset(torch.from_numpy(testx2).float(), torch.from_numpy(testy2).long())

train_loader2 = DataLoader(dataset=train_dataset2,
                          batch_size=BATCH_SIZE
)
val_loader2 = DataLoader(dataset=val_dataset2, batch_size=1)
test_loader2 = DataLoader(dataset=test_dataset2, batch_size=1)
print("Begin training.")
for e in tqdm(range(1, EPOCHS+1)):
    gc.collect()
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    torch_model2.train()
    for X_train_batch, y_train_batch in train_loader2:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()
        
        y_train_pred = torch_model2(X_train_batch)
        
        train_loss = criterion(y_train_pred, y_train_batch)
        train_acc = multi_acc(y_train_pred, y_train_batch)
        
        train_loss.backward()
        optimizer.step()
        
        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()
        
        
    # VALIDATION    
    with torch.no_grad():
        
        val_epoch_loss = 0
        val_epoch_acc = 0
        
        torch_model2.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            
            y_val_pred = torch_model2(X_val_batch)
                        
            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)
            
            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
            loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))
                              
    gc.collect()
    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
y_pred_list = []
with torch.no_grad():
    torch_model2.eval()
    for X_batch, _ in test_loader2:
        X_batch = X_batch.to(device)
        y_test_pred = torch_model2(X_batch)
        y_pred_softmax = torch.log_softmax(y_test_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)
        y_pred_list.append(y_pred_tags.cpu().numpy())
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

print(classification_report(testy, y_pred_list))
import pickle
pickle.dump(scaler, open('/kaggle/working/torch-2-vlarge.scaler.pickle', 'wb'))
torch.save(torch_model2.state_dict(), '/kaggle/working/torch-2-vlarge.dict')
gc.collect()
def load_img(names):
    print(names)
    imgs = []
    for i, image_name in enumerate(names):
        if i% 50 == 0 :
            print(f"Loading Image {i}")
        img = image.load_img(f'/kaggle/input/vernacular-set/{image_name}.jpeg', target_size=(IMG_SIZE, IMG_SIZE))
        if img is None:
            continue
        img = np.array(img)
        imgs.append(img)
    return np.array(imgs)
pred_imgs = load_img(list(range(1,17)))
pred_imgs.shape
gc.collect()
pred_vecs = get_vectors(pred_imgs)
preds = abc.predict(pred_vecs)
map_to_names = lambda x: map(lambda y: "Shirt" if y == 0 else "T-Shirt" if y == 1 else "Pant", x)
pred_results = list(zip(map_to_names([0,1,0,0,0,0,1,0,1,1,1,0,2,2,0,0]), map_to_names(preds)))
print(pred_results)
for result, image in zip(pred_results, pred_imgs):
    plt.figure()
    plt.title(f"Actual: {result[0]} | Predicted: {result[1]}")
    plt.imshow(image)
    
gc.collect()

