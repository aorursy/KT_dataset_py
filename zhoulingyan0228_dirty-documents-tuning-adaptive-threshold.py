import numpy as np 
import scipy as sp 
import pandas as pd
import matplotlib.pyplot as plt
import glob
import cv2
import sklearn
import seaborn as sns
from sklearn.metrics import mean_squared_error

train_imgs = glob.glob("../input/train/*.png")
train_imgs.sort()
train_cleaned_imgs = glob.glob("../input/train_cleaned/*.png")
train_cleaned_imgs.sort()
test_imgs= glob.glob("../input/test/*.png")
# Any results you write to the current directory are saved as output.
mean_rmses = []
windows = np.arange(5, 65, 6)
Cs = np.arange(0, 100, 10)
for i, window in enumerate(windows):
    mean_rmses.append([])
    for j, C in enumerate(Cs): 
        rmses = []
        for k, files  in enumerate(zip(train_imgs, train_cleaned_imgs)):
            #plt.figure(figsize=(30, 60))
            #plt.subplot(1,3,1)
            train_img = plt.imread(files[0])
            #_ = plt.imshow(train_img, cmap='gray')
            #plt.subplot(1,3,2)
            train_cleaned_img = plt.imread(files[1])
            #_ = plt.imshow(train_cleaned_img, cmap='gray')
            processed_img = cv2.adaptiveThreshold(np.multiply(train_img, 256).astype(np.uint8), 255., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                             cv2.THRESH_BINARY, window, C)
            #plt.subplot(1,3,3)
            #_ = plt.imshow(processed_img, cmap='gray')

            rmses.append(np.sqrt(mean_squared_error(processed_img/255., train_cleaned_img)))
        mean_rmses[i].append(np.mean(rmses))
plt.imshow(mean_rmses);
min_index = np.argmin(mean_rmses)
window_min = windows[min_index // Cs.size]
C_min = Cs[min_index % Cs.size]
rmse_min = mean_rmses[min_index // Cs.size][min_index % Cs.size]
print("window_min=", window_min, ", C_min=", C_min, ", rmse_min=", rmse_min)
for files  in zip(train_imgs[:5], train_cleaned_imgs[:5]):
    plt.figure(figsize=(20, 40))
    plt.subplot(1,3,1)
    train_img = plt.imread(files[0])
    _ = plt.imshow(train_img, cmap='gray')
    plt.subplot(1,3,2)
    train_cleaned_img = plt.imread(files[1])
    _ = plt.imshow(train_cleaned_img, cmap='gray')
    processed_img = cv2.adaptiveThreshold(np.multiply(train_img, 256).astype(np.uint8), 255., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, window_min, C_min)
    plt.subplot(1,3,3)
    _ = plt.imshow(processed_img, cmap='gray')
idColumn = []
valColumn = []
for file in test_imgs:
    test_img = plt.imread(file)
    id=file.replace('../input/test/', '').replace('.png', '')
    processed_img = cv2.adaptiveThreshold(np.multiply(test_img, 256).astype(np.uint8), 255., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, window_min, C_min) // 255
    for r in range(processed_img.shape[0]):
        for c in range(processed_img.shape[1]):
            idColumn.append(str(id)+'_'+str(r+1)+'_'+str(c+1))
            valColumn.append(processed_img[r][c])
pd.DataFrame({'id': idColumn, 'value': valColumn}).to_csv('submission.csv', index=False)
