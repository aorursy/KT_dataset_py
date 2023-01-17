import time
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
data_dir = '../input/'

# load csv file
def load_data(row_nums):
    train = pd.read_csv(data_dir + 'train.csv').values
    x_test = pd.read_csv(data_dir + 'test.csv').values

    x_train = train[:row_nums, 1:]
    y_train = train[:row_nums, 0]
    return x_train, y_train, x_test

Origin_x_train, Origin_y_train, Origin_x_test = load_data(1500)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num = 5

for y, cls in enumerate(classes):
    # select all labels equal to current class
    idxs = np.nonzero([i == y for i in Origin_y_train])
    idxs = np.random.choice(idxs[0], num)
    for i, idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(num, len(classes), plt_idx)
        pixels = Origin_x_train[idx].reshape((28,28))
        plt.imshow(pixels)
        plt.axis("off")
        if i == 0:
            plt.title(cls)

plt.show()
x_train, x_vali, y_train, y_vali = train_test_split(
    Origin_x_train, 
    Origin_y_train, 
    test_size = 0.2, 
    random_state = 0)

print(x_train.shape, x_vali.shape, y_train.shape, y_vali.shape)
k_range = range(1,10)
accuracies = []

for k in k_range:
    print('k = {} classifier begins:'.format(k))
    # get start timestamp
    start_time = time.time() 
    
    # create a knn object:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_vali)
    
    # get the accuracy
    accuracy = accuracy_score(y_vali, y_pred)
    accuracies.append(accuracy)
    
    # get end timestamp
    end_time = time.time()
    
    print('Running time: {} sec'.format(start_time - end_time))
print(accuracies)
plt.plot(k_range, accuracies)
plt.xlabel('k value')
plt.ylabel('accuracy')
plt.show()
best_k = np.argmax(accuracies) + 1

knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(Origin_x_train, Origin_y_train)
y_pred = knn.predict(Origin_x_test)
# test the final result
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
num = 10

for y, cls in enumerate(classes):
    # select all labels equal to current class
    idxs = np.nonzero([i == y for i in y_pred])

    # random select one
    idxs = np.random.choice(idxs[0], num)
    for i, idx in enumerate(idxs):
        plt_idx = i * len(classes) + y + 1
        plt.subplot(num, len(classes), plt_idx)
        pixels = Origin_x_test[idx].reshape((28,28))
        plt.imshow(pixels)
        plt.axis("off")
        if i == 0:
            plt.title(cls)

plt.show()
# output the prediction result
pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),
              "Label": y_pred}).to_csv('./Digit_Recogniser_Result.csv', 
                                       index=False,
                                       header=True)