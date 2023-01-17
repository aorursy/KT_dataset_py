import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
import matplotlib.pyplot as plt # 画图常用库
data_dir = "../input/"
# a function to load data and show some information of data
def load_data(data_dir, data_rows):
    data = pd.read_csv(data_dir + "train.csv", header = 0, sep = ',')
    print(data.head())
    print(data.shape)
    x_train = data.values[0:data_rows, 1:]
    y_train = data.values[0:data_rows, 0]
    x_test = pd.read_csv(data_dir + "test.csv", header = 0, sep = ',').values
    print(x_test.shape)
    return x_train, y_train, x_test

data_rows = 5000
origin_x_train, origin_y_train, origin_x_test = load_data(data_dir, data_rows)

# for origin test data, we just take first 300 rows
test_rows = 300
origin_x_test = origin_x_test[0: test_rows]
print(origin_x_train.shape, origin_y_train.shape, origin_x_test.shape)
# given index, show the number picture
pic_index = 222
print("label = {}".format(origin_y_train[pic_index]))
plt.imshow(origin_x_train[pic_index].reshape((28, 28)))
plt.axis("off")
plt.show()

# now we show more pics in rows 
rows = 6
labels = [str(i) for i in range(10)]
print(labels)
for index, label in enumerate(labels):
    indices = np.nonzero([i == index for i in origin_y_train])[0]
    indices = np.random.choice(indices, rows)
    for i, v in enumerate(indices):
        id = i * len(labels) + index + 1
        plt.subplot(rows, len(labels), id)
        plt.imshow(origin_x_train[v].reshape((28, 28)))
        plt.axis("off")
        if i == 0:
            plt.title(label)

plt.show()
from sklearn.model_selection import train_test_split
x_train, x_vali, y_train, y_vali = train_test_split(origin_x_train, origin_y_train, test_size = 0.2, random_state = 0)
print(x_train.shape, x_vali.shape, y_train.shape, y_vali.shape)
import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier

# we run test on k from 1 to 8
k_range = range(1, 9)
scores = list()
for k in k_range:
    start = time.time()
    print("k = {} now start...".format(k))
    
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_vali)
    
    accuracy = accuracy_score(y_vali, y_pred)
    scores.append(accuracy)
    print(confusion_matrix(y_vali, y_pred))
    print(classification_report(y_vali, y_pred))
    
    end = time.time()
    print("k = {} now end, time spent = {}".format(k, end - start))

print(scores)
plt.title("Accuracy on k")
plt.plot(k_range, scores)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()

sorted_scores_indices = np.array(scores).argsort()
best_accuracy = scores[sorted_scores_indices[-1]]
best_k = sorted_scores_indices[-1] + 1
print("best accuracy = {}, best k = {}".format(best_accuracy, best_k))
start = time.time()
knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(origin_x_train, origin_y_train)
final_y_pred = knn.predict(origin_x_test)
end = time.time()
print("calculations finished, time spent = {}".format(end - start))
print(final_y_pred)
# pick an index within 300, and see the picture
index = 66
print("prediction = {}".format(final_y_pred[index]))
plt.imshow(origin_x_test[index].reshape((28, 28)))
plt.axis("off")
plt.show()
df = pd.DataFrame({"ImageId": range(1, len(final_y_pred) + 1), "Label":final_y_pred})
print(df.head())
df.to_csv("Digit_Recognizer_Result.csv", header = True, index = False)