import pandas as pd

import matplotlib.pyplot as plt

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.losses import MeanSquaredError

from keras.metrics import Recall,Precision,Accuracy
train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")

sample_submission = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
test = test.drop(['ID'], axis = 1)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(train.iloc[:,2:-2],train.iloc[:,-1],test_size=.33, random_state = 101)



X_train
model = Sequential()

model.add(Dense(10, input_dim = 2, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(

    optimizer='adam',

    loss='mean_squared_error',

    metrics=[

        Accuracy(),

        Precision(),

        Recall()

    ]

)



history = model.fit(

    X_train, 

    y_train, 

    validation_data = (X_test, y_test), 

    epochs = 2, 

    batch_size = 8

)
fig, ax = plt.subplots()

ax.plot(history.history['accuracy'], label = 'acc')

ax.plot(history.history['loss'], label = 'loss')





ax.legend()

ax.grid(True)

ax.set_xlabel("Epoch")

ax.set_ylabel("Rate")

plt.show()
model.evaluate(X_train, y_train)
pred = model.predict(X_test)
pred
test_pred = model.predict(test)
test_pred
from sklearn import metrics



fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=2)

metrics.auc(fpr, tpr)
from sklearn.metrics import f1_score



f1_score(y_test, pred, average='macro')
f1_score(y_test, pred, average='micro')
from sklearn.metrics import accuracy_score



accuracy_score(y_test, pred)
sample_submission = sample_submission.drop(['item_cnt_month'], axis = 1)
my_sub = pd.DataFrame(test_pred,columns=['item_cnt_month'])

my_sub = pd.concat([sample_submission, my_sub], axis = 1)

my_sub.head()

my_sub.to_csv('submission.csv', index = False)