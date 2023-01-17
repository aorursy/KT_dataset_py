import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten



import os

print(os.listdir("../input"))
main_train = pd.read_csv("../input/train.csv")

main_test = pd.read_csv("../input/test.csv")
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split



# Create a train/test validation split

train, test = train_test_split(main_train, test_size=0.2)



# Seperate output and input from DataFrame

train_output = to_categorical(train['label'])

test_output = to_categorical(test['label'])



train_input_raw = train.drop('label', axis=1)

test_input_raw = test.drop('label', axis=1)



def resize1DArray(in_array):

    out_array = []

    for index, row in in_array.iterrows():

        out_row = row.values.reshape(28,28)

        out_array.append(out_row)

        

    return(out_array)



# Train & test arrays

train_input = resize1DArray(train_input_raw)

test_input = resize1DArray(test_input_raw)

test_final = resize1DArray(main_test)



# Convert to numpy arrays

train_input = np.array(train_input).reshape(-1, 28, 28, 1)

test_input = np.array(test_input).reshape(-1, 28, 28, 1)

test_final = np.array(test_final).reshape(-1, 28, 28, 1)

print(test_input.shape)
model = Sequential()



# Input 28x28 matrecies representing images at grayscale

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=3, activation='relu'))

model.add(Flatten())

model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train and test model

model.fit(train_input, train_output, validation_data=(test_input, test_output), epochs=3)
results = model.predict(test_final)
results_nums = []

for result in results:

    largestIndex = np.argmax(result)

    results_nums.append(largestIndex)

    

ids = range(1, len(results_nums) + 1)

result_dataframe = pd.DataFrame({'ImageId': ids, 'Label': results_nums})
result_dataframe.to_csv('submission.csv', index=False)