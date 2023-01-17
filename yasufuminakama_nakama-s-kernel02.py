import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
# Measure run time
import time
start_time = time.time() 
# Load the data
train = pd.read_csv("../input/fashion-mnist-dataset/fashion-mnist_train.csv")
test = pd.read_csv("../input/fashion-mnist-dataset/fashion-mnist_test.csv")

# X for Features, Y for Labels
X_train = train.drop(["label"], axis=1) 
Y_train = train["label"]
X_test = test.drop(["label"], axis=1)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)

# Split the train and the validation set for the fitting
random_seed = 0
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = pd.get_dummies(Y_train)
Y_val_dummies = pd.get_dummies(Y_val)
# n*models
def make_models(n, X_train, Y_train, X_val, Y_val, batch_size, epochs):
    for i in range(n):
        
        # Create Model
        model = Sequential()
        
        # Add Layers
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Dropout(0.666))
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        
        # Compile
        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        
        # checkpoint
        filepath="weights.best"+str(i)+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]   
        
        # Fit the model to the train data
        model.fit(X_train, Y_train,
        batch_size=batch_size,
        verbose=1,
        epochs=epochs,
        validation_data=(X_val, Y_val),
        callbacks=callbacks_list)    
# What we want in this cell is "Majority rule"
# By using n*models, we can predict n*answers
# So take "Majority rule" like RandomForest
# And then we will have more accuracy

# Y_test n*answers for X_test
def answers(n, X_test, X_val, Y_val):
    
    # answer array
    answer = [[] for p in range(n)]
    
    # Create Model
    model = Sequential()
        
    # Add Layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
        
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.666))
        
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
        
    # Compile
    model.compile(loss='categorical_crossentropy',
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
    # load weights and insert predict results for each
    for q in range(n):
        print("----------Loading weights.best" + str(q) + ".hdf5----------")
        model.load_weights("weights.best"+str(q)+".hdf5")
        print("["+"val_loss, val_acc"+"]"+" ---> "+str(model.evaluate(X_val, Y_val, verbose=0)))
        results = model.predict(X_test)
        results = np.argmax(results, axis=1)
        results = pd.Series(results, name="label")
        answer[q] = results
        
    # Concat results
    results_concat = answer[0]
    for r in range(n-1):
        results_concat = pd.concat([results_concat, answer[r+1]], axis=1)
        
    # Mode of concat results
    results_concat_mode = results_concat.mode(axis=1).iloc[ : , 0]
    results_concat_mode = pd.Series(results_concat_mode, name="label")
    
    # Concat id as you can submit
    submission = pd.concat([pd.Series(range(1,10001), name='id'), results_concat_mode], axis=1)
    
    # float ---> int
    submission = submission.astype(np.int64)
    
    # to_csv
    submission.to_csv('submission.csv', index=False)
    print("----------complete----------")
# Y_val n*answers for X_val
def val_answers(n, X_val, Y_val):
    
    # answer array
    answer = [[] for p in range(n)]
    
    # Create Model
    model = Sequential()
        
    # Add Layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
        
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(0.666))
        
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
        
    # Compile
    model.compile(loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])
    # load weights and insert predict results for each
    for q in range(n):
        print("----------Loading weights.best" + str(q) + ".hdf5----------")
        model.load_weights("weights.best"+str(q)+".hdf5")
        print("["+"val_loss, val_acc"+"]"+" ---> "+str(model.evaluate(X_val, Y_val, verbose=0)))
        results = model.predict(X_val)
        results = np.argmax(results, axis=1)
        results = pd.Series(results, name="label")
        answer[q] = results
        
    # Concat results
    results_concat = answer[0]
    for r in range(n-1):
        results_concat = pd.concat([results_concat, answer[r+1]], axis=1)
        
    # Mode of concat results
    results_concat_mode = results_concat.mode(axis=1).iloc[ : , 0]
    results_concat_mode = pd.Series(results_concat_mode, name="label")
    
    # Concat id as you can submit
    results_concat_mode = pd.concat([pd.Series(range(1,6001), name='id'), results_concat_mode], axis=1)
    
    # float ---> int
    results_concat_mode = results_concat_mode.astype(np.int64)
    
    # to_csv
    results_concat_mode.to_csv("val_results_concat_mode.csv", index=True)
    print("----------complete----------")
# set parameters
n = 5
batch_size = 128
epochs = 350

# make models
make_models(n, X_train, Y_train, X_val, Y_val_dummies, batch_size, epochs)
# Majority rule for val
val_answers(n, X_val, Y_val_dummies)
# check val accuracy after Majority rule
val_results_concat_mode = pd.read_csv("val_results_concat_mode.csv")
accuracy_score(Y_val, val_results_concat_mode["label"])
# Majority rule for test
answers(n, X_test, X_val, Y_val_dummies)
# Measure run time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"elapsed_time：{elapsed_time}")