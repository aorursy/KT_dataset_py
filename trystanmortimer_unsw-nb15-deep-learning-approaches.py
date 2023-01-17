import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_probability as tfp

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import average_precision_score, f1_score

from sklearn.metrics import recall_score, accuracy_score



# Load up the data

train_data = pd.read_csv("/kaggle/input/unswnb15/UNSW_NB15/UNSW_NB15_train-set.csv")

test_data =  pd.read_csv("/kaggle/input/unswnb15/UNSW_NB15/UNSW_NB15_test-set.csv")

features = pd.read_csv("/kaggle/input/unswnb15/UNSW_NB15/NUSW-NB15_features.csv", encoding = "ISO-8859-1")



df = pd.concat([train_data, test_data], ignore_index=True)



# Remove unwanted columns

df.drop(['id', 'attack_cat'], inplace=True, axis=1)



# Perform one-hot encoding on categorical columns and join back to main train_data

one_hot = pd.get_dummies(df[["proto", "state", "service"]])

df = df.join(one_hot)



# Remove the original categorical columns

df.drop(["proto", "state", "service"], inplace=True, axis=1)



# Re split the data back into train / test

train_data = df.iloc[0:175341, 0:]

test_data = df.iloc[175341:, 0:]



# Create y_train and then drop the label from the training data

y_train = np.array(train_data["label"])

train_data.drop(['label'], inplace=True, axis=1)



y_test = np.array(test_data["label"])

test_data.drop(['label'], inplace=True, axis=1)



# Scale the data

# Only fit the scaler on the train data!!

scaler = StandardScaler()

X_train = scaler.fit_transform(train_data)



# Scale the testing data

X_test = scaler.transform(test_data)



# Ensure our dataset splits are still correct

print(f"Train data shape: {X_train.shape} Train label shape: {y_train.shape}")

print(f"Test data shape: {X_test.shape} Test label shape: {y_test.shape}")



# Set some variables for ease of use

FEATURES = X_train.shape[1]

train_length = X_train.shape[0]

test_length = X_test.shape[0]
# Define the layers

input_layer = tf.keras.Input(shape=(FEATURES))

x = tf.keras.layers.Dense(FEATURES, activation="relu")(input_layer)

x = tf.keras.layers.Dense(64, activation="relu")(x)

x = tf.keras.layers.Dense(32, activation="relu")(x)

output_layer = tf.keras.layers.Dense(2, activation="softmax")(x)



# Create the model

fcn = tf.keras.models.Model(input_layer, output_layer, name="Fully_Connected_Network")



# Compile the model

fcn.compile(optimizer = 'adam', loss="SparseCategoricalCrossentropy", metrics = ['accuracy'])

fcn.summary()



# Fit the model

fcn.fit(X_train, y_train, epochs=3, batch_size=256)



loss, accuracy = fcn.evaluate(X_test,y_test, batch_size=64)

print("Accuracy", accuracy)
# Test+

preds = fcn.predict(X_test) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test, pred_labels), 3)

recall = np.round(recall_score(y_test, pred_labels), 3)

precision = np.round(average_precision_score(y_test, pred_labels), 3)

f1 = np.round(f1_score(y_test, pred_labels), 3)



accuracy_dict = {"FCN":accuracy}

recall_dict = {"FCN":recall}

precision_dict = {"FCN":precision}

f1_dict = {"FCN":f1}



print(f"Accuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")
encoded_size=32



# ENCODER

encoder_input_layer = tf.keras.Input(shape=(FEATURES))

dense_encoder = tf.keras.layers.Dense(FEATURES, activation="relu")(encoder_input_layer)

latent_output = tf.keras.layers.Dense(encoded_size, activation=None)(dense_encoder)

encoder = tf.keras.models.Model(encoder_input_layer, latent_output, name="Encoder")



# DECODER

decoder_input_layer = tf.keras.Input(shape=(encoded_size))

dense_decoder = tf.keras.layers.Dense(64, activation="relu")(decoder_input_layer)

decoder_output = tf.keras.layers.Dense(FEATURES, activation=None)(dense_decoder)

decoder = tf.keras.models.Model(decoder_input_layer, decoder_output, name="Decoder")



# AUTOENCODER

input_layer = tf.keras.Input(shape=(FEATURES))

encoder_layer = encoder(input_layer)

decoder_layer = decoder(encoder_layer)

autoencoder = tf.keras.models.Model(input_layer, decoder_layer, name="Autoencoder")



autoencoder.compile(optimizer='adam', loss="MSE")

autoencoder.summary()



# We are trying to recreate the input so the target is also X_train

autoencoder.fit(X_train, X_train, epochs=5, batch_size=256)



# Now we need to freeze the weights of the encoder

encoder.trainable = False



# Use the encoder with a FCN attached for classification

x = tf.keras.layers.Dense(32, activation="relu")(encoder_layer)

x = tf.keras.layers.Dense(16, activation="relu")(x)

fcn_output = tf.keras.layers.Dense(2, activation="softmax")(x)



autoencoder_fcn = tf.keras.models.Model(input_layer, fcn_output, name="Autoencoder_FCN")

autoencoder_fcn.compile(optimizer="adam", loss="SparseCategoricalCrossentropy", metrics=["accuracy"])

autoencoder_fcn.summary()

autoencoder_fcn.fit(X_train, y_train, epochs=5, batch_size=256)



loss, accuracy = autoencoder_fcn.evaluate(X_test,y_test, batch_size=256)

print("Accuracy", accuracy)
# Test+

preds = autoencoder_fcn.predict(X_test) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test, pred_labels), 3)

recall = np.round(recall_score(y_test, pred_labels), 3)

precision = np.round(average_precision_score(y_test, pred_labels), 3)

f1 = np.round(f1_score(y_test, pred_labels), 3)



accuracy_dict["AE-FCN"] = accuracy

recall_dict["AE-FCN"] = recall

precision_dict["AE-FCN"] = precision

f1_dict["AE-FCN"] = f1



print(f"Accuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")
# Reshape the data input for LSTM input

X_train = X_train.reshape(-1,1,FEATURES)

y_train = y_train.reshape(-1,1)



print(f"X_train shape: {X_train.shape} y_train shape: {y_train.shape}")



X_test = X_test.reshape(-1,1,FEATURES)

y_test = y_test.reshape(-1,1)



print(f"X_test shape: {X_test.shape} y_test shape: {y_test.shape}")
# ENCODER

# Layers

input_layer = tf.keras.Input(shape=(1,FEATURES), name="Input_Layer")

encoder_lstm_one = tf.keras.layers.LSTM(FEATURES, return_sequences=True, name="Encoder_LSTM_One", input_shape=(1,FEATURES))(input_layer)

encoder_lstm_two = tf.keras.layers.LSTM(FEATURES, return_sequences=True, name="Encoder_LSTM_Two")(encoder_lstm_one)

output, state_h, state_c = tf.keras.layers.LSTM(FEATURES, return_state=True, name="Encoder_LSTM_Three")(encoder_lstm_two)

encoder_state = [state_h, state_c]

output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="Encoder_Classifier")(output)



# Model

encoder = tf.keras.models.Model(input_layer, output_layer, name="Encoder")

encoder.compile(optimizer = 'adam', loss="MSE", metrics = ['accuracy'])





# DECODER

# Layers

decoder_lstm_one = tf.keras.layers.LSTM(FEATURES, name="Decoder_LSTM_One", input_shape=(1,FEATURES), return_sequences=True)(input_layer, initial_state=encoder_state)

decoder_lstm_two = tf.keras.layers.LSTM(FEATURES, name="Decoder_LSTM_Two", return_sequences=True)(decoder_lstm_one)

decoder_lstm_three = tf.keras.layers.LSTM(FEATURES, name="Decoder_LSTM_Three")(decoder_lstm_two)

decoder_output_layer = tf.keras.layers.Dense(2, activation="softmax", name="Decoder_Classifier")(decoder_lstm_three)



# Model

seq2seq = tf.keras.models.Model(input_layer, decoder_output_layer, name="Seq2Seq")

seq2seq.compile(optimizer = 'adam', loss="SparseCategoricalCrossentropy", metrics = ['accuracy'])

seq2seq.summary()



# Fit

seq2seq.fit(X_train, y_train, epochs=5, batch_size=64)



# EVALUATE

loss, accuracy = seq2seq.evaluate(X_test,y_test, batch_size=64)

print("Accuracy", accuracy)

# Reshape the test data back to normal

y_test = y_test.flatten()



# Test+

preds = seq2seq.predict(X_test) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test, pred_labels), 3)

recall = np.round(recall_score(y_test, pred_labels), 3)

precision = np.round(average_precision_score(y_test, pred_labels), 3)

f1 = np.round(f1_score(y_test, pred_labels), 3)



accuracy_dict["Seq2Seq"] = accuracy

recall_dict["Seq2Seq"] = recall

precision_dict["Seq2Seq"] = precision

f1_dict["Seq2Seq"] = f1



print(f"Test+\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")
# We currently have one-hot encoded columns for categorical variables

# Convert each group into a one hot vector in a single column



# Get a list of each group of columns

proto_cols = [col for col in df.columns if 'proto_' in col]

service_cols = [col for col in df.columns if 'service_' in col]

state_cols = [col for col in df.columns if 'state_' in col]



# Function to convert each group of cols into a single col

def vectorize_cols(col_list, dataframe):



    col = []

    # Iterate over each row

    for row in dataframe[col_list].itertuples():

        row_entry = []

        

        # Iterate over each column

        for i in range(1,len(col_list)+1):

            row_entry.append(row[i])

        

        col.append(row_entry)

    

    return col



df["state"] = vectorize_cols(state_cols, df)

df["service"] = vectorize_cols(service_cols, df)

df["proto"] = vectorize_cols(proto_cols, df)



# Drop the individual columns

df.drop(proto_cols, inplace=True, axis=1)

df.drop(service_cols, inplace=True, axis=1)

df.drop(state_cols, inplace=True, axis=1)



# resplit the data

train_plus = df.iloc[0:train_length, :]

test_plus = df.iloc[train_length:, :]



# Create the training target for each dataset

y_train = np.array(train_plus["label"])

y_test = np.array(test_plus["label"])



# Scale the data with standard scaler 

# do not include the training targets or the already vectorized columns

scaler = StandardScaler()

X_train = scaler.fit_transform(train_plus.drop(["label", "state", "service", "proto"], axis=1))

X_test = scaler.transform(test_plus.drop(["label", "state", "service", "proto"], axis=1))



# Discretize the data into 10 bins

X_train = np.digitize(X_train, np.arange(0,0.9,.1))

X_test = np.digitize(X_test, np.arange(0,0.9,.1))



# One hot encode each value

X_train = tf.keras.utils.to_categorical(X_train, num_classes=10)

X_test = tf.keras.utils.to_categorical(X_test, num_classes=10)



# Reshape the data ready to concatenate categorical data back in

X_train = X_train.reshape(train_length, -1)

X_test = X_test.reshape(test_length, -1)





# Add the categorical data back in

# X_train

proto = np.vstack(train_plus["proto"].apply(lambda x: np.asarray(x)))

service = np.vstack(train_plus["service"].apply(lambda x: np.asarray(x)))

state = np.vstack(train_plus["state"].apply(lambda x: np.asarray(x)))

X_train = np.concatenate((X_train, proto, service, state), axis=1)



# X_test

proto = np.vstack(test_plus["proto"].apply(lambda x: np.asarray(x)))

service = np.vstack(test_plus["service"].apply(lambda x: np.asarray(x)))

state = np.vstack(test_plus["state"].apply(lambda x: np.asarray(x)))

X_test = np.concatenate((X_test, proto, service, state), axis=1)





print("We now have the correct shape described in the paper for the number of features")

print(X_train.shape, X_test.shape)



# * We now have the correct shape described in the paper for the number of features

# * Convert each 8 bits into a pixel

# * Then pad the array's so we can have the correct size for a square image

# * We currently have data for 464 / 8 = 58 pixels

# * We need data for 8x8 = 64 pixels

# * 6x8 = 48 -> we must pad the current 464 dimension to 512 length with zeroes



X_train = np.pad(X_train, ((0,0), (0,110)), "constant", constant_values=0)

X_test = np.pad(X_test, ((0,0), (0,110)), "constant", constant_values=0)

print("\nPad out the data so we can have 9x9")

print(X_train.shape, X_test.shape)



# We need to convert each 8bit segment into a 0-255 value, reshape so we can.

X_train = X_train.reshape(-1, 81, 8).astype("int")

X_test = X_test.reshape(-1, 81, 8).astype("int")



# Convert binary array to 0-255

X_train = np.packbits(X_train, axis=2)

X_test = np.packbits(X_test, axis=2)



# Reshape into an 8x8 image

X_train = X_train.reshape(-1, 9, 9)

X_test = X_test.reshape(-1, 9, 9)

print("\nNow we have the correct shape of an image")

print(X_train.shape, X_test.shape)



# Have a look at the first example

print("\nVisualize the first observation")

plt.imshow(X_train[0,:,:], cmap='gray', vmin=0, vmax=255)

plt.show()



# The paper leaves out some information regarding preprocessing

# ResNet50 was trained on RGB images and requires three channels of data

# To get around this I will repeat the greyscale image three times



X_train = np.repeat(X_train[..., np.newaxis], 3, -1)

X_test = np.repeat(X_test[..., np.newaxis], 3, -1)

print("\nRepeat the greyscale layer three times so we immitate an RGB image")

print(X_train.shape, X_test.shape)
# Create and compile the base resnet model using imagenet weights

resnet = tf.keras.applications.ResNet50(

    include_top=False, weights=None, input_tensor=tf.keras.Input(shape=(9,9,3)), 

    input_shape=None, pooling="max")



# Now we can compile the model

# resnet.compile(optimizer='SGD', loss="SparseCategoricalCrossentropy", metrics="accuracy")



inputs = tf.keras.Input(shape=(9,9,3))

resnet_layer = resnet(inputs)

# flatten = tf.keras.layers.Flatten()(resnet_layer)

x = tf.keras.layers.Dense(256, activation="relu")(resnet_layer)

x = tf.keras.layers.Dense(64, activation="relu")(x)

outputs = tf.keras.layers.Dense(2, activation="softmax")(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='SGD', loss="SparseCategoricalCrossentropy", metrics="accuracy")



model.summary()



model.fit(x=X_train, y=y_train.astype("bool"), epochs=50, batch_size=256)



loss, accuracy = model.evaluate(X_test,y_test, batch_size=256)

print("Accuracy Test+", accuracy)
# Test+

preds = model.predict(X_test) 

pred_labels = []



for prediction in preds:

    pred_labels.append(np.argmax(prediction))



accuracy = np.round(accuracy_score(y_test, pred_labels), 3)

recall = np.round(recall_score(y_test, pred_labels), 3)

precision = np.round(average_precision_score(y_test, pred_labels), 3)

f1 = np.round(f1_score(y_test, pred_labels), 3)



accuracy_dict["ResNet"] = accuracy

recall_dict["ResNet"] = recall

precision_dict["ResNet"] = precision

f1_dict["ResNet"] = f1



print(f"Test+\nAccuracy: {accuracy}\nPrecsion: {precision}\nRecall: {recall}\nF1: {f1}")
import seaborn as sns



# Plot accuracies

plt.figure(figsize=(8,5))

sns.barplot(x=list({k: v for k, v in sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True)}.keys()), 

            y=list({k: v for k, v in sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True)}.values()),

           palette="viridis")

plt.xticks(rotation=45)

plt.title("UNSW-NB15")

plt.ylabel("Accuracy")

plt.xlabel("Model")

plt.show()



# Plot Precision

plt.figure(figsize=(8,5))

sns.barplot(x=list({k: v for k, v in sorted(precision_dict.items(), key=lambda item: item[1], reverse=True)}.keys()), 

            y=list({k: v for k, v in sorted(precision_dict.items(), key=lambda item: item[1], reverse=True)}.values()), 

            palette="cool_r")

plt.xticks(rotation=45)

plt.title("UNSW-NB15")

plt.ylabel("Precision")

plt.xlabel("Model")

plt.show()



# Plot Recall

plt.figure(figsize=(8,5))

sns.barplot(x=list({k: v for k, v in sorted(recall_dict.items(), key=lambda item: item[1], reverse=True)}.keys()), 

            y=list({k: v for k, v in sorted(recall_dict.items(), key=lambda item: item[1], reverse=True)}.values()), 

            palette="magma")

plt.xticks(rotation=45)

plt.title("UNSW-NB15")

plt.ylabel("Recall")

plt.xlabel("Model")

plt.show()



# Plot F1

plt.figure(figsize=(8,5))

sns.barplot(x=list({k: v for k, v in sorted(f1_dict.items(), key=lambda item: item[1], reverse=True)}.keys()), 

            y=list({k: v for k, v in sorted(f1_dict.items(), key=lambda item: item[1], reverse=True)}.values()), 

            palette="mako")

plt.xticks(rotation=45)

plt.title("UNSW-NB15")

plt.ylabel("F1")

plt.xlabel("Model")

plt.show()