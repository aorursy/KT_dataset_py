import os
from IPython.display import Image, display
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.utils import class_weight
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121, densenet
path = '/kaggle/input/insulation-joint-training-set-prorail/trainset_insulation_joint/labels.csv'
ROOT_DATA = '/kaggle/input/insulation-joint-training-set-prorail/trainset_insulation_joint/images/'
df = pd.read_csv(path, sep=';')
df_train, df_test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=42, shuffle = True)
df_train, df_val = train_test_split(df_train, train_size=0.7, test_size=0.3, random_state=42, shuffle = True)
# Class count
target_count = df_train['label'].value_counts()
count_class_0, count_class_1 = target_count

# Divide by class
df_class_0 = df_train[df_train['label'] == 'n']
df_class_1 = df_train[df_train['label'] == 'p']

df_class_0_under = df_class_0.sample(count_class_1)
df_train_under = pd.concat([df_class_0_under, df_class_1], axis=0)

target_countRe = df_train_under['label'].value_counts()
target_countRe.plot(kind='bar', title='Count (Training data)')
Input_shape = (224, 224, 3)
Generator = ImageDataGenerator(preprocessing_function = densenet.preprocess_input)
train_generator = Generator.flow_from_dataframe(df_train_under,
                                                directory = ROOT_DATA,
                                                   x_col='filepath', y_col='label',
                                                   target_size=Input_shape[:2],
                                                    classes=['p', 'n'],
                                                   batch_size=16,
                                                    shuffle = True,
                                                   class_mode='categorical')
val_generator = Generator.flow_from_dataframe(df_val,
                                              directory = ROOT_DATA,
                                                   x_col='filepath', y_col='label',
                                                   target_size=Input_shape[:2],
                                                    classes=['n', 'p'],
                                                   batch_size=32,
                                                    shuffle = True,
                                                   class_mode='categorical')
test_generator = Generator.flow_from_dataframe(df_test, directory = ROOT_DATA, target_size=Input_shape[:2],
                                               x_col="filepath", y_col=None,
                                               batch_size=32, shuffle=False, class_mode=None)
Input_shape = (224,224,3)
densenet121_model = DenseNet121(include_top=False, weights=None, input_shape=Input_shape)
densenet121_model.summary()
del model
x = densenet121_model.output
x = layers.GlobalAveragePooling2D()(x)
#x = layers.Dense(128, activation='relu', name = 'FC1')(x) 
#x = layers.Dense(128, activation='relu', name = 'FC2')(x)
preds = layers.Dense(2, activation='softmax', name = 'Output')(x) 
model = keras.Model(inputs = densenet121_model.input, outputs = preds)
model.trainable = True
for layer in model.layers:
    layer.trainable = True
for layer in model.layers:
    print(layer.name, "  " , layer.trainable)
model.summary()
model.compile(loss=losses.CategoricalCrossentropy(), 
              optimizer = optimizers.Adam(learning_rate = 0.00001), 
              metrics=['accuracy'])
History = model.fit_generator(train_generator, 
                                epochs=10,
                                verbose=1,
                                validation_data = val_generator)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

epoch_list = list(range(1,11))
ax1.plot(epoch_list, History.history['accuracy'], label='Train Accuracy')
ax1.plot(epoch_list, History.history['val_accuracy'], label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 11, 1))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, History.history['loss'], label='Train Loss')
ax2.plot(epoch_list, History.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 11, 1))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
test_generator.reset()
y_pred = model.predict(x = test_generator, verbose = 1)
Y_pred = np.argmax(y_pred, axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in Y_pred]
from sklearn.metrics import confusion_matrix
import itertools

conf_mat = confusion_matrix(y_true = df_test['label'].values, y_pred=predictions)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
cm = np.array(conf_mat)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels)
plt.yticks(tick_marks, labels)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
TN = conf_mat[0][0]
FP = conf_mat[0][1]
FN = conf_mat[1][0]
TP = conf_mat[1][1]
#calculate the accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)
print("Prediction_Accuracy=",accuracy,"%")
#calculate sensitivity = TPR
sensitivity = TP/(TP+FN)
print("Sensitivity=",sensitivity,"%")
#calculate specificity = 1 - FPR
specificity = TN/(TN+FP)
print("Specificity=",specificity,"%")
#calculate precision
precision = TP/(TP+FP)
print("Precision=",precision,"%")
df_test['label'] = df.label.apply(lambda x: 1 if x=='p' else 0)
y_test = df_test['label'].values
from sklearn.metrics import roc_curve, roc_auc_score
#Calculate AUC
dense_auc = roc_auc_score(y_test, Y_pred)
print('MobileNetV2 (chance) Prediction: AUROC = %.3f' % (dense_auc))
#Calculate ROC
r_fpr, r_tpr, _ = roc_curve(y_test, Y_pred)

#Plot The ROC curve
plt.plot(r_fpr, r_tpr, linestyle='--', label='Model prediction (AUROC = %0.3f)' % dense_auc)


# Title
plt.title('ROC Plot')
# Axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# Show legend
plt.legend() # 
# Show plot
plt.show()
y_true = df_test['label'].values # ground truth labels
y_probas = y_pred# predicted probabilities generated by sklearn classifier
skplt.metrics.plot_roc_curve(y_true, y_probas)
plt.show()
