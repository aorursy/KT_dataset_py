import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import os
import glob
import random
import shutil
from keras.preprocessing.image import load_img, img_to_array

img_path="data/images/"
img_height=224 #299
img_width=224 #299

# credit: https://github.com/fastai/fastai/blob/9e9ffbd49eb6490bb1168ce2ff32b10a81498ba9/fastai/utils.py
import bcolz
def save_array(fname, arr): c=bcolz.carray(arr, rootdir=fname, mode='w'); c.flush()
def load_array(fname): return bcolz.open(fname)[:]

def plot_img(img, title, count, cols, plot_axis=False):
    a = fig.add_subplot(1, cols, count)
    # if 'img' is a NumPy array, then it has already been loaded; just show it
    if type(img).__module__ == np.__name__:
        plt.imshow(img)
    else:
        plt.imshow(load_img(img))
    a.set_title(title,fontsize=10)
    if plot_axis is False:
        plt.axis('off')
import csv

def create_label_directories(csv_filename, img_path, is_one_v_all=False, one_v_all_label="Fibrosis"):
    directories = set()
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(csvfile) # skip header row
        for row in reader:
            img_filename = str(row[0])
            labels = str(row[1])
            for label in labels.split('|'):
                if (is_one_v_all is True) and (label != one_v_all_label):
                    label = "ALL"
                src_file = os.path.join(img_path,img_filename)
                label = "_".join(label.split())
                dst_train_dir = os.path.join(img_path,"train",label)
                dst_train_file = os.path.join(dst_train_dir,img_filename)
                dst_valid_dir = os.path.join(img_path,"valid",label)
                dst_test_dir = os.path.join(img_path,"test",label)
                if not os.path.exists(dst_train_dir):
                    os.makedirs(dst_train_dir)
                    directories.add(label)
                if not os.path.exists(dst_valid_dir):
                    os.makedirs(dst_valid_dir)
                if not os.path.exists(dst_test_dir):
                    os.makedirs(dst_test_dir)
                src_file_abs = os.path.join(os.getcwd(),src_file)
                dst_train_file_abs = os.path.join(os.getcwd(),dst_train_file)
                #print("copy: " + src_file_abs + " to: " + dst_train_file_abs)
                if not os.path.exists(dst_train_file_abs):
                    os.symlink(src_file_abs, dst_train_file_abs)
    return list(directories)

is_one_v_all = True
one_v_all_label = "Fibrosis"
print(img_path)
directories = create_label_directories("data/Data_Entry_2017.csv", img_path, is_one_v_all, one_v_all_label)
print(directories)
def get_per_label_count(directories):
    per_label_count = []
    for ii in range(len(directories)):
        #print(directories[ii])
        path, dirs, files = os.walk(os.path.join(img_path,"train",directories[ii])).__next__()
        file_count = len(files)
        per_label_count.append(file_count)
    return per_label_count
        
print(directories)
per_label_count = get_per_label_count(directories) 
print(per_label_count)
import subprocess

def upsample(directories, per_label_count, iqr):
    for ii in range(len(per_label_count)):
        label = directories[ii]
        count = per_label_count[ii]
        if count < iqr:
            offset = iqr-count
            subprocess.call(['./batch-augment.sh', os.path.join(os.getcwd(),img_path,"train",label), str(offset)])
    return get_per_label_count(directories)

print(directories)
print(per_label_count)
#print(label_batch_size)
per_label_count_upsampled = upsample(directories, per_label_count, 8430)#label_batch_size)
print(per_label_count_upsampled)
def downsample(directories, per_label_count):
    label_idx = np.argmin(per_label_count)
    #print(label_idx)
    downsample_count = per_label_count[label_idx]
    #print(downsample_count)
    for ii in range(len(per_label_count)):
        label = directories[ii]
        src_train_dir = os.path.join(img_path,"train",label)
        all_img_paths = glob.glob(os.path.join(src_train_dir,"*.*"))
        np.random.shuffle(all_img_paths)
        if len(all_img_paths) != downsample_count:
            imgs_to_remove = all_img_paths[downsample_count:]
            #print(len(imgs_to_remove))
            for file in imgs_to_remove:
                file_abs = os.path.join(os.getcwd(),file)
                #print("remove file: " + file_abs)
                os.remove(file_abs)
    return get_per_label_count(directories)
                
print(directories)
print(per_label_count_upsampled)
per_label_count_downsampled = downsample(directories, per_label_count_upsampled)
print(per_label_count_downsampled)
def split_train_valid_test(directories, per_label_count, valid_pct, test_pct):
    for ii in range(len(directories)):
        all_img_paths = glob.glob(os.path.join(img_path,"train",directories[ii],"*.*"))
        np.random.shuffle(all_img_paths)
        label_count = per_label_count[ii]
        valid_count = int(label_count*valid_pct)
        valid_files = all_img_paths[:valid_count]
        all_img_paths[:valid_count] = []
        test_count = int(label_count*test_pct)
        test_files = all_img_paths[:test_count]
        all_img_paths[:test_count] = []
        #print(len(valid_files))
        #print(len(test_files))
        train_files = all_img_paths
        all_img_paths = []
        #print(len(train_files))
        for valid_file in valid_files:
            valid_file_abs = os.path.join(os.getcwd(),valid_file)
            #print("move: '" + valid_file_abs + "' to: '" + os.path.join(img_path,"valid",directories[ii]))
            shutil.move(valid_file_abs, os.path.join(img_path,"valid",directories[ii]))
        for test_file in test_files:
            test_file_abs = os.path.join(os.getcwd(),test_file)
            #print("move: '" + test_file_abs + "' to: '" + os.path.join(img_path,"test",directories[ii]))
            shutil.move(test_file_abs, os.path.join(img_path,"test",directories[ii]))
        

valid_pct = 0.01 # 0.1
test_pct = 0.98 # 0.1       
#print(resampled_directories)
#print(resampled_per_label_count)
#split_train_valid_test(resampled_directories, resampled_per_label_count, valid_pct, test_pct)
print(directories)
print(per_label_count_downsampled)
split_train_valid_test(directories, per_label_count_downsampled, valid_pct, test_pct)
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.densenet import preprocess_input

train_datagen = ImageDataGenerator(
    #rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True, 
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    'data/images/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
    #color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(
    'data/images/valid',
    target_size=(img_height, img_width),
    batch_size=batch_size, #val_batch_size,
    class_mode='categorical')
    
test_generator = test_datagen.flow_from_directory(
    'data/images/test',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
from keras.applications.densenet import DenseNet121
model = DenseNet121(include_top=True, weights=None, input_shape=(img_width,img_height,3), classes=2)
model.summary()
model.load_weights('weights.best.DenseNet121-nih-one-v-all-fibrosis.20180305-r1.hdf5')
# only missing -> cmd:option('-weightDecay', 1e-4, 'weight decay')
from keras import optimizers
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True), metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
filepath="weights.best.DenseNet121-nih-one-v-all-fibrosis.20180305-r1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min') #mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=500)
#tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=64, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
callbacks_list = [checkpoint,early_stopping]#,tensorboard]
nb_train_samples = 16188 #8094 #3036 #18046 #111589 #113243 #139987 
nb_validation_samples= 336 
epochs = int(nb_train_samples/batch_size)*3
history = model.fit_generator(
    train_generator,
    steps_per_epoch=batch_size, #nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=batch_size, #nb_validation_samples/batch_size, #val_batch_size,
    callbacks=callbacks_list,
    verbose=1)
def plot_history(history):
    plt.plot(history.history['loss'],'r--')
    plt.plot(history.history['val_loss'],'b-')
    plt.title('Model Loss')
    plt.legend(['Train Loss', 'Valid Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show();
    plt.plot(history.history['acc'],'r--')
    plt.plot(history.history['val_acc'],'b-')
    plt.title('Model Accuracy')
    plt.legend(['Train Acc', 'Valid Acc'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

# requires history=model.fit, fit_generator...
plot_history(history)
#nb_test_samples = 168
#steps = nb_test_samples/batch_size
#scores = model.evaluate(X_test, y_test)
scores = model.evaluate_generator(test_generator, steps=10, max_queue_size=10, workers=1, use_multiprocessing=False)
print("score = Loss: %f, Acc@1: %.2f" % (scores[0],scores[1]))
preds = model.predict_generator(test_generator, steps=10, max_queue_size=10, workers=1, use_multiprocessing=False)
preds.shape
#print(test_generator.classes)
steps = 11 #int(SIZE?/batch_size)
preds = np.zeros((0,2))
y_test = np.zeros((0,2))
step_count = 0
for batch_x, batch_y in test_generator:
    if step_count < steps:
        batch_preds = model.predict(batch_x)
        #print(batch_preds.shape)
        preds = np.vstack((preds,batch_preds))
        #print(batch_y)
        y_test = np.vstack((y_test,batch_y))
        step_count = step_count + 1
    else:
        break
        
print(preds.shape)
print(y_test.shape)
import itertools
from sklearn.metrics import confusion_matrix

y_trues = [np.argmax(ii) for ii in y_test]
y_preds = [np.argmax(ii) for ii in preds]

# credit: https://tatwan.github.io/How-To-Plot-A-Confusion-Matrix-In-Python/    
def plot_binary_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    classNames = classes
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    
cm = confusion_matrix(y_trues, y_preds)
#print(cm)
tn, fp, fn, tp = cm.ravel()
#print(tn, fp, fn, tp)

# Plot non-normalized confusion matrix
plt.figure()
plot_binary_confusion_matrix(cm, classes=['Negative','Positive'], title='Confusion matrix, without normalization', cmap=plt.cm.Greens)
plt.show()


