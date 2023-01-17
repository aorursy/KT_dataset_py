import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import glob
import pydicom as dcm
train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
train.head()
train.dtypes
train_cols = ['pe_present_on_image', 'negative_exam_for_pe', 'qa_motion',
       'qa_contrast', 'flow_artifact', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
       'leftsided_pe', 'chronic_pe', 'true_filling_defect_not_pe',
       'rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']

def plot_grid(cols = train_cols):
    fig=plt.figure(figsize=(12, 12))
    columns = 3
    rows = 5
    for i in range(1, columns*rows):
        col = cols[i-1]
        fig.add_subplot(rows, columns, i)
        train[col].value_counts().plot(kind = "bar")
        indices = train[col].value_counts().index.tolist()
        count_0 = train[col].value_counts()[0]
        count_1 = train[col].value_counts()[1]
        plt.xlabel(f"{col}\n {indices[0]}: {count_0}\n {indices[1]}: {count_1}")
    plt.tight_layout()
    plt.show()

plot_grid()
corr_mat = train[train_cols].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
f, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr_mat, mask = mask, annot = True, vmax = 0.3, square = False, linewidths = 0.5, center = 0)
def read_dicom(file_path, show = False, cmap = 'gray'):
    im = dcm.dcmread(file_path)
    image = im.pixel_array
    if show:
        plt.imshow(image, cmap = 'gray')
    return image
def show_images_with_specific_condition(column_name):
    
    rows = 2
    cols = 5
    
    train_with_condition = train[train[column_name] == 0]
    train_with_condition = train_with_condition.sample(n = rows*cols) 
    train_image_file_paths = []
    for _, entry in train_with_condition.iterrows():
        train_image_file_paths.append('../input/rsna-str-pulmonary-embolism-detection/train/'+str(entry['StudyInstanceUID'])+'/'+str(entry['SeriesInstanceUID'])+'/'+str(entry['SOPInstanceUID'])+'.dcm')
    counter  = 0
    fig = plt.figure(figsize=(25,15))
    fig.suptitle('Samples with ' + column_name + ' = 0', fontsize=40)
    for path in train_image_file_paths:
        fig.add_subplot(rows, cols, counter+1)
        plt.imshow(read_dicom(path), cmap='gray')
        plt.axis(False)
        fig.add_subplot
        counter += 1
    
    
    train_with_condition = train[train[column_name] == 1]
    train_with_condition = train_with_condition.sample(n = rows*cols) 
    train_image_file_paths = []
    for _, entry in train_with_condition.iterrows():
        train_image_file_paths.append('../input/rsna-str-pulmonary-embolism-detection/train/'+str(entry['StudyInstanceUID'])+'/'+str(entry['SeriesInstanceUID'])+'/'+str(entry['SOPInstanceUID'])+'.dcm')
    counter  = 0
    fig = plt.figure(figsize=(25,15))
    fig.suptitle('Samples with ' + column_name + ' = 1', fontsize=40)
    for path in train_image_file_paths:
        fig.add_subplot(rows, cols, counter+1)
        plt.imshow(read_dicom(path), cmap='gray')
        plt.axis(False)
        fig.add_subplot
        counter += 1
show_images_with_specific_condition('qa_motion')
show_images_with_specific_condition('qa_contrast')
show_images_with_specific_condition('indeterminate')
show_images_with_specific_condition('rightsided_pe')
show_images_with_specific_condition('leftsided_pe')
show_images_with_specific_condition('central_pe')