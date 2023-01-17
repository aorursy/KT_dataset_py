# Thanks to https://www.kaggle.com/wcukierski/example-metric-implementation

import csv
import numpy as np
from scipy import ndimage as ndi
from tqdm import tqdm

def masks2labels(masks, roll = False):
    num_mask = masks.shape[2 if roll else 0]
    rolled = np.rollaxis(masks, 2) if roll else masks
    mask_list = list(rolled)
    labels = np.zeros((masks.shape[0 if roll else 1], masks.shape[1 if roll else 2]), dtype=np.uint16)
    for i in range(num_mask):
        mask_list[i] = np.where(mask_list[i] > 0, 1, 0)
        labels = np.where(mask_list[i] > 0, i + 1, labels)
    return labels

def label_loss(label_true, label_pred):
    true_objects = len(np.unique(label_true))
    pred_objects = len(np.unique(label_pred))
    
    intersection = np.histogram2d(label_true.flatten(), label_pred.flatten(), bins=(true_objects, pred_objects))[0]

    area_true = np.histogram(label_true, bins = true_objects)[0]
    area_pred = np.histogram(label_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    
    union = area_true + area_pred - intersection

    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9
    
    iou = intersection / union

    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp + fn)
        prec.append(p)
    return np.mean(prec)


def compute_map_nuclei(true_masks, pred_masks):
    true_labels = masks2labels(true_masks)
    pred_labels = masks2labels(pred_masks)
    return label_loss(true_labels, pred_labels)

def get_stage1_masks(true_filename, pred_filename):
    stage1_mask_list = {}
    stage1_test_sizes = {}
    stage1_pred_mask_list = {}
    with open(true_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rleNumbers = [int(s) for s in row['EncodedPixels'].split(' ')]
            rlePairs = np.array(rleNumbers).reshape(-1,2)
            if row['ImageId'] not in stage1_test_sizes:
                stage1_test_sizes[row['ImageId']] = [int(row['Height']), int(row['Width'])]
            height = stage1_test_sizes[row['ImageId']][0]
            width = stage1_test_sizes[row['ImageId']][1]
            
            mask = np.zeros(height*width,dtype=np.uint8)
            for index,length in rlePairs:
                index -= 1
                mask[index:index+length] = 1
            mask = mask.reshape(width,height)
            mask = mask.T
            if row['ImageId'] not in stage1_mask_list:
                stage1_mask_list[row['ImageId']] = []
            stage1_mask_list[row['ImageId']].append(mask)
    
    with open(pred_filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rleNumbers = [int(s) for s in row['EncodedPixels'].split(' ')] if len(row['EncodedPixels'])>0 else []
            rlePairs = np.array(rleNumbers).reshape(-1,2)
            height = stage1_test_sizes[row['ImageId']][0]
            width = stage1_test_sizes[row['ImageId']][1]
            mask = np.zeros(height*width,dtype=np.uint8)
            for index,length in rlePairs:
                index -= 1
                mask[index:index+length] = 1
            mask = mask.reshape(width,height)
            mask = mask.T
            if row['ImageId'] not in stage1_pred_mask_list:
                stage1_pred_mask_list[row['ImageId']] = []
            stage1_pred_mask_list[row['ImageId']].append(mask)
    
    APs = []
    for imageId in tqdm(stage1_mask_list):
        true_masks = np.array(stage1_mask_list[imageId])
        pred_masks = np.array(stage1_pred_mask_list[imageId])
        APs.append(compute_map_nuclei(true_masks, pred_masks))
    
    return np.mean(APs)
    
# Put the solution of stage1 here
true_filename = '../input/ds2018-stage-1-solution/stage1_solution.csv'
# Put your solution here
pred_filename = '../input/with-empty-mask/dsbowl2018-1.csv'
print(get_stage1_masks(true_filename, pred_filename))
