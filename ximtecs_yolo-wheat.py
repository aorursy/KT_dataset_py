from __future__ import division

from terminaltables import AsciiTable



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import matplotlib.patches as patches



import tqdm

import math

import glob

import random

import os

import sys

import time

import datetime

import argparse

from PIL import Image



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.autograd import Variable

from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from torchvision import datasets

from torchvision import transforms

from torch.autograd import Variable

import torch.optim as optim
epochs = 1

batch_size = 32

gradient_accumulations = 2

model_def = "/kaggle/input/custom-files/yolov3-Wheat-tiny.cfg"

data_config_str = "/kaggle/input/custom-files/Wheat_online.data"

#pretrained_weights = "/kaggle/input/custom-files/yolov3-tiny.weights"

pretrained_weights = "/kaggle/input/custom-files/yolov3_ckpt_final.pth"

img_size = 416

optimizer_str = "Adam"
def pad_to_square(img, pad_value):

    c, h, w = img.shape

    dim_diff = np.abs(h - w)

    # (upper / left) padding and (lower / right) padding

    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding

    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)

    # Add padding

    img = F.pad(img, pad, "constant", value=pad_value)



    return img, pad





def resize(image, size):

    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)

    return image





def random_resize(images, min_size=288, max_size=448):

    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]

    images = F.interpolate(images, size=new_size, mode="nearest")

    return images





class ImageFolder(Dataset):

    def __init__(self, folder_path, img_size=416):

        self.files = sorted(glob.glob("%s/*.*" % folder_path))

        self.img_size = img_size



    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]

        # Extract image as PyTorch tensor

        img = transforms.ToTensor()(Image.open(img_path))

        # Pad to square resolution

        img, _ = pad_to_square(img, 0)

        # Resize

        img = resize(img, self.img_size)



        return img_path, img



    def __len__(self):

        return len(self.files)





class ListDataset(Dataset):

    def __init__(self, list_path, img_size=416, multiscale=True, normalized_labels=True):

        with open(list_path, "r") as file:

            self.img_files = file.readlines()



        self.label_files = [

            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")

            for path in self.img_files

        ]

        self.img_size = img_size

        self.max_objects = 100

        self.multiscale = multiscale

        self.normalized_labels = normalized_labels

        self.min_size = self.img_size - 3 * 32

        self.max_size = self.img_size + 3 * 32

        self.batch_count = 0



    def __getitem__(self, index):



        # ---------

        #  Image

        # ---------



        img_path = self.img_files[index % len(self.img_files)].rstrip()



        # Extract image as PyTorch tensor

        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))



        # Handle images with less than three channels

        if len(img.shape) != 3:

            img = img.unsqueeze(0)

            img = img.expand((3, img.shape[1:]))



        _, h, w = img.shape

        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # Pad to square resolution

        img, pad = pad_to_square(img, 0)

        _, padded_h, padded_w = img.shape



        # ---------

        #  Label

        # ---------



        label_path = self.label_files[index % len(self.img_files)].rstrip()



        targets = None

        if os.path.exists(label_path):

            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            # Extract coordinates for unpadded + unscaled image

            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)

            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)

            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)

            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

            # Adjust for added padding

            x1 += pad[0]

            y1 += pad[2]

            x2 += pad[1]

            y2 += pad[3]

            # Returns (x, y, w, h)

            boxes[:, 1] = ((x1 + x2) / 2) / padded_w

            boxes[:, 2] = ((y1 + y2) / 2) / padded_h

            boxes[:, 3] *= w_factor / padded_w

            boxes[:, 4] *= h_factor / padded_h



            targets = torch.zeros((len(boxes), 6))

            targets[:, 1:] = boxes



        return img_path, img, targets



    def collate_fn(self, batch):

        paths, imgs, targets = list(zip(*batch))

        # Remove empty placeholder targets

        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets

        for i, boxes in enumerate(targets):

            boxes[:, 0] = i

        targets = torch.cat(targets, 0)

        # Selects new image size every tenth batch

        if self.multiscale and self.batch_count % 10 == 0:

            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape

        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        self.batch_count += 1

        return paths, imgs, targets



    def __len__(self):

        return len(self.img_files)

def to_cpu(tensor):

    return tensor.detach().cpu()





def load_classes(path):

    """

    Loads class labels at 'path'

    """

    fp = open(path, "r")

    names = fp.read().split("\n")[:-1]

    return names





def weights_init_normal(m):

    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm2d") != -1:

        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

        torch.nn.init.constant_(m.bias.data, 0.0)





def rescale_boxes(boxes, current_dim, original_shape):

    """ Rescales bounding boxes to the original shape """

    orig_h, orig_w = original_shape

    # The amount of padding that was added

    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))

    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))

    # Image height and width after padding is removed

    unpad_h = current_dim - pad_y

    unpad_w = current_dim - pad_x

    # Rescale bounding boxes to dimension of original image

    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w

    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h

    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w

    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes





def xywh2xyxy(x):

    y = x.new(x.shape)

    y[..., 0] = x[..., 0] - x[..., 2] / 2

    y[..., 1] = x[..., 1] - x[..., 3] / 2

    y[..., 2] = x[..., 0] + x[..., 2] / 2

    y[..., 3] = x[..., 1] + x[..., 3] / 2

    return y





def ap_per_class(tp, conf, pred_cls, target_cls):

    """ Compute the average precision, given the recall and precision curves.

    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.

    # Arguments

        tp:    True positives (list).

        conf:  Objectness value from 0-1 (list).

        pred_cls: Predicted object classes (list).

        target_cls: True object classes (list).

    # Returns

        The average precision as computed in py-faster-rcnn.

    """



    # Sort by objectness

    i = np.argsort(-conf)

    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]



    # Find unique classes

    unique_classes = np.unique(target_cls)



    # Create Precision-Recall curve and compute AP for each class

    ap, p, r = [], [], []

    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):

        i = pred_cls == c

        n_gt = (target_cls == c).sum()  # Number of ground truth objects

        n_p = i.sum()  # Number of predicted objects



        if n_p == 0 and n_gt == 0:

            continue

        elif n_p == 0 or n_gt == 0:

            ap.append(0)

            r.append(0)

            p.append(0)

        else:

            # Accumulate FPs and TPs

            fpc = (1 - tp[i]).cumsum()

            tpc = (tp[i]).cumsum()



            # Recall

            recall_curve = tpc / (n_gt + 1e-16)

            r.append(recall_curve[-1])



            # Precision

            precision_curve = tpc / (tpc + fpc)

            p.append(precision_curve[-1])



            # AP from recall-precision curve

            ap.append(compute_ap(recall_curve, precision_curve))



    # Compute F1 score (harmonic mean of precision and recall)

    p, r, ap = np.array(p), np.array(r), np.array(ap)

    f1 = 2 * p * r / (p + r + 1e-16)



    return p, r, ap, f1, unique_classes.astype("int32")





def compute_ap(recall, precision):

    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.



    # Arguments

        recall:    The recall curve (list).

        precision: The precision curve (list).

    # Returns

        The average precision as computed in py-faster-rcnn.

    """

    # correct AP calculation

    # first append sentinel values at the end

    mrec = np.concatenate(([0.0], recall, [1.0]))

    mpre = np.concatenate(([0.0], precision, [0.0]))



    # compute the precision envelope

    for i in range(mpre.size - 1, 0, -1):

        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])



    # to calculate area under PR curve, look for points

    # where X axis (recall) changes value

    i = np.where(mrec[1:] != mrec[:-1])[0]



    # and sum (\Delta recall) * prec

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap





def get_batch_statistics(outputs, targets, iou_threshold):

    """ Compute true positives, predicted scores and predicted labels per sample """

    batch_metrics = []

    for sample_i in range(len(outputs)):



        if outputs[sample_i] is None:

            continue



        output = outputs[sample_i]

        pred_boxes = output[:, :4]

        pred_scores = output[:, 4]

        pred_labels = output[:, -1]



        true_positives = np.zeros(pred_boxes.shape[0])



        annotations = targets[targets[:, 0] == sample_i][:, 1:]

        target_labels = annotations[:, 0] if len(annotations) else []

        if len(annotations):

            detected_boxes = []

            target_boxes = annotations[:, 1:]



            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):



                # If targets are found break

                if len(detected_boxes) == len(annotations):

                    break



                # Ignore if label is not one of the target labels

                if pred_label not in target_labels:

                    continue



                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)

                if iou >= iou_threshold and box_index not in detected_boxes:

                    true_positives[pred_i] = 1

                    detected_boxes += [box_index]

        batch_metrics.append([true_positives, pred_scores, pred_labels])

    return batch_metrics





def bbox_wh_iou(wh1, wh2):

    wh2 = wh2.t()

    w1, h1 = wh1[0], wh1[1]

    w2, h2 = wh2[0], wh2[1]

    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area





def bbox_iou(box1, box2, x1y1x2y2=True):

    """

    Returns the IoU of two bounding boxes

    """

    if not x1y1x2y2:

        # Transform from center and width to exact coordinates

        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2

        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2

        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2

        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    else:

        # Get the coordinates of bounding boxes

        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]

        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]



    # get the corrdinates of the intersection rectangle

    inter_rect_x1 = torch.max(b1_x1, b2_x1)

    inter_rect_y1 = torch.max(b1_y1, b2_y1)

    inter_rect_x2 = torch.min(b1_x2, b2_x2)

    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(

        inter_rect_y2 - inter_rect_y1 + 1, min=0

    )

    # Union Area

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)

    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)



    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)



    return iou





def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):

    """

    Removes detections with lower object confidence score than 'conf_thres' and performs

    Non-Maximum Suppression to further filter detections.

    Returns detections with shape:

        (x1, y1, x2, y2, object_conf, class_score, class_pred)

    """



    # From (center x, center y, width, height) to (x1, y1, x2, y2)

    prediction[..., :4] = xywh2xyxy(prediction[..., :4])

    output = [None for _ in range(len(prediction))]

    for image_i, image_pred in enumerate(prediction):

        # Filter out confidence scores below threshold

        image_pred = image_pred[image_pred[:, 4] >= conf_thres]

        # If none are remaining => process next image

        if not image_pred.size(0):

            continue

        # Object confidence times class confidence

        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]

        # Sort by it

        image_pred = image_pred[(-score).argsort()]

        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)

        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

        # Perform non-maximum suppression

        keep_boxes = []

        while detections.size(0):

            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres

            label_match = detections[0, -1] == detections[:, -1]

            # Indices of boxes with lower confidence scores, large IOUs and matching labels

            invalid = large_overlap & label_match

            weights = detections[invalid, 4:5]

            # Merge overlapping bboxes by order of confidence

            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()

            keep_boxes += [detections[0]]

            detections = detections[~invalid]

            #Handle NaN values:

            nans = torch.isnan(detections)

            detections[nans] = 0

        if keep_boxes:

            output[image_i] = torch.stack(keep_boxes)



    return output





def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):



    ByteTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor

    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor



    nB = pred_boxes.size(0)

    nA = pred_boxes.size(1)

    nC = pred_cls.size(-1)

    nG = pred_boxes.size(2)



    # Output tensors

    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(False)

    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(True)

    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)

    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)

    tx = FloatTensor(nB, nA, nG, nG).fill_(0)

    ty = FloatTensor(nB, nA, nG, nG).fill_(0)

    tw = FloatTensor(nB, nA, nG, nG).fill_(0)

    th = FloatTensor(nB, nA, nG, nG).fill_(0)

    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)



    # Convert to position relative to box

    target_boxes = target[:, 2:6] * nG

    gxy = target_boxes[:, :2]

    gwh = target_boxes[:, 2:]

    # Get anchors with best iou

    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])

    best_ious, best_n = ious.max(0)

    # Separate target values

    b, target_labels = target[:, :2].long().t()

    gx, gy = gxy.t()

    gw, gh = gwh.t()

    gi, gj = gxy.long().t()

    # Set masks

    obj_mask[b, best_n, gj, gi] = 1

    noobj_mask[b, best_n, gj, gi] = 0



    # Set noobj mask to zero where iou exceeds ignore threshold

    for i, anchor_ious in enumerate(ious.t()):

        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0



    # Coordinates

    tx[b, best_n, gj, gi] = gx - gx.floor()

    ty[b, best_n, gj, gi] = gy - gy.floor()

    # Width and height

    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)

    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # One-hot encoding of label

    tcls[b, best_n, gj, gi, target_labels] = 1

    # Compute label correctness and iou at best anchor

    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()

    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)



    tconf = obj_mask.float()

    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf



def parse_model_config(path):

    """Parses the yolo-v3 layer configuration file and returns module definitions"""

    file = open(path, 'r')

    lines = file.read().split('\n')

    lines = [x for x in lines if x and not x.startswith('#')]

    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces

    module_defs = []

    for line in lines:

        if line.startswith('['): # This marks the start of a new block

            module_defs.append({})

            module_defs[-1]['type'] = line[1:-1].rstrip()

            if module_defs[-1]['type'] == 'convolutional':

                module_defs[-1]['batch_normalize'] = 0

        else:

            key, value = line.split("=")

            value = value.strip()

            module_defs[-1][key.rstrip()] = value.strip()



    return module_defs



def parse_data_config(path):

    """Parses the data configuration file"""

    options = dict()

    options['gpus'] = '0,1,2,3'

    options['num_workers'] = '10'

    with open(path, 'r') as fp:

        lines = fp.readlines()

    for line in lines:

        line = line.strip()

        if line == '' or line.startswith('#'):

            continue

        key, value = line.split('=')

        options[key.strip()] = value.strip()

    return options

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):

    model.eval()



    # Get dataloader

    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)

    dataloader = torch.utils.data.DataLoader(

        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn

    )



    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor



    labels = []

    sample_metrics = []  # List of tuples (TP, confs, pred)

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):



        # Extract labels

        labels += targets[:, 1].tolist()

        # Rescale target

        targets[:, 2:] = xywh2xyxy(targets[:, 2:])

        targets[:, 2:] *= img_size



        imgs = Variable(imgs.type(Tensor), requires_grad=False)



        with torch.no_grad():

            outputs = model(imgs)

            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)



        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)



    # Concatenate sample statistics

    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)



    return precision, recall, AP, f1, ap_class
def create_modules(module_defs):

    """

    Constructs module list of layer blocks from module configuration in module_defs

    """

    hyperparams = module_defs.pop(0)

    output_filters = [int(hyperparams["channels"])]

    module_list = nn.ModuleList()

    for module_i, module_def in enumerate(module_defs):

        modules = nn.Sequential()



        if module_def["type"] == "convolutional":

            bn = int(module_def["batch_normalize"])

            filters = int(module_def["filters"])

            kernel_size = int(module_def["size"])

            pad = (kernel_size - 1) // 2

            modules.add_module(

                f"conv_{module_i}",

                nn.Conv2d(

                    in_channels=output_filters[-1],

                    out_channels=filters,

                    kernel_size=kernel_size,

                    stride=int(module_def["stride"]),

                    padding=pad,

                    bias=not bn,

                ),

            )

            if bn:

                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))

            if module_def["activation"] == "leaky":

                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))



        elif module_def["type"] == "maxpool":

            kernel_size = int(module_def["size"])

            stride = int(module_def["stride"])

            if kernel_size == 2 and stride == 1:

                modules.add_module(f"_debug_padding_{module_i}", nn.ZeroPad2d((0, 1, 0, 1)))

            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))

            modules.add_module(f"maxpool_{module_i}", maxpool)



        elif module_def["type"] == "upsample":

            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")

            modules.add_module(f"upsample_{module_i}", upsample)



        elif module_def["type"] == "route":

            layers = [int(x) for x in module_def["layers"].split(",")]

            filters = sum([output_filters[1:][i] for i in layers])

            modules.add_module(f"route_{module_i}", EmptyLayer())



        elif module_def["type"] == "shortcut":

            filters = output_filters[1:][int(module_def["from"])]

            modules.add_module(f"shortcut_{module_i}", EmptyLayer())



        elif module_def["type"] == "yolo":

            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]

            # Extract anchors

            anchors = [int(x) for x in module_def["anchors"].split(",")]

            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]

            anchors = [anchors[i] for i in anchor_idxs]

            num_classes = int(module_def["classes"])

            img_size = int(hyperparams["height"])

            # Define detection layer

            yolo_layer = YOLOLayer(anchors, num_classes, img_size)

            modules.add_module(f"yolo_{module_i}", yolo_layer)

        # Register module list and number of output filters

        module_list.append(modules)

        output_filters.append(filters)



    return hyperparams, module_list





class Upsample(nn.Module):

    """ nn.Upsample is deprecated """



    def __init__(self, scale_factor, mode="nearest"):

        super(Upsample, self).__init__()

        self.scale_factor = scale_factor

        self.mode = mode



    def forward(self, x):

        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

        return x





class EmptyLayer(nn.Module):

    """Placeholder for 'route' and 'shortcut' layers"""



    def __init__(self):

        super(EmptyLayer, self).__init__()





class YOLOLayer(nn.Module):

    """Detection layer"""



    def __init__(self, anchors, num_classes, img_dim=416):

        super(YOLOLayer, self).__init__()

        self.anchors = anchors

        self.num_anchors = len(anchors)

        self.num_classes = num_classes

        self.ignore_thres = 0.5

        self.mse_loss = nn.L1Loss()

        self.bce_loss = nn.BCELoss()

        self.obj_scale = 1

        self.noobj_scale = 1

        self.metrics = {}

        self.img_dim = img_dim

        self.grid_size = 0  # grid size



    def compute_grid_offsets(self, grid_size, cuda=True):

        self.grid_size = grid_size

        g = self.grid_size

        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.stride = self.img_dim / self.grid_size

        # Calculate offsets for each grid

        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)

        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)

        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])

        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))

        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))



    def forward(self, x, targets=None, img_dim=None):



        # Tensors for cuda support

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor



        self.img_dim = img_dim

        num_samples = x.size(0)

        grid_size = x.size(2)



        prediction = (

            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)

            .permute(0, 1, 3, 4, 2)

            .contiguous()

        )



        # Get outputs

        x = torch.sigmoid(prediction[..., 0])  # Center x

        y = torch.sigmoid(prediction[..., 1])  # Center y

        w = prediction[..., 2]  # Width

        h = prediction[..., 3]  # Height

        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf

        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.



        # If grid size does not match current we compute new offsets

        if grid_size != self.grid_size:

            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)



        # Add offset and scale with anchors

        pred_boxes = FloatTensor(prediction[..., :4].shape)

        pred_boxes[..., 0] = x.data + self.grid_x

        pred_boxes[..., 1] = y.data + self.grid_y

        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w

        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h



        output = torch.cat(

            (

                pred_boxes.view(num_samples, -1, 4) * self.stride,

                pred_conf.view(num_samples, -1, 1),

                pred_cls.view(num_samples, -1, self.num_classes),

            ),

            -1,

        )



        if targets is None:

            return output, 0

        else:

            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(

                pred_boxes=pred_boxes,

                pred_cls=pred_cls,

                target=targets,

                anchors=self.scaled_anchors,

                ignore_thres=self.ignore_thres,

            )



            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)

            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])

            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])

            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])

            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])

            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])

            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj

            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf_obj + loss_conf_noobj



            # Metrics

            cls_acc = 100 * class_mask[obj_mask].mean()

            conf_obj = pred_conf[obj_mask].mean()

            conf_noobj = pred_conf[noobj_mask].mean()

            conf50 = (pred_conf > 0.5).float()

            iou50 = (iou_scores > 0.5).float()

            iou75 = (iou_scores > 0.75).float()

            detected_mask = conf50 * class_mask * tconf

            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)

            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)

            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)



            self.metrics = {

                "loss": to_cpu(total_loss).item(),

                "x": to_cpu(loss_x).item(),

                "y": to_cpu(loss_y).item(),

                "w": to_cpu(loss_w).item(),

                "h": to_cpu(loss_h).item(),

                "conf": to_cpu(loss_conf).item(),

                "cls": to_cpu(loss_cls).item(),

                "cls_acc": to_cpu(cls_acc).item(),

                "recall50": to_cpu(recall50).item(),

                "recall75": to_cpu(recall75).item(),

                "precision": to_cpu(precision).item(),

                "conf_obj": to_cpu(conf_obj).item(),

                "conf_noobj": to_cpu(conf_noobj).item(),

                "grid_size": grid_size,

            }



            return output, total_loss





class Darknet(nn.Module):

    """YOLOv3 object detection model"""



    def __init__(self, config_path, img_size=416):

        super(Darknet, self).__init__()

        self.module_defs = parse_model_config(config_path)

        self.hyperparams, self.module_list = create_modules(self.module_defs)

        self.yolo_layers = [layer[0] for layer in self.module_list if hasattr(layer[0], "metrics")]

        self.img_size = img_size

        self.seen = 0

        self.header_info = np.array([0, 0, 0, self.seen, 0], dtype=np.int32)



    def forward(self, x, targets=None):

        img_dim = x.shape[2]

        loss = 0

        layer_outputs, yolo_outputs = [], []

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):

            if module_def["type"] in ["convolutional", "upsample", "maxpool"]:

                x = module(x)

            elif module_def["type"] == "route":

                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)

            elif module_def["type"] == "shortcut":

                layer_i = int(module_def["from"])

                x = layer_outputs[-1] + layer_outputs[layer_i]

            elif module_def["type"] == "yolo":

                x, layer_loss = module[0](x, targets, img_dim)

                loss += layer_loss

                yolo_outputs.append(x)

            layer_outputs.append(x)

        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

        return yolo_outputs if targets is None else (loss, yolo_outputs)



    def load_darknet_weights(self, weights_path):

        """Parses and loads the weights stored in 'weights_path'"""



        # Open the weights file

        with open(weights_path, "rb") as f:

            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values

            self.header_info = header  # Needed to write header when saving weights

            self.seen = header[3]  # number of images seen during training

            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights



        # Establish cutoff for loading backbone weights

        cutoff = None

        if "darknet53.conv.74" in weights_path:

            cutoff = 75



        ptr = 0

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):

            if i == cutoff:

                break

            if module_def["type"] == "convolutional":

                conv_layer = module[0]

                if module_def["batch_normalize"]:

                    # Load BN bias, weights, running mean and running variance

                    bn_layer = module[1]

                    num_b = bn_layer.bias.numel()  # Number of biases

                    # Bias

                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)

                    bn_layer.bias.data.copy_(bn_b)

                    ptr += num_b

                    # Weight

                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)

                    bn_layer.weight.data.copy_(bn_w)

                    ptr += num_b

                    # Running Mean

                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)

                    bn_layer.running_mean.data.copy_(bn_rm)

                    ptr += num_b

                    # Running Var

                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)

                    bn_layer.running_var.data.copy_(bn_rv)

                    ptr += num_b

                else:

                    # Load conv. bias

                    num_b = conv_layer.bias.numel()

                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)

                    conv_layer.bias.data.copy_(conv_b)

                    ptr += num_b

                # Load conv. weights

                num_w = conv_layer.weight.numel()

                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)

                conv_layer.weight.data.copy_(conv_w)

                ptr += num_w



    def save_darknet_weights(self, path, cutoff=-1):

        """

            @:param path    - path of the new weights file

            @:param cutoff  - save layers between 0 and cutoff (cutoff = -1 -> all are saved)

        """

        fp = open(path, "wb")

        self.header_info[3] = self.seen

        self.header_info.tofile(fp)



        # Iterate through layers

        for i, (module_def, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):

            if module_def["type"] == "convolutional":

                conv_layer = module[0]

                # If batch norm, load bn first

                if module_def["batch_normalize"]:

                    bn_layer = module[1]

                    bn_layer.bias.data.cpu().numpy().tofile(fp)

                    bn_layer.weight.data.cpu().numpy().tofile(fp)

                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)

                    bn_layer.running_var.data.cpu().numpy().tofile(fp)

                # Load conv bias

                else:

                    conv_layer.bias.data.cpu().numpy().tofile(fp)

                # Load conv weights

                conv_layer.weight.data.cpu().numpy().tofile(fp)



        fp.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



data_config = parse_data_config(data_config_str)

train_path = data_config["train"]

valid_path = data_config["valid"]

class_names = load_classes(data_config["names"])
model = Darknet(model_def).to(device)

model.apply(weights_init_normal)

#model.load_darknet_weights(pretrained_weights)

model.load_state_dict(torch.load(pretrained_weights)) #Use this if its a checkpoint
#TODO - Fix train.txt and valid.txt to have correct image name (missing '/WheatDetection_data/')

"""

dataset = ListDataset(train_path, multiscale=True)

dataloader = torch.utils.data.DataLoader(

    dataset,

    batch_size=batch_size,

    shuffle=True,

    num_workers=8,

    pin_memory=True,

    collate_fn=dataset.collate_fn,

)

learning_rate = model.hyperparams['learning_rate']

if optimizer_str == "Adam":

    optimizer = torch.optim.Adam(model.parameters(),lr=float(learning_rate))

else:

    optimizer = torch.optim.SGD(model.parameters(),lr=float(learning_rate))

    

for epoch in range(epochs):

        model.train()

        start_time = time.time()

        tot_loss = 0

        for batch_i, (_, imgs, targets) in enumerate(dataloader):

            batches_done = len(dataloader) * epoch + batch_i



            imgs = Variable(imgs.to(device))

            targets = Variable(targets.to(device), requires_grad=False)



            loss, outputs = model(imgs, targets)

            loss.backward()



            if batches_done % gradient_accumulations:

                # Accumulates gradient before each step

                optimizer.step()

                optimizer.zero_grad()

"""
model.eval()  # Set in evaluation mode

image_folder = "/kaggle/input/global-wheat-detection/test/"

class_path = "/kaggle/input/custom-files/classes.names"



dataloader = DataLoader(

    ImageFolder(image_folder, img_size=416),

    batch_size=1,

    shuffle=False,

    num_workers=0,

)

classes = load_classes(class_path)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

imgs = []  # Stores image paths

img_detections = []

for batch_i, (img_paths, input_imgs) in enumerate(dataloader):

    # Configure input

    input_imgs = Variable(input_imgs.type(Tensor))

    # Get detections

    with torch.no_grad():

        detections = model(input_imgs)

        detections = non_max_suppression(detections, 0.8, 0.4)

    # Save image and detections

    imgs.extend(img_paths)

    img_detections.extend(detections)



submission = []

for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        # Draw bounding boxes and labels of detections

        prediction_string = []

        if detections is not None:

            detections = rescale_boxes(detections, 416, (1024,1024))

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                if x1 < 0:

                    x1 = torch.FloatTensor([0.001])[0]

                if y1 < 0:

                    y1 = torch.FloatTensor([0.001])[0]

                if x2 > 1024:

                    x2 = torch.FloatTensor([1023.999])[0]

                if y2 > 1024:

                    y2 = torch.FloatTensor([1023.999])[0]

                x = torch.round(x1)

                y = torch.round(y1)

                h = torch.round(y2-y1)

                w = torch.round(x2-x1)

                prediction_string.append(f"{conf} {x} {y} {w} {h}")

        

        prediction_string = " ".join(prediction_string)

        

        filename = path.split("/")[-1].split(".")[0]



        submission.append([filename,prediction_string])
out_of_order_df = pd.DataFrame(submission, columns=["image_id","PredictionString"])

submission = pd.read_csv(f'/kaggle/input/global-wheat-detection/sample_submission.csv')

root_image = "/kaggle/input/custom-files/test/test/"

order = [f"{img}" for img in submission.image_id]

list_order = list()

for name in order:

    list_order.extend(out_of_order_df[out_of_order_df['image_id'] == name].values)

in_order_df = pd.DataFrame(list_order,columns=['image_id', 'PredictionString'])

in_order_df.to_csv('submission.csv', index=False)