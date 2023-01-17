import mxnet as mx





def load_image(path):

    with open(path, "rb") as f:

        buf = f.read()

    return mx.image.imdecode(buf)

import mxnet as mx

import gluoncv as gcv





def load_model(path, ctx=mx.cpu()):

    net = gcv.model_zoo.yolo3_darknet53_custom(["wheat"], pretrained_base=False)

    net.set_nms(post_nms=150)

    net.load_parameters(path, ctx=ctx)

    return net

!ln -snf /kaggle/input/weighted-boxes-fusion/ensemble_boxes && ls -lh
import os

import random

import numpy as np

import mxnet as mx

import pandas as pd

import gluoncv as gcv

from ensemble_boxes import *



threshold = 0.1

img_s = 608

context = mx.gpu()



print("Loading test images...")

images = [

    (os.path.join(dirname, filename), os.path.splitext(filename)[0])

        for dirname, _, filenames in os.walk('/kaggle/input/global-wheat-detection/test') for filename in filenames

]



print("Loading model...")

model = load_model("/kaggle/input/global-wheat-detection-private/global-wheat-yolo3-darknet53_608x608_clean.params", ctx=context)



print("Inference...")

results = []

for path, image_id in images:

    print(path)

    raw = load_image(path)

    rh, rw, _ = raw.shape

    classes_list = []

    scores_list = []

    bboxes_list = []

    for _ in range(5):

        img, flips = gcv.data.transforms.image.random_flip(raw, px=0.5, py=0.5)

        x, _ = gcv.data.transforms.presets.yolo.transform_test(img, short=img_s)

        _, _, xh, xw = x.shape

        rot = random.randint(0, 3)

        if rot > 0:

            x = np.rot90(x.asnumpy(), k=rot, axes=(2, 3))

        classes, scores, bboxes = model(mx.nd.array(x, ctx=context))

        if rot > 0:

            if rot == 1:

                raw_bboxes = bboxes.copy()

                bboxes[0, :, [0, 2]] = xh - raw_bboxes[0, :, [1, 3]]

                bboxes[0, :, [1, 3]] = raw_bboxes[0, :, [2, 0]]

            elif rot == 2:

                bboxes[0, :, [0, 1, 2, 3]] = mx.nd.array([[xw], [xh], [xw], [xh]], ctx=context) - bboxes[0, :, [2, 3, 0, 1]]

            elif rot == 3:

                raw_bboxes = bboxes.copy()

                bboxes[0, :, [0, 2]] = raw_bboxes[0, :, [1, 3]]

                bboxes[0, :, [1, 3]] = xw - raw_bboxes[0, :, [2, 0]]

            raw_bboxes = bboxes.copy()

            bboxes[0, :, 0] = raw_bboxes[0, :, [0, 2]].min(axis=0)

            bboxes[0, :, 1] = raw_bboxes[0, :, [1, 3]].min(axis=0)

            bboxes[0, :, 2] = raw_bboxes[0, :, [0, 2]].max(axis=0)

            bboxes[0, :, 3] = raw_bboxes[0, :, [1, 3]].max(axis=0)

        bboxes[0, :, :] = gcv.data.transforms.bbox.flip(bboxes[0, :, :], (xw, xh), flip_x=flips[0], flip_y=flips[1])

        bboxes[0, :, 0::2] = (bboxes[0, :, 0::2] / (xw - 1)).clip(0.0, 1.0)

        bboxes[0, :, 1::2] = (bboxes[0, :, 1::2] / (xh - 1)).clip(0.0, 1.0)

        classes_list.append([

            int(classes[0, i].asscalar()) for i in range(classes.shape[1])

                if classes[0, i].asscalar() >= 0.0



        ])

        scores_list.append([

            scores[0, i].asscalar() for i in range(classes.shape[1])

                if classes[0, i].asscalar() >= 0.0



        ])

        bboxes_list.append([

            bboxes[0, i].asnumpy().tolist() for i in range(classes.shape[1])

                if classes[0, i].asscalar() >= 0.0

        ])

    bboxes, scores, classes = weighted_boxes_fusion(bboxes_list, scores_list, classes_list)

    bboxes[:, 0::2] *= rw - 1

    bboxes[:, 1::2] *= rh - 1

    bboxes[:, 2:4] -= bboxes[:, 0:2]

    results.append({

        "image_id": image_id,

        "PredictionString": " ".join([

            " ".join([str(x) for x in [scores[i]] + [round(x) for x in bboxes[i].tolist()]])

                for i in range(classes.shape[0])

                    if model.classes[int(classes[i])] == "wheat" and scores[i] > threshold

        ])

    })

pd.DataFrame(results, columns=['image_id', 'PredictionString']).to_csv('submission.csv', index=False)
