import os

import cv2

import copy

import json

import random

import numpy as np

import mxnet as mx

import pandas as pd

import gluoncv as gcv

from multiprocessing import cpu_count

from multiprocessing.dummy import Pool





def load_dataset(root):

    csv = pd.read_csv(os.path.join(root, "train.csv"))

    data = {}

    for i in csv.index:

        key = csv["image_id"][i]

        bbox = json.loads(csv["bbox"][i])

        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], 0.0]

        if key in data:

            data[key].append(bbox)

        else:

            data[key] = [bbox]

    return sorted(

        [(k, os.path.join(root, "train", k + ".jpg"), v) for k, v in data.items()],

        key=lambda x: x[0]

    )



def load_image(path):

    with open(path, "rb") as f:

        buf = f.read()

    return mx.image.imdecode(buf)



def get_batches(dataset, batch_size, width=512, height=512, net=None, ctx=mx.cpu()):

    batches = len(dataset) // batch_size

    sampler = Sampler(dataset, width, height, net)

    stack_fn = [gcv.data.batchify.Stack()]

    pad_fn = [gcv.data.batchify.Pad(pad_val=-1)]

    if net is None:

        batchify_fn = gcv.data.batchify.Tuple(*(stack_fn + pad_fn))

    else:

        batchify_fn = gcv.data.batchify.Tuple(*(stack_fn * 6 + pad_fn))

    with Pool(cpu_count() * 2) as p:

        for i in range(batches):

            start = i * batch_size

            samples = p.map(sampler, range(start, start + batch_size))

            batch = batchify_fn(samples)

            yield [x.as_in_context(ctx) for x in batch]



def gauss_blur(image, level):

    return cv2.blur(image, (level * 2 + 1, level * 2 + 1))



def gauss_noise(image):

    for i in range(image.shape[2]):

        c = image[:, :, i]

        diff = 255 - c.max();

        noise = np.random.normal(0, random.randint(1, 6), c.shape)

        noise = (noise - noise.min()) / (noise.max() - noise.min())

        noise = diff * noise

        image[:, :, i] = c + noise.astype(np.uint8)

    return image





# This class was modified from YOLO3DefaultTrainTransform of GluonCV

class YOLO3TrainTransform:

    def __init__(self, width, height, net, mean=(0.485, 0.456, 0.406),

                 std=(0.229, 0.224, 0.225), **kwargs):

        self._width = width

        self._height = height

        self._mean = mean

        self._std = std



        # in case network has reset_ctx to gpu

        self._fake_x = mx.nd.zeros((1, 3, height, width))

        net = copy.deepcopy(net)

        net.collect_params().reset_ctx(None)

        with mx.autograd.train_mode():

            _, self._anchors, self._offsets, self._feat_maps, _, _, _, _ = net(self._fake_x)

        self._target_generator = gcv.model_zoo.yolo.yolo_target.YOLOV3PrefetchTargetGenerator(

            num_class=len(net.classes), **kwargs)



    def __call__(self, img, label):

        # random expansion with prob 0.5

        if np.random.uniform(0, 1) > 0.5:

            img, expand = gcv.data.transforms.image.random_expand(img, max_ratio=1.5, fill=114, keep_ratio=False)

            bbox = gcv.data.transforms.bbox.translate(label, x_offset=expand[0], y_offset=expand[1])

        else:

            img, bbox = img, label



        # random cropping

        h, w, _ = img.shape

        bbox, crop = gcv.data.transforms.experimental.bbox.random_crop_with_constraints(bbox, (w, h))

        x0, y0, w, h = crop

        img = mx.image.fixed_crop(img, x0, y0, w, h)



        # resize with random interpolation

        h, w, _ = img.shape

        interp = np.random.randint(0, 5)

        img = gcv.data.transforms.image.imresize(img, self._width, self._height, interp=interp)

        bbox = gcv.data.transforms.bbox.resize(bbox, (w, h), (self._width, self._height))



        # random horizontal&vertical flip

        h, w, _ = img.shape

        img, flips = gcv.data.transforms.image.random_flip(img, px=0.5, py=0.5)

        bbox = gcv.data.transforms.bbox.flip(bbox, (w, h), flip_x=flips[0], flip_y=flips[1])



        # random color jittering

        img = gcv.data.transforms.experimental.image.random_color_distort(img)



        # to tensor

        img = mx.nd.image.to_tensor(img)

        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)



        # generate training target so cpu workers can help reduce the workload on gpu

        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])

        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])

        gt_mixratio = mx.nd.array(bbox[np.newaxis, :, -1:])

        objectness, center_targets, scale_targets, weights, class_targets = self._target_generator(

            self._fake_x, self._feat_maps, self._anchors, self._offsets,

            gt_bboxes, gt_ids, gt_mixratio)

        return (img, objectness[0], center_targets[0], scale_targets[0], weights[0],

                class_targets[0], gt_bboxes[0])





class Sampler:

    def __init__(self, dataset, width, height, net=None, **kwargs):

        self._dataset = dataset

        if net is None:

            self._training_mode = False

            self._transform = gcv.data.transforms.presets.yolo.YOLO3DefaultValTransform(width, height, **kwargs)

        else:

            self._training_mode = True

            self._transform = YOLO3TrainTransform(width, height, net, **kwargs)



    def __call__(self, idx):

        if self._training_mode:

            raw, bboxes = self._load_mixup(idx)

            raw = raw.asnumpy()

            blur = random.randint(0, 3)

            if blur > 0:

                raw = gauss_blur(raw, blur)

            raw = gauss_noise(raw)

            h, w, _ = raw.shape

            rot = random.randint(0, 3)

            if rot > 0:

                raw = np.rot90(raw, k=rot)

                if rot == 1:

                    raw_bboxes = bboxes.copy()

                    bboxes[:, [0, 2]] = raw_bboxes[:, [1, 3]]

                    bboxes[:, [1, 3]] = w - raw_bboxes[:, [2, 0]]

                elif rot == 2:

                    bboxes[:, [0, 1, 2, 3]] = np.array([[w, h, w, h]]) - bboxes[:, [2, 3, 0, 1]]

                elif rot == 3:

                    raw_bboxes = bboxes.copy()

                    bboxes[:, [0, 2]] = h - raw_bboxes[:, [1, 3]]

                    bboxes[:, [1, 3]] = raw_bboxes[:, [2, 0]]

                raw_bboxes = bboxes.copy()

                bboxes[:, 0] = np.min(raw_bboxes[:, [0, 2]], axis=1)

                bboxes[:, 1] = np.min(raw_bboxes[:, [1, 3]], axis=1)

                bboxes[:, 2] = np.max(raw_bboxes[:, [0, 2]], axis=1)

                bboxes[:, 3] = np.max(raw_bboxes[:, [1, 3]], axis=1)

            raw = mx.nd.array(raw)

        else:

            raw = load_image(self._dataset[idx][1])

            bboxes = np.array(self._dataset[idx][2])

        res = self._transform(raw, bboxes)

        return [mx.nd.array(x) for x in res]



    def _load_mixup(self, idx1):

        r = random.gauss(0.5, 0.5 / 1.96)

        if r > 0.0:

            raw1 = load_image(self._dataset[idx1][1])

            bboxes1 = np.array(self._dataset[idx1][2])

            if r >= 1.0:

                return raw1, np.hstack([bboxes1, np.full((bboxes1.shape[0], 1), 1.0)])

        idx2 = random.randint(0, len(self._dataset) - 1)

        raw2 = load_image(self._dataset[idx2][1])

        bboxes2 = np.array(self._dataset[idx2][2])

        if r <= 0.0:

            return raw2, np.hstack([bboxes2, np.full((bboxes2.shape[0], 1), 1.0)])

        h = max(raw1.shape[0], raw2.shape[0])

        w = max(raw1.shape[1], raw2.shape[1])

        mix_raw = mx.nd.zeros(shape=(h, w, 3), dtype="float32")

        mix_raw[:raw1.shape[0], :raw1.shape[1], :] += raw1.astype("float32") * r

        mix_raw[:raw2.shape[0], :raw2.shape[1], :] += raw2.astype("float32") * (1.0 - r)

        mix_bboxes = np.vstack([

            np.hstack([bboxes1, np.full((bboxes1.shape[0], 1), r)]),

            np.hstack([bboxes2, np.full((bboxes2.shape[0], 1), 1.0 - r)])

        ])

        return mix_raw.astype("uint8"), mix_bboxes

import mxnet as mx

import gluoncv as gcv





def load_model(path, ctx=mx.cpu()):

    net = gcv.model_zoo.yolo3_darknet53_custom(["wheat"], pretrained_base=False)

    net.set_nms(post_nms=150)

    net.load_parameters(path, ctx=ctx)

    return net

!ln -snf /kaggle/input/weighted-boxes-fusion/ensemble_boxes && ls -lh
import random

import numpy as np

import gluoncv as gcv

from ensemble_boxes import *





def inference(models, path):

    raw = load_image(path)

    rh, rw, _ = raw.shape

    classes_list = []

    scores_list = []

    bboxes_list = []

    for _ in range(16):

        img, flips = gcv.data.transforms.image.random_flip(raw, px=0.5, py=0.5)

        x, _ = gcv.data.transforms.presets.yolo.transform_test(img, short=img_s)

        _, _, xh, xw = x.shape

        rot = random.randint(0, 3)

        if rot > 0:

            x = np.rot90(x.asnumpy(), k=rot, axes=(2, 3))

        for model in models:

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

    return bboxes, scores, classes

import os

import time

import random

import mxnet as mx

import pandas as pd



rounds = 2

max_epochs = 5

learning_rate = 0.001

batch_size = 2

img_s = 1024

threshold = 0.1

context = mx.gpu()



print("Loading pre-trained model...")

model = load_model("../input/yolov3/global-wheat-yolo3-darknet53_best2.params", ctx=context)



print("Loading training set...")

dataset = load_dataset("/kaggle/input/global-wheat-detection")



print("Loading test images...")

test_images = [

    (os.path.join(dirname, filename), os.path.splitext(filename)[0])

        for dirname, _, filenames in os.walk('/kaggle/input/global-wheat-detection/test') for filename in filenames

]



if len(test_images) > 10:

    best_score = 0.0

    metrics = [gcv.utils.metrics.VOCMApMetric(iou_thresh=iou) for iou in [0.5, 0.55, 0.6]]

    for pseudo_models in [[model]] * rounds:

        print("Pseudo labaling...")

        pseudo_set = []

        for path, image_id in test_images:

            print(path)

            bboxes, scores, classes = inference(pseudo_models, path)

            label = [

                [round(x) for x in bboxes[i].tolist()] + [0.0] for i in range(classes.shape[0])

                    if model.classes[int(classes[i])] == "wheat" and scores[i] > threshold

            ]

            if len(label) > 0:

                pseudo_set.append((image_id, path, label))



        training_set = pseudo_set

        validation_set = dataset[int(len(dataset) * 0.9):]

        print("Training set: ", len(training_set))

        print("Validation set: ", len(validation_set))



        print("Re-training...")

        trainer = mx.gluon.Trainer(model.collect_params(), "Nadam", {

            "learning_rate": learning_rate

        })

        for epoch in range(max_epochs):

            ts = time.time()

            random.shuffle(training_set)

            training_total_L = 0.0

            training_batches = 0

            for x, objectness, center_targets, scale_targets, weights, class_targets, gt_bboxes in get_batches(training_set, batch_size, width=img_s, height=img_s, net=model, ctx=context):

                training_batches += 1

                with mx.autograd.record():

                    obj_loss, center_loss, scale_loss, cls_loss = model(x, gt_bboxes, objectness, center_targets, scale_targets, weights, class_targets)

                    L = obj_loss + center_loss + scale_loss + cls_loss

                    L.backward()

                trainer.step(x.shape[0])

                training_batch_L = mx.nd.mean(L).asscalar()

                if training_batch_L != training_batch_L:

                    raise ValueError()

                training_total_L += training_batch_L

                print("[Epoch %d  Batch %d]  batch_loss %.10f  average_loss %.10f  elapsed %.2fs" % (

                    epoch, training_batches, training_batch_L, training_total_L / training_batches, time.time() - ts

                ))

            training_avg_L = training_total_L / training_batches

            for metric in metrics:

                metric.reset()

            for x, label in get_batches(validation_set, batch_size, width=img_s, height=img_s, ctx=context):

                classes, scores, bboxes = model(x)

                for metric in metrics:

                    metric.update(

                        bboxes,

                        classes.reshape((0, -1)),

                        scores.reshape((0, -1)),

                        label[:, :, :4],

                        label[:, :, 4:5].reshape((0, -1))

                    )

            score = mx.nd.array([metric.get()[1] for metric in metrics], ctx=context).mean()

            print("[Epoch %d]  training_loss %.10f  validation_score %.10f  best_score %.10f  duration %.2fs" % (

                epoch + 1, training_avg_L, score.asscalar(), best_score, time.time() - ts

            ))

            if score.asscalar() > best_score:

                best_score = score.asscalar()

                model.save_parameters("global-wheat-yolo3-darknet53.params")



        print("Loading re-trained model...")

        model = load_model("global-wheat-yolo3-darknet53.params", ctx=context)



print("Inference...")

results = []

for path, image_id in test_images:

    print(path)

    bboxes, scores, classes = inference([model], path)

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
