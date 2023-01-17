DEBUG = True
import pandas as pd

import numpy as np

import cv2

import os

import re



from PIL import Image



import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2



import torch

import torchvision



from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator



from torch.utils.data import DataLoader, Dataset

from torch.utils.data.sampler import SequentialSampler



from matplotlib import pyplot as plt



import warnings

warnings.filterwarnings('ignore')



DIR_INPUT = '/kaggle/input/global-wheat-detection'

DIR_TRAIN = f'{DIR_INPUT}/train'

DIR_TEST = f'{DIR_INPUT}/test'
train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')

train_df.shape
train_df['x'] = -1

train_df['y'] = -1

train_df['w'] = -1

train_df['h'] = -1



def expand_bbox(x):

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r



train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

train_df.drop(columns=['bbox'], inplace=True)

train_df['x'] = train_df['x'].astype(np.float)

train_df['y'] = train_df['y'].astype(np.float)

train_df['w'] = train_df['w'].astype(np.float)

train_df['h'] = train_df['h'].astype(np.float)
image_ids = train_df['image_id'].unique()

valid_ids = image_ids[-10:] if DEBUG else image_ids[-665:]

train_ids = image_ids[:10] if DEBUG else image_ids[:-665]
valid_df = train_df[train_df['image_id'].isin(valid_ids)]

train_df = train_df[train_df['image_id'].isin(train_ids)]
valid_df.shape, train_df.shape
class WheatDataset(Dataset):



    def __init__(self, dataframe, image_dir, transforms=None):

        super().__init__()



        self.image_ids = dataframe['image_id'].unique()

        self.df = dataframe

        self.image_dir = image_dir

        self.transforms = transforms



    def __getitem__(self, index: int):



        image_id = self.image_ids[index]

        records = self.df[self.df['image_id'] == image_id]



        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        image /= 255.0



        boxes = records[['x', 'y', 'w', 'h']].values

#         boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

#         boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        

#         area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        area = boxes[:,2]*boxes[:,3]

        area = torch.as_tensor(area, dtype=torch.float32)



        # there is only one class

        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        

        # suppose all instances are not crowd

        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        

        target = {}

        target['boxes'] = boxes

        target['labels'] = labels

        # target['masks'] = None

        target['image_id'] = torch.tensor([index])

        target['area'] = area

        target['iscrowd'] = iscrowd



        if self.transforms:

            sample = {

                'image': image,

                'bboxes': target['boxes'],

                'labels': labels

            }

            sample = self.transforms(**sample)

            image = sample['image']

            

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)



        return image, target, image_id



    def __len__(self) -> int:

        return self.image_ids.shape[0]
# Albumentations

def get_train_transform():

    return A.Compose([

        A.Flip(0.5),

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'coco', 'label_fields': ['labels']})



def get_valid_transform():

    return A.Compose([

        ToTensorV2(p=1.0)

    ], bbox_params={'format': 'coco', 'label_fields': ['labels']})

# load Detection Transformer model; pre-trained on COCO

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
num_classes = 2  # 1 class (wheat) + background



# # get number of input features for the classifier

# in_features = model.roi_heads.box_predictor.cls_score.in_features



# # replace the pre-trained head with a new one

# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
import torch.nn as nn



model.class_embed = nn.Linear(256, num_classes, bias = True)
from torchvision.ops.boxes import box_area



def box_iou(boxes1, boxes2):

    area1 = box_area(boxes1)

    area2 = box_area(boxes2)



    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]

    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]



    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]



    union = area1[:, None] + area2 - inter



    iou = inter / union

    return iou, union
def generalized_box_iou(boxes1, boxes2):

    """

    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)

    and M = len(boxes2)

    """

    # degenerate boxes gives inf / nan results

    # so do an early check

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()

    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    iou, union = box_iou(boxes1, boxes2)



    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])

    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])



    wh = (rb - lt).clamp(min=0)  # [N,M,2]

    area = wh[:, :, 0] * wh[:, :, 1]



    return iou - (area - union) / area
def box_cxcywh_to_xyxy(x):

    x_c, y_c, w, h = x.unbind(-1)

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),

         (x_c + 0.5 * w), (y_c + 0.5 * h)]

    return torch.stack(b, dim=-1)
from scipy.optimize import linear_sum_assignment



class HungarianMatcher(nn.Module):

    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,

    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,

    while the others are un-matched (and thus treated as non-objects).

    """



    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):

        """Creates the matcher

        Params:

            cost_class: This is the relative weight of the classification error in the matching cost

            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost

            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost

        """

        super().__init__()

        self.cost_class = cost_class

        self.cost_bbox = cost_bbox

        self.cost_giou = cost_giou

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"



    @torch.no_grad()

    def forward(self, outputs, targets):

        """ Performs the matching

        Params:

            outputs: This is a dict that contains at least these entries:

                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits

                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:

                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth

                           objects in the target) containing the class labels

                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:

            A list of size batch_size, containing tuples of (index_i, index_j) where:

                - index_i is the indices of the selected predictions (in order)

                - index_j is the indices of the corresponding selected targets (in order)

            For each batch element, it holds:

                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

        """

        bs, num_queries = outputs["pred_logits"].shape[:2]



        # We flatten to compute the cost matrices in a batch

        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]



        # Also concat the target labels and boxes

        tgt_ids = torch.cat([v["labels"] for v in targets])

        tgt_bbox = torch.cat([v["boxes"] for v in targets])



        # Compute the classification cost. Contrary to the loss, we don't use the NLL,

        # but approximate it in 1 - proba[target class].

        # The 1 is a constant that doesn't change the matching, it can be ommitted.

        cost_class = -out_prob[:, tgt_ids]



        # Compute the L1 cost between boxes

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)



        # Compute the giou cost betwen boxes

        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))



        # Final cost matrix

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        C = C.view(bs, num_queries, -1).cpu()



        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
@torch.no_grad()

def accuracy(output, target, topk=(1,)):

    """Computes the precision@k for the specified values of k"""

    if target.numel() == 0:

        return [torch.zeros([], device=output.device)]

    maxk = max(topk)

    batch_size = target.size(0)



    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))



    res = []

    for k in topk:

        correct_k = correct[:k].view(-1).float().sum(0)

        res.append(correct_k.mul_(100.0 / batch_size))

    return res
import torch.nn.functional as F





class SetCriterion(nn.Module):

    """ This class computes the loss for DETR.

    The process happens in two steps:

        1) we compute hungarian assignment between ground truth boxes and the outputs of the model

        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):

        """ Create the criterion.

        Parameters:

            num_classes: number of object categories, omitting the special no-object category

            matcher: module able to compute a matching between targets and proposals

            weight_dict: dict containing as key the names of the losses and as values their relative weight.

            eos_coef: relative classification weight applied to the no-object category

            losses: list of all the losses to be applied. See get_loss for list of available losses.

        """

        super().__init__()

        self.num_classes = num_classes

        self.matcher = matcher

        self.weight_dict = weight_dict

        self.eos_coef = eos_coef

        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)

        empty_weight[-1] = self.eos_coef

        self.register_buffer('empty_weight', empty_weight)



    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):

        """Classification loss (NLL)

        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]

        """

        assert 'pred_logits' in outputs

        src_logits = outputs['pred_logits']



        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,

                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o



        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

        losses = {'loss_ce': loss_ce}



        if log:

            # TODO this should probably be a separate loss, not hacked in this one here

            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses



    @torch.no_grad()

    def loss_cardinality(self, outputs, targets, indices, num_boxes):

        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients

        """

        pred_logits = outputs['pred_logits']

        device = pred_logits.device

        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)

        # Count the number of predictions that are NOT "no-object" (which is the last class)

        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)

        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())

        losses = {'cardinality_error': card_err}

        return losses



    def loss_boxes(self, outputs, targets, indices, num_boxes):

        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss

           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]

           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.

        """

        assert 'pred_boxes' in outputs

        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs['pred_boxes'][idx]

        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)



        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')



        losses = {}

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes



        loss_giou = 1 - torch.diag(generalized_box_iou(

            box_cxcywh_to_xyxy(src_boxes),

            box_cxcywh_to_xyxy(target_boxes)))

        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses



    def loss_masks(self, outputs, targets, indices, num_boxes):

        """Compute the losses related to the masks: the focal loss and the dice loss.

           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]

        """

        assert "pred_masks" in outputs



        src_idx = self._get_src_permutation_idx(indices)

        tgt_idx = self._get_tgt_permutation_idx(indices)



        src_masks = outputs["pred_masks"]



        # TODO use valid to mask invalid areas due to padding in loss

        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()

        target_masks = target_masks.to(src_masks)



        src_masks = src_masks[src_idx]

        # upsample predictions to the target size

        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],

                                mode="bilinear", align_corners=False)

        src_masks = src_masks[:, 0].flatten(1)



        target_masks = target_masks[tgt_idx].flatten(1)



        losses = {

            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),

            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),

        }

        return losses



    def _get_src_permutation_idx(self, indices):

        # permute predictions following indices

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])

        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx



    def _get_tgt_permutation_idx(self, indices):

        # permute targets following indices

        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])

        tgt_idx = torch.cat([tgt for (_, tgt) in indices])

        return batch_idx, tgt_idx



    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):

        loss_map = {

            'labels': self.loss_labels,

            'cardinality': self.loss_cardinality,

            'boxes': self.loss_boxes,

            'masks': self.loss_masks

        }

        assert loss in loss_map, f'do you really want to compute {loss} loss?'

        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)



    def forward(self, outputs, targets):

        """ This performs the loss computation.

        Parameters:

             outputs: dict of tensors, see the output specification of the model for the format

             targets: list of dicts, such that len(targets) == batch_size.

                      The expected keys in each dict depends on the losses applied, see each loss' doc

        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}



        # Retrieve the matching between the outputs of the last layer and the targets

        indices = self.matcher(outputs_without_aux, targets)



        # Compute the average number of target boxes accross all nodes, for normalization purposes

        num_boxes = sum(len(t["labels"]) for t in targets)

        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

#         if is_dist_avail_and_initialized():

#             torch.distributed.all_reduce(num_boxes)

        num_boxes = torch.clamp(num_boxes / 1, min=1).item()



        # Compute all the requested losses

        losses = {}

        for loss in self.losses:

            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))



        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.

        if 'aux_outputs' in outputs:

            for i, aux_outputs in enumerate(outputs['aux_outputs']):

                indices = self.matcher(aux_outputs, targets)

                for loss in self.losses:

                    if loss == 'masks':

                        # Intermediate masks losses are too costly to compute, we ignore them.

                        continue

                    kwargs = {}

                    if loss == 'labels':

                        # Logging is enabled only for the last layer

                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)

                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}

                    losses.update(l_dict)



        return losses
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



model.train()

matcher = HungarianMatcher()



# Play with these weights to improve the results

weight_dict = weight_dict = {'loss_ce': 1, 'loss_bbox': 1 , 'loss_giou': 1}



losses = ['labels', 'boxes', 'cardinality']



#eos_coef is the weightage for the background class

criterion = SetCriterion(num_classes-1, matcher, weight_dict, eos_coef = 0.1, losses=losses)



criterion.to(device)
class Averager:

    def __init__(self):

        self.current_total = 0.0

        self.iterations = 0.0



    def send(self, value):

        self.current_total += value

        self.iterations += 1



    @property

    def value(self):

        if self.iterations == 0:

            return 0

        else:

            return 1.0 * self.current_total / self.iterations



    def reset(self):

        self.current_total = 0.0

        self.iterations = 0.0

def collate_fn(batch):

    return tuple(zip(*batch))



train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform())

valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform())





# split the dataset in train and test set

indices = torch.randperm(len(train_dataset)).tolist()



train_data_loader = DataLoader(

    train_dataset,

    batch_size=2,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)



valid_data_loader = DataLoader(

    valid_dataset,

    batch_size=2,

    shuffle=False,

    num_workers=4,

    collate_fn=collate_fn

)
images, targets, image_ids = next(iter(train_data_loader))

images = list(image.to(device) for image in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
boxes = targets[0]['boxes'].cpu().numpy().astype(np.int32)

sample = images[0].permute(1,2,0).cpu().numpy()
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2]+box[0], box[3]+box[1]),

                  (220, 0, 0), 3)

    

ax.set_axis_off()

ax.imshow(sample)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

lr_scheduler = None



num_epochs = 1 if DEBUG else 10
loss_hist = Averager()

itr = 1



for epoch in range(num_epochs):

    loss_hist.reset()

    

    for images, targets, image_ids in train_data_loader:



        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]



        output = model(images)

        

        loss_dict = criterion(output, targets)

        weight_dict = criterion.weight_dict

        

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_value = losses.item()



        loss_hist.send(loss_value)



        optimizer.zero_grad()

        losses.backward()

        optimizer.step()



        if itr % 50 == 0:

            print(f"Iteration #{itr} loss: {loss_value}")



        itr += 1

    

    # update the learning rate

    if lr_scheduler is not None:

        lr_scheduler.step()



    print(f"Epoch #{epoch} loss: {loss_hist.value}")   
images, targets, image_ids = next(iter(valid_data_loader))
images = list(img.to(device) for img in images)

targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
boxes = targets[1]['boxes'].cpu().numpy().astype(np.int32)

sample = images[1].permute(1,2,0).cpu().numpy()
model.eval()

model.to(device)

cpu_device = torch.device("cpu")



outputs = model(images)

outputs = [{k: v.to(cpu_device) for k, v in outputs.items()}]
fig, ax = plt.subplots(1, 1, figsize=(16, 8))



for box in boxes:

    cv2.rectangle(sample,

                  (box[0], box[1]),

                  (box[2]+box[0], box[3]+box[1]),

                  (220, 0, 0), 3)

    

ax.set_axis_off()

ax.imshow(sample)
torch.save(model.state_dict(), 'detr_baseline.pth')