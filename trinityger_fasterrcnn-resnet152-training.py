import pandas as pd

import numpy as np

import cv2

import os, re



import torch



import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone



from torch.utils.data import DataLoader, Dataset



from matplotlib import pyplot as plt 

plt.rcParams['figure.figsize'] = (10.0, 10.0)
DATA_DIR = "/kaggle/input/global-wheat-detection"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ON_CPU = DEVICE == torch.device('cpu')

TRAIN_BATCH_SIZE = 4 if ON_CPU else 6

VALID_BATCH_SIZE = 2 if ON_CPU else 2

NUM_EPOCHS = 5 if ON_CPU else 17

NEW_COLUMNS = ['x', 'y', 'w', 'h']
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

test_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))



UNIQ_TRAIN_IMAGE_IDS = train_df["image_id"].unique()



train_df.shape, test_df.shape
# expand the bbox coordinates into x, y, w, h

def expand_bbox(x):

    # also convert everything to np.float

    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x), dtype=np.float)

    if len(r) == 0:

        r = [-1, -1, -1, -1]

    return r



# initialize new columns with -1

for new_column in NEW_COLUMNS:

    train_df[new_column] = -1



train_df[NEW_COLUMNS] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))

train_df.drop(columns=['bbox'], inplace=True)

train_df.head()
def load_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    assert image is not None, f"IMAGE NOT FOUND AT {image_path}"

    return image
def draw_bboxes(boxes, image, color=(255,0,0)):

    for box in boxes:

        cv2.rectangle(

            image,

            (int(box[0]), int(box[1])),

            (int(box[2]), int(box[3])),

            color, 3

        )

    return image
def plot_random_train_sample():

    image_id = np.random.choice(UNIQ_TRAIN_IMAGE_IDS)

    plt.title(image_id)

    image = load_image(os.path.join(DATA_DIR, "train", f"{image_id}.jpg"))

    bboxes = (train_df[train_df["image_id"] == image_id][NEW_COLUMNS]).to_numpy()

    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]

    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

    plt.imshow(draw_bboxes(bboxes, image))
plot_random_train_sample()
class WheatDataset(Dataset):

    def __init__(self, df, image_dir, transforms=None):

        super().__init__()



        self.df = df

        self.image_dir = image_dir

        self.transforms = transforms

        self.image_ids = df['image_id'].unique()



    def __len__(self) -> int:

        return len(self.image_ids)



    def __getitem__(self, idx: int):

        image_id = self.image_ids[idx]

        image = load_image(os.path.join(self.image_dir, f"{image_id}.jpg")).astype(np.float32)

        image /= 255.0

        # change the shape from [h,w,c] to [c,h,w]  

        image = torch.from_numpy(image).permute(2,0,1)



        records = self.df[self.df['image_id'] == image_id]



        boxes = records[NEW_COLUMNS].values

        area = boxes[:, 2] * boxes[:, 3]

        area = torch.as_tensor(area, dtype=torch.float32)

        # change the co-ordinates into expected [x, y, x+w, y+h] format

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]

        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        

        boxes = torch.as_tensor(boxes, dtype=torch.float32)



        # since all the boxes are wheat, it's all 1s

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        

        # consider iscrowd false for all the boxes

        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)



        target = {}

        target["boxes"] = boxes

        target["labels"] = labels

        target["image_id"] = torch.tensor([idx])

        target["area"] = area

        target["iscrowd"] = iscrowd



        if self.transforms:

            pass



        return image, target
def get_model():

    """

    https://stackoverflow.com/questions/58362892/resnet-18-as-backbone-in-faster-r-cnn

    """

    backbone = resnet_fpn_backbone('resnet152', pretrained=True)

    model = FasterRCNN(backbone, num_classes=2)

    return model
# 12% into validation

n_validation = int(0.13* len(UNIQ_TRAIN_IMAGE_IDS))

valid_ids = UNIQ_TRAIN_IMAGE_IDS[-n_validation:]

train_ids = UNIQ_TRAIN_IMAGE_IDS[:-n_validation]



df_in_valid = train_df[train_df['image_id'].isin(valid_ids)]

df_in_train = train_df[train_df['image_id'].isin(train_ids)]



print("%i training samples\n%i validation samples" % (len(df_in_train["image_id"].unique()), len(df_in_valid["image_id"].unique())))
train_dataset = WheatDataset(df_in_train, os.path.join(DATA_DIR, "train"))

valid_dataset = WheatDataset(df_in_valid, os.path.join(DATA_DIR, "train"))



# since our single getitem returns image, targets. [shape of targets is different depending on the number of bounding boxes in the image] ?

def collate_fn(batch):

    return tuple(zip(*batch))



train_data_loader = DataLoader(

    train_dataset,

    batch_size=TRAIN_BATCH_SIZE,

    shuffle=True,

    num_workers=1,

    collate_fn=collate_fn

)



valid_data_loader = DataLoader(

    valid_dataset,

    batch_size=VALID_BATCH_SIZE,

    shuffle=False,

    num_workers=1,

    collate_fn=collate_fn

)
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
# Taken from https://github.com/Bjarten/early-stopping-pytorch

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):

        """

        Args:

            patience (int): How long to wait after last time validation loss improved.

                            Default: 7

            verbose (bool): If True, prints a message for each validation loss improvement. 

                            Default: False

            delta (float): Minimum change in the monitored quantity to qualify as an improvement.

                            Default: 0

            path (str): Path for the checkpoint to be saved to.

                            Default: 'checkpoint.pt'

            trace_func (function): trace print function.

                            Default: print            

        """

        self.patience = patience

        self.verbose = verbose

        self.counter = 0

        self.best_score = None

        self.early_stop = False

        self.val_loss_min = np.Inf

        self.delta = delta

        self.path = path

        self.epoch = 0

        self.trace_func = trace_func

    def __call__(self, val_loss, epoch, model):



        score = -val_loss



        if self.best_score is None:

            self.best_score = score

            self.epoch = epoch

            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:

            self.counter += 1

            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:

                self.early_stop = True

        else:

            self.best_score = score

            self.epoch = epoch

            self.save_checkpoint(val_loss, model)

            self.counter = 0



    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''

        if self.verbose:

            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss
model = get_model()

model.to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
%%time

train_losses = Averager()

totalTrainLosses = []

val_losses = Averager()

totalValLosses = []

lowest_val_loss = float('inf')

early_stopping = EarlyStopping(patience = 7, path = "checkpoint.pth", verbose=True)

# has to be in train mode for both train and valid coz the outputs are different in two cases

model.train()

for epoch in range(NUM_EPOCHS):

    train_losses.reset()

    val_losses.reset()

    for batch_index, (images, targets) in enumerate(train_data_loader):

        # move the images and targets to device

        images = list(image.to(DEVICE) for image in images)

        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        loss = sum(loss for loss in loss_dict.values())

        # track the loss

        train_losses.send(loss.item())



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        if batch_index % 50 == 0:

            print(f"Epoch: {epoch} Batch Index: {batch_index} Loss: {loss.item()}")



    # evaluate

    with torch.no_grad():

        for _, (images, targets) in enumerate(valid_data_loader):

            # move the images and targets to device

            images = list(image.to(DEVICE) for image in images)

            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]



            val_loss_dict = model(images, targets)

            val_loss = sum(loss for loss in val_loss_dict.values())



            # track the loss

            val_losses.send(val_loss.item())



    if val_losses.value >= lowest_val_loss and lr_scheduler is not None:

        lr_scheduler.step()



    # print stats

    print(f"Epoch #{epoch} TRAIN LOSS: {train_losses.value} VALIDATION LOSS: {val_losses.value}\n")

    totalTrainLosses.append(train_losses.value)

    totalValLosses.append(val_losses.value)

    early_stopping(val_losses.value, epoch, model)

    if early_stopping.early_stop:

        print("Early stopping!")

        break



plt.plot(totalTrainLosses, label = "Training Loss", color = "deepskyblue")

plt.plot(totalValLosses, label = "Validation Loss", color = "darkorange")

plt.axvline(early_stopping.epoch, label = "Early Stop", color = "red")

plt.legend()

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss vs Validation Loss")

plt.show()

os.rename("checkpoint.pth", "resnet152.pth")

print("Training finished!")