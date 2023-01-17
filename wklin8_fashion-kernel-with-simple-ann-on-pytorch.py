import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from pathlib import Path
from fastai.column_data import *
from fastai.structured import *
import fastai


PATH = Path('../input')

train_raw = pd.read_csv(PATH/"fashion-mnist_train.csv")
test_raw = pd.read_csv(PATH/"fashion-mnist_test.csv")
labels_dict={
'0': 'T-shirt/top',
'1': 'Trouser',
'2': 'Pullover',
'3': 'Dress',
'4': 'Coat',
'5': 'Sandal',
'6': 'Shirt',
'7': 'Sneaker',
'8': 'Bag',
'9': 'Ankle boot'
}

def display_img(df, idx):
    l = str(df.iloc[idx].values[0])
    plt.imshow(df.iloc[idx][1:].values.reshape(28, 28))
    plt.title(labels_dict[l])

class FashionNN(nn.Module):
    def __init__(self, layers, dropout=0.5):
        super().__init__()
        # first layer must be of the image flatten size
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i, l in enumerate(range(len(layers)-1))])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, cat, cont):
        # we assume there is no cat for simplicity
        for l in self.layers:
            cont = l(cont)
            d = self.dropout(cont)
            xs = F.relu(d)
        return F.log_softmax(xs, dim=-1)
do_scale=True
y_label = "label"

bs, val_ratio, img_size = 64, 0.2, int(math.sqrt(train_raw.shape[1]-1))

y_label = "label"

if (do_scale):
    train_x, train_y, nas, mapper = proc_df(train_raw, y_label, do_scale=True)
    test_x, _, nas, mapper = proc_df(test_raw, y_label, do_scale=True, na_dict=nas, mapper=mapper)
else:
    train_x, train_y, nas = proc_df(train_raw, y_label)
    test_x, _, nas = proc_df(test_raw, y_label, na_dict=nas)

val_idxs = get_cv_idxs(train_raw.shape[0], val_pct=val_ratio)
md = ColumnarModelData.from_data_frame(PATH, val_idxs, train_x, train_y.astype(np.int64), [], bs=bs, is_reg=False, is_multi=False, test_df=test_x)

net = FashionNN([1*img_size*img_size, 300, 200, 10], dropout=0.1).cuda()

lr = 1e-3
optim = torch.optim.Adam(net.parameters(), lr)

fit(net, md, 20, optim, F.nll_loss, metrics=[accuracy])
# set learning rate to lower value and keep training!
set_lrs(optim, 1e-4)

fit(net, md, 30, optim, F.nll_loss, metrics=[accuracy])
# predict
net.eval()
x_vv = VV(test_x.values)
y_vv = net(None, x_vv)
net.train()

y = y_vv.cpu().data.numpy()
y = np.argmax(y, axis=1)
      
