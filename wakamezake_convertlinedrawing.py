%ls -a ../input/danbooru-images/
# reference 

# https://qiita.com/koshian2/items/f53088d1dedaa72f9dca

import numpy as np

from PIL import Image

import torch

import torch.nn.functional as F

import matplotlib.pyplot as plt

from torchvision.utils import make_grid



def load_tensor(image_path):

    with Image.open(image_path) as img:

        array = np.asarray(img, np.float32) / 255.0 # [0, 1]

        array = np.expand_dims(array, axis=0)

        array = np.transpose(array, [0, 3, 1, 2]) # PyTorchはNCHW

        return torch.as_tensor(array)



def show_tensor(input_image_tensor):

    img = input_image_tensor.numpy() * 255.0

    img = img.astype(np.uint8)[0,0,:,:]    

    plt.imshow(img, cmap="gray")

    plt.show()



def linedraw(image_path):

    # データの読み込み

    x = load_tensor(image_path)

    # Y = 0.299R + 0.587G + 0.114B　でグレースケール化

    gray_kernel = torch.as_tensor(

        np.array([0.299, 0.587, 0.114], np.float32).reshape(1, 3, 1, 1))

    x = F.conv2d(x, gray_kernel) # 行列積は畳み込み関数でOK

    # 3x3カーネルで膨張1回（膨張はMaxPoolと同じ）

    dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)

    # 膨張の前後でL1の差分を取る

    diff = torch.abs(x-dilated)    

    # ネガポジ反転

    x = 1.0 - diff

    # 結果表示

    show_tensor(x)
from pathlib import Path



image_root_path = Path('../input/moeimouto-faces/moeimouto-faces/')
for img_idx, img_path in enumerate(image_root_path.glob("*/*.png")):

    linedraw(img_path)

    if img_idx == 10:

        break

    