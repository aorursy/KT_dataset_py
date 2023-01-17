import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import json
import matplotlib
import requests
from io import BytesIO
import time
import os
from torchvision.utils import make_grid, save_image
from PIL import Image
labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}
def decode_target(target, text_labels=False, threshold=0.5):
    result = []
    for i, x in enumerate(target):
        if (x >= threshold):
            if text_labels:
                result.append(labels[i] + "(" + str(i) + ")")
            else:
                result.append(str(i))
    return ' '.join(result)
def predict_single(image):
    xb = image.unsqueeze(0)
    preds = model(xb)
    prediction = preds[0]
    result = decode_target(prediction,text_labels=True)
    return result
def F_score(output, label, threshold=0.5, beta=1):
    prob = output > threshold
    label = label > threshold

    TP = (prob & label).sum(1).float()
    TN = ((~prob) & (~label)).sum(1).float()
    FP = (prob & (~label)).sum(1).float()
    FN = ((~prob) & label).sum(1).float()

    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    F2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-12)
    return F2.mean(0)

class MultilabelImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch 
        out = self(images)                      
        loss = F.binary_cross_entropy(out, targets)      
        return loss
    
    def validation_step(self, batch):
        images, targets = batch 
        out = self(images)                           # Generate predictions
        loss = F.binary_cross_entropy(out, targets)  # Calculate loss
        score = F_score(out, targets)
        return {'val_loss': loss.detach(), 'val_score': score.detach() }
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_scores = [x['val_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_score: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_score']))
        
class ProteinCnnModel2(MultilabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet152(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 10)
    
    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
# Install gradcam

!pip install pytorch-gradcam
import gradcam
from gradcam.utils import visualize_cam
from gradcam.gradcam import GradCAMpp
import numpy as np
# Transform test image

norm_vals = ([0.0793, 0.0530, 0.0545], [0.1290, 0.0886, 0.1376])

trans = transforms.Compose([
    transforms.Resize(256), 
    transforms.ToTensor(), 
    transforms.Normalize(*norm_vals),
])
# Function used to create activation map

def activation_map(img_path):
    
    img = Image.open(img_path)
    
    print(predict_single(trans(img)))

    torch_img = transforms.Compose([ transforms.Resize((224)),transforms.ToTensor()])(img).to('cpu')

    normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
    
    model.eval()

    model_conf = dict(model_type='resnet', arch=model, layer_name='layer4')

    mask_pp, _ = GradCAMpp.from_config(**model_conf)(normed_torch_img)
    
    _ , map = visualize_cam(mask_pp, torch_img)

    map_image = make_grid([torch_img.cpu() , map])
    
    return transforms.ToPILImage()(map_image)
# Upload your model to kernel and load the path.
model_path = '../input/stratified-sampling-and-normalization/protien_model.pth'
# Setup model

model = torch.load(model_path, map_location='cpu').network
model.eval()
import glob

# Load test image directory
test_img = glob.glob('../input/jovian-pytorch-z2g/Human protein atlas/test/*')
# Visualize output 

activation_map(test_img[110])
activation_map(test_img[112])
activation_map(test_img[12])
