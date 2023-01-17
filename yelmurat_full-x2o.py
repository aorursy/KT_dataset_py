import torch

import torchvision

import PIL

from torchvision import transforms

from torchvision.models.detection import FasterRCNN

from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models import resnet18 as resnet
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

classes = ['Negative', 'Positive']
mobilenet_v2 = torchvision.models.mobilenet_v2(pretrained=False)

backbone = mobilenet_v2.features

backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),

                                   aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],

                                                output_size=7,

                                                sampling_ratio=2)
marker_detection_model = FasterRCNN(backbone,

                                    num_classes=2,

                                    rpn_anchor_generator=anchor_generator,

                                    box_roi_pool=roi_pooler)

marker_detection_model.load_state_dict(torch.load('/kaggle/input/x2o-shape-3-custom-aug-detection-bbox/marker_detection_v3.4.torch', map_location=device))
marker_detection_model = marker_detection_model.float().to(device)

marker_detection_model = marker_detection_model.eval()
marker_classification_model = resnet(pretrained=False)

marker_classification_model.fc = torch.nn.Linear(512, 2)

marker_classification_model.load_state_dict(torch.load('/kaggle/input/x2o-color/marker_classification_v2.8.torch', map_location=device))

marker_classification_model = marker_classification_model.eval()
tfms = transforms.Compose([transforms.ToPILImage(),

                           transforms.Resize((224, 224), PIL.Image.LANCZOS),

                           transforms.ToTensor()])
def predict(img):

    img = transforms.ToTensor()(img).to(device)*255    

    with torch.no_grad():

        output = marker_detection_model([img])

        boxes_inds = torchvision.ops.nms(output[0]['boxes'], output[0]['scores'], 0.2)

        boxes = output[0]['boxes'][boxes_inds]

        classes = []

        for bbox in boxes:

            cropped_img = img[:,int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])].permute(1, 2, 0)

            transformed_input = tfms(cropped_img)

            output = marker_classification_model(transformed_input.unsqueeze(0).to(device))

            classes.append(output.argmax().item())

    

    return [{'min_x':box[0].item(), 'min_y':box[1].item(), 'max_x':box[2].item(), 'max_y':box[3].item(), 'cls':cls} for box, cls in zip(boxes, classes)]
import cv2

import matplotlib.pyplot as plt
img = cv2.cvtColor(cv2.imread('/kaggle/input/x2o-markers-v1/Negative/DSC_0197.jpg'), cv2.COLOR_BGR2RGB) # Check if your color space is BGR
boxes = predict(img)
boxes
plt.imshow(img)

for bbox in boxes:

    rect = plt.Rectangle((bbox['min_x'], bbox['min_y']), bbox['max_x'] - bbox['min_x'], bbox['max_y'] - bbox['min_y'], fill=False, color='r')

    text = plt.text(bbox['min_x'], bbox['min_y']*0.9, classes[bbox['cls']])

    plt.gcf().gca().add_artist(rect)

    plt.gcf().gca().add_artist(text)

    