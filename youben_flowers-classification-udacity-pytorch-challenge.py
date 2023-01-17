import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
data_dir = '../input/flower-data/flower_data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
scrapped_data_dir = "../input/flowers102species/google_test_data"
batch_size = 64
# TODO: Define your transforms for the training and validation sets
train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

validation_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# TODO: Load the datasets with ImageFolder
# Dataset provided by udacity
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
validation_datasets = datasets.ImageFolder(valid_dir, transform=validation_transforms)

# Dataset scrapped from the web by
scrapped_datasets = datasets.ImageFolder(scrapped_data_dir, transform=train_transforms)
# split it to train and validation
train_size = int(len(scrapped_datasets) * 0.8)
valid_size = len(scrapped_datasets) - train_size
scrapped_train_datasets, scrapped_valid_datasets = torch.utils.data.random_split(scrapped_datasets, [train_size, valid_size])

all_train_datasets = torch.utils.data.ConcatDataset([train_datasets, scrapped_train_datasets])
all_validation_datasets = torch.utils.data.ConcatDataset([validation_datasets, scrapped_valid_datasets])

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(all_train_datasets, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(all_validation_datasets, batch_size=batch_size)
import json
f = open("cat_to_name.json","w")
f.write('{"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt\'s foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}')
f.close()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#make sure to use pytorch 0.4.0
!pip install fastai==0.7.0 --no-deps
!pip install torch==0.4.1 torchvision==0.2.1
# TODO: Build and train your network
from torch import nn
from collections import OrderedDict

net = torchvision.models.densenet121(pretrained=True)

# make sure to train the feature part
for param in net.parameters():
    param.requires_grad = True

net.classifier = nn.Sequential(OrderedDict([
    ('fcl1', nn.Linear(1024,512)),
    ('dp1', nn.Dropout(0.3)),
    ('r1', nn.ReLU()),
    ('fcl2', nn.Linear(512,256)),
    ('dp2', nn.Dropout(0.3)),
    ('r2', nn.ReLU()),
    ('fcl3', nn.Linear(256,102)),
    ('out', nn.LogSoftmax(dim=1)),
]))
optimizer = torch.optim.Adadelta(params=net.parameters())
criterion = nn.NLLLoss()
# moving to gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
# to not output the net
dont_print = None
# some cst for training
epochs = 30
running_loss = 0
steps = 0
print_every = 10
last_loss = np.inf
#Training
for epoch in range(epochs):
    # stop training the feature CNN after epochs/2 epochs
    if epoch == epochs // 2:
        net.load_state_dict(torch.load('state_dict_with_scrapped_data.pt'))
        for param in net.features.parameters():
            param.requires_grad = False
    for inputs, labels in train_loader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = net.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            net.eval()
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = net.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            if valid_loss < last_loss:
                last_loss = valid_loss
                torch.save(net.state_dict(), 'state_dict_with_scrapped_data.pt')
                print("[+] New Best Loss, state dict saved")
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validation_loader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validation_loader):.3f}")
            running_loss = 0
            net.train()
# TODO: Save the checkpoint 
from torchvision.models import densenet121
from torch import nn
from collections import OrderedDict
import torch


def load_saved_model(state_dict_path="state_dict_with_scrapped_data.pt"):
    model = densenet121(pretrained=True)
    model.classifier = nn.Sequential(OrderedDict([
                ('fcl1', nn.Linear(1024,512)),
                ('dp1', nn.Dropout(0.3)),
                ('r1', nn.ReLU()),
                ('fcl2', nn.Linear(512,256)),
                ('dp2', nn.Dropout(0.3)),
                ('r2', nn.ReLU()),
                ('fcl3', nn.Linear(256,102)),
                ('out', nn.LogSoftmax(dim=1)),
                ]))
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = image.convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    return image
    
    # TODO: Process a PIL image for use in a PyTorch model
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
from PIL import Image

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(open(image_path, "rb"))
    image = process_img(image)
    with torch.no_grad():
        probs = model(image)[0]
    probs = probs.exp()
    probs, cats = probs.topk(topk)
    probs, cats = probs.numpy().astype('float'), cats.numpy()
    probs_cats = {cat_to_name[str(cat+1)]: prob for prob,cat in zip(probs, cats)}

    return probs_cats
# TODO: Display an image along with the top 5 classes