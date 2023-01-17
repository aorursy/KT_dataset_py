# Imports here
import os
import copy
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data_dir = '../input/pytorch-challange-flower-dataset/flower_data/flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
## TODO: Define your transforms for the training and validation sets

# Transforms for the train data
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomRotation(25),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# Transforms for the validation data
valid_transforms = transforms.Compose([transforms.Resize(260),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
# Transforms for test data
test_transforms = valid_transforms

# Dictionary that stores references to our train_transforms and validation_transforms
data_transforms = dict({"train_transforms": train_transforms, 
                        "valid_transforms": valid_transforms, "test_transforms": test_transforms})

## TODO: Load the datasets with ImageFolder

# Data loader for train images
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms["train_transforms"])
# Data loader for validation images
valid_dataset = datasets.ImageFolder(valid_dir, transform=data_transforms["valid_transforms"])
# Data loader for test images (same as validation but raw without transforms)
test_dataset = valid_dataset

# Dictionary that stores the train_dataset and validation_dataset.
image_datasets = dict({"train_dataset": train_dataset, "valid_dataset": valid_dataset, "test_dataset": test_dataset})

## TODO: Using the image datasets and the trainforms, define the dataloaders

# Load the train data into a DataLoader
train_loader = torch.utils.data.DataLoader(image_datasets["train_dataset"], batch_size=25, shuffle=True)
# Load the validation data into a Dataloader
valid_loader = torch.utils.data.DataLoader(image_datasets["valid_dataset"], batch_size=25, shuffle=True)
# Load the test data into a Dataloader
test_loader = torch.utils.data.DataLoader(image_datasets["test_dataset"], batch_size=25, shuffle=True)

# Save the loaders in a dictionary.
dataloaders = dict({"train_loader": train_loader, "valid_loader": valid_loader, "test_loader": test_loader})
import json

with open('../input/pytorch-challange-flower-dataset/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Check some of the entries in our dictionary
print("1:", cat_to_name["1"], ",23:", cat_to_name["23"])
# TODO: Build your network

# Load ResNet pretrained model
model_resnet152 = models.resnet152(pretrained=False)
model_resnet152.load_state_dict(torch.load("../input/resnet152/resnet152_pretrained.pth"))

# Freeze parameters in pre trained ResNET.
for param in model_resnet152.parameters():
    param.requires_grad = False

# Use our custom fc layer in the Resnet model instead of the default.
model_resnet152.fc = nn.Sequential(
    nn.Linear(2048, 1000),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(1000, 102))

# Print our model to screen.
model_resnet152.fc
# Load VGG19 pretrained model
model_vgg19 = models.vgg19(pretrained=False)
model_vgg19.load_state_dict(torch.load("../input/vgg19/vgg19.pth"))

# Freeze parameters in pre trained ResNET.
for param in model_vgg19.parameters():
    param.requires_grad = False
    
# Use our custom fc layer in the VGG19 model instead of the default.
model_vgg19.classifier = nn.Sequential(
    nn.Linear(25088, 3264),
    nn.ReLU(),
    nn.Dropout(p=0.0),
    nn.Linear(3264, 1632),
    nn.ReLU(),
    nn.Dropout(p=0.0),
    nn.Linear(1632, 816),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(816, 102))

model_vgg19.classifier
# Return the model we are going to be using
def current_model (model):
    if (model == "resnet152"):
        return model_resnet152;
    elif (model == "vgg19"):
        return model_vgg19

current_model = current_model("resnet152")
# Checks if GPU is available.
isGPUAvailable = torch.cuda.is_available()
device = "cpu"

if isGPUAvailable:
    device = "cuda"
    print("Training on GPU")
else:
    device = "cpu"
    print("Training on CPU")
    
# If GPU available move the model to GPU. Otherwise CPU.
current_model.to(device)

print(current_model.fc)
# Mapping of classes to indices
current_model.class_to_idx = image_datasets['train_dataset'].class_to_idx

# Saving function
def save_model (model, val_loss):
    #other information about the model
    checkpoint = {
        'hidden_layer': 2048,
        'fc': model.fc,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'min_val_loss': val_loss
    }
    
    torch.save(checkpoint, 'checkpoint_resnet152_e.pth')
# TODO: Write a function that loads a checkpoint and rebuilds the model.

# Function to load a saved model.
def load_checkpoint (filepath):
    # load the checkpoint.
    checkpoint = torch.load(filepath)
    # make the model like our saved model.
    model = models.resnet152(pretrained=False)
    model_resnet152.load_state_dict(torch.load("../input/resnet152/resnet152_pretrained.pth"))
    # Freeze parameters in pre trained ResNET.
    for param in model.parameters():
        param.requires_grad = False
    # Add custom fc layers
    model.fc = checkpoint['fc']
    # Load the satate in our model.
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.min_val_loss = checkpoint['min_val_loss']
    
    return model
# TODO: Train your network.
def train (my_model, epochs = 30, min_valid_loss=np.Inf, lr=0.001):
    # Optimizer to update weights during backpropagation.
    optimizer = optim.Adam(my_model.parameters(), lr = lr)
    # Criterion: loss function.
    criterion = nn.CrossEntropyLoss()
    
    best_model = copy.deepcopy(my_model.state_dict())
    
    for epoch in range(epochs):
        train_loss = 0.0
        validation_loss = 0.0
        accuracy = 0.0
        
        # Train the model.
        my_model.train()
        for inputs, labels in dataloaders["train_loader"]:
            # Reset optimizer for every iteration.
            optimizer.zero_grad()
            # Move tensors to GPU if available.
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass.
            output = my_model(inputs)
            # Loss.
            loss = criterion(output, labels)
            # Backward pass
            loss.backward()
            # Update the weights. "Take a step".
            optimizer.step()

            # Update the loss.
            train_loss += loss.item() * inputs.size(0)

        # Validate the model
        my_model.eval()
        
        for inputs, labels in  dataloaders["valid_loader"]:
            # Move tensors to GPU.
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass.
            output = my_model(inputs)
            # Loss.
            loss = criterion(output, labels)
            
            # Check accuracy.
            _, predicted = output.topk(1, dim=1)
            equals = predicted == labels.view(*predicted.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

            # Update the validation loss.
            validation_loss += loss.item() * inputs.size(0)
            
        # Calculate the losses.
        train_loss = train_loss/len(dataloaders["train_loader"].dataset)
        validation_loss = validation_loss/len(dataloaders["valid_loader"].dataset)
        accuracy = (accuracy/len(dataloaders["valid_loader"]))*100
        # Print the losses
        print("Epoch {0}".format(epoch + 1 ))
        print("Train loss = {0}".format(train_loss))
        print("Validation loss = {0}".format(validation_loss))
        print("Accuracy: {0}%".format(accuracy))
        

        # Check if validation loss has reduced, and therefore the model predicts better
        if validation_loss < min_valid_loss:
            min_valid_loss = validation_loss
            save_model(my_model, validation_loss)
            print("Validation loss has decreased. Saving the model...")
        print("------------------------------------")
        
    my_model.load_state_dict(best_model)
    return my_model
# Train the model
model_1 = train(my_model=current_model, epochs=45, lr=0.0001)
# Load model
model = load_checkpoint('../input/checkpoint-resnet152-cpth/checkpoint_resnet152_c.pth').to(device)

# Train our model some more
#model_1 = train(my_model=model_1, epochs=20, min_valid_loss=model_1.min_val_loss, lr=0.001)
# TODO: Do validation on the test set
def check_accuracy_on_test(model, data, cuda=False):
    model.eval()
    model.to(device) 
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in (dataloaders[data]):
            images, labels = data
            #images, labels = images.to('cuda'), labels.to('cuda')
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.topk(output, 1)
            
            equals = predicted == labels.view(*predicted.shape)
            correct += int(torch.sum(equals))
            total += len(images)
           
    accuracy = (correct / total)*100    
    print(f'Accuracy of the network on the test images: {correct}/{total} --> {accuracy}%')
    
check_accuracy_on_test(model_1, 'test_loader', True)
import PIL
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Load the image in PIL
    image_pil = Image.open(image)
    # Apply changes with transforms
    changes = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image_tensor = changes(image_pil)

    return image_tensor


def process_image_2(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    
    # Load the image in PIL
    image_pil = Image.open(image)
    # Resize image
    width, height = image_pil.size
    if width > height:
        image_pil.thumbnail((np.Inf, 256))
    else:
        image_pil.thumbnail((256, np.Inf))
    
    # Crop image
    image_pil = image_pil.resize((224, 224))
    
    # Convert to numpy and normalize
    np_image = np.array(image_pil)/255
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean)/std
    
    # Transpose for image to have the correct dimensions, depth first.
    np_image = np_image.transpose(2, 0, 1)

    return np_image

test_image = process_image("../input/pytorch-challange-flower-dataset/flower_data/flower_data/valid/1/image_06739.jpg")
test_image_2 = process_image_2("../input/pytorch-challange-flower-dataset/flower_data/flower_data/valid/1/image_06739.jpg") 
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

# It is a tensor so we convert it to numpy.
imshow(test_image.numpy())
imshow(test_image_2)
# Makes a prediction given a raw image
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to("cpu")
    
    # Load the image
    image = process_image(image_path)
    image = image.unsqueeze(0)

    # Forward pass.
    output = model.forward(image)
    # Get the top 5 elements
    top_prob, top_class = torch.topk(output, topk)
    # Get the probabilities
    top_prob = top_prob.exp()
    # Convert to arrays.
    top_prob = top_prob.squeeze().detach().numpy()
    top_class = top_class.detach().squeeze().numpy()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Get the actual labels
    top_classes = [idx_to_class[i] for i in top_class]
    
    return top_prob, top_classes
    

image_path = "../input/pytorch-challange-flower-dataset/flower_data/flower_data/valid/23/image_03444.jpg"
image_path2 = "../input/pytorch-challange-flower-dataset/flower_data/flower_data/valid/102/image_08002.jpg"

_, labels = predict(image_path, model_1)
print("prediction is:", cat_to_name[str(labels[0])])
_, labels = predict(image_path2, model_1)
print("prediction is:", cat_to_name[str(labels[0])])
# Same as predict above but takes an image tensor rather than an image path
# this way we can feed it tensors from an imageloader.
def predict_test(image, model, topk=5):
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    model.to("cpu")
    
    # Load the image
    image = image.unsqueeze(0)

    # Forward pass.
    output = model.forward(image)
    # Get the top 5 elements
    top_prob, top_class = torch.topk(output, topk)
    # Get the probabilities
    top_prob = top_prob.exp()
    # Convert to arrays.
    top_prob = top_prob.squeeze().detach().numpy()
    top_class = top_class.detach().squeeze().numpy()
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    
    # Get the actual labels
    top_classes = [idx_to_class[i] for i in top_class]
    
    return top_prob, top_classes
    
# TODO: Display an image along with the top 5 classes
def check_sanity(img, top_prob_array, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    # imshow expects the 3rd dimension at the end.
    img = img.numpy().transpose(1,2,0)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    flower_name = cat_to_name[str(classes[0])]
    
    ax1.set_title(flower_name)
    ax1.imshow(img)
    
    y_pos = np.arange(len(top_prob_array))
    ax2.barh(y_pos, top_prob_array)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in classes])
    ax2.invert_yaxis()
# Load the batch, and pick one image.
dataiter = iter(dataloaders["test_loader"])
images, labels = dataiter.next()
img = images[0]
# Get the probabilities and classes
probs, classes = predict_test(img, model_1)
check_sanity(img, probs, classes, cat_to_name)
