import torchvision
from torchvision import transforms,models,datasets
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from PIL import Image
import seaborn as sns
data_dir = {
            'train': '../input/chest_xray/chest_xray/test',
            'test': '../input/chest_xray/chest_xray/test',
            'valid': '../input/chest_xray/chest_xray/val',
            }
batch_size = 32 # we will set the batch size to 64 

data_transforms = {
            'train': transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.CenterCrop(224),
                        transforms.RandomHorizontalFlip(), # randomly flip and rotate
                        transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]),
    
            'test': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ]),
    
            'valid': transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
            }
# Load the datasets with ImageFolder

data_set={
        'train': torchvision.datasets.ImageFolder(data_dir['train'] ,data_transforms['train']),
        'test': torchvision.datasets.ImageFolder(data_dir['test'], data_transforms['test']),
        'valid': torchvision.datasets.ImageFolder(data_dir['valid'], data_transforms['valid']),
         }

# Using the image datasets and the trainforms, define the dataloaders
data_loader={
        'train': torch.utils.data.DataLoader(data_set['train'], batch_size=batch_size,shuffle=True),
        'test': torch.utils.data.DataLoader(data_set['test'], batch_size=batch_size,shuffle=True),
        'valid': torch.utils.data.DataLoader(data_set['test'], batch_size=batch_size,shuffle=True),
        }

### we get the class_to_index in the data_Set but what we really need is the cat_to_names  so we will create
_ = data_set['valid'].class_to_idx
cat_to_name = {_[i]: i for i in list(_.keys())}

def showimage(data_loader, number_images, cat_to_name):
    dataiter = iter(data_loader)
    images, labels = dataiter.next()
    images = images.numpy() # convert images to numpy for display
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(number_images, 4))
    # display 20 images
    for idx in np.arange(number_images):
        ax = fig.add_subplot(2, number_images/2, idx+1, xticks=[], yticks=[])
        img = np.transpose(images[idx])
        plt.imshow(img)
        ax.set_title(cat_to_name[labels.tolist()[idx]])
        

#### to show some  images
showimage(data_loader['valid'],2,cat_to_name)
model = models.densenet121(pretrained=True) # we will use a pretrained model and we are going to change only the last layer
for param in model.parameters():
    param.requires_grad = True
model.classifier = nn.Sequential(OrderedDict([
    ('fcl1', nn.Linear(1024,256)),
    ('dp1', nn.Dropout(0.3)),
    ('r1', nn.ReLU()),
    ('fcl2', nn.Linear(256,32)),
    ('dp2', nn.Dropout(0.3)),
    ('r2', nn.ReLU()),
    ('fcl3', nn.Linear(32,2)),
    ('out', nn.LogSoftmax(dim=1)),
]))
def train_function(model, train_loader, valid_loader, criterion, optimizer, scheduler=None,
                       train_on_gpu=False, n_epochs=30, save_file='mymodel.pth'):
    
    valid_loss_min = 0.218098#np.Inf
    if train_on_gpu:
        model.cuda()
    for epoch in range(1, n_epochs + 1):
        # stop training the feature CNN after epochs/2 epochs    
        if epoch == n_epochs // 2:
            model.load_state_dict(torch.load(save_file))
            for param in model.features.parameters():
                param.requires_grad = False
            
        train_loss = 0.0
        valid_loss = 0.0
        if scheduler != None:
            scheduler.step()
        model.train()
        for data, target in train_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()   
            train_loss += loss.item() * data.size(0)

        ######################    
        # validate the model #
        ######################
        model.eval()
        number_correct, number_data = 0, 0
        for data, target in valid_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)
            ############# calculate the accurecy
            _, pred = torch.max(output, 1) 
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu \
                                    else np.squeeze(correct_tensor.cpu().numpy())
            number_correct += sum(correct)
            number_data += correct.shape[0]
            ###################################
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        accuracy = (100 * number_correct / number_data)
        print('Epoch: {} \n-----------------\n \tTraining Loss: {:.6f} \t Validation Loss: {:.6f} \t accuracy : {:.4f}% '.format(epoch, train_loss, valid_loss,accuracy))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
            torch.save(model.state_dict(), save_file)
            valid_loss_min = valid_loss
    model.to('cpu')
    return torch.load(save_file)
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('GPU is  available :)   Training on GPU ...')
else:
    print('GPU is not available :(  Training on CPU ...')
criterion = nn.NLLLoss()
optimizer = optim.Adadelta(model.parameters())

model_state_dict = train_function(
                            model,
                            data_loader['train'],
                            data_loader['valid'],
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=None,
                            train_on_gpu=train_on_gpu,
                            n_epochs=40,
                            save_file='saved_state.pth'
                            )

model.load_state_dict(model_state_dict)
def test_function(model, test_loader, train_on_gpu, criterion,classes):
    test_loss = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    if train_on_gpu:
        model.cuda()
    model.eval()
    cat_accuracy = {}
    if train_on_gpu:
        model.cuda()
    for data,target in test_loader:
        if train_on_gpu:
            data,target = data.cuda(),target.cuda()
        output = model(data)
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        
        _, pred = torch.max(output, 1) 
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu \
                                else np.squeeze(correct_tensor.cpu().numpy())
        for i in range(data.shape[0]):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    test_loss = test_loss/len(test_loader.dataset)
    print('Test Loss: {:.6f}'.format(test_loss))
    print('Test Accuracy (Overall): %2d%% (%2d/%2d) \n ----------------------' % (100. * np.sum(class_correct) / np.sum(class_total),np.sum(class_correct), np.sum(class_total)))
    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %s : %d%% (%2d/%2d)' % (classes[i], 100 * class_correct[i] / class_total[i],np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %s: N/A (no training examples)' % (classes[str(i+1)]))

  
criterion = nn.NLLLoss()
test_function(model, data_loader['test'], train_on_gpu, criterion, cat_to_name)
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    ##########Scales 
    if img.size[0] > img.size[1]:
        img.thumbnail((1000000, 256))
    else:
        img.thumbnail((256 ,1000000))
    #######Crops: to crop the image we have to specifiy the left,Right,button and the top pixels because the crop function take a rectongle ot pixels
    Left = (img.width - 224) / 2
    Right = Left + 224
    Top = (img.height - 244) / 2
    Buttom = Top + 224
    img = img.crop((Left, Top, Right, Buttom))
    img = np.stack((img,)*3, axis=-1)# to repeate the the one chanel of a gray image to be RGB image 
    #img = np.repeat(image[..., np.newaxis], 3, -1)
    #print(np.array(img).shape)
    #normalization (divide the image by 255 so the value of the channels will be between 0 and 1 and substract the mean and divide the result by the standtared deviation)
    img = ((np.array(img) / 255) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = img.transpose((2, 0, 1))
    return img
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    image = image.transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    #image=np.transpose(image)
    ax.imshow(image)

    return ax
def predict(image_path, model, topk=2):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img = process_image(image_path)
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    model_input = image_tensor.unsqueeze(0)
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    #top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[lab] for lab in top_labs]

    return top_probs, top_flowers
    # TODO: Implement the code to predict the class from an image file
def plot(image_path,model,top_k=2):
    proba, flowers = predict(image_path, model, top_k)
    plt.figure(figsize=(6,10))
    ax = plt.subplot(2,1,1)
    
    title = image_path.split('/')[5]
    imshow(process_image(image_path), ax, title=title)
    
    plt.subplot(2,1,2)
    sns.barplot(x=proba, y=flowers, color=sns.color_palette()[0]);
    plt.show()
model.to('cpu')
plot('../input/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg',model,2)
