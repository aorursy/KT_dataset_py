!python --version
import torch # For Convolutional Neural Network
import torchvision # ""
import torchvision.transforms as transforms # ""
import torch.nn as nn # ""
import torch.nn.functional as F # ""
import torch.optim as optim # Neural net optimizer
import pandas as pd # For loading csv data
import matplotlib.pyplot as plt # For showing images
import os, os.path # For sorting training images
import shutil # ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
train_data = pd.read_csv("../input/melanoma-image-data/train.csv")
test_data = pd.read_csv("../input/melanoma-image-data/test.csv")
train_data.head()
test_data.head()
def move_images(data, root):
    images = [name for name in os.listdir(root) 
              if os.path.isfile(os.path.join(root, name))]
    if images:
        file_names = data['image_name'].tolist()
        for file in file_names:
            classification = data[data['image_name'] == 
                                  file]['benign_malignant'].values[0]
            shutil.move(root + file + '.jpg', 
                        root + classification + '/' + file + '.jpg')
move_images(train_data, '../input/melanoma-image-data/train/train/')
train_image_data = torchvision.datasets.ImageFolder(
        '../input/melanoma-image-data/train/train/',
        transform=transforms.Compose([
                  transforms.Resize((512, 512)),
                  transforms.RandomHorizontalFlip(),
                  transforms.RandomVerticalFlip(),
                  transforms.ToTensor()])
)
n = len(train_image_data)  # total number of examples
n_valid = int(0.1 * n)  # take ~10% for validation
valid_set, train_set = torch.utils.data.random_split(train_image_data, [n_valid, n - n_valid])

validationloader = torch.utils.data.DataLoader(valid_set, shuffle=True)
images, labels = next(iter(validationloader))
plt.imshow(images[0][0])
print(labels[0].item())
print(images[0][0].shape)
class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.norm1 = nn.BatchNorm2d(32)
        self.norm2 = nn.BatchNorm2d(64)
        self.norm3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout1 = nn.Dropout(p=0.1, inplace=False)

    def forward(self, x):
        x = self.norm1(self.pool(F.relu(self.conv1(x))))
        x = self.norm2(self.pool(F.relu(self.conv2(x))))
        x = self.norm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout1(x)
        x = x.view(-1, 6272)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    acc = (y_pred_tag == y_test).sum().float()
    acc = torch.round(acc * 100)
    
    return acc
count = train_data.groupby('benign_malignant').count()
for index, row in count.iterrows():
    print(f'Count {index}: {row[0]}')
# got to get array of labels. This is slowwwwwwwwwwwwwwwwwwwwwwwwwww
class_arr = [0 if label == 0 else 1 for _, label in train_set]
reciprocal_weights = [0 for index in range(len(train_set))]
count0, count1 = 32542, 584

marg = count0 + count1
prob = [count0 / marg, count1 / marg]
for index in range(len(train_set)):
    reciprocal_weights[index] = prob[class_arr[index]]

weights = (1 / torch.Tensor(reciprocal_weights))
print(torch.unique(weights))
num_samples = 1024
batch_size = 64
nets = [Net() for i in range(3)]
criterion = nn.BCEWithLogitsLoss()

for n in nets:
    n.cuda()
for index, n in enumerate(nets):
    PATH = f'../input/net-data/ensemble_net_{index}.pth'
    n.load_state_dict(torch.load(PATH))
for index, net in enumerate(nets):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    EPOCHS = 10
    print(f'STARTING TRAINING NET {index}')
    running_loss, running_acc = 0, 0
    for e in range(EPOCHS):
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 
                             num_samples, replacement=True)
        sampler = torch.utils.data.BatchSampler(sampler, batch_size, 
                             drop_last=True)
        loader = torch.utils.data.DataLoader(train_set,
                                     batch_sampler = sampler,
                                     num_workers = 8) # stable
        for i, (x, y) in enumerate(loader, 0):
            images, labels = x.cuda(), y.cuda()
            labels = labels.float()
            optimizer.zero_grad()

            y_pred = net(images).cuda()
            loss = criterion(y_pred, labels.unsqueeze(1))
            acc = binary_acc(y_pred, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() 
            running_acc += acc.item() 

        print('#', end='')
        if e % 10 == 9:
            print('|', end='')
    print('\n[net %d epochs %d], loss: %.3f, acc: %2.2f%%' %
          (index, EPOCHS, 
           running_loss / (EPOCHS * num_samples), 
           running_acc / (EPOCHS * num_samples)))
    PATH = f'./ensemble_net_{index}.pth'
    torch.save(net.state_dict(), PATH)  
    
def get_majority_vote(pred_list):
    return 1 if sum(pred_list) >= (len(pred_list) / 2) else 0
for net in nets:
    net.eval()
correct, total, tp, tn, fn, fp = 0, 0, 0, 0, 0, 0
with torch.no_grad():
    for (x, y) in validationloader:
        images, labels = x.to(device), y.to(device)
        pred_list = [0 for i in nets]
        for index, net in enumerate(nets):
            y_pred = net(images)
            pred_list[index] = torch.round(torch.sigmoid(y_pred))
        y_pred_tag =  get_majority_vote(pred_list)
        if y_pred_tag == labels.unsqueeze(1):
            correct += 1
            if labels.item() == 1:
                tp += 1
            else:
                tn += 1
        else:
            if labels.item() == 1:
                fn += 1
            else:
                fp += 1
        total += 1
print()     
print(f'*********************************')
print('| TP = %2.2f%%  \t| FP = %2.2f%% \t|' % (tp/total*100, fp/total*100))
print(f'*********************************')
print('| FN = %2.2f%%  \t| TN = %2.2f%% \t|' % (fn/total*100, tn/total*100))
print(f'*********************************')  
print('Validation Accuracy: %2.2f%%' % (100*correct//total))
print('Precision: %2.2f%%    Recall: %2.2f%%' % (100 * tp / (tp + fp),
                                                100 * tp / (tp + fn)))
test_image_data = torchvision.datasets.ImageFolder(
        '../input/melanoma-image-data/test/test/',
        transform=transforms.Compose([
                  transforms.Resize((512, 512)),
                  transforms.ToTensor()])
    )
testloader = torch.utils.data.DataLoader(test_image_data, shuffle=False)
test_frame = pd.DataFrame(test_data['image_name'])
test_frame['target'] = [0 for i in range(len(test_frame))]
test_frame.head()
for net in nets:
    net.eval()
with torch.no_grad():
    for j, (x, y) in enumerate(testloader):
        images, labels = x.cuda(), y.cuda()
        pred_list = [0 for i in nets]
        for index, net in enumerate(nets):
            y_pred = net(images).cuda()
            pred_list[index] = torch.round(torch.sigmoid(y_pred))
        test_frame.at[j, 'target'] =  get_majority_vote(pred_list)
        
test_frame.to_csv('./ensemble_net_submission.csv', index = False)