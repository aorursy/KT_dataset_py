import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
data_set = torchvision.datasets.ImageFolder(  #making train set
    root = '/kaggle/input/lfw-dataset-cropped-faces/croped_dataset/', 
    transform = transforms.Compose([
                                    transforms.Resize((220,220)),
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
)
sample = next(iter(data_set))
image, lable = sample
print (image.shape)
npimg = image.numpy()
npimg = np.transpose(npimg, (1, 2, 0))
print(npimg.shape)
plt.imshow(npimg)
print ('lable : ',lable)
def triplets_train_loader(train_set, train_label, number_of_new_dp):
    List, new_dataset = [], []
    datapoints, total_permutations, train_size = 0, number_of_new_dp, len(train_set) - 1
    while datapoints < total_permutations: 
        
        end = True
        while end:
            A = random.randint(1, train_size - 1)

            if train_label[A - 1] ==  train_label[A]:
                P = A - 1
                end = False
            elif train_label[A + 1] ==  train_label[A]:
                P = A + 1
                end = False
        List.append((train_set[A]))
        List.append((train_set[P]))
        
        N = random.randint(0, train_size)
        while train_label[A] ==  train_label[N]:
            N = random.randint(0, train_size)
        List.append((train_set[N]))
        
        new_dataset.append(List) 
        List = []
        datapoints  += 1
        torch.cuda.empty_cache()
    train_loader = torch.utils.data.DataLoader(new_dataset, batch_size=30,num_workers=8, shuffle = True)
    return train_loader
train_data, train_label, x, y = [], [], 0, 1
KNN_train_data, KNN_test_data = [], []
previous_img = data_set[0]
start = False
z = 0
for sample in iter(data_set):
    if start:
        if previous_img[1] == sample[1] and x < 20:
            if y == 1:  
                KNN_train_data.append((previous_img))
                y = 0
            KNN_test_data.append((sample))
        elif previous_img[1] != sample[1] and x < 20:
            if y == 1:
                train_data.append(previous_img[0])
                train_label.append(previous_img[1])            
            if y == 0:
                x += 1
                #print(x,z, sample[1])
                
            previous_img = sample
            y = 1
            
        else:
            train_data.append(sample[0])
            train_label.append(sample[1])
    else:
        start = True
    z+=1
print (len(train_data))
print (len(KNN_train_data))
print (len(KNN_test_data))
train_loader = triplets_train_loader(train_data, train_label ,number_of_new_dp = 20000)
print(len(train_loader))
class network(nn.Module):

    def __init__(self):
        super(network, self).__init__()
       
        self.conv1 = nn.Conv2d( 3, 64, (7,7), padding=3, stride=(2, 2))
        self.pool1 = nn.MaxPool2d( (2,2))
        self.conv_bn1 = nn.BatchNorm2d(64)
        
        
        
        self.conv2a= nn.Conv2d( 64, 64, (1,1))
        self.conv2 = nn.Conv2d( 64, 192, (3,3), padding = 1)
        self.conv_bn2 = nn.BatchNorm2d(192)

        self.pool2 = nn.MaxPool2d( (2,2))
        
        self.conv3a= nn.Conv2d( 192, 192, (1,1))
        self.conv3 = nn.Conv2d( 192, 384, (3,3), padding = 1)
        self.pool3 = nn.MaxPool2d( (2,2))
        
        self.conv4a= nn.Conv2d( 384, 384, (1,1))
        self.conv4 = nn.Conv2d( 384, 256, (3,3) , padding = 1)
        
        self.conv5a= nn.Conv2d( 256, 256, (1,1))
        self.conv5 = nn.Conv2d( 256, 256, (3,3), padding = 1)
        
        self.conv6a= nn.Conv2d( 256, 256, (1,1))
        self.conv6 = nn.Conv2d( 256, 256, (3,3), padding = 1)
        
        self.pool4 = nn.MaxPool2d( (2,2))
  
        self.fc1 = nn.Linear(256*6*6, 128*1*32) 
        self.fc2 = nn.Linear(128*1*32, 128*1*4)
        self.out  = nn.Linear(128*1*4, 128)


    def forward_once(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = (self.pool1(x))
        x = self.conv_bn1(x)
        
        
        
        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2(x))
        x = self.conv_bn2(x)
        x = (self.pool2(x))
        
        
        
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3(x))
        x = (self.pool3(x))
        
        
        
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4(x))
        
       
        
        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5(x))
        
        
        
        x = F.relu(self.conv6a(x))
        x = F.relu(self.conv6(x))
        x = (self.pool4(x))
        
        
        x = x.view(-1, 256*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        #x = F.normalize(self.out(x),dim=0,p=2)
        return x
    
    def forward(self, in_A, in_P, in_N):
        
        return (self.forward_once(in_A), self.forward_once(in_P), self.forward_once(in_N))
from sklearn.metrics.pairwise import euclidean_distances
import scipy.stats as ss

class KNearestNeighbor:
    ''' Implements the KNearest Neigbours For Classification... '''
    def __init__(self, k, scalefeatures=False):        
       

        self.K=k
        return
        
        #pass    
        
        
    
    def train(self, X, Y):
        ''' Train K Nearest Neighbour classifier using the given 
            X [m x d] data matrix and Y labels matrix
            
            Input:
            ------
            X: [m x d] a data matrix of m d-dimensional examples.
            Y: [m x 1] a label vector.
            
            Returns:
            -----------
            Nothing
            '''
        
        #print(X.shape)
        #nexamples,nfeatures=X.shape
        
        # YOUR CODE HERE

        #Your code goes here...
       
        self.X_train=X
        self.Y_train=Y
        
        #define self.X_train to store the training data...
                
    
    def predict(self, X):
        
        """
        Test the trained K-Nearset Neighoubr classifier result on the given examples X
        
                   
            Input:
            ------
            X: [m x d] a matrix of m  d-dimensional test examples.
            Returns:
            -----------
                pclass: the predicted class for the given set of examples, i.e. to which it belongs
        """
        
        num_test = X.shape[0]
        
        y_pred = np.zeros(self.K, dtype = self.Y_train.dtype)
        pclass=[]

        
        compute_distance = euclidean_distances(X, self.X_train)
        # YOUR CODE HERE
        #print(compute_distance)
        for x in range(num_test):
            SortedDist=np.sort(compute_distance[x])
            for y in range(self.K):
                index=np.where(SortedDist[y] == compute_distance[x])
                y_pred[y]=self.Y_train[index][0]
                # print(self.Y_train[index][0])
            pclass.append(ss.mode(y_pred)[0][0])
            # print (np.min(compute_distance[x]))

        return pclass
        
class Loss_Function(torch.nn.Module):

    def __init__(self,  margin = 0.2):
        
        super(Loss_Function, self).__init__()
        self.margin = margin

    def forward(self, out_A, out_P, out_N):
        
        dist = nn.PairwiseDistance(p=2)
        distance = dist(out_A, out_P) - dist(out_A, out_N) + self.margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance))) 
        return loss
x = [1,2,3,4,5,6,7,8,9,10]
y = [11,17,42,32,53,57,45,60,65,59]
plt.plot(x,y) 
plt.show()
class Face_Net():
    def __init__(self, learning_rate = 0.01):
    
        self.net = network()
        self.loss_func = Loss_Function()
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        
        self.train_data_set = None
        self.test_data_set = None
        self.Classifier_KNN = KNearestNeighbor(1)
        
        self.acc_vector = []
        self.loss_vector = []
        self.epoch_vector = []
    
    def train_NN(self, nepochs, loss_thresh, train_loader):
        
        use_cuda = True
        optm_net = network()
        best_acc = 0.0
        if use_cuda and torch.cuda.is_available():
            self.net.cuda()
            print('cuda')
        
        epoch, end = 0, False
        while epoch < nepochs and not end:
            epoch += 1
            total_loss = 0
            train_loader = triplets_train_loader(train_data, train_label ,number_of_new_dp = 20000)
            for sample in (iter(train_loader)):
                images_A, images_P, images_N = sample
                        
                if use_cuda and torch.cuda.is_available():
                    images_A = images_A.cuda()
                    images_P = images_P.cuda()
                    images_N = images_N.cuda()

                
                out_A, out_P, out_N = self.net.forward(images_A, images_P, images_N)
                
                
                loss = self.loss_func(out_A, out_P, out_N)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
            
                total_loss += loss.item()
                torch.cuda.empty_cache()
            
            
            print('epoch : ', epoch ,' loss : ',total_loss*1.0/len(train_loader))

            
            self.train_Classifier()
            current_acc = self.test_Classifier()
            
            self.loss_vector.append(total_loss*1.0/len(train_loader))
            self.acc_vector.append(current_acc)
            self.epoch_vector.append(epoch)
            
            if current_acc > best_acc:
                
                best_acc = current_acc
                optm_net.load_state_dict(self.net.state_dict())
                
                if current_acc > 0.8 :
                    end = True
                
                  
        self.net.load_state_dict(optm_net.state_dict())
        
    
    def load_data(self):

          self.train_data_set, self.test_data_set = KNN_train_data, KNN_test_data
        
    def train_Classifier(self):
        
        Xtrain, Ytrain = [], []
        for data in iter(self.train_data_set):
            image, label = data
            if torch.cuda.is_available():
                image = image.cuda()
            out = self.net.forward_once(torch.unsqueeze(image, 0))
            out = (out.cpu()).detach().numpy()
            Xtrain.append(out[0])
            Ytrain.append(label)
            
        self.Classifier_KNN.train(np.asarray(Xtrain), np.asarray(Ytrain))
        
    
    def test_Classifier(self):
        
        Xtest, Ytest = [], []
        for data in iter(self.test_data_set):
            image, label = data
            if torch.cuda.is_available():
                image = image.cuda()
            out = self.net.forward_once(torch.unsqueeze(image, 0))
            out = (out.cpu()).detach().numpy()
            Xtest.append(out[0])
            Ytest.append(label)
        pred = self.Classifier_KNN.predict(np.asarray(Xtest))
        #print(pred)
        #Lets see how good we are doing, by finding the accuracy on the test set..
        Total=len(Ytest)
        correct=y=0
        for x in pred:
            if x==Ytest[y]:
                correct+=1
            y+=1
        #print('Test Set Accuracy:',int(correct/Total*100),"%")
        return correct/Total

fn = Face_Net(learning_rate = 0.01)
fn.load_data()

fn.train_NN(30,0.0,train_loader)

#Graph of Loss
plt.plot(fn.epoch_vector,fn.loss_vector) 
plt.show()
#Graph of Accuracy
plt.plot(fn.epoch_vector,fn.acc_vector) 
plt.show()
#Now testing Time
fn.train_Classifier()
print('Test Set Accuracy:',(fn.test_Classifier()*100),"%")
