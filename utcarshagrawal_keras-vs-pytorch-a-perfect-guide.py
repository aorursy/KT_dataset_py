import numpy as np 

import pandas as pd 

import os

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import StandardScaler

import numbers
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import RMSprop, SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

from torchvision.utils import make_grid
df1 = pd.read_csv("../input/titanic/train.csv")
df1.head()
val = ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']

plt.figure(figsize=(15,15))

plt.subplots_adjust(right=1.5)

for i in range(5):

    plt.subplot(2,3,i+1), sns.countplot(x=val[i], hue='Survived', data = df1)

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 10})

    plt.title('Count of Survival in {} Feature'.format(val[i]), size=10, y=1.05)
# Sex Encoding

binar = LabelBinarizer().fit(df1.loc[:, "Sex"])

df1["Sex"] = binar.transform(df1["Sex"])

# Embarked Encoding

df1["Embarked"] = df1["Embarked"].fillna('S')

df_Embarked = pd.get_dummies(df1.Embarked)

df1 = pd.concat([df1, df_Embarked], axis=1)

#Family

df1['Family'] = df1['SibSp'] + df1['Parch'] + 1

df1['Alone'] = df1['Family'].apply(lambda x : 0 if x>1 else 1 )

#Age

df1['Age'] = df1['Age'].fillna(-0.5)
features = ["Pclass","Sex", "Age", "C", "Q", "S", "Alone"]



X_train = df1[features].values

Y_train = df1["Survived"].values
# Feature Scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
model1 = Sequential()

model1.add(Dense(100, input_dim=7, activation="relu"))

model1.add(Dropout(0.5))

model1.add(Dense(200, activation="relu"))

model1.add(Dropout(0.5))

model1.add(Dense(1, activation="sigmoid"))
class ANN(nn.Module):

    def __init__(self):

        super(ANN, self).__init__()

        self.features = nn.Sequential(

            nn.Linear(7,100),

            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),

            nn.Linear(100,200),

            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),

            nn.Linear(200,2),

        )

    def forward(self, x):

        x = self.features(x)

        

        return x

    

model2 = ANN()
#For Keras Model



optimizer1 = SGD(lr = 0.01, momentum = 0.9)

# Compiling our model

model1.compile(optimizer = optimizer1, loss = 'binary_crossentropy', metrics = ['accuracy'])



#For PyTorch Model



# Define the optimier

optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

# Define our loss function

criterion = nn.CrossEntropyLoss()
batch_size=64

epochs=15
# Keras Model

model1.fit(X_train, Y_train, batch_size = batch_size, epochs = epochs, verbose=2)
# PyTorch Model

batch_no = len(X_train) // batch_size

train_loss_plot, train_acc_plot = [], []

train_loss = 0

train_loss_min = np.Inf

for epoch in range(epochs):

    for i in range(batch_no):

        start = i*batch_size

        end = start+batch_size

        x_ten = Variable(torch.FloatTensor(X_train[start:end]))

        y_ten = Variable(torch.LongTensor(Y_train[start:end])) 

        

        # Prevent accumulation of gradients

        optimizer2.zero_grad()

        # Make predictions

        output = model2(x_ten)

        loss = criterion(output,y_ten)

        #backprop

        loss.backward()

        optimizer2.step()

        

        values, labels = torch.max(output, 1)

        num_right = np.sum(labels.data.numpy() == Y_train[start:end])

        train_loss += loss.item()*batch_size

        

    train_loss = train_loss / len(X_train)

    train_loss_plot.append(train_loss)

    train_acc_plot.append((num_right / len(Y_train[start:end])))

    if train_loss <= train_loss_min:

        print("Validation loss decreased ({:6f} ===> {:6f})".format(train_loss_min,train_loss))

        torch.save(model2.state_dict(), "model.pt")

        train_loss_min = train_loss



    print('')

    print("Epoch: {} \tTrain Loss: {} \tTrain Accuracy: {}".format(epoch+1, train_loss,num_right / len(Y_train[start:end]) ))
metrics=pd.DataFrame(model1.history.history)

plt.plot(metrics['loss'], label='Keras_loss')

plt.plot(train_loss_plot, label='PyTorch_loss')

plt.legend()

plt.show()
plt.plot(metrics['accuracy'], label='Keras_acc')

plt.plot(train_acc_plot, label='PyTorch_acc')

plt.legend()

plt.show()
df2 = pd.read_csv('../input/digit-recognizer/train.csv')

df3 = pd.read_csv('../input/digit-recognizer/test.csv')
Y_train = df2["label"]

X_train = df2.drop(labels = ["label"],axis = 1).values



X_train = X_train/255.0

X_test = df3/255.0



X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)



Y_train = to_categorical(Y_train, num_classes = 10)



X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=35)
targets_np = df2.label.values

features_np = df2.loc[:, df2.columns != 'label'].values/255

test_np = df3.values/255



features_np = features_np.reshape(-1,1,28,28)

test_np = test_np.reshape(-1,1,28,28)

fake_labels = np.zeros(test_np.shape)

fake_labels = torch.from_numpy(fake_labels)



features_train, features_test, target_train, target_test = train_test_split(features_np, targets_np, test_size=0.1, random_state=35)



featuresTrain = torch.from_numpy(features_train)

targetsTrain = torch.from_numpy(target_train)



featuresTest = torch.from_numpy(features_test)

targetsTest = torch.from_numpy(target_test)



test_data = torch.from_numpy(test_np)



train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)

test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

submission_data = torch.utils.data.TensorDataset(test_data, fake_labels)



train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)

submission_loader = torch.utils.data.DataLoader(submission_data, batch_size = batch_size, shuffle = False)
model3 = Sequential()



model3.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))

model3.add(BatchNormalization())

model3.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))

model3.add(BatchNormalization())

model3.add(MaxPool2D(pool_size=(2,2)))

model3.add(Dropout(0.25))



model3.add(Flatten())

model3.add(Dense(256, activation='relu'))

model3.add(BatchNormalization())

model3.add(Dropout(0.5))

model3.add(Dense(10, activation='softmax'))
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=(5,5), stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=(5,5), stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

        )

          

        self.classifier = nn.Sequential(

            nn.Dropout(p = 0.5),

            nn.Linear(32 * 12 * 12, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(p = 0.5),

            nn.Linear(512, 10),

        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        

        return x



model4 = CNN()
#For Keras Model



optimizer3 = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compiling our model

model3.compile(optimizer = optimizer3 , loss = "categorical_crossentropy", metrics=["accuracy"])



#For PyTorch Model



# Define the optimier

optimizer4 = optim.RMSprop(model4.parameters(), lr=0.001)

# Define our loss function

criterion = nn.CrossEntropyLoss()
model3.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data = (X_val, Y_val))
if torch.cuda.is_available():

    model4 = model4.cuda()

    criterion = criterion.cuda()

    

model4.train()



steps = 0

print_every = 50

train_losses, test_losses = [], []



for e in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        steps += 1

        if torch.cuda.is_available():

            images = images.cuda()

            labels = labels.cuda()

        # Prevent accumulation of gradients

        optimizer4.zero_grad()

        # Make predictions

        log_ps = model4(images.float())

        loss = criterion(log_ps, labels.long())

        #backprop

        loss.backward()

        optimizer4.step()

        

        running_loss += loss.item()

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0



            # Turn off gradients for validation

            with torch.no_grad():

                model4.eval()

                for images, labels in test_loader:

                    if torch.cuda.is_available():

                        images = images.cuda()

                        labels = labels.cuda()

                    log_ps = model4(images.float())

                    test_loss += criterion(log_ps, labels.long())



                    ps = torch.exp(log_ps)

                    # Get our top predictions

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor))



    train_losses.append(running_loss/len(train_loader))

    test_losses.append(test_loss/len(test_loader))



    print("Epoch: {}/{}.. ".format(e+1, epochs),

          "Training Loss: {:.4f}.. ".format(train_losses[-1]),

          "Test Loss: {:.4f}.. ".format(test_losses[-1]),

          "Test Accuracy: {:.4f}".format(accuracy/len(test_loader)))
metrics2=pd.DataFrame(model3.history.history)

plt.plot(metrics2['loss'], label='Keras_loss')

plt.plot(train_losses, label='PyTorch_loss')

plt.legend()

plt.show()
plt.plot(metrics2['val_loss'], label='Keras_val_loss')

plt.plot(test_losses, label='PyTorch_val_loss')

plt.legend()

plt.show()
results = model3.predict(X_test)
def make_prediction(data):

    images, labels = next(iter(data))

    if torch.cuda.is_available():

        images = images.cuda()

        labels = labels.cuda()



    img = images

    # Turn off gradients to speed up this part

    with torch.no_grad():

        logps = model4(images.float())



    ps = torch.exp(logps)

make_prediction(submission_loader)
class CNN_2(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)

        )

          

        self.classifier = nn.Sequential(

            nn.Dropout(p = 0.5),

            nn.Linear(64 * 7 * 7, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(p = 0.5),

            nn.Linear(512, 512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(p = 0.5),

            nn.Linear(512, 10),

        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        

        return x