import pandas as pd

# import matplotlib.pyplot as plt

# import matplotlib.image as mpimg

# %matplotlib inline

# %config InlineBackend.figure_format = 'retina'
labels_df = pd.read_csv('../input/train_v2.csv')

labels_df.head(10)
from itertools import chain

labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))

labels_set = set(labels_list)
labels = sorted(labels_set)

labels_map = {l: i for i, l in enumerate(labels)}

y_map = {v: k for k, v in labels_map.items()}
y_map
labels_s = pd.Series(labels_list).value_counts() # To sort them by count

# fig, ax = plt.subplots(figsize=(16, 8))

# sns.barplot(x=labels_s, y=labels_s.index, orient='h')
# images_title = [labels_df[labels_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 

#                 for i, label in enumerate(labels_set)]
# train_jpeg_dir = './data/train-jpg'

# plt.rc('axes', grid=False)

# _, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))

# axs = axs.ravel()

# for i, (image_name, label) in enumerate(zip(images_title, labels_set)):

#     img = mpimg.imread(train_jpeg_dir + '/' + image_name)

#     axs[i].imshow(img)

#     axs[i].set_title('{} - {}'.format(image_name, label))
# img_resize = (224, 224) # The resize size of each image

# x_train, y_train, y_map = preprocess_train_data(train_jpeg_dir, train_csv_file, img_resize)

# # Free up all available memory space after this heavy operation

# gc.collect();
import torch.nn as nn

from torch.autograd import Variable

import torch

import torchvision.models as models
import scipy.misc

from concurrent.futures import ThreadPoolExecutor

from multiprocessing import cpu_count

def get_imgs(*args):

#     batch_img = []

#     for path in path_batch:

    path,size_ = list(args[0])

    img = scipy.misc.imread(path, mode='RGB')

    img = scipy.misc.imresize(img, size_, interp='bilinear', mode=None)

    img = img/ 255.0

#         batch_img.append(img)

    return img

def get_batch(files_path,img_resize,dir_path):

    x_train = []

    with ThreadPoolExecutor(cpu_count()) as pool:

        for img_array in pool.map(get_imgs,[(dir_path+file_path+'.jpg',img_resize) for file_path in files_path]):

                x_train.append(img_array)

    return x_train
import numpy as np

labels_df = pd.read_csv("../input/train_v2.csv")

labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))

labels_map = {l: i for i, l in enumerate(labels)}



files_path = []

y_label = []

for file_name, tags in labels_df.values:

    files_path.append(file_name)

    targets = np.zeros(len(labels_map))

    for t in tags.split(' '):

        targets[labels_map[t]] = 1

    y_label.append(targets)
from sklearn.model_selection import train_test_split

X_train, X_valid, Y_train, Y_valid = train_test_split(files_path, y_label,test_size=0.1)
class Amazon(nn.Module):

    def __init__(self, pretrained_model_1):

        super(Amazon, self).__init__()

        self.pretrained_model_1 = pretrained_model_1

        # self.pretrained_model_2 = pretrained_model_2

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(2000,1000)

        self.fc2 = nn.Linear(1000,len(labels_set)) # create layer

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        model_1 = self.relu(self.pretrained_model_1(x))

        #model_2 = self.relu(self.pretrained_model_2(x))

        #out1 = torch.cat((model_1,model_2),1)

        return self.sigmoid(self.fc2(self.relu(model_1)))



#pretrained_model1 = models.densenet169(pretrained=True)

pretrained_model1 = models.resnet18(pretrained=False)#in fact, this should be set as true



model = Amazon(pretrained_model1)
dir_path = '../input/train-jpg/'
input_ = Variable(torch.from_numpy(np.transpose(get_batch(X_train[0:32],(224,224),dir_path), (0, 3,1, 2)))).float()

o = model(input_)

o.size()
from sklearn.metrics import fbeta_score
def train(train_x,train_y,valid_x, valid_y,epoch,num_model,img_resize,dir_path):

    batch_size = 128

    pretrained_model1 = models.resnet18(pretrained=False)

    model = Amazon(pretrained_model1)

    torch.cuda.set_device(2)

    criterion = nn.BCELoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.cuda()

    best_score = 0

    for epo in range(epoch):

        num_shuffle = np.random.permutation(range(len(train_y)))

        for step in range(len(train_x)/batch_size):

            x_batch = np.transpose(get_batch(train_x[num_shuffle[step*batch_size:(step+1)*batch_size]],img_resize,dir_path), (0, 3,1, 2))

            input_var = Variable(torch.from_numpy(x_batch)).float().cuda()

            target_var = Variable(torch.from_numpy(train_y[num_shuffle[step*batch_size:(step+1)*batch_size]])).cuda().float()

            output = model(input_var)

            loss = criterion(output, target_var)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if step % 30 ==0:

                valid_pred = validate(model,valid_x, valid_y,32,img_resize,dir_path)

                threshhold = [0.2]*17

                score = fbeta_score(np.array(valid_y)[:len(valid_pred)], np.array(valid_pred) >threshhold, beta=2, average='samples')

                print("epo: "+str(epo)+" step: "+str(step)+"  score: "+str(score))

                print('loss: '+ str(loss.data.cpu().numpy()[0].astype(float)))

                path = './model_'+str(num_model)+'.pkl'

                if score > best_score:

                    best_score = score

                    torch.save(model.state_dict(), path)

                    print("save in : "+ path)
def validate(model_,x_valid, y_valid,batch_val_size,img_resize,dir_path):

    p_valid = []

    pred_true = []

    for i in range(len(x_valid)/batch_val_size-1):

        #target = target.cuda(async=True)

        x_batch = np.transpose(get_batch(x_valid[i*batch_val_size:(i+1)*batch_val_size],img_resize,dir_path), (0, 3,1, 2))

        input_var = Variable(torch.from_numpy(x_batch)).float().cuda()

        target_var = Variable(torch.from_numpy(y_valid[i*batch_val_size:(i+1)*batch_val_size])).cuda().float()

        output = model_(input_var).data.cpu().numpy().astype(float)

        p_valid.extend(output)

        pred_true.extend(y_valid[i*batch_val_size:(i+1)*batch_val_size])

    return p_valid
def test_pred(x_test,batch_test_size,path,dir_path):

    pretrained_model1 = models.resnet18(pretrained=True)

    model = Amazon(pretrained_model1)

    model.load_state_dict(torch.load(path))

    torch.cuda.set_device(2)

    model.cuda()

    p_test = []

    for step in range(len(x_test)/batch_test_size):

        if step%20==0:

            print(step)

        x_batch = np.transpose(get_batch(x_test[step*batch_test_size:(step+1)*batch_test_size],img_resize,dir_path), (0, 3,1, 2))

        input_var = Variable(torch.from_numpy(x_batch)).float().cuda()

        output = model(input_var).data.cpu().numpy().astype(float)

        p_test.extend(output)

    left_data = get_batch(x_test[-(len(x_test)- len(x_test)/batch_test_size*batch_test_size):],img_resize,dir_path)

    input_var = Variable(torch.from_numpy(np.transpose(left_data, (0, 3,1, 2)))).float().cuda()

    output = model(input_var).data.cpu().numpy().astype(float)

    p_test.extend(output)

    return p_test
# del x_train, y_train

# gc.collect();
img_resize = (224,224)
epoch=0

num_model =0

train(np.array(X_train),np.array(Y_train),np.array(X_valid),np.array(Y_valid),epoch,num_model,img_resize,dir_path)
path = './model_0.pkl'

y_pred = test_pred(np.array(X_valid),128,path,dir_path)
import numpy as np

from sklearn.metrics import fbeta_score

def get_optimal_threshhold(true_label, prediction, iterations = 100):



    best_threshhold = [0.2]*17    

    for t in range(17):

        best_fbeta = 0

        temp_threshhold = [0.2]*17

        for i in range(iterations):

            temp_value = i / float(iterations)

            temp_threshhold[t] = temp_value

            temp_fbeta = fbeta(true_label, prediction >temp_threshhold)

            if  temp_fbeta>best_fbeta:

                best_fbeta = temp_fbeta

                best_threshhold[t] = temp_value

    return best_threshhold



def fbeta(true_label, prediction):

    return fbeta_score(true_label, prediction, beta=2, average='samples')
best_threshhold = get_optimal_threshhold(np.array(Y_valid)[:len(y_pred)], np.array(y_pred), iterations = 100)
path = './model_0.pkl'
import pandas as pd
test_sub = pd.read_csv('../input/sample_submission_v2.csv')
sample_sub = test_sub.image_name.values
import os

files_name_test1 = sample_sub[:len(os.listdir("../test-jpg/"))]
files_name_test2 = sample_sub[len(os.listdir("./data/test-jpg/")):]
predictions = test_pred(files_name_test1,128,path,"./data/test-jpg/")
new_predictions = test_pred(files_name_test2,128,path,"./data/test-jpg-additional/")
predictions = np.vstack((predictions, new_predictions))

x_test_filename = sample_sub
def map_predictions(predictions, labels_map, thresholds):

    """

    Return the predictions mapped to their labels

    :param predictions: the predictions from the predict() method

    :param labels_map: the map

    :param thresholds: The threshold of each class to be considered as existing or not existing

    :return: the predictions list mapped to their labels

    """

    predictions_labels = []

    for prediction in predictions:

        labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]

        predictions_labels.append(labels)



    return predictions_labels
#np.save('ss.npy',predictions)
#threshhold = [0.2]*17

predicted_labels = map_predictions(predictions, y_map, best_threshhold)
tags_list = [None] * len(predicted_labels)

for i, tags in enumerate(predicted_labels):

    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]
final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])

final_df.head()
final_df.to_csv('./sub/submission_0.csv', index=False)