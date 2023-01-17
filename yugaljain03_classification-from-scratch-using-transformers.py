# IMPORTING LIBRARIES
import torch 
from torch import nn
from torch.nn import LayerNorm,Dropout,Linear,Softmax
# import activation function named softmax
import torch.nn.functional as F
from torch.nn import Embedding
import torch.nn.functional as F
from sklearn.metrics import f1_score
# Self Attention Architecture
class selfattention(nn.Module):
    
    def __init__(self,k,heads=9):
        super().__init__()
        self.k , self.heads = k,heads
        # initialize dimensions for keys,values and queries as we are building multi-head attention layers for modern transformer
        self.tokeys = nn.Linear(k,k*heads,bias=False)
        self.toqueries = nn.Linear(k,k*heads,bias=False)
        self.tovalues = nn.Linear(k,k*heads,bias=False)
        
        self.unified = nn.Linear(k*heads,k,bias=False)
        
    def forward(self,x):  # pass values into __init__ function
        b,t,k = x.size()  # where b is number of batches, t is size of input sequence length and k is number of dimensions
        # now determine keys,values,queries and heads
        h = self.heads
        
        keys = self.tokeys(x).view(b,t,h,k) # initially we don't have head next to each other
        queries = self.toqueries(x).view(b,t,h,k)
        values = self.tovalues(x).view(b,t,h,k)
    
        # now we want head next to batch, its highly computational expensive but we have to do it, so transform keys,values and queries
        keys = keys.transpose(1,2).contiguous().view(b*h,t,k)
        queries = queries.transpose(1,2).contiguous().view(b*h,t,k)
        values = values.transpose(1,2).contiguous().view(b*h,t,k)
        # now calculate dot product using bmm function in pytorch
        dot = torch.bmm(queries,keys.transpose(1,2))
        
        dot = F.softmax(dot,dim=2)
        out = torch.bmm(dot,values).view(b,h,t,k)
        
        # now finally we want unified output in k dimensions as we have initially, so to do this again transspose out
        out = out.transpose(1,2).contiguous().view(b,t,h*k)
        out = self.unified(out)
        return out  
# now build architecture for transformer block
class transformer(nn.Module):
    def __init__(self,k,heads=9):
        super().__init__()
        self.attention = selfattention(k) # pass class named selfattention
        # now add normalization layer to normalize outputs of attention layer
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        # now make a fully connected multi layer perceptron
        self.ff = nn.Sequential(nn.Linear(k,5*k),nn.ReLU(),nn.Linear(5*k,k)) # fully connected layer for hidden states
        self.drop = nn.Dropout(0.5)
    def forward(self,x):
        
        attention = self.attention(x)
        
        x = self.norm1(attention+x)
        x = self.drop(x) # dropout layer after normalization(drop some neurons to prevent from overfitting)
        perceptron = self.ff(x)
        x = self.norm2(perceptron + x)
        x = self.drop(x)
        return x
# now add classification and add embedding layer initially to it.
# REMEMBER INPUT SHOULD BE INDEXES OF WORDS 
class classify(nn.Module):
    def __init__(self,k,seq_length,num_tokens,depth,num_classes,max_pool=True,heads=9):
        super().__init__()
        # initialize tokens
        self.num_tokens = num_tokens
        self.maxpool = max_pool
        # it needs input in word indexes using tokenizer
        self.tokenemb = Embedding(embedding_dim=k,num_embeddings=num_tokens)
        # position embedding
        self.posemb = Embedding(embedding_dim=k, num_embeddings=seq_length)
        tfblocks = []
        for i in range(depth):
            tfblocks.append(transformer(k))
        # now add sequential layer of tfblocks
        self.transform = nn.Sequential(*tfblocks)
        # add linear layer to convert into desired number of class i.e. 2 for binary class
        self.prob = nn.Linear(k,num_classes)
        self.drop = nn.Dropout(0.5)
    
    def forward(self,x,y):
        
        tokens = self.tokenemb(x)
        b,t,k = tokens.size()
        positions = self.posemb(torch.arange(t, device=torch.device('cuda')))[None, :, :].expand(b, t, k)
        x = tokens+positions # adding tokens and positions to get proper embeddings of each indexes of words
        x = self.drop(x)
        x = self.transform(x)
        x = x.max(dim=1)[0] if self.maxpool else x.mean(dim=1)
        x = self.prob(x)
        x = F.softmax(x,dim=1)
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(x,y)
        return loss,x        
if torch.cuda.is_available():
    device = torch.device('cuda')
device
# Now we have base modern transformer architecture and we can use this architecture for low level end task like sentiment classification,
# machine language translation by adding some layers before input(for word embeddings) and after output(to classify or translation).

# In this way GOogle stacked 18 transformers to train Bidirectional transformer for language understanding whereas deepspeed by microsoft take 48 transformers block in stack form
# This architecture can't handle parallelly more than 512 tokens initially so mind your length of input tokens

# read sentimental analysis data and generating padded sequences of training data
import pandas as pd
import numpy as np
train_data = pd.read_csv('../input/traini/cleaned_train_fina.csv')['user_review'].values
labels = list(pd.read_csv('../input/traini/train.csv')['user_suggestion'].values)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_data,labels,test_size=0.2)
train_data = np.array(x_train)
train_data = list(train_data)
labels = list(y_train)

test_data = train_data
print(test_data[0])
final_data = []
import re
for data in test_data:
    new = re.sub('[0-9]','',str(data))
    final_data.append(new)
print(final_data[0])
from tensorflow.keras.preprocessing import text,sequence

tokenize = text.Tokenizer()

tokenize.fit_on_texts(texts = (final_data))
index_data = tokenize.texts_to_sequences(final_data)

word_index = tokenize.word_index
print(len(word_index))

pad_sequences = sequence.pad_sequences(index_data,maxlen=50)
len(pad_sequences)
# initializing classify model for binary classification
classifier = classify(2,200,len(word_index)+1,12,2)
classifier.to(device)
from torch.utils.data import DataLoader,SequentialSampler,TensorDataset,RandomSampler
dataset = torch.utils.data.TensorDataset(torch.LongTensor(pad_sequences), torch.LongTensor(labels))
batch_data = DataLoader(batch_size=128,dataset = dataset, sampler = SequentialSampler(dataset))
# declare optimizer 
from torch.autograd import no_grad,Variable
from torch.optim import Optimizer
from torch.optim import Adam
optimizer = Adam(classifier.parameters(),lr=0.003)
x_test = np.array(x_test)
x_test = list(x_test)
y_test = np.array(y_test)
y_test = list(y_test)
test_data = x_test
final_data_test = []
# generating padded sequences for test data
import re
for data in test_data:
    new = re.sub('[0-9]','',str(data))
    final_data_test.append(new)
print(final_data_test[0])
from tensorflow.keras.preprocessing import text,sequence

tokenize = text.Tokenizer()

tokenize.fit_on_texts(texts = (final_data_test))
index_data_test = tokenize.texts_to_sequences(final_data_test)

word_index_test = tokenize.word_index
print(len(word_index_test))

pad_sequences_test = sequence.pad_sequences(index_data_test,maxlen=50)

test_dataset = TensorDataset(torch.LongTensor(pad_sequences_test),torch.LongTensor(y_test))
test_loader = DataLoader(test_dataset,batch_size=128,sampler = RandomSampler(test_dataset))
for test in test_loader:
    print(test)
    break
# set scheduler for learning rate, for that calculate total steps
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler
epochs = 40 # researchers suggest to take epochs should be in range of 4-7 for fine tuning pretrained model as we have concern 
# just for last layer which is untrained classification layer.
total_steps = len(batch_data) * epochs
# scheduler take care of linear schedule of learning rate 
scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)
# Training 
epochs = 1000
final_loss = []
output = []
testing_accuracy = []
for epoc in range(epochs):
    print('epoch-',epoc)
    total_loss = 0
    classifier.train()
    for step,batch in enumerate(batch_data):

        
        classifier.zero_grad()
        loss,outputs = classifier.forward(x=(batch[0].to(device)),y=(batch[1].to(device)))
        
        loss.backward()
        total_loss+=loss.item()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(),1.0)
        optimizer.step()
        scheduler.step()
        
    avg_loss = total_loss/len(batch_data)
    print(avg_loss)
    final_loss.append(avg_loss) # testing outputs
    # Validating
    classifier.eval()
    test_accuracy = 0
    for step,batch_t in enumerate(test_loader):
        with torch.no_grad():
            outputs = classifier.forward((batch_t[0].to(device)) ,y=(batch_t[1].to(device)))
            predictions = outputs[1]
            # no need for backward loss and zero grad
            test_accuracy += f1_score(y_pred=np.argmax(predictions.cpu().detach().numpy(),axis=1),y_true=batch_t[1].cpu().detach().numpy())
            
            output.append(predictions)
    avg_accuracy = test_accuracy/len(test_loader)
    testing_accuracy.append(avg_accuracy)    
    
# saving entire model
torch.save(classifier,'tf_classify.pth')