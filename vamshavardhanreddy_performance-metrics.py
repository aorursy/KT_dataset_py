### Importing Required Packages
import pandas  as pd
## Setting up the files
CANCERDATA = "../input/DS_WDBC_NOIDFIELD.data"
# Function to convert Categorical variables into numeric variables
def labelConvert(s):
    s = s.strip().lower()
    if s == "m":
        return 0
    if s == "b":
        return 1
    return -1

data = pd.read_csv(CANCERDATA, header = None, converters={30:labelConvert})
data[:10]
def split_Train_Test(data):
    import random
    TRAIN_TEST_RATIO = 0.8
    train = []
    test = []
    for d in data:
        if random.random() < TRAIN_TEST_RATIO:
            train.append(d)
        else:
            test.append(d)
    return train, test
def predict(row, weights):
    activation = weights[0]
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

### We are using an algorithm called gradient descent.

def train_weights(train, l_rate, n_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error**2
            weights[0] = weights[0] + l_rate * error
            for i in range(len(row)-1):
                weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
        if epoch % 100 == 0:
            print('> epoch = %4d, lrate = %.4f, error =%6.1f' % (epoch, l_rate, sum_error))
    return weights

## Do not worry about the hyperparameters of this algorithm right now.
train, test = split_Train_Test(data.values)
print("Training", len(train))
print("Testing", len(test))
weights = train_weights(train, l_rate=0.001, n_epoch=2000)
print(weights)
### You have preds and actuals now :)
### Your Code Here
pred_list=[]
pass_cnt=0
for row in test:
    pred = predict(row,weights)
    pred_list.append(pred)
    if(pred == row[-1]):
        pass_cnt +=1
print("accuracy", pass_cnt/len(test))

actual = [row[-1] for row in test]

def confusionmatrix(actuals, prediction):
    TruePositive = sum([int(a == 1 and p == 1) for a, p in zip(actuals, prediction)])
    TrueNegative = sum([int(a == 0 and p == 0) for a, p in zip(actuals, prediction)])
    FalsePositive = sum([int(a == 0 and p == 1) for a, p in zip(actuals, prediction)])
    FalseNegative = sum([int(a == 1 and p == 0) for a, p in zip(actuals, prediction)])
    return TruePositive, TrueNegative, FalsePositive, FalseNegative
#confusionmatrix(actual, pred)
confusionmatrix(actual, pred_list)
### Your Code Here
tp,tn,fp,fn = confusionmatrix(actual, pred_list)

p = tp/(tp+fp)
p
### Your Code Here
r = tp/(tp+fn)
r
### Your Code Here
m = 1-r
m
### Your Code Here
a = (tp+tn)/(tp+tn+fp+fn)
a
F = 2 * ((p*r)/p+r)
F
TPR = tp/tp+fn

TPR
FPR = fp/fp+tn

FPR