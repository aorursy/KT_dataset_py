import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler,LabelEncoder, label_binarize

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, auc, f1_score, precision_score,recall_score, roc_auc_score

from keras.datasets import mnist

import warnings 

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/breast-cancer-dataset/breast_cancer.csv')

del df['Unnamed: 32']

x = df.iloc[:,2:]



y = df['diagnosis']

y = y.map({'M': 0, 'B': 1})

y = y.values





sc = MinMaxScaler()

X = sc.fit_transform(x)

X = np.c_[np.ones(len(X)),X]

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

x.head()
def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def train_perceptron(X,Y,alpha = 0.1,iter = 2500):

    theta = np.random.uniform(size=(X.shape[1], 1))

    Y = Y.reshape(Y.shape[0], 1)

    for i in range(iter):

        z = X @ theta

        Y_pred = sigmoid(z)

        error = Y - Y_pred

        temp = alpha * error

        theta += X.T @ temp

    return theta
def evaluateClassifier(x,y,y_pred,y_score):

    cm = pd.DataFrame(

        confusion_matrix(y, y_pred),

        columns=['Predicted Benign', 'Predicted Malignant'],

        index=['True Benign', 'True Malignant']

    )

    print('\nConfusion Matrix: \n')

    sns.set(font_scale=1.4) # for label size

    sns.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size

    plt.show()



    w1 = cm['Predicted Benign']['True Benign'] / (cm['Predicted Benign']['True Benign'] + cm['Predicted Malignant']['True Benign'])

    w2 = cm['Predicted Malignant']['True Malignant'] / (cm['Predicted Benign']['True Malignant'] + cm['Predicted Malignant']['True Malignant'])

    print('\nClasswise accuracy: ')

    print('\nBenign: ',w1 * 100)

    print('\nMalignant: ',w2 * 100)

    

    indices = ['Accuracy','Precision','F1 score','Recall  score']

    eval = pd.DataFrame([accuracy_score(y,y_pred) * 100,precision_score(y,y_pred,average = 'macro') * 100,f1_score(y,y_pred,average = 'macro') * 100,recall_score(y,y_pred,average = 'macro') * 100],columns=['Value'],index=indices)

    eval.index.name = 'Metrics'

    print('\n',eval)

  

    fpr,tpr,_ = roc_curve(y, y_score)

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)        

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.05])

    plt.ylim([0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend()

    plt.title('ROC curve')

    plt.show()
weights = train_perceptron(x_train,y_train)

y_pred = sigmoid(x_test @ weights)

y_probs = y_pred

y_pred[y_pred >=  0.5] = 1

y_pred[y_pred <  0.5] = 0

evaluateClassifier(x_test,y_test,y_pred,y_probs)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255

x_test = x_test / 255



nsamples, nx, ny = x_train.shape

x_train = x_train.reshape((nsamples,nx*ny))



nsamples, nx, ny = x_test.shape

x_test = x_test.reshape((nsamples,nx*ny))

training_acc = np.zeros(11)

testing_acc = np.zeros(11)

iter = 0

for i in range(30,41): 

    mlp = MLPClassifier(hidden_layer_sizes = (i), max_iter = 10, alpha=0.001, solver='sgd', verbose=False, learning_rate_init=0.01)

    

    mlp.fit(x_train, y_train)

    training_acc[iter] = mlp.score(x_train, y_train)

    

    mlp.fit(x_test, y_test)

    testing_acc[iter] = mlp.score(x_test, y_test)

    iter += 1

plt.plot(range(30,41),training_acc * 100,'b-')

plt.xlabel('No. of hidden layer nodes')

plt.ylabel('Accuracy')

plt.title('Training accuracy v/s no. of hidden nodes')

plt.show()

s = pd.Series(training_acc * 100,range(30,41))

df = pd.DataFrame({'No. of hidden nodes':s.index, 'Training Accuracy':s.values})

df

plt.plot(range(30,41),testing_acc * 100,'b-')

plt.xlabel('No. of hidden layer nodes')

plt.ylabel('Accuracy')

plt.title('Testing accuracy v/s no. of hidden nodes')

plt.show()

testing_acc * 100



s = pd.Series(testing_acc * 100,range(30,41))

df = pd.DataFrame({'No. of hidden nodes':s.index, 'Testing Accuracy':s.values})

df

mlp = MLPClassifier(hidden_layer_sizes = (40), max_iter = 10, alpha=0.001, solver='sgd', verbose=False, learning_rate_init=0.01)

mlp.fit(x_train,y_train)

y_pred = mlp.predict(x_test)

y_probs = mlp.predict_proba(x_test)

indices = ['Accuracy','Precision','F1 score','Recall  score']

eval = pd.DataFrame([accuracy_score(y_test,y_pred) * 100,precision_score(y_test,y_pred,average = 'macro') * 100,f1_score(y_test,y_pred,average = 'macro') * 100,recall_score(y_test,y_pred,average = 'macro') * 100],columns=['Value'],index=indices)

eval.index.name = 'Metrics'

print('\n',eval)


classes = range(10)

probabs = y_probs

y_test2 = label_binarize(y_test, classes)

for i in range(10):

    preds = probabs[:,i]    

    fpr, tpr, threshold = roc_curve(y_test2[:, i], preds)

    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'Class ' + str(i + 1))

    plt.legend(loc = 'lower right')

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

plt.show()
