import sys, numpy as np

from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data() 
images, labels = (x_train[0:1000].reshape(1000,28*28)/ 

                                                  255, y_train[0:1000])
one_hot_labels=np.zeros((len(labels),10))
for i, l in enumerate (labels):

    one_hot_labels[i][l] =1

    labels=one_hot_labels
test_images=x_test.reshape(len(x_test),28*28)/255

test_labels=np.zeros((len(y_test),10))

for i,l in enumerate(y_test):

    test_labels[i][l]=1
np.random.seed(1)

relu=lambda x:(x>0)*x

relu2deriv=lambda x: x>=0

alpha, iterations, hidden_size, pixels_per_image, num_labels=(0.005,350,40,784,10)
weights01=0.2*np.random.random((pixels_per_image,hidden_size))-0.1

weights12=0.2*np.random.random((hidden_size, num_labels))-0.1
for j in range(iterations):

    error, correct_cnt=(0.0,0)

    

    for i in range (len(images)):

        layer_0 =images[i:i+1]

        layer_1 =relu(np.dot(layer_0, weights01))

        layer_2=np.dot(layer_1, weights12)

        error+= np.sum((labels[i:i+1]-layer_2)**2)

        correct_cnt+= int(np.argmax(layer_2)==np.argmax(labels[i:i+1]))

        

        layer_2_delta=(labels[i:i+1]-layer_2)

        layer_1_delta=layer_2_delta.dot(weights12.T)*relu2deriv(layer_1)

        

        weights12+=alpha*layer_1.T.dot(layer_2_delta)

        weights01+=alpha*layer_0.T.dot(layer_1_delta)
sys.stdout.write("\r"+"I:"+str(j)+"Error:" +str(error/float(len(images)))[0:5]+"Correct:"+str(correct_cnt/float(len(images))))