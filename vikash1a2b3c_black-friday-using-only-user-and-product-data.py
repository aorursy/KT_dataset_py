# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Importing the dataset

train= pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')



#Join test and train for preprocessing benefit

train['source']='train'

test['source']='test'

data = pd.concat([train, test],ignore_index=True,sort=False)
#creating data matrix

data=data.fillna(0)

data_rec = np.zeros((783667,3))

for line in data.itertuples(): 

    m=line[2]

    data_rec[line[0],0]=int(line[1]%1000000)

    data_rec[line[0],1]=int(m[1:])

    data_rec[line[0],2]=line[12]



product_id=data_rec[:,1]

product_id=np.unique(product_id)

product_id_1=np.zeros((3677,2))

for i in range(0,3677):

    product_id_1[i][0]=i

    product_id_1[i][1]=product_id[i]

    

product_id_2=pd.DataFrame(data=product_id_1[:,0],index=product_id_1[:,1])



data_matrix= np.zeros((6040,3677))

for i in range(0,783667):

    data_matrix[int(data_rec[i][0])-1,int(product_id_2[0][data_rec[i][1]])] = data_rec[i][2]
#matrix factorization code

class MF():



    # Initializing the user-movie rating matrix, no. of latent features, alpha and beta.

    def __init__(self, R, K, alpha, beta, iterations):

        self.R = R

        self.num_users, self.num_items = R.shape

        self.K = K

        self.alpha = alpha

        self.beta = beta

        self.iterations = iterations



    # Initializing user-feature and movie-feature matrix 

    def train(self):

        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))

        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))



        # Initializing the bias terms

        self.b_u = np.zeros(self.num_users)

        self.b_i = np.zeros(self.num_items)

        self.b = np.mean(self.R[np.where(self.R != 0)])



        # List of training samples

        self.samples = [

        (i, j, self.R[i, j])

        for i in range(self.num_users)

        for j in range(self.num_items)

        if self.R[i, j] > 0

        ]



        # Stochastic gradient descent for given number of iterations

        training_process = []

        for i in range(self.iterations):

            np.random.shuffle(self.samples)

            self.sgd()

            mse = self.mse()

            training_process.append((i, mse))

            if (i+1) % 20 == 0:

                print("Iteration: %d ; error = %.4f" % (i+1, mse))



        return training_process



    # Computing total mean squared error

    def mse(self):

        xs, ys = self.R.nonzero()

        predicted = self.full_matrix()

        error = 0

        for x, y in zip(xs, ys):

            error += pow(self.R[x, y] - predicted[x, y], 2)

        return np.sqrt(error)



    # Stochastic gradient descent to get optimized P and Q matrix

    def sgd(self):

        for i, j, r in self.samples:

            prediction = self.get_rating(i, j)

            e = (r - prediction)



            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])

            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])



            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])

            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])



    # Ratings for user i and moive j

    def get_rating(self, i, j):

        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)

        return prediction



    # Full user-movie rating matrix

    def full_matrix(self):

        return mf.b + mf.b_u[:,np.newaxis] + mf.b_i[np.newaxis:,] + mf.P.dot(mf.Q.T)





#Now let us predict all the missing ratings. Let’s take K=20, alpha=0.001, beta=0.01 and iterations=100.

data_matrix_1=data_matrix/10000

mf = MF(data_matrix_1, K=20, alpha=0.001, beta=0.01, iterations=100)

training_process = mf.train()

print()

print("P x Q:")

print(mf.full_matrix())

full_matrix=(mf.full_matrix())*10000

print()
#finding the predicted data

for i in range(550068,783667):

     data_rec[i][2]=full_matrix[int(data_rec[i][0])-1,int(product_id_2[0][data_rec[i][1]])] 

data_rec_1=data_rec[550068:][:]
#submission file

submission=pd.DataFrame()

submission['User_ID']=test['User_ID']

submission['Product_ID']=test['Product_ID']

submission['Purchase']=data_rec_1[:,2]

submission.to_csv("submission.csv",index=False)