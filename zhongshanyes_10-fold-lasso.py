'Shan Zhong'

from sklearn import datasets

import numpy as np

Boston = datasets.load_boston()



###############################################################################

'Get Data'

feature = Boston['feature_names']



def scaler(X):

    return ( X - np.mean(X) )/( np.std(X , ddof=1) )



X = np.apply_along_axis(scaler, 0, Boston['data'])



'assume 0 intercept'

y = Boston['target'] - np.mean(Boston['target'])
###############################################################################

'lasso model with 0 intercept'

class mk_model(object):



    def __init__(self, lr , X, y ):

        self.lr = lr

        self.X = X

        self.y = y

        self.params = np.zeros(shape = [len(feature),] , dtype = np.float64 )

        self.loss =  np.float32()

        self.lam =  np.float32()

       

    def predict(self,X):

        kernal = self.params

        bias = 0

        return np.matmul(X, kernal ) + bias

  

    def get_loss(self):

        output = self.predict(self.X)    

        return ((self.y-output)**2).sum() + (self.lam * abs(self.params[:-1])).sum() 



    def get_updates(self, params, lam):

        

        def soft_threshold(theta, lam):

            if theta > lam:

                return theta - lam

            elif theta < -lam:

                return theta + lam

            else:

                return 0

            

        for i in range(len(params)):

            other = np.setdiff1d(range(len(params)),i)

            tehta = np.matmul(X[:,i] ,y - np.matmul(X[:,other], params[other] )) / np.matmul( np.transpose(X[:,i]) , X[:,i]) 

            params[i] = soft_threshold(tehta,lam)



        return params



    def fit(self, lam): 

        self.params = np.zeros(shape = [len(feature),] , dtype = np.float64 )

        self.lam = lam

        for i in range(100000):

            self.params = self.get_updates(self.params,self.lam)

            loss = self.get_loss()

            if abs(loss-self.loss) < 0.0001:

                #print('converge after ' + str(i) +' iter')

                break

            self.loss = self.get_loss()

         
###############################################################################

from matplotlib import pyplot as plt

lam = np.arange(0,10,0.05)



params = np.zeros(shape = [ len(lam),len(feature)] ,  dtype = np.float64)

model = mk_model(lr = 0.0001, X = X, y = y) 



for i in range(len(lam)):

    model.fit(lam = lam[i])

    params[i] = model.params



for i in range(len(feature)):

    plt.plot(lam, np.transpose(params)[i])

plt.show()   
###############################################################################

'k fold cross validation'

class n_fold(object):



    def __init__(self, n_splits = 10):

        self.n_splits = n_splits



    def _iter_indices(self, X, y=None, groups=None):

            n_samples = X.shape[0]

            indices = np.arange(n_samples)

            np.random.shuffle(indices)

            n_splits = self.n_splits

            fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=np.int)

            fold_sizes[:n_samples % n_splits] += 1

            current = 0

            for fold in range(n_splits):

                fold_size = fold_sizes[fold]

                start, stop = current, current + fold_size

                yield fold , np.setdiff1d(indices,indices[start:stop]) , indices[start:stop] 

                current = stop
###############################################################################

n_splits = 10

error = np.zeros(shape = [n_splits , len(lam)] ,dtype = np.float64)



for fold , train_idx, valid_idx in n_fold(n_splits = n_splits)._iter_indices(X):



    train_inputs = X[train_idx]

    train_outputs = y[train_idx]

    

    valid_inputs = X[valid_idx]

    valid_outputs = y[valid_idx] 

    

    model = mk_model(lr = 0.0001, X = train_inputs, y = train_outputs) 

    

    for i in range(len(lam)):

        model.fit(lam = lam[i])

        error[fold,i] = sum(np.square(model.predict(valid_inputs) - valid_outputs))



plt.plot(lam, np.mean(error,axis = 0))