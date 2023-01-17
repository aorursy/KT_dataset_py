import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
from scipy.io import wavfile

#Se extrae los datos y la frecuenci de muestreo 
fs, data = wavfile.read("../input/voice-example/voz yosef.wav")
data = data/2.0**15
#fs es igual a 100 frames de 160 muestreos cada uno.
data=data[0:fs*67]
# Data de entrenamiento (1 min =60 Frame Samples(fs))
training_data = data[0:fs*20]
training_data = training_data.reshape(2000,160)
%matplotlib inline

plt.plot(data[fs*0:fs*67])

from IPython.display import Audio
Audio(rate=fs, data=data[fs*0:fs*67])
# Ejemplo de frame
plt.plot(training_data[2])
data2=training_data[2].reshape(-1,1)
Audio(data=training_data[2],rate=fs)
plt.plot(training_data[14])
plt.plot(training_data[15])
plt.plot(training_data[16])
plt.title("Señales $y_i$ de $Y$")
plt.legend(["$y_{14}$", "$y_{15}$","$y_{16}$"])
import numpy as np
import scipy as sp
import itertools
from sklearn.linear_model import orthogonal_mp_gram, OrthogonalMatchingPursuit

T=0
Error=[]
Error2=[]
Error_n_K=[]
class KSVD(object):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements
        max_iter:
            Maximum number of iterations
        tol:
            tolerance for error
        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, Y, D, X):
        
        Error3=[]
        
        for j in range(self.n_components):
            I = X[:, j] != 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = X[I, j].T
            r = Y[I, :] - X[I, :].dot(D)
            if(j==0):
                print("I:", I)
                print("I:", I.shape)
                print("Y:",Y.shape)
                print("X:",X.shape)
                print("D:",D.shape)
                print("g,X_R^k:" ,g)
                print("g,X_R^k:" ,g.shape)
                print("R:" , r.shape)
            u,s,vh=  np.linalg.svd(r.T, full_matrices=True)
            #d = r.T.dot(g)
            #d /= np.linalg.norm(d)
            #g = r.dot(d)
            #print("R.T:",r.T.shape)
            #print("U:",u.shape)
            #print("S:",s.shape)
            #print("V:",vh.shape)
            
            #vh=vh.T
            
            delta=s[0]
            D[j, :] = u[:,0]
            X[I, j] = delta*np.array(vh[0,:])
            
            #calculo de eror
            Y_prima_k= D.T.dot(X.T)
            #print(Y_prima_k.shape)
            #Y_prima2= D.T@X.T
            #print(Y_prima2.shape)
            #if(Y_prima==Y_prima2):
                #print("Si es igual la operacion")
            Error_k= np.sum((Y.T-Y_prima_k)**2,axis=0)
            #print(Error_k.shape)
            Error_k=np.sum(Error_k)
            #print(Error_k.shape)
            #print(Error_k)
            Error3.append(Error_k)
            Error_n_K.append(Error_k)
            
            #if(j==0):
             #   plt.figure()
              #  print(r.shape[0])
               # for i in range(r.shape[0]):
                #    plt.plot(r[i,:])
                #plt.show()
        #plt.figure()
        #plt.xlabel("K") 
        #plt.ylabel("$\sum{E^k}$") 
        #plt.plot(Error3)
        #plt.show()
        return D, X

    def _initialize(self, Y):
        if min(Y.shape) < self.n_components:
            D = np.random.randn(self.n_components, Y.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(Y, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, Y):
        gram = D.dot(D.T)
        Xy = D.dot(Y.T)
        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * Y.shape[1])
        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T
    

    def fit(self, Y):
        """
        Parameters
        ----------
        Y: shape = [n_samples, n_features]
        """
        D = self._initialize(Y)
        for i in range(self.max_iter):
            T=i
            X = self._transform(D, Y)
            e = np.linalg.norm(Y - X.dot(D))
            if e < self.tol:
                break
            D, X = self._update_dict(Y, D, X)
            
            print(D.T.shape)
            print(X.shape)
            out= np.sum(D.T*X[0,:],axis=1)
            print(out.shape)
            print(np.sum((out-Y[0,:])**2))
            Error.append(np.sum((out-Y[0,:])**2))
            #plt.figure()
            #plt.plot((out-Y[0,:])**2)
            #plt.show()
            Y_prima= D.T.dot(X.T)
            print(Y_prima.shape)
            #Y_prima2= D.T@X.T
            #print(Y_prima2.shape)
            #if(Y_prima==Y_prima2):
                #print("Si es igual la operacion")
            Error_N= np.sum((Y.T-Y_prima)**2,axis=0)
            print(Error_N.shape)
            Error_N=np.sum(Error_N)
            print(Error_N.shape)
            print(Error_N)
            Error2.append(Error_N)
                
            
        fig, axes = plt.subplots(2,5)
        fig.set_size_inches(16,12)
        for i, j in itertools.product(range(0,2), range(5)):
            axes[i][j].plot(Error_n_K[(((4*i)+j)*640):(((4*i)+j+1)*640)])   
        #plt.figure()
        #plt.xlabel("K") 
        #plt.ylabel("$\sum{e_k}$") 
        #plt.plot(Error)
        #plt.show()
        
        #plt.figure()
        #plt.xlabel("K") 
        #plt.ylabel("$\sum{E_R^k}$") 
        #plt.plot(Error2)
        #plt.show()
           
        self.components_ = D
        return self

    def transform(self, Y):
        return self._transform(self.components_, Y)
A=np.random.rand(640*10)
A.shape
fig, axes = plt.subplots(2,5)
fig.set_size_inches(16,12)
for i, j in itertools.product(range(0,2), range(5)):
    #print(i)
    #print(j)
    axes[i-1][j].plot(Error_n_K[(((4*i)+j)*640):(((4*i)+j+1)*640)]) 
    
fig, axes = plt.subplots(2,5)
fig.set_size_inches(16,12)
for i, j in itertools.product(range(0,2), range(5)):
    #print(i)
    #print(j)
    axes[i][j].plot(Error_n_K[(((4*i)+j)*640):(((4*i)+j+1)*640)]) 

# Y ~ X.dot(D)
aksvd = KSVD(n_components=640,transform_n_nonzero_coefs=10)
D = aksvd.fit(training_data).components_
print(D.shape)

import itertools
# Ejemplos de diccionario
fig, axes = plt.subplots(6,6)
fig.set_size_inches(16,12)
for i, j in itertools.product(range(1,7), range(6)):
    axes[i-1][j].plot(D[6*(i-1)+j])

# Datos a representar
test_data = data[fs*60:fs*67].reshape(700, 160)

# Reconstruccion con 10 atomos

from sklearn.decomposition import SparseCoder

coder = SparseCoder(dictionary = D, transform_n_nonzero_coefs=20, transform_alpha=None, transform_algorithm="omp")


result = coder.transform(test_data)

plt.plot(result[436])
orig = data[fs*60:fs*67]
out = np.zeros(orig.shape)

for n in range(result.shape[0]):
    out[n*160:(n+1)*160] = np.sum(D.T*result[n],axis=1)
plt.plot(out, 'g')
Audio(data=out,rate=fs)
fig, axes = plt.subplots(3)
fig.set_size_inches(10,8)

axes[0].plot(orig)
axes[1].plot(out, 'g')
axes[2].plot((out-orig)**2, 'r')
frame=0
idx_r, = result[frame].nonzero()
plt.xlim(0, 600)
plt.title("Primer frame de sonido")
plt.stem(idx_r, result[frame][idx_r], use_line_collection=True)
plt.suptitle('Reconstrucción de señal con OMP',
             fontsize=16)
plt.show()
Residuals=[]
for i in range(100):
  test_data = data[fs*60:fs*60+160].reshape(1, 160)
  result = np.ndarray((test_data.shape[0],512))
  from sklearn.decomposition import SparseCoder
  coder = SparseCoder(dictionary = D, transform_n_nonzero_coefs=i+1, transform_alpha=None, transform_algorithm="omp")
  result = coder.transform(test_data)
  orig = data[fs*60:fs*60+160]
  out = np.zeros(orig.shape)
  for n in range(result.shape[0]):
      out[n*160:(n+1)*160] = np.sum(D.T*result[n],axis=1)
  Residuals.append(np.sum((test_data-out)**2)**(1/2))
print(Residuals)
plt.plot(np.arange(160),out,np.arange(160),test_data.reshape(160))
plt.plot(np.arange(100),Residuals)