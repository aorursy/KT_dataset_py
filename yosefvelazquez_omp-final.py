import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
from scipy.io import wavfile

#Se extrae los datos y la frecuenci de muestreo 
fs, data = wavfile.read("../input/voz-yosef/voz yosef.wav")
data = data/2.0**15
#fs es igual a 100 frames de 160 muestreos cada uno.
data=data[0:fs*67]
# Data de entrenamiento (1 min =60 Frame Samples(fs))
training_data = data[0:fs*60]
training_data = training_data.reshape(6000,160)
%matplotlib inline

plt.plot(data[fs*0:fs*67])

from IPython.display import Audio
Audio(rate=fs, data=data[fs*0:fs*67])
# Ejemplo de frame
plt.plot(training_data[10])
data2=training_data[10].reshape(-1,1)
Audio(data=training_data[10],rate=fs)
# Construccion del diccionario
from sklearn.decomposition import MiniBatchDictionaryLearning
from time import time
t0 = time()
dico = MiniBatchDictionaryLearning(n_components=160*4, alpha=1, n_iter=200)
D = dico.fit(training_data).components_
t1 = time() - t0
print('Tiempo de entrenamiento en %d segundos' % t1)
import itertools
# Ejemplos de diccionario
fig, axes = plt.subplots(6,6)
fig.set_size_inches(16,12)
for i, j in itertools.product(range(1,7), range(6)):
    axes[i-1][j].plot(D[6*(i-1)+j])

# Datos a representar
test_data = data[fs*60:fs*67].reshape(700, 160)

# Reconstruccion con 100 atomos
result = np.ndarray((test_data.shape[0],512))

from sklearn.decomposition import SparseCoder

coder = SparseCoder(dictionary = D, transform_n_nonzero_coefs=100, transform_alpha=None, transform_algorithm="omp")

t0 = time()
result = coder.transform(test_data)
t1 = time() - t0
print('Compresión terminada en %d segundos.' % t1)
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