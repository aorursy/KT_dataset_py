import numpy as np

from scipy.io import loadmat

from scipy.cluster import vq

from scipy.spatial.distance import euclidean as d_eucli



def reconstruction_error(orig, reconstructed):

    return d_eucli(orig, reconstructed)



def list_reconstruction_error(list_orig, list_reconstruct):

    errors = []

    for i in range(len(list_orig)):

        errors.append(reconstruction_error(list_orig[i], list_reconstruct[i]))

    return errors

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import csv



# Input data files are available in the "../input/" directory.



## initializing paths

inputFolderPath = '/kaggle/input/data-series-summarization-project-v3/'

outputFolderPath = '/kaggle/working/'



filename = 'synthetic_size50k_len256_znorm.bin'



inputFilePath = inputFolderPath + filename
class PQEncoder():

    """

    Product Quantization Encoder

    """

    def __init__(self):

        super(PQEncoder, self).__init__()

    

        

    



    def __del__(self):

        pass



    def build(self, pardic=None):

        # training data

        vals = pardic['vals']

        # the number of subquantizers

        nsubq = pardic['nsubq']

        # the number bits of each subquantizer

        nsubqbits = pardic.get('nsubqbits', 8)



        # vector dimension

        self.dim = vals.shape[1]

        dim = vals.shape[1]



        # dimension of the subvectors to quantize

        dsub = dim / nsubq

        # number of centroids per subquantizer

        ksub = 2 ** nsubqbits



        """

        Initializing indexer data

        """

        ecdat = dict()

        ecdat['nsubq'] = nsubq

        ecdat['ksub'] = ksub

        ecdat['dsub'] = dsub

        ecdat['centroids'] = [None for q in range(nsubq)]



        for q in range(nsubq):

            vs = np.require(vals[:, int(q*dsub):int((q+1)*dsub)],

                            requirements='C', dtype=np.float32)

            ecdat['centroids'][q] = kmeans(vs, ksub, niter=100)



        self.ecdat = ecdat



    def encode(self, vals):

        dsub = self.ecdat['dsub']

        nsubq = self.ecdat['nsubq']

        centroids = self.ecdat['centroids']



        num_vals = vals.shape[0]

        codes = np.zeros((num_vals, nsubq), np.uint8)

        for q in range(nsubq):

            vsub = vals[:, int(q*dsub):int((q+1)*dsub)]

            codes[:, q] = pq_kmeans_assign(centroids[q], vsub)

        return codes

    

    def decode(self, codes):

        centroids = self.ecdat['centroids']

        dsub = self.ecdat['dsub']

        decodes = np.zeros((codes.shape[0],self.dim))

        i = 0

        for centroid in centroids:

            j=0

            for  decode in decodes:



                decode[int(i*dsub):int((i + 1)*dsub)] = centroid[codes[j][i]]

                j+=1

                

            i+=1

        

        return decodes



def kmeans(vs, ks, niter):

    centers, labels = vq.kmeans2(vs, ks, niter)

    return centers





def pq_kmeans_assign(centroids, query):

    

    dist = euclidean(centroids, query)

    return dist.argmin(1)



def euclidean(feat, query, featl2norm=None, qryl2norm=None):



    dotprod = query.dot(feat.T)



    if qryl2norm is None:

        qryl2norm = (query ** 2).sum(1).reshape(-1, 1)

    if featl2norm is None:

        featl2norm = (feat ** 2).sum(1).reshape(1, -1)



    return - 2 * dotprod + qryl2norm + featl2norm

np.random.seed(79)



time_series50k = np.fromfile(inputFilePath, dtype=np.float32).reshape(-1, 256)



encoder32b = PQEncoder()



encoder64b = PQEncoder()



encoder128b = PQEncoder()



encoder32b.build({'vals': time_series50k, 'nsubq': 32,'nsubqbits': 7})

encoder64b.build({'vals': time_series50k, 'nsubq': 64, 'nsubqbits': 6})

encoder128b.build({'vals': time_series50k, 'nsubq': 128, 'nsubqbits': 5})

def sum32(inputFilePath):

    """ Summurizes 50k 256*float32 time series read from a binary file to 50k 32-bytes summaries 

    

    Parameters: 

    inputFilePath (string): the path of the binary file containing the 50k time series to summarize

  

    Returns: 

    string: the path of the binary file of the 50k summaries



    """



    summary_filepath =  outputFolderPath + filename + '_sum32'    

    ##############################################################################

    ##change the following code for 32-bytes summarization 

    

    #read binary file

    time_series50k = np.fromfile(inputFilePath, dtype=np.float32).reshape(-1, 256)

        

    summary50k = encoder32b.encode(time_series50k)



    

    #write the result in a binary file, then return the output file path

    summary50knp = np.array(summary50k,dtype=np.int8)

    summary50knp.tofile(summary_filepath)    



    ##############################################################################

    return summary_filepath



def rec32(summary_filepath):

    """ reconstructs 50k 256*float32 time series based on 50k 32-bytes summaries read from binary file

    

    Parameters: 

    summary_filepath (string): the path of the binary file containing the 50k 32-bytes summaries

  

    Returns: 

    string: the path of the binary file of the 50k reconstructed time series

    """

    reconstructed_filepath = summary_filepath + '_rec32'



    ##############################################################################

    ##change the following code for reconstruction from 32-bytes summaries 

    

    #read binary file

    summary50k = np.fromfile(summary_filepath, dtype=np.int8).reshape(-1, 32)



    

    reconstructed50k = encoder32b.decode(summary50k)

    

    #write the result in a binary file, then return the output file path

    reconstructed50knp = np.array(reconstructed50k,dtype=np.float32)

    reconstructed50knp.tofile(reconstructed_filepath) 



    ##############################################################################

    return reconstructed_filepath

    



def sum64(inputFilePath):

    """ Summurizes 50k 256*float32 time series read from a binary file to 50k 64-bytes summaries 

    

    Parameters: 

    inputFilePath (string): the path of the binary file containing the 50k time series to summarize

  

    Returns: 

    string: the path of the binary file of the 50k summaries



    """



    summary_filepath =  outputFolderPath + filename + '_sum64'    

    ##############################################################################

    ##change the following code for 64-bytes summarization 

    

    #read binary file

    time_series50k = np.fromfile(inputFilePath, dtype=np.float32).reshape(-1, 256)

        

    summary50k = encoder64b.encode(time_series50k)



    

    #write the result in a binary file, then return the output file path

    summary50knp = np.array(summary50k,dtype=np.int8)

    summary50knp.tofile(summary_filepath)    



    ##############################################################################

    return summary_filepath



def rec64(summary_filepath):

    """ reconstructs 50k 256*float32 time series based on 50k 64-bytes summaries read from binary file

    

    Parameters: 

    summary_filepath (string): the path of the binary file containing the 50k 64-bytes summaries

  

    Returns: 

    string: the path of the binary file of the 50k reconstructed time series

    """

    reconstructed_filepath = summary_filepath + '_rec64'



    ##############################################################################

    ##change the following code for reconstruction from 64-bytes summaries 

    

    #read binary file

    summary50k = np.fromfile(summary_filepath, dtype=np.int8).reshape(-1, 64)



    reconstructed50k = encoder64b.decode(summary50k)



    

    #write the result in a binary file, then return the output file path

    reconstructed50knp = np.array(reconstructed50k,dtype=np.float32)

    reconstructed50knp.tofile(reconstructed_filepath) 



    ##############################################################################

    return reconstructed_filepath

    



def sum128(inputFilePath):

    """ Summurizes 50k 256*float32 time series read from a binary file to 50k 128-bytes summaries 

    

    Parameters: 

    inputFilePath (string): the path of the binary file containing the 50k time series to summarize

  

    Returns: 

    string: the path of the binary file of the 50k summaries



    """



    summary_filepath =  outputFolderPath + filename + '_sum128'    

    ##############################################################################

    ##change the following code for 128-bytes summarization 

    

    #read binary file

    time_series50k = np.fromfile(inputFilePath, dtype=np.float32).reshape(-1, 256)

        

    summary50k = encoder128b.encode(time_series50k)

    

    #write the result in a binary file, then return the output file path

    summary50knp = np.array(summary50k,dtype=np.int8)

    summary50knp.tofile(summary_filepath)    



    ##############################################################################

    return summary_filepath



def rec128(summary_filepath):

    """ reconstructs 50k 256*float32 time series based on 50k 128-bytes summaries read from binary file

    

    Parameters: 

    summary_filepath (string): the path of the binary file containing the 50k 128-bytes summaries

  

    Returns: 

    string: the path of the binary file of the 50k reconstructed time series

    """

    reconstructed_filepath = summary_filepath + '_rec128'



    ##############################################################################

    ##change the following code for reconstruction from 128-bytes summaries 

    

    #read binary file

    summary50k = np.fromfile(summary_filepath, dtype=np.int8).reshape(-1, 128)



    reconstructed50k = encoder128b.decode(summary50k)

    

    #write the result in a binary file, then return the output file path

    reconstructed50knp = np.array(reconstructed50k,dtype=np.float32)

    reconstructed50knp.tofile(reconstructed_filepath) 



    ##############################################################################

    return reconstructed_filepath

    

########################## Submission ################################################

############ follow this templateand do not modify this cell code ####################

s32= sum32(inputFilePath)

r32 = rec32(s32)

pred32=np.fromfile(r32, dtype=np.float32)



s64= sum64(inputFilePath)

r64 = rec64(s64)

pred64=np.fromfile(r64, dtype=np.float32)



s128= sum128(inputFilePath)

r128 = rec128(s128)

pred128=np.fromfile(r128, dtype=np.float32)



#preparing submission 

output = []

globalCsvIndex = 0 



#append reconstruction from 32-bytes summaries (pred32) to csv

for i in range (len(pred32)) :

    output.append([globalCsvIndex,pred32[i]])

    globalCsvIndex = globalCsvIndex+1



#append reconstruction from 64-bytes summaries (pred64) to csv

for i in range (len(pred64)) :

    output.append([globalCsvIndex,pred64[i]])

    globalCsvIndex = globalCsvIndex+1



#append reconstruction from 128-bytes summaries (pred128) to csv

for i in range (len(pred128)) :

    output.append([globalCsvIndex,pred128[i]])

    globalCsvIndex = globalCsvIndex+1



with open('submission.csv', 'w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow(['id','expected'])

    writer.writerows(output)