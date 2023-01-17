import multiprocessing 

import os 

  

def worker1(): 

    # printing process id 

    print("ID of process running worker1: {}".format(os.getpid())) 

  

def worker2(): 

    # printing process id 

    print("ID of process running worker2: {}".format(os.getpid())) 

  

if __name__ == "__main__": 

    # printing main program process id 

    print("ID of main process: {}".format(os.getpid())) 

  

    # creating processes 

    p1 = multiprocessing.Process(target=worker1) 

    p2 = multiprocessing.Process(target=worker2) 

  

    # starting processes 

    p1.start() 

    p2.start() 



    # process IDs 

    print("ID of process p1: {}".format(p1.pid)) 

    print("ID of process p2: {}".format(p2.pid)) 

  

    # wait until processes are finished 

    p1.join() 

    p2.join() 



    # both processes finished 

    print("Both processes finished execution!") 

  

    # check if processes are alive 

    print("Process p1 is alive: {}".format(p1.is_alive())) 

    print("Process p2 is alive: {}".format(p2.is_alive()))

    

# for file_chunk in file_chunks:

#     p = Process(target=my_func, args=(file_chunk, my_other_arg))

#     p.start()

#     p.join()
# from multiprocessing import Pool



# pool = Pool(ncores)



# for file_chunk in file_chunks:

#     pool.apply_async(my_func, args=(file_chunk, arg1, arg2)) 
# # Pool example skeleton code:

# def eval_formula...





# p=multiprocessing.Pool(multiprocessing.cpu_count)

# result=p.map(eval_formula, expression_list)

# p.close()

# p.join()
# # Process example skeleton code:



# def eval_formula...



# for i in range (len(expression_list)):

#     p=Process(target=proces_eval,args=(expression_list[i],))

#     p.start()

#     p.join()



# from multiprocessing import Pool

# from PIL import Image



# SIZE = (75,75)

# SAVE_DIRECTORY = 'thumbs'



# def get_image_paths(folder):

#   return (os.path.join(folder, f)

#       for f in os.listdir(folder)

#       if 'jpeg' in f)



# def create_thumbnail(filename):

#   im = Image.open(filename)

#   im.thumbnail(SIZE, Image.ANTIALIAS)

#   base, fname = os.path.split(filename)

#   save_path = os.path.join(base, SAVE_DIRECTORY, fname)

#   im.save(save_path)



# if __name__ == '__main__':

#   folder = os.path.abspath(

#     '11_18_2013_R000_IQM_Big_Sur_Mon__e10d1958e7b766c3e840')

#   os.mkdir(os.path.join(folder, SAVE_DIRECTORY))



#   images = get_image_paths(folder)



#   pool = Pool()

#     pool.map(create_thumbnail, images)

#     pool.close()

#     pool.join()

# # only thing that we had to replace is 

# for image in images:

#     create_thumbnail(image)

# #and instead we used



#  pool = Pool()

#     pool.map(create_thumbnail, images)

#     pool.close()

#     pool.join()
# df.shape

# # (100, 100)

# dfs = [df.iloc[i*25:i*25+25, 0] for i in range(4)]

# with Pool(4) as p:

#     res = p.map(np.exp, dfs)

# for i in range(4): df.iloc[i*25:i*25+25, 0] = res[i]

# from numba import njit, jit

# @njit      # or @jit(nopython=True)

# def function(a, b):

#     # your loop or numerically intensive computations

#     return result
# @vectorize(target="parallel")

# def func(a, b):

#     # Some operation on scalars

#     return result
# @vectorize(target="cuda")

# def func(a, b):

#     # Some operation on scalars

#     return result
%time

from math import sqrt

from joblib import Parallel, delayed

Parallel(n_jobs=2)(delayed(sqrt)(i**2) for i in range(100000))

%time

from math import sqrt

from joblib import Parallel, delayed

Parallel(n_jobs=2,backend='threading')(delayed(sqrt)(i**2) for i in range(100000))

for i in range(100000):

    print(sqrt(i**2))
[sqrt(x**2) for x in range(1000000)]
from math import modf

from joblib import Parallel, delayed

r = Parallel(n_jobs=1,verbose=10)(delayed(modf)(i/2.) for i in range(100000))

res, i = zip(*r)
