### HW0002: Random Number

import numpy as np



# Make the output reproducible

np.random.seed(42)
### HW0002-1: 1. 產生五個亂數，並將其輸出。



#output 5 float numbers between 0~1

rand_float = np.random.rand(5)

print('rand_float: {}'.format(rand_float))

#[0.37454012 0.95071431 0.73199394 0.59865848 0.15601864]



#output 5 integers between -n ~ +n

def rand_integer(n):

    return np.random.randint(-n, n, 5)



print('rand_integer: {}'.format(rand_integer(10)))

#[ 8  0  0 -7 -3]

### HW0002-2: 產生N個介於-1與1之間的亂數，計算其平均值與標準差並輸出，

### 每個亂數的值則不用輸出。N=10**1, 10**2, 10**3, 10**4, 10**5。

### HW0002-3(進階): 一併輸出產生每N個亂數前後的系統時間，並計算所需的時間。

import time



def rand_N_numbers(N):

    np.random.seed(42)

    # -1 + 2 * [0, 1) -> [-1, 1)

    return -1 + 2*np.random.rand(N)



for i in range(5):

    start = time.time()

    N = 10**(i+1)

    end = time.time()

    period = end - start

    

    mean = rand_N_numbers(N).mean()

    std = rand_N_numbers(N).std()

    

    print('{}. Number Dimension: {}'.format(i+1,rand_N_numbers(N).shape))

    print('    run time:{} s'.format(round(period,8)))

    print('    mean:{}'.format(round(mean, 5)))

    print('     std: {}\n'.format(round(std, 5)))

### HW0002-4(進階)自己寫一個亂數產生器。(without using of the random module)

class rand_num_generator(object):

    pass