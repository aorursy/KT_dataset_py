import numpy as np

from numpy import linalg

from numpy import random

#from numpy.linalg import cholesky

import matplotlib.pyplot as plt

#from sklearn import datasets 

#generate the data set

import matplotlib

import math
#0.5||Ax-b||^2  范数球约束

#LASSO 问题
# 随机数据设置

d = 200

n = 500

t = 25

A = np.random.rand(d,n)    #200x500

# print(A.shape)

xstar = np.concatenate([np.ones((t,1)),-np.ones((t,1)),np.zeros((n-2*t,1))]) #数组拼接

print(xstar.shape)

np.random.shuffle(xstar)  #数组打乱

#print(xstar)

noise = 0.1 * np.random.randn(d,1)

b = np.dot(A,xstar) + noise

print(b.shape)

one_norm_of_xstar = linalg.norm(xstar,ord=1) #xstar的一范数

r = 20 #一范数球半径 radius

Aall = r*np.concatenate([A,-A]); #2n个atoms
#定义一些函数

def objective_fun(x,b):

    obj = sum((x-b)*(x-b))/2

    return obj 

    #obj = .5 * norm(x - b, 2)  #gives a different result



def grad_fun(x,b):

    grad = x-b

    return grad



#LMO returns the index of, and the atom of max inner product with the negative gradient

def LMO(A,grad):

    ips = np.dot(grad.transpose(),A)

    id = np.argmin(ips)

    id = id % 500

    #print (id)

    s = A[:,id:id+1]

#     s = s.reshape(200,1)

    return id,s



# returns the id of the active atom with the worst value w.r.t. the

# gradient

def away_step(grad, A, I_active):

#     A_I = (A[:,I_active]).reshape(200,2)

    

    I_active = np.array(I_active)

    print(I_active,'fffffffff')

    lis = []

    for i in I_active:

        lis.append(A[:,int(i)])

#     print(lis)

    A_I = np.array(lis)

#     print(A_I)

#     A_I = A[:,I_active[0]:I_active[0]+1]

    ips = np.dot(grad.transpose(), A_I.transpose())

    id = np.argmax(ips)

    id = id % 500

    #id = I_active(id[1]) 

    v = A[:,id:id+1]

    v = v.reshape(200,1)

    return id,v 





# lis = []

# for i in I_active:

#     lis.append(A[:,int(i)])

# print(lis)

# A_I = np.array(lis)

# print(C)
# opts.Tmax  = 1000 #迭代步数

# opts.TOL   = 1e-8 #精度

# opts.verbose = true

# opts.pureFW = 1 # 直接用FW算法

Tmax  = 1000 #迭代步数

TOL   = 1e-8#精度

eps = 1e-8

verbose = True

pureFW = 1 # 直接用FW算法
#def FW(A, b, opts):

    # [x_t,f_t,res] = FW(A, b, opts)

    # res: objective tracking

    # solves min ||x-b|| with x constrained to a finite atomic norm ball (here for the lasso, A is the set of scaled lasso columns and their negatives)

    # 默认运行AFW

    # Set opts.pureFW to 1 to run standard FW. 运行FW



    # [d,n] = A.shape() # n is the number of atoms  (twice the number of lasso data columns)

    # 初始化

    # alpha_t 是权重向量  x_t = S_t * alpha_t

alpha_t = np.zeros(n)

alpha_t[0] = 1

x_t = A[:,0:1]   # this is tracking A\alpha in the lasso case

# x_t = x_t.reshape(200,1)

# print(x_t.shape)   #200x1 dim

# each column of S_t = A(:,I_active) is a potential vertex

# I_active -> contains index of active vertices (same as alpha_t > 0)

# alpha_t(i) == 0 表示vertex is not active anymore

# alpha_t will be a weight vector so that x_t = A * alpha_t

I_active = np.where(alpha_t > 0) 

I_active = np.array(I_active)

#print(type(I_active),type(alpha_t)) #分别返回tuple和array



print(I_active,'~0~')  #标示



# tracking results:



fvalues = [] 

gap_values = []   #list

number_away = 0  #away步

number_drop = 0  #drop步 (max stepsize for away step)



#pureFW = isfield(opts, 'pureFW') and opts.pureFW  判断是否在域中

# pureFW = 'pureFW' in opts and opts.pureFW 

pureFW = True  #run classical FW, or False run AFW

if pureFW:

    #print('running plain FW, for at most #d iterations\n', opts.Tmax) 

    print('running plain FW, for at most #d iterations\n', Tmax)

else : # use away step

    #print('running FW with away steps, for at most #d iterations\n', opts.Tmax) 

    print('running FW with away steps, for at most #d iterations\n', Tmax)



# optimization: 



it = 1   #迭代数

while it <= Tmax :

#while it <= opts.Tmax :

    it = it + 1  



    # cost function:

    f_t = objective_fun(x_t,b) 

    # print(x_t.shape) #200x1 dim

    # gradient = x-b:

    grad = grad_fun(x_t,b)

    # print(A.shape)  #200x500

    # print(x_t.shape)

    # print(b.shape)

    # print(grad.shape) # 200x200

    # print(grad)

    # towards direction search:

    #[id_FW, s_FW]   = LMO(A,grad) # the linear minimization oracle, returning an atom

    id_FW,s_FW = LMO(A,grad)  #某一列值

    # print(s_FW.shape)  #***************为什么是一个矩阵

    d_FW = s_FW - x_t 

    # print(d_FW.shape)

    # duality gap:

    gap = - d_FW.transpose()@ grad 

    #print(gap)

#         fvalues[it-1] = f_t 

#         gap_values[it-1] = gap 

    fvalues.append(f_t)

    gap_values.append(gap)



    if gap < TOL:   #达到精度要求

        print('end of FW: reach small duality gap (gap=#g)\n', gap) 



    # away direction search:

    

    id_A,v_A = away_step(grad, A, I_active)

    d_A = x_t - v_A 

    alpha_max = alpha_t[id_A]



    # construct direction (between towards and away):



    if pureFW or - gap <= d_A.transpose() * grad:

        is_aw = False 

        d = d_FW  

        max_step = 1 

    else :# away step

        is_aw = True 

        number_away = number_away+1 

        d = d_A 

        max_step = alpha_max / (1 - alpha_max) 





    # line search:

    #step = -  (grad.T * d) / ( d.T * A * d )  # was this for the video QP

    # print(grad.shape)

    # print(d.shape)

    # print(A.shape)



    step = - (grad.transpose() @ d) / ( d.transpose() @ d ) 

#     print(step.shape)

    # simpler predefined stepsize

    #stepSize = 2 / (t+2) 

    step = max((0, min(step, max_step )))



    #if opts.verbose :

    if verbose :

        print('it = %d -  f = %g - gap=%g - stepsize=%g\n', it, f_t, gap, step) 



    if step < -eps :

      # not a descent direction???

        print('ERROR -- not descent direction???')

        keyboard   #终止



    # doing steps and updating active set:



    if is_aw :

        #print(' AWAY step from index #d\n',id_A) 

        # away step:

        alpha_t = (1+step)*alpha_t  # note that inactive should stay at 0 

        if abs(step - max_step) < 10*eps :

            # drop step:

            number_drop = number_drop+1 

            alpha_t[id_A] = 0 

            I_active = np.delete(I_active,id_A) #[I_active == id_A] = []   remove from active set

            print(I_active.shape,'~11~')

#             I_active[I_active == id_A] = None

        else:

            alpha_t[id_A] = alpha_t[id_A] - step 

    else:

      # FW step:

        alpha_t = (1-step)*alpha_t 



        # alpha_t[id_FW] = alpha_t[id_FW] + step 

        alpha_t[id_FW] = alpha_t[id_FW] + step



        I_active = np.where(alpha_t > 0)  #TODO: could speed up by checking if id_FW is new

        

      # exceptional case: stepsize of 1, this collapses the active set!

        if step > 1-eps :

            I_active = [id_FW] 



    x_t = x_t + step * d  



# res.primal = fvalues 

# res.gap = gap_values 

# res.number_away = number_away 

# res.number_drop = number_drop 

# res.S_t = A[:,I_active]

# res.alpha_t = alpha_t 

# res.x_t = x_t 

for i in I_active:

    print(int(i))

B = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

lis = []

for i in I_active:

    lis.append(B[:,int(i)])

print(lis)

C = np.array(lis)

print(C)
x=np.linspace(1,1000,1000)

plt.plot(x,gap_values,ls="-",lw=2,label="AFW")

#ls:line style ,lw:line width,label:line name

#plt.yscale('log')

plt.xlabel('iter.no.')

plt.ylabel('gap')

plt.title('AFW')

plt.xlim((0,1000))

plt.ylim((1e-8,1))

plt.grid()

plt.show()