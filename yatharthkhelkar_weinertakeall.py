import numpy as np
xi=[np.array([0.8,0.6]).transpose(),np.array([0.17,-0.98]).transpose(),np.array([0.707,0.707]).transpose(),np.array([0.34,-0.93]),
    np.array([0.6,0.8]).transpose()]

w1=np.array([1,0]).transpose()
w2=np.array([0,-1]).transpose()

a = 0.1
Empty = np.array([])
W = np.array([0,0])
k = np.array([1,1,1,1,1])
iteration = 1
cluster = 1
j = 0
i = 0
while(j < 5):
    net1=sum(w1.transpose()*xi[i])
    net2=sum(w2.transpose()*xi[i])
    if net1>net2:
        dw=a*(xi[i]-w1)
        Empty = np.append(Empty,[1])
        W=W+dw
    if net1<net2:
        dw=a*(xi[i]-w2)
        Empty = np.append(Empty,[0])
        W=W+dw
    i += 1
    if i > 4:
        i = 0
        j += 1
        print("1 stands for net1 while 2 stands for net2 : {}".format(Empty))
        if Empty.all() == k.all():
            break
        k = Empty
        Empty = np.array([])
        cluster += 1
    iteration += 1
print("Final Weight matrix: {}".format(W))
print("Iterations: {}".format(iteration))
print("Clusters iterations: {}".format(cluster))
