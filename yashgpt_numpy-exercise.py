import numpy as np
np.__version__
a=np.array([1,2,3,4,5,6,7,8,9])
a
np.full((3,3),True,dtype=bool)
a[a%2==1]
a[a%2==1]=-1
a
b=np.arange(10)
b
out=np.where(b%2==1,-1,b)
out
print(b)

print(out)
c=np.arange(10).reshape(2,-1)
c
d=np.repeat(1,10).reshape(2,-1)
d
np.vstack([c,d])
np.hstack([c,d])
np.concatenate([c,d],axis=0)
np.concatenate([c,d],axis=1)
e=np.array([1,2,3])
np.r_[np.repeat(e,3),np.tile(e,3)]
a = np.array([1,2,3,2,3,4,3,4,5,6])

b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)
np.setdiff1d(a,b)
np.where(a==b)
a = np.array([2, 6, 1, 9, 10, 3, 27])
a[(a>=5)&(a<=10)]
def maxx(x, y):

    """Get the maximum of two items"""

    if x >= y:

        return x

    else:

        return y



pair_max = np.vectorize(maxx, otypes=[float])



a = np.array([5, 7, 9, 8, 6, 4, 5])

b = np.array([6, 3, 4, 8, 9, 7, 1])



pair_max(a, b)
arr = np.arange(9).reshape(3,3)
arr
arr[:,[1,0,2]]
arr[:,[0,2,1]]
arr[[1,2,0],:]
arr
arr[::-1]
arr[::1]
arr[:,::-1]
np.random.uniform(5,10,size=(5,3))
rand_arr = np.random.random([5,3])
np.set_printoptions(precision=3)
rand_arr[:4]
np.random.seed(100)
np.random.random([3,3])/1e3
np.set_printoptions(suppress=True,precision=6)
np.random.random([3,3])/1e3
a=np.arange(15)
np.set_printoptions(threshold=6)
a
np.set_printoptions(threshold=100)
a
iris = np.genfromtxt('../input/irisdata/iris.csv', delimiter=',', dtype='object')
iris
iris[:3]
iris.shape
species=np.array([row[4] for row in iris])
species[:5]
iris_2d=np.array([row.tolist()[:4] for row in iris])
iris_2d[:4]
sepallength=np.genfromtxt("../input/irisdata/iris.csv",delimiter=',',dtype='float',usecols=[0])
mean,median,std=np.mean(sepallength),np.median(sepallength),np.std(sepallength)
print(mean,median,std)
max,min=sepallength.max(),sepallength.min()
S=sepallength-min/max-min
S
def softmax(x):

    e_x=np.exp(x-x.max())

    return e_x/e_x.sum(axis=0)

    
softmax(sepallength)
np.percentile(sepallength,[5,95])
np.random.seed(100)
i,j=np.where(iris_2d)
iris_2d[np.random.choice((i),20),np.random.choice((j),20)]=np.nan
iris_2d
iris_2d = np.genfromtxt("../input/irisdata/iris.csv", delimiter=',', dtype='float', usecols=[0,1,2,3])

iris_2d[np.random.randint(150, size=20), np.random.randint(4, size=20)] = np.nan
print("Number=",np.isnan(iris_2d[:,0]).sum())

print("Where=",np.where(np.isnan(iris_2d[:,0])))
iris_2d = np.genfromtxt("../input/irisdata/iris.csv", delimiter=',', dtype='float', usecols=[0,1,2,3])
cond=(iris_2d[:,2]>1.5) & (iris_2d[:,0]<5.0)
iris_2d[cond]
condt=iris_2d[:,:]!=np.nan
nan_row=np.array([~np.any(np.isnan(row)) for row in iris_2d])
iris_2d[nan_row][:5]
iris = np.genfromtxt("../input/irisdata/iris.csv", delimiter=',', dtype='float', usecols=[0,1,2,3])

from scipy.stats.stats import pearsonr
corr,pvalue=pearsonr(iris[:,0],iris[:,2])
corr
pvalue
np.isnan(iris_2d).any()
nan_row=([np.any(np.isnan(iris_2d)) for row in iris_2d])
iris_2d[np.isnan(iris_2d)]=0
iris_2d[:10]
iris=np.genfromtxt("../input/irisdata/iris.csv", delimiter=',', dtype='object')
species=np.array([row.tolist()[4] for row in iris])
np.unique(species,return_counts=True)
petal_length_bin = np.digitize(iris[:, 2].astype('float'), [0, 3, 5, 10])

petal_length_bin
label_map={1:'Small',2:'Medium',3:'Large',4:np.nan}
petal_length_cat = [label_map[x] for x in petal_length_bin]

petal_length_cat