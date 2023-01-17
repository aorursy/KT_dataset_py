japanese28oldMenHeight_mean = 172.14    #平均
japanese28oldMenHeight_std     = 5.50        #標準偏差
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(32)
histBins = 50
samples = np.random.normal(japanese28oldMenHeight_mean,
                                                   japanese28oldMenHeight_std,
                                                   size = 10000,
                                                  )
plt.clf()
plt.hist(samples, bins=histBins, density=True)
plt.xlabel("height")
plt.ylabel("probability")
plt.show()
samples_dist, samples_dist_bin_edges = np.histogram(
                                                                            samples, 
                                                                            bins=histBins,
                                                                            density=True
                                                                         )
plt.clf()
plt.bar(
        x=samples_dist_bin_edges[0:-1],
        width=samples_dist_bin_edges[1]-samples_dist_bin_edges[0],
        height=samples_dist,
        linewidth=1, edgecolor="black"
)
plt.xlabel("height")
plt.ylabel("probability")
plt.show()
import scipy.stats as ss
real_normal_dist = [ss.norm.pdf(x,
                                        loc=japanese28oldMenHeight_mean,
                                        scale=japanese28oldMenHeight_std
                                    )for x in samples_dist_bin_edges[0:-1]
                                  ]
plt.clf()
plt.plot(samples_dist_bin_edges[0:-1], real_normal_dist, '.-', color="r")
plt.xlabel("height")
plt.ylabel("probability")
plt.show()
import scipy.stats as ss
plt.clf()
plt.plot(samples_dist_bin_edges[0:-1], real_normal_dist, '-', color="gray")
plt.xlabel("height")
plt.ylabel("probability")

some_normal_meanPlus5_stdPlus0=[
    ss.norm.pdf(
        x,
        loc=japanese28oldMenHeight_mean+5.0,
        scale=japanese28oldMenHeight_std
    )for x in samples_dist_bin_edges[0:-1]
]
plt.plot(samples_dist_bin_edges[0:-1], some_normal_meanPlus5_stdPlus0, '.-', color="y")

some_normal_meanMinus3_stdPlus0=[
    ss.norm.pdf(
        x,
        loc=japanese28oldMenHeight_mean-3.0,
        scale=japanese28oldMenHeight_std
    )for x in samples_dist_bin_edges[0:-1]
]
plt.plot(samples_dist_bin_edges[0:-1], some_normal_meanMinus3_stdPlus0, '.-', color="g")

some_normal_meanPlus0_stdPlus3=[
    ss.norm.pdf(
        x,
        loc=japanese28oldMenHeight_mean,
        scale=japanese28oldMenHeight_std+3
    )for x in samples_dist_bin_edges[0:-1]
]
plt.plot(samples_dist_bin_edges[0:-1], some_normal_meanPlus0_stdPlus3, '.-', color="b")

some_normal_meanPlus0_stdMinus2=[
    ss.norm.pdf(
        x,
        loc=japanese28oldMenHeight_mean,
        scale=japanese28oldMenHeight_std-2
    )for x in samples_dist_bin_edges[0:-1]
]
plt.plot(samples_dist_bin_edges[0:-1], some_normal_meanPlus0_stdMinus2, '.-', color="c")


plt.show()
def calcKL(p, q):
    # assertEqual(len(p), len(q), msg="len(p) != len(q)")
    N = len(p)
    
    KLdiv = np.sum(-np.log(q) + np.log(p)) / N
    
    return KLdiv

# calcKL(samples_dist, real_normal_dist)
plt.clf()
plt.bar(
        x=samples_dist_bin_edges[0:-1],
        width=samples_dist_bin_edges[1]-samples_dist_bin_edges[0],
        height=samples_dist,
        linewidth=1, edgecolor="black"
)
plt.plot(samples_dist_bin_edges[0:-1], real_normal_dist, '.-', color="r")
plt.xlabel('$x$')
plt.ylabel('p')
plt.text(x=180,y=0.04,s='mean='+str(japanese28oldMenHeight_mean))
plt.text(x=180,y=0.035,s='std='+str(japanese28oldMenHeight_std))
plt.text(x=180,y=0.07,s='KLdiv='+str(calcKL(samples_dist, real_normal_dist)))
plt.title('True normal distribution')
plt.show()

plt.clf()
plt.bar(
        x=samples_dist_bin_edges[0:-1],
        width=samples_dist_bin_edges[1]-samples_dist_bin_edges[0],
        height=samples_dist,
        linewidth=1, edgecolor="black"
)
plt.plot(samples_dist_bin_edges[0:-1], some_normal_meanPlus5_stdPlus0, '.-', color="y")
plt.xlabel('$x$')
plt.ylabel('p')
plt.text(x=180,y=0.04,s='mean='+str(japanese28oldMenHeight_mean+5))
plt.text(x=180,y=0.035,s='std='+str(japanese28oldMenHeight_std))
plt.text(x=180,y=0.07,s='KLdiv='+str(calcKL(samples_dist, some_normal_meanPlus5_stdPlus0)))
plt.title('mean + 5    std+0')
plt.show()

plt.clf()
plt.bar(
        x=samples_dist_bin_edges[0:-1],
        width=samples_dist_bin_edges[1]-samples_dist_bin_edges[0],
        height=samples_dist,
        linewidth=1, edgecolor="black"
)
plt.plot(samples_dist_bin_edges[0:-1], some_normal_meanMinus3_stdPlus0, '.-', color="g")
plt.xlabel('$x$')
plt.ylabel('p')
plt.text(x=180,y=0.04,s='mean='+str(japanese28oldMenHeight_mean-3))
plt.text(x=180,y=0.035,s='std='+str(japanese28oldMenHeight_std))
plt.text(x=180,y=0.07,s='KLdiv='+str(calcKL(samples_dist, some_normal_meanMinus3_stdPlus0)))
plt.title('mean - 3    std+0')
plt.show()

plt.clf()
plt.bar(
        x=samples_dist_bin_edges[0:-1],
        width=samples_dist_bin_edges[1]-samples_dist_bin_edges[0],
        height=samples_dist,
        linewidth=1, edgecolor="black"
)
plt.plot(samples_dist_bin_edges[0:-1], some_normal_meanPlus0_stdPlus3, '.-', color="b")
plt.xlabel('$x$')
plt.ylabel('p')
plt.text(x=180,y=0.04,s='mean='+str(japanese28oldMenHeight_mean))
plt.text(x=180,y=0.035,s='std='+str(japanese28oldMenHeight_std+3))
plt.text(x=180,y=0.07,s='KLdiv='+str(calcKL(samples_dist, some_normal_meanPlus0_stdPlus3)))
plt.title('mean + 0    std+3')
plt.show()

plt.clf()
plt.bar(
        x=samples_dist_bin_edges[0:-1],
        width=samples_dist_bin_edges[1]-samples_dist_bin_edges[0],
        height=samples_dist,
        linewidth=1, edgecolor="black"
)
plt.plot(samples_dist_bin_edges[0:-1], some_normal_meanPlus0_stdMinus2, '.-', color="c")
plt.xlabel('$x$')
plt.ylabel('p')
plt.text(x=180,y=0.04,s='mean='+str(japanese28oldMenHeight_mean))
plt.text(x=180,y=0.035,s='std='+str(japanese28oldMenHeight_std-2))
plt.text(x=180,y=0.07,s='KLdiv='+str(calcKL(samples_dist, some_normal_meanPlus0_stdMinus2)))
plt.title('mean + 0    std-2')
plt.show()