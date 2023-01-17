import pandas as pd #数据分析库
import hypertools as hyp #高维可视化库
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))
#加载muschrooms数据
data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
# 单独的降维数据(针对无标签的数据)
hyp.plot(data,"o")
# 根据类别给降维的数据着色(适合有标签的数据)
class_labels = data["class"]
hyp.plot(data,'o',group=class_labels,legend=class_labels.unique().tolist())
#经过kemeans聚类后，然后通过降维.观察数据的形状
hyp.plot(data,"o",n_clusters=23)
#上述的数据等价于如下的操作
cluster_labels = hyp.tools.cluster(data, n_clusters=23)
hyp.plot(data, "o",group=cluster_labels)
#默认情况下，hypertools采用pca降维方法
#自定义方法,对数据进行降维
from sklearn.manifold import TSNE #流形方法--TSNE算法
tsne_model = TSNE(n_components=3) #降为3维
reduced_data_tsne = tsne_model.fit_transform(hyp.tools.df2mat(data))
hyp.plot(reduced_data_tsne,"o",group=class_labels,legend=class_labels.unique().tolist())