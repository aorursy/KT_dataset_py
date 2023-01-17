!pip install shap #安装shap库
import shap
import pandas as pd
from sklearn import metrics
from sklearn.metrics import  roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier  #导入MLP
train = pd.read_csv('../input/data-7-222/train.csv')
test = pd.read_csv('../input/data-7-222/test.csv')
###选取前50条

train = train.iloc[:50]
# #删除group=6的行
# df_train = train[~(train['group']==6)]
# df_test = test[~(test['group']==6)]
# #label编码
# df['label'] = df['group'].map({2:0,3:0,1:1,4:1,5:1})
#初始化jupyter shap
shap.initjs() 
# 选取特征
col = [i for i in train.columns if i not in ['sample','Class']]
nn = MLPClassifier(alpha=1e-1, hidden_layer_sizes=(12, 2, 2), random_state=0) #构造MLP分类器 ，隐藏层三层分别，12,2,2 
nn.fit(train[col], train['Class']) #拟合数据
pre = nn.predict(test[col])

##50条数据量很小，预测的效果很不好，全部数据预测大概auc0.7

#绘图ROC曲线
fpr,tpr,threshold = metrics.roc_curve(test['Class'],pre)
print('AUC:',roc_auc_score(test['Class'],pre))
plt.plot(fpr,tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
# 解释数据集中的所有预测
explainer = shap.KernelExplainer(nn.predict_proba, train[col])
shap_values = explainer.shap_values(train[col])
#打印具体的每个特征shap值
def f1():
    a = {}
    for i,c in enumerate(train[col].columns):
        a.update({f'{c}':shap_values[0][:,i].sum()})
    return a
f1()
shap.summary_plot(shap_values[1], train[col])  #前20特征可视化
shap.summary_plot(shap_values[0], train[col], plot_type="bar") #前20特征条形图可视化



