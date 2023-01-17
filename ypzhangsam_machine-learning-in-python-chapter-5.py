#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')
# 获取红酒数据集
df = pd.read_csv('../input/winequality/winequality-red.csv')
df.columns = ['非挥发性酸','挥发性酸','柠檬酸', '残留糖分', '氯化物', '游离二氧化硫','总二氧化硫', '密度', 
              'PH值', '亚硝酸盐', '酒精含量', '品质']
# 标准化 切分属性和标签
norm_df = (df - df.mean())/df.std()
xData = df.values[:,:-1]; yData = df.values[:,-1] 
xNormData = norm_df.values[:,:-1]; yNormData = norm_df.values[:,-1] 
m, n = xData.shape
#  标准化与否的选择
# xx = xData; 
xx = xNormData
#y = yData
y = yNormData

# 调用sklearn.linear_model中的LassoCV 
wineModel = LassoCV(cv=10).fit(xx, y)

# 显示结果
plt.figure()
plt.plot(wineModel.alphas_, wineModel.mse_path_, ':')
plt.plot(wineModel.alphas_, wineModel.mse_path_.mean(axis=-1),
         label='Average MSE Across Folds', linewidth=2)
plt.axvline(wineModel.alpha_, linestyle='--',
            label='CV Estimate of Best alpha')
plt.semilogx()
plt.legend()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Mean Square Error')
plt.axis('tight')
plt.show()

#print out the value of alpha that minimizes the Cv-error
print("最小均方误差对应的alpha值  ",wineModel.alpha_)
print("最小均方误差：", min(wineModel.mse_path_.mean(axis=-1)))
#  标准化与否的选择
# xx = xData; 
xx = xNormData
y = yData
# y = yNormData

# 调用sklearn.linear_model中的LassoCV 
wineModel = LassoCV(cv=10).fit(xx, y)

plt.figure()
plt.plot(wineModel.alphas_, wineModel.mse_path_, ':')
plt.plot(wineModel.alphas_, wineModel.mse_path_.mean(axis=-1),
         label='Average MSE Across Folds', linewidth=2)
plt.axvline(wineModel.alpha_, linestyle='--',
            label='CV Estimate of Best alpha')
plt.semilogx()
plt.legend()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Mean Square Error')
plt.axis('tight')
plt.show()

#print out the value of alpha that minimizes the Cv-error
print("最小均方误差对应的alpha值  ",wineModel.alpha_)
print("最小均方误差：", min(wineModel.mse_path_.mean(axis=-1)))
#  标准化与否的选择
xx = xData; 
# xx = xNormData
y = yData
# y = yNormData

# 调用sklearn.linear_model中的LassoCV 
wineModel = LassoCV(cv=10).fit(xx, y)

plt.figure()
plt.plot(wineModel.alphas_, wineModel.mse_path_, ':')
plt.plot(wineModel.alphas_, wineModel.mse_path_.mean(axis=-1),
         label='Average MSE Across Folds', linewidth=2)
plt.axvline(wineModel.alpha_, linestyle='--',
            label='CV Estimate of Best alpha')
plt.semilogx()
plt.legend()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Mean Square Error')
plt.axis('tight')
plt.show()

#print out the value of alpha that minimizes the Cv-error
print("最小均方误差对应的alpha值  ",wineModel.alpha_)
print("最小均方误差：", min(wineModel.mse_path_.mean(axis=-1)))
#  标准化与否的选择
# xx = xData; 
xx = xNormData
# y = yData
y = yNormData


alphas, coefs, _  = linear_model.lasso_path(xx, y,  return_models=False)

plt.plot(alphas,coefs.T)

plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.axis('tight')
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()
plt.show()

nattr, nalpha = coefs.shape

# 回归系数排序
nzList = []
for iAlpha in range(1,nalpha):
    coefList = coefs[: ,iAlpha]
    
    # 记录回归系数刚好变成非零的属性
    nzCoef = np.where(coefList!=0)[0]
    for q in nzCoef:
        if q not in nzList:
            nzList.append(q)

print("系数进入模型的次序所对应的属性排序 :",)
for idx in nzList:
    print(df.columns[idx],end=' ')
print("")

# 根据前面已获得的最佳\alpha，寻找此例中对应的索引
alphaStar = 0.013561387700964642
indexLTalphaStar = [index for index in range(100) if alphas[index] > alphaStar]
indexStar = max(indexLTalphaStar)

# 进而根据上面“对应的索引”，获得最佳回归系数值
coefStar = coefs[:,indexStar]
print("最佳回归系数：", coefStar)

# 回归系数给出了另外一组稍微不同的顺序
absCoef =  np.abs(coefStar)
idxs = np.argsort(-absCoef)

print("最优alpha的系数尺度所对应的属性排序 :",)
for idx in idxs:
    if absCoef[idx]!=0:
        print(df.columns[idx],end=' ')
print ("")
#  标准化与否的选择
xx = xData; 
# xx = xNormData
# y = yData
y = yNormData


alphas, coefs, _  = linear_model.lasso_path(xx, y,  return_models=False)

plt.plot(alphas,coefs.T)

plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.axis('tight')
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()
plt.show()

nattr, nalpha = coefs.shape

# 回归系数排序
nzList = []
for iAlpha in range(1,nalpha):
    coefList = coefs[: ,iAlpha]
    
    # 记录回归系数刚好变成非零的属性
    nzCoef = np.where(coefList!=0)[0]
    for q in nzCoef:
        if q not in nzList:
            nzList.append(q)

print("系数进入模型的次序所对应的属性排序 :",)
for idx in nzList:
    print(df.columns[idx], end=' ')
print("")

# 根据前面已获得的最佳\alpha，寻找此例中对应的索引
alphaStar = 0.013561387700964642
indexLTalphaStar = [index for index in range(100) if alphas[index] > alphaStar]
indexStar = max(indexLTalphaStar)

# 进而根据上面“对应的索引”，获得最佳回归系数值
coefStar = coefs[:,indexStar]
print("最佳回归系数：", coefStar)

# 回归系数给出了另外一组稍微不同的顺序
absCoef =  np.abs(coefStar)
idxs = np.argsort(-absCoef)

print("最优alpha的系数尺度所对应的属性排序 :",)
for idx in idxs:
    if absCoef[idx]!=0:
        print(df.columns[idx],end=' ')
print ("")
# 沿用前面已经获得的数据

# 扩展2个新属性
xExtData = np.concatenate((xData, np.zeros((m,2))), axis=1)
xExtData[:, n] = xData[:,-1]**2
xExtData[:, n+1] = xData[:,-1]*xData[:,1]

xNormExtData = (xExtData - xExtData.mean(axis=0))/np.std(xExtData,axis=0)

m, n = xData.shape

# 更新属性名
names = list(df.columns[idx])
names[-1] = "alco^2"
names.append("alco*volAcid")

#  标准化与否的选择
# xx = xExtData
xx = xNormExtData
# y = yData
y = yNormData

# 调用sklearn.linear_model中的LassoCV 
wineModel = LassoCV(cv=10).fit(xx, y)

# 显示结果
plt.figure()
plt.plot(wineModel.alphas_, wineModel.mse_path_, ':')
plt.plot(wineModel.alphas_, wineModel.mse_path_.mean(axis=-1),
         label='Average MSE Across Folds', linewidth=2)
plt.axvline(wineModel.alpha_, linestyle='--',
            label='CV Estimate of Best alpha')
plt.semilogx()
plt.legend()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Mean Square Error')
plt.axis('tight')
plt.show()

#print out the value of alpha that minimizes the Cv-error
print("最小均方误差对应的alpha值  ",wineModel.alpha_)
print("最小均方误差：", min(wineModel.mse_path_.mean(axis=-1)))
from sklearn.linear_model import enet_path
from sklearn.metrics import roc_auc_score, roc_curve

df = pd.read_csv('../input/sonaralldata/sonar.all-data.csv', header=None, prefix='V')

# 分类标签数值化
df['V60'] = df.iloc[:,-1].apply(lambda v: 1.0 if v=='M' else 0.0)

# 切分属性和标签 然后 标准化 
xData = df.values[:,:-1]; yData = df.values[:,-1] 

xData = (xData - xData.mean(axis=0))/xData.std(axis=0)
yData = (yData - yData.mean())/yData.std()

m, n = xData.shape

# 手工构建10折交叉验证循环
nxval = 10
for ixval in range(nxval):
    
    # 第ixval折验证的训练集和测试集的切分
    idxTest = [i for i in range(m) if i%nxval == ixval%nxval]
    idxTrain = [i for i in range(m) if i%nxval != ixval%nxval]    
    xTest = xData[idxTest,:]; yTest = yData[idxTest]
    xTrain = xData[idxTrain,:]; yTrain = yData[idxTrain]

    # enet_path 就是 ElasticNet正规化路径
    # 参数：l1_ratio就是套索惩罚项的占比
    alphas, coefs, _ = enet_path(xTrain, yTrain,l1_ratio=0.8, fit_intercept=False, return_models=False)

    # 将所有额预测及其标签归集到一起
    if ixval == 0:
        pred = np.dot(xTest, coefs)
        yOut = yTest
    else:
        pred = np.concatenate((pred, np.dot(xTest, coefs)), axis = 0)
        yOut = np.concatenate((yOut, yTest), axis=0)  

# 计算误分率
misClassRate = 1.0*((pred>=0)^(np.repeat(yOut, pred.shape[1]).reshape(pred.shape)>=0)).sum(axis=0)/m

# 寻找最小误差
idxMin = misClassRate.argmin()
minError = misClassRate[idxMin]

plt.figure()
plt.plot(alphas[1:], misClassRate[1:], label='Misclassification Error Across Folds', linewidth=2)
plt.axvline(alphas[idxMin], linestyle='--', label='CV Estimate of Best alpha')
plt.legend()
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Misclassification Error')
plt.axis('tight')
plt.show()

# 计算AUC
auc = []
for iPred in range(0, pred.shape[1]):
    predList = list(pred[:, iPred])
    aucCalc = roc_auc_score((yOut>0.0), predList)
    auc.append(aucCalc)

# 最大auc 及其对应的最小误分率
idxMax = np.argmax(auc)
minError = misClassRate[idxMin]

plt.figure()
plt.plot(alphas[1:], auc[1:], label='AUC Across Folds', linewidth=2)
plt.axvline(alphas[idxMax], linestyle='--', label='CV Estimate of Best alpha')
plt.legend()
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()
plt.xlabel('alpha')
plt.ylabel('Area Under the ROC Curve')
plt.axis('tight')
plt.show()


# 绘制最佳性能分类器的ROC曲线
fpr, tpr, thresh = roc_curve((yOut>0.0), list(pred[:, idxMax]))
ctClass = [i*0.01 for i in range(101)]
plt.plot(fpr, tpr, linewidth=2)
plt.plot(ctClass, ctClass, linestyle=':')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

print('最佳误分率 = ', misClassRate[idxMin])
print('最佳alpha(对应最佳误分率) = ', alphas[idxMin])
print('')
print('最佳AUC = ', auc[idxMax])
print('最佳alpha(对应最佳AUC)   =  ', alphas[idxMax])

print('')
print('不同阈值对应的混淆矩阵' )

# 正例数（标签为水雷）
P = len(yOut[yOut>0.0])

thr_idx = 20
print('')
print('阈值 =   ', thresh[thr_idx])
print('真正数(TP) = ', tpr[thr_idx]*P, '假负数(FN) = ', (1-tpr[thr_idx])*P)
print('假正数(FP) = ', fpr[thr_idx]*(m-P),'真负数(TN) = ', (1-fpr[thr_idx])*(m-P) )

thr_idx = 40
print('')
print('阈值 =   ', thresh[thr_idx])
print('真正数(TP) = ', tpr[thr_idx]*P, '假负数(FN) = ', (1-tpr[thr_idx])*P)
print('假正数(FP) = ', fpr[thr_idx]*(m-P),'真负数(TN) = ', (1-fpr[thr_idx])*(m-P) )

thr_idx = 60
print('')
print('阈值 =   ', thresh[thr_idx])
print('真正数(TP) = ', tpr[thr_idx]*P, '假负数(FN) = ', (1-tpr[thr_idx])*P)
print('假正数(FP) = ', fpr[thr_idx]*(m-P),'真负数(TN) = ', (1-fpr[thr_idx])*(m-P) )
# 沿用已有的数据

# 在整个数据集上进行路径分析
alphas, coefs, _ = enet_path(xData, yData,l1_ratio=0.8, fit_intercept=False, return_models=False)

plt.plot(alphas,coefs.T)
plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.axis('tight')
plt.semilogx()
ax = plt.gca()
ax.invert_xaxis()
plt.show()

nattr, nalpha = coefs.shape

# 回归系数排序
nzList = []
for iAlpha in range(1,nalpha):
    coefList = coefs[: ,iAlpha]
    
    # 记录回归系数刚好变成非零的属性
    nzCoef = np.where(coefList!=0)[0]
    for q in nzCoef:
        if q not in nzList:
            nzList.append(q)

print("系数进入模型的次序所对应的属性排序 :",)
for idx in nzList:
    print(df.columns[idx],end=' ')
print("")

# 根据前面已获得的最佳\alpha，寻找此例中对应的索引
alphaStar = 0.020334883589342503
indexLTalphaStar = [index for index in range(100) if alphas[index] > alphaStar]
indexStar = max(indexLTalphaStar)
print(indexStar)

# 进而根据上面“对应的索引”，获得最佳回归系数值
coefStar = coefs[:,indexStar]
print("最佳回归系数：", coefStar)

# 回归系数给出了另外一组稍微不同的顺序
absCoef =  np.abs(coefStar)
idxs = np.argsort(-absCoef)

print("最优alpha的系数尺度所对应的属性排序 :",)
for idx in idxs:
    if absCoef[idx]!=0:
        print(df.columns[idx],end=' ')
print("")
def S(z,gamma):
    if gamma >= np.abs(z):
        return 0.0
    if z > 0.0:
        return z - gamma
    else:
        return z + gamma

def glmnetIRLS(xx, y, beta, lam, alp):
    '''
        Glmnet和IRLS两个算法结合在一起，用于解决惩罚逻辑回归问题. 这是一个单步迭代公式
    参数：
        xx:  属性矩阵X
        y:   标签向量
        beta: 回归系数向量(更新后返回)
        lam:  \lambda
        alp:  \alpha
    '''

    m,n = xx.shape
       
    eta = np.dot(xx, beta) # = X beta
    eta[eta<-100] = -100
    
    mu = 1.0/(1.0+np.exp(-eta))
    
    w = mu * (1.0 - mu)
    
    cond1 = (np.abs(mu)<1e-5)
    cond2 = (np.abs(1.0-mu)<1e-5)
    mu[cond1] = 0.0
    mu[cond2] = 1.0
    w[cond1|cond2] = 1e-5
    
    ww = np.diag(w)
    
    z = (y-mu)/w + eta # = eta + W^{-1} (y - mu)
    
    for j in range(n):
        r = z - np.dot(xx,beta) # = z - X beta
        
        sumWxr = (w*xx[:,j]*r).sum()  # = (X^T W r)[j]
        sumWxx =  (w*xx[:,j]*xx[:,j]).sum() # = (X^T W X)[j,j]
        
        beta[j] = beta[j] + sumWxr / sumWxx
        
        if j > 0:
            avgWxx = sumWxx / m
            beta[j] = S(beta[j]*avgWxx, lam * alp) / (avgWxx + lam * (1.0 - alp))
    
    return beta

# 分类标签数值化
#df['V60'] = df.iloc[:,-1].apply(lambda v: 1.0 if v=='M' else 0.0)

# 切分属性和标签 然后 标准化 
xData = df.values[:,:-1]; yData = df.values[:,-1] 

# 只对X正规化, y只计算均值和标准差
xData = (xData - xData.mean(axis=0))/xData.std(axis=0)
yMean = yData.mean();  yStd = yData.std()

# 选择alpha参数
alpha = 0.8  # ElasticNet回归

# 确定lambda初始值：导致所有beta值都为零的lambda值（最简模型）
lam = (np.abs(yData.dot(xData))/xData.shape[0]).max()/alpha

# 扩展X成[1,X]
xData = np.concatenate((np.ones((xData.shape[0],1)), xData), axis=1)
nrow, ncol = xData.shape

# beta初始化
beta = np.zeros(ncol)
beta[0] = np.log(yMean/(1-yMean))

# 记录历史beta
betaMat = []
betaMat.append(list(beta[1:]))
beta0Mat = []
beta0Mat.append(beta[0])

# 迭代步数
nSteps = 100

# lambda缩减乘子：Fredman建议每步迭代后都要稍微减小lambda
lamMult = 0.93 # Fredman建议: lamMult^nSteps = 0.001

# 记录属性回归系数变成非零先后次序
nzList = []

# 开始进行lam迭代计算
for iStep in range(nSteps):
    lam = lam * lamMult
    
    deltaBeta = 100.0
    eps = 0.01
    iterStep = 0
    
    # 开始Glmnet算法迭代
    while deltaBeta > eps:
        iterStep += 1
        if iterStep > 100: #100:
            break
        
        # 上一步beta
        _beta = beta.copy()

        # IRL算法递归公式
        beta = glmnetIRLS(xData, yData, beta, lam, alpha)
        
        # 计算精度
        deltaBeta = np.abs(beta[1:]-_beta[1:]).sum() / np.abs(beta[1:]).sum()
    
    # 记录beta历史
    betaMat.append(list(beta[1:]))
    beta0Mat.append(beta[0])

    # 记录回归系数刚好变成非零的属性
    nzBeta = np.where(beta[1:]!=0)[0]
    for q in nzBeta:
        if (q in nzList) == False:
            nzList.append(q)

# 打印属性重要性排序
for idx in nzList:
    print(df.columns[idx],end=' ')
print("")

# lambda-误差曲线
plt.plot(betaMat)
plt.xlabel("Step Taken")
plt.ylabel("Coefficient Values")
plt.show()
df = pd.read_csv('../input/glassdata/glass.data.csv', header=None, prefix="V")
df.columns = ['Id','RI','Na', 'Mg', 'Al', 'Si','K', 'Ca', 'Ba', 'Fe', 'Type']
df = df.set_index("Id")

# 切分属性和标签
xData = df.values[:,:-1]; yLabel = df.values[:,-1] 

# 标签值向量化(一对所有)
labelList = list(set(yLabel))
labelList.sort()
nlabels = len(labelList)
def mapFunc(label):
    idx = labelList.index(label)
    row = [0]*nlabels
    row[idx] = 1
    return row
yData = np.array([mapFunc(label) for label in yLabel])

# 标准化
xData = (xData - xData.mean(axis=0))/xData.std(axis=0)
yMean = yData.mean(axis=0); yStd = yData.std(axis=0)
yData = (yData - yMean)/yStd

# 数据规模
m, n = xData.shape

# 手工构建n折交叉验证循环
nxval = 10
nAlphas= 100
misClass = [0.0] * nAlphas
for ixval in range(nxval):   
    # 第ixval折验证的训练集和测试集的切分
    idxTest = [i for i in range(m) if i%nxval == ixval%nxval]
    idxTrain = [i for i in range(m) if i%nxval != ixval%nxval]    
    xTest = xData[idxTest,:]; yTest = yData[idxTest,:]
    xTrain = xData[idxTrain,:]; yTrain = yData[idxTrain,:]
    labelTest = yLabel[idxTest]    

    # 为yTrain的每列建立模型
    # enet_path 就是 ElasticNet正规化路径
    # 参数：l1_ratio就是套索惩罚项的占比
    models = [enet_path(xTrain, yTrain[:,k] ,l1_ratio=1.0, fit_intercept=False, 
                        eps=0.5e-3, n_alphas=nAlphas , return_models=False) 
              for k in range(nlabels)]
    
    lenTest = m - len(yTrain)
    for iStep in range(1,nAlphas):
        # 组合所有模型的预测
        allPredictions = []
        for iModel in range(nlabels):
            # 模型预测
            _, coefs, _ = models[iModel]
            predTemp = np.dot(xTest, coefs[:,iStep])
            
            # 去标准化后比较
            predUnNorm = predTemp*yStd[iModel]+yMean[iModel]
            allPredictions.append(list(predUnNorm))
        allPredictions = np.array(allPredictions)

        # 找出最大的预测和误差        
        predictions = []
        for i in range(lenTest):
            listOfPredictions = allPredictions[:,i]
            idxMax = listOfPredictions.argmax()
            if labelList[idxMax] != labelTest[i]:
                misClass[iStep] += 1.0

misClassPlot = [misClass[i]/m for i in range(1, nAlphas)]

plt.plot(misClassPlot)

plt.xlabel("Penalty Parameter Steps")
plt.ylabel(("Misclassification Error Rate"))
plt.show()