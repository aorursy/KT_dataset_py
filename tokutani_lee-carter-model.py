#ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#国立社会保障人口問題研究所より日本全国の死亡率をCSV化したもの。100歳以上は削除した。
mort = pd.read_csv("../input/JapanMort_modified.csv")
print(mort.describe())
#死亡率データから特定の観察年度・性別の死亡率を取得する関数を定義
def select_mort(year,sex):
    tmp= mort[mort["Year"]==year]
    tmp2=tmp.values
    if sex==2: #2:female, 3:male
        tmp3=tmp2[:,2]
    else:
        tmp3=tmp2[:,3] 
    return tmp3.reshape(101,1)

mort_idx=np.arange(101)
mort2016m=select_mort(2016,3)
mort1986m=select_mort(1986,3)
mort1956m=select_mort(1956,3)
#mxを重ねて表示
fig, ax = plt.subplots()
ax.plot(mort_idx,mort2016m)
ax.plot(mort_idx,mort1986m)
ax.plot(mort_idx,mort1956m)
plt.show()
#log(mx)を重ねて表示
fig, ax = plt.subplots()
ax.plot(mort_idx,np.log(mort2016m))
ax.plot(mort_idx,np.log(mort1986m))
ax.plot(mort_idx,np.log(mort1956m))
plt.show()
#男女それぞれのlog(mx)を、年齢×観察年度のマトリックスにする
mx_m=select_mort(1947,3)
mx_f=select_mort(1947,2)

#観察年度をfor文で逐次水平結合する
for i in np.arange(1948,2017):
    mx_m=np.hstack([mx_m,select_mort(i,3)])
    mx_f=np.hstack([mx_f,select_mort(i,2)])

#mxに対してlogをとる
logmx_m=np.log(mx_m)
logmx_f=np.log(mx_f)

#軸もリストに格納
age=[x for x in range(101)]
year=[x for x in range(1947,2017,1)]
#ヒートマップで男性の死亡率変遷を確認する
fig, ax = plt.subplots()
sns.heatmap(logmx_m,annot=False,xticklabels=year,yticklabels=age)
plt.show()
#第５９号　第２分冊（２００６年） 将来死亡率推定に関する一考察　＜及川桂＞ 
#Lee Carter Model
#logmx = ax + bx*kt + ε
#Σbx=1, Σkx=0

#Parameter設定
T=70
N=101
logm_xt=logmx_m+0 #ここでは男性の実績死亡率を使用

#ここから計算
a_x=logm_xt.sum(axis=1) / T #論文(1.4)式より
z_xt=logm_xt - a_x.reshape(N,1) #ブロードキャストにより計算　(101,70) - (101,1)の形式

U, S, V = np.linalg.svd(z_xt, full_matrices=True)

bxkt = S[0] * np.dot(U[:,0].reshape(N,1),V[0,:].reshape(T,1).T) #論文(1.5)式より
eps = z_xt - bxkt

#Lee Carter Modelでの予測値
logm_xt_lcfitted = bxkt + a_x.reshape(N,1)

#bx,kxの出力
b_x = U[:,0]/U[:,0].sum() #論文(1.7)式より
k_t = V[0,:]*S[0]*U[:,0].sum()
a_x = a_x + k_t.sum()*b_x
k_t = k_t - k_t.sum()

#ax+bx*ktが予測値を再現することの確認
chk = logm_xt_lcfitted - a_x.reshape(N,1)
chk = chk - np.dot(b_x.reshape(N,1),k_t.reshape(T,1).T)
chk = chk*chk
print(chk.sum())
#ax,bx,ktのプロット
fig, ax = plt.subplots(1,3,figsize=(15,5))
ax[0].plot(age,a_x)
ax[1].plot(age,b_x)
ax[2].plot(year,k_t) #ktの2000年以降のピークは東日本大震災の影響
plt.show()
#ヒートマップで予測値を確認する
fig, ax = plt.subplots()
sns.heatmap(logm_xt_lcfitted,annot=False,xticklabels=year,yticklabels=age)
plt.show()
#ヒートマップで残差を確認する 斜めに走る線=生年によるcohort効果がはっきり見える
fig, ax = plt.subplots()
sns.heatmap(eps,annot=False,xticklabels=year,yticklabels=age)
plt.show()
