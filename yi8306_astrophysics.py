import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
filename = "/kaggle/input/mini_bss.txt"



with open(filename) as fn:

    content = fn.readlines()



print(content) 
import numpy as np

cat_1 = np.loadtxt('/kaggle/input/mini_bss.txt', usecols=range(1, 7))



print(cat_1[0:3])
filename = "/kaggle/input/mini_super.csv"



with open(filename) as fn:

    content = fn.readlines()



print(content)
import numpy as np

cat_2 = np.loadtxt('/kaggle/input/mini_super.csv', delimiter=',', skiprows=1, usecols=[0, 1])

print(cat_2)
import numpy as np

import time



# 將赤經的 HMS 系統轉換為角度

def hms2dec(hr, m, s):

    dec = hr + m/60 + s/3600

    return dec*15



# 將赤緯的 DMS 系統轉換為角度

def dms2dec(d, m, s):

    sign = d/abs(d)

    dec = abs(d) + m/60 + s/3600

    return sign*dec



# 匯入 BSS 資料

def import_bss():

    res = []

    data = np.loadtxt('/kaggle/input/bss.txt', usecols=range(1, 7))

    for i, row in enumerate(data, 1):

        res.append((i, hms2dec(row[0], row[1], row[2]), 

                    dms2dec(row[3], row[4], row[5])))

    return res



# 匯入 SuperCOSMOS 資料

def import_super():

    data = np.loadtxt('/kaggle/input/super.csv', delimiter=',', 

                      skiprows=1, usecols=(0, 1))

    res = []

    for i, row in enumerate(data, 1):

        res.append((i, row[0], row[1]))

    return res



def angular_dist(ra1, dec1, ra2, dec2):

    # 將角度轉換成弧度單位

    r1 = np.radians(ra1)

    d1 = np.radians(dec1)

    r2 = np.radians(ra2)

    d2 = np.radians(dec2)

    # 計算2個資料集彼此訊號間的角距離

    a = np.sin(np.abs(d1 - d2)/2)**2

    b = np.cos(d1)*np.cos(d2)*np.sin(np.abs(r1 - r2)/2)**2

    angle = 2*np.arcsin(np.sqrt(a + b)) 

    # 最後再將弧度轉換回角度

    return np.degrees(angle)

  

# crossmatch程式接受3個參數匯入：

# cat1(BSS資料)、cat2(SuperCOSMOS資料)、max_radius(最大角距離)，

# 並匯出3個參數：

# (1)配對清單(列出哪些 BSS的訊號與哪些 SuperCOSMOS的訊號指到同一天體，與彼此的角距離)、

# (2)非配對清單、(3)程式執行時間



def crossmatch(cat1, cat2, max_radius):

    # 開始計時

    start = time.perf_counter()

    

    matches = []

    no_matches = []

    

    # 以下為上述的程式流程(1-4 步驟)

    for id1, ra1, dec1 in cat1:

        closest_dist = np.inf

        closest_id2 = None

        for id2, ra2, dec2 in cat2:

            dist = angular_dist(ra1, dec1, ra2, dec2)

            if dist < closest_dist:

                closest_id2 = id2

                closest_dist = dist

        

        # 若最短角距離大於程式設定的最大角距離，則忽略此配對

        if closest_dist > max_radius:

            # 此為非配對清單：僅列出 BSS中沒有被配對到的訊號ID

            no_matches.append(id1)

        else:

            # 此為配對清單，列出：

            # BSS的訊號ID、對應的SuperCOSMOS的訊號ID、這2個訊號的角距離

            matches.append((id1, closest_id2, closest_dist))

    

    # 停止計時

    time_taken = time.perf_counter() - start

    return matches, no_matches, time_taken



bss_cat = import_bss()

super_cat = import_super()

    

max_dist = 3/3600   # 此為設定的最大角距離

matches, no_matches, time_taken = crossmatch(bss_cat, super_cat, max_dist)



# 列出前5對的配對

print('配對清單:', matches[:5], '\n')

print(len(matches), '\n')



# 列出前5個沒有被配對到的BSS資料集的訊號ID

print('無配對的BSS資料集ID:', no_matches[:5], '\n')

print(len(no_matches), '\n')

print('程式執行時間 (秒):', time_taken)