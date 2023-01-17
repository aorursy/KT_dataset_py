import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import torch

import torch.optim as optim

import torch.nn.functional as F  
my_na_values = ['불명']

data = pd.read_csv('../input/aidefensegame18011862/18011862Aitrain.csv',na_values=my_na_values)

data
data['관할경찰서'][data.관할경찰서.str.contains('광주')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('서울')] = '0'

data['관할경찰서'][data.관할경찰서.str.contains('부산')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('대구')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('인천')] = '2'

data['관할경찰서'][data.관할경찰서.str.contains('대전')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('울산')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('수원')|data.관할경찰서.str.contains('일산')|data.관할경찰서.str.contains('성남')|data.관할경찰서.str.contains('용인')|data.관할경찰서.str.contains('안양')|data.관할경찰서.str.contains('안산')|data.관할경찰서.str.contains('과천')|data.관할경찰서.str.contains('광명')|data.관할경찰서.str.contains('군포')|data.관할경찰서.str.contains('부천')|data.관할경찰서.str.contains('시흥')|data.관할경찰서.str.contains('김포')|data.관할경찰서.str.contains('안성')|data.관할경찰서.str.contains('오산')|data.관할경찰서.str.contains('의왕')|data.관할경찰서.str.contains('이천')|data.관할경찰서.str.contains('평택')|data.관할경찰서.str.contains('하남')|data.관할경찰서.str.contains('화성')|data.관할경찰서.str.contains('여주')|data.관할경찰서.str.contains('양평')|data.관할경찰서.str.contains('고양')|data.관할경찰서.str.contains('구리')|data.관할경찰서.str.contains('남양주')|data.관할경찰서.str.contains('동두천')|data.관할경찰서.str.contains('양주')|data.관할경찰서.str.contains('의정부')|data.관할경찰서.str.contains('파주')|data.관할경찰서.str.contains('포천')|data.관할경찰서.str.contains('연천')|data.관할경찰서.str.contains('가평')|data.관할경찰서.str.contains('분당')] = '1'

data['관할경찰서'][data.관할경찰서.str.contains('춘천')|data.관할경찰서.str.contains('원주')|data.관할경찰서.str.contains('강릉')|data.관할경찰서.str.contains('동해')|data.관할경찰서.str.contains('태백')|data.관할경찰서.str.contains('속초')|data.관할경찰서.str.contains('삼척')|data.관할경찰서.str.contains('홍천')|data.관할경찰서.str.contains('횡성')|data.관할경찰서.str.contains('영월')|data.관할경찰서.str.contains('평창')|data.관할경찰서.str.contains('정선')|data.관할경찰서.str.contains('철원')|data.관할경찰서.str.contains('화천')|data.관할경찰서.str.contains('양구')|data.관할경찰서.str.contains('인제')|data.관할경찰서.str.contains('고성')|data.관할경찰서.str.contains('양양')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('청주')|data.관할경찰서.str.contains('상당')|data.관할경찰서.str.contains('서원')|data.관할경찰서.str.contains('흥덕')|data.관할경찰서.str.contains('청원')|data.관할경찰서.str.contains('충주')|data.관할경찰서.str.contains('제천')|data.관할경찰서.str.contains('보은')|data.관할경찰서.str.contains('옥천')|data.관할경찰서.str.contains('영동')|data.관할경찰서.str.contains('증평')|data.관할경찰서.str.contains('진천')|data.관할경찰서.str.contains('괴산')|data.관할경찰서.str.contains('음성')|data.관할경찰서.str.contains('단양')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('천안')|data.관할경찰서.str.contains('공주')|data.관할경찰서.str.contains('보령')|data.관할경찰서.str.contains('아산')|data.관할경찰서.str.contains('서산')|data.관할경찰서.str.contains('논산')|data.관할경찰서.str.contains('계룡')|data.관할경찰서.str.contains('당진')|data.관할경찰서.str.contains('금산')|data.관할경찰서.str.contains('부여')|data.관할경찰서.str.contains('서천')|data.관할경찰서.str.contains('청양')|data.관할경찰서.str.contains('홍성')|data.관할경찰서.str.contains('예산')|data.관할경찰서.str.contains('태안')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('전주')|data.관할경찰서.str.contains('군산')|data.관할경찰서.str.contains('익산')|data.관할경찰서.str.contains('정읍')|data.관할경찰서.str.contains('정읍')|data.관할경찰서.str.contains('남원')|data.관할경찰서.str.contains('김제')|data.관할경찰서.str.contains('완주')|data.관할경찰서.str.contains('진안')|data.관할경찰서.str.contains('무주')|data.관할경찰서.str.contains('장수')|data.관할경찰서.str.contains('임실')|data.관할경찰서.str.contains('순창')|data.관할경찰서.str.contains('고창')|data.관할경찰서.str.contains('부안')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('목포')|data.관할경찰서.str.contains('여수')|data.관할경찰서.str.contains('순천')|data.관할경찰서.str.contains('나주')|data.관할경찰서.str.contains('광양')|data.관할경찰서.str.contains('담양')|data.관할경찰서.str.contains('곡성')|data.관할경찰서.str.contains('구례')|data.관할경찰서.str.contains('고흥')|data.관할경찰서.str.contains('보성')|data.관할경찰서.str.contains('화순')|data.관할경찰서.str.contains('장흥')|data.관할경찰서.str.contains('강진')|data.관할경찰서.str.contains('해남')|data.관할경찰서.str.contains('영암')|data.관할경찰서.str.contains('무안')|data.관할경찰서.str.contains('함평')|data.관할경찰서.str.contains('영광')|data.관할경찰서.str.contains('장성')|data.관할경찰서.str.contains('완도')|data.관할경찰서.str.contains('진도')|data.관할경찰서.str.contains('신안')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('포항')|data.관할경찰서.str.contains('경주')|data.관할경찰서.str.contains('김천')|data.관할경찰서.str.contains('안동')|data.관할경찰서.str.contains('구미')|data.관할경찰서.str.contains('영주')|data.관할경찰서.str.contains('영천')|data.관할경찰서.str.contains('영천')|data.관할경찰서.str.contains('상주')|data.관할경찰서.str.contains('문경')|data.관할경찰서.str.contains('경산')|data.관할경찰서.str.contains('군위')|data.관할경찰서.str.contains('의성')|data.관할경찰서.str.contains('청송')|data.관할경찰서.str.contains('영양')|data.관할경찰서.str.contains('영덕')|data.관할경찰서.str.contains('청도')|data.관할경찰서.str.contains('고령')|data.관할경찰서.str.contains('성주')|data.관할경찰서.str.contains('칠곡')|data.관할경찰서.str.contains('예천')|data.관할경찰서.str.contains('봉화')|data.관할경찰서.str.contains('울진')|data.관할경찰서.str.contains('울릉')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('창원')|data.관할경찰서.str.contains('진주')|data.관할경찰서.str.contains('통영')|data.관할경찰서.str.contains('사천')|data.관할경찰서.str.contains('김해')|data.관할경찰서.str.contains('밀양')|data.관할경찰서.str.contains('거제')|data.관할경찰서.str.contains('양산')|data.관할경찰서.str.contains('의령')|data.관할경찰서.str.contains('함안')|data.관할경찰서.str.contains('창녕')|data.관할경찰서.str.contains('고성')|data.관할경찰서.str.contains('남해')|data.관할경찰서.str.contains('하동')|data.관할경찰서.str.contains('산청')|data.관할경찰서.str.contains('함양')|data.관할경찰서.str.contains('거창')|data.관할경찰서.str.contains('합천')|data.관할경찰서.str.contains('마산')|data.관할경찰서.str.contains('진해')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('제주')|data.관할경찰서.str.contains('서귀포')] = 'hi'

data['관할경찰서'][data.관할경찰서.str.contains('반곡')|data.관할경찰서.str.contains('소담')|data.관할경찰서.str.contains('보람')|data.관할경찰서.str.contains('대평')|data.관할경찰서.str.contains('가람')|data.관할경찰서.str.contains('한솔')|data.관할경찰서.str.contains('나성')|data.관할경찰서.str.contains('새롬')|data.관할경찰서.str.contains('다정')|data.관할경찰서.str.contains('어진')|data.관할경찰서.str.contains('종촌')|data.관할경찰서.str.contains('고운')|data.관할경찰서.str.contains('아름')|data.관할경찰서.str.contains('도담')|data.관할경찰서.str.contains('조치원')|data.관할경찰서.str.contains('연기')|data.관할경찰서.str.contains('연동')|data.관할경찰서.str.contains('부강')|data.관할경찰서.str.contains('금남')|data.관할경찰서.str.contains('장군')|data.관할경찰서.str.contains('연서')|data.관할경찰서.str.contains('전의')|data.관할경찰서.str.contains('전동')|data.관할경찰서.str.contains('소정')|data.관할경찰서.str.contains('세종')] = 'hi'

replace_values = {'남자' : 0,'여자' : 1}

data = data.replace({"성별": replace_values})

data1 = data[data['관할경찰서'] == 'hi'].index

data = data.drop(data1) 

data = data.dropna(axis=0) 

data = data[['성별','적발횟수','나이','측정일시','관할경찰서']]
m_tmp = data[['측정일시']]

d_tmp = data[['측정일시']]

d_tmp['측정일시'] = data['측정일시'].str.slice(8,10) 

m_tmp['측정일시'] = data['측정일시'].str.slice(5,7) 

replace_values = {'01' :13,'02':14} # 3월 부분은 replace_values = {'03' :3}로 변경

m_tmp = m_tmp.replace({"측정일시": replace_values})

m = m_tmp['측정일시']

d = data['측정일시'].str.slice(8,10) 

d = d.apply(pd.to_numeric)

days = {1:'SUN' , 2:'MON', 

        3:'TUE', 4:'WED', 

        5:'THU', 6:'FRI', 0:'SAT'}



var = ((m+1)*13)/5

var = var.astype('int')

h = (d +  var + 19 + int(19/4) + int(20/4) - 2*20 ) % 7 # 3월부분은 h = (d +  var + 20 + int(20/4) + int(20/4) - 2*20 ) % 7로 변경

data['측정일시'] = h
newdata = data[['나이']]

newdata[(newdata['나이'] < 40)] = 0

newdata[(newdata['나이'] >= 40)] = 1
data = data[['성별','적발횟수','측정일시','관할경찰서']]

data = pd.concat([data,newdata], axis=1) #열합치기

data
data = data.astype('float32')

data = data.to_numpy()

data
x_data = data[: , 0:-1]

y_data = data[: , -1]



x_train = torch.FloatTensor(x_data)

y_train = torch.LongTensor(y_data)
import torch.nn.functional as F  



nb_class = 2

nb_data = len(y_train)



W = torch.zeros((4, nb_class), requires_grad=True)

b = torch.zeros(1, requires_grad=True)



optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 10000

for epoch in range(nb_epochs + 1):

    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)

    

    y_one_hot = torch.zeros(nb_data, nb_class)

    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

    cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()

    

    optimizer.zero_grad()

    cost.backward()

    optimizer.step()



    if epoch % 1000 == 0:

        print('Epoch {:4d}/{} Cost: {:.6f}'.format(

            epoch, nb_epochs, cost.item()

        ))
testdata = pd.read_csv('../input/aidefensegame18011862/kaggle_18011862test.csv')

testdata
testdata = testdata.astype('float32')

testdata = testdata.to_numpy()
x_data = testdata[: , :]

x_train = torch.FloatTensor(x_data)
hypothesis = F.softmax(x_train.matmul(W) + b, dim=1) 

predict = torch.argmax(hypothesis, dim=1)
predict = predict.detach().numpy().reshape(-1,1)

id=np.array([i for i in range(2719)]).reshape(-1,1)

result=np.hstack([id,predict])

df=pd.DataFrame(result,columns=["ID","Label"])

df['Label'] = df['Label'].astype('float32')

df.to_csv("baseline.csv",index=False,header=True)