# 이야기의 흐름

# 지난 시간엔 데이터 분석 기법에 대해 

# 이번시간엔 데이터와 표본분포(p.63)에 대해 알아보고 이와 관련된 예제를 알아본다.



# 표본추출 : 모집단(데이터를 구성하는 전체 집합)으로 부터 얻은 부분 집합(p.64)

# 표본추출방법 : 대표성을 담보하기 위한 여러방법이 있지만, 결국 핵심은 랜덤표본추출(p.65)



# 표본분포 : 동일 모집단에서 얻은 여러 샘플에 대한 표본통계량의 분포(p.74), (작은) 표본을 가지고 (매우 큰) 모집단을 추론하는 것과 관련..

# 중심극한정리 : 모집단이 정규분포가 아니더라도 표본크기가 충분하고 데이터가 정규성을 크게 이탈하지 않은 경우, 여러표본에서 추출한 평균은 종모양의 정규곡선을 따른다.(p.75, 76)



# 표준오차 : s/루트n (표본크기:n, 표준편차s) - 표본크기가 커지면 표준오차가 줄어듬 : n제곱근의 법칙 (p.77), 표준오차를 2배 줄이려면 표본크기를 4배증가

# 부트스트랩 재표본 : 표준오차를 줄이기 위해 표본크기를 늘리는건 불가능하거나 비효율적일때가 있다. 이때 사용. 관측데이터 집합에서 얻은 복원추출 표본



# 정규분포 : 종형곡선의 데이터 분포(p.86), 데이터의 68%는 평균 표준편차내에 속하며 95%는 표준편차 두 배수내에 존재

# 표준정규분포 : x축의 단위가 평균의 표준편차로 표현되는 정규분포 , 평균=0, 표준편차=1인 정규분포

# 정규화 : 데이터를 표준정규분포와 비교할려면 데이터에서 평균을 빼고, 표준편차로 나눈다. 이것이 곧 정규화(표준화) => 변화된 값 = z 점수

# QQ 그림 : 표본분포가 정규분포에 얼마나 가까운지를 시각적으로 보여줌

#          z점수를 오름차순으로 정렬하고 각 점수를 y축에 표시 (p.87)

#          x축은 정규분포에서의 해당 분위수를 나타냄

#          점(z점수)들이 대략 대각선 위에 놓이면 표본분포가 정규분포에 가까운 것으로 간주할 수 있다.

# 긴 꼬리 분포 : 왜도와 첨도

# http://blog.naver.com/PostView.nhn?blogId=chochila&logNo=40144022678&parentCategoryNo=&categoryNo=66&viewDate=&isShowPopularPosts=true&from=search



# 사용 데이터 : https://www.kaggle.com/c/house-prices-advanced-regression-techniques

# 참고 커널 : https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python



# house prices 데이터 선정 이유

# 1. 제공 데이터가 일반적으로 익숙한 점 예) 연면적, 부동산의 크기, 건설연도, ...

# 2. 데이터 간에 직관적으로 파악되는 연관관계가 있다는점 예) 면적이 크면 가격도 비싸겠지...

# 3. 교제 예제 중에 대입가능한 것이 있는지 여부 예) 2번을 토대로 양의 상관관계로 미루어 볼때 linear regression 을 써볼수 있겠다.



# 다음과 같이 진행해봤습니다. 2,3항목에서 위에 본 개념들이 등장 합니다.

# 1. 데이터를 훝어보고

# 2. 히트맵을 통해 상관관계가 있는 데이터를 찾아보고

# 3. 데이터 클린징

# 4. linear regression 적용해보기 (p.151)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# 제공 데이터는 아래와 같습니다. data_description.txt

# 다양하며 수치, 비수치 데이터가 혼재되어 있습니다.



# MSSubClass : 주거지 유형 ex) 1층 1946 새로운스타일 , 1층 1945 구식, ...

# MSZoning : 구역분류 ex) 상업용지, 농업용지, ...

# LotFrontage : 부동산과 연결된 거리의 길이

# LotArea : 크기, 면적

# Street : 도로유형 ex) 포장, 비포장

# Alley : 골목유형 ex) 포장, 비포장, 없음

# LotShape : 부동산 모양, 형태  ex) 규칙적, 불규칙적...

# LandContour : 평탄한 정도

# Utilities : 수도,전기,가스 등의 접근성 ex) 모두 가능, 전기/가스만, 등...

# LotConfig : 배치 ex) 안쪽, 코너, ...

# LandSlope : 땅 기울기

# Neighborhood : 에임스 도시내의 물리적 위치 ex) 서울에 "구"에 해당하는 정도?

# Condition1 : 근접성 ex) 주거리와 인접함, 철도인접함, ...

# Condition2 : 근접성 2번째 항목, 상동

# BldgType : 주거유형 ex) 단독가구, 2가구 1주택, 2세대 가구, ...

# HouseStyle : 주거형태 ex) 단층, 1.5층, 2층, ...

# OverallQual : 주택의 전체적인 재질과 마감에 대한 평가 ex) 1~10 클수록 좋음

# OverallCond : 집 상태에 대한 평가 ex) 1~10 클수록 좋음

# YearBuilt : 건설연도

# YearRemodAdd : 리모델연도

# RoofStyle : 지붕형태 ex) 평탄, ...

# RoofMatl : 지붕 재료 ex) 타일, 금속, ...

# Exterior1st : 익스테리어 ex) 석면, 아스팔트, ..

# Exterior2nd : 상동

# .... 이 아래로 너무 많음 ...
# 2. 위 데이터에 따른 가격(Y) 분포를 봐본다.

# 가격 분포는?? 대략 정규분포를 따르는것으로 보인다.

df_train = pd.read_csv('../input/train.csv')

#df_train['SalePrice'].describe()

sns.distplot(df_train['SalePrice']);

# 면적에 따른 분포는?

# 부동산 부지의 크기와 가격간의 관계는 애매해 보인다.

var = 'LotArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# 지상거주지 면적 에 따른 분포는? 연면적

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# 지하거주지 면적 에 따른 분포는? 연면적

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# 주택의 질과 마감에 대한 평가는?

var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(8, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
# 추측에 의한 관계를 살펴본 후 전체 데이터에 대한 상관관계를 히트맵을 통해 보았다.

# https://seaborn.pydata.org/generated/seaborn.heatmap.html

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(corrmat, vmin=0, vmax=1, square=True);



# TotalBsmtSF, 1stFlrSF, GrLivArea, GarageCars,GarageArea
# 관계가 있어보이는 주요 항목에 대해 수치적으로 자세히 확인해보자.

# saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=False, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

# TotalBsmtSF, 1stFlrSF, GrLivArea, GarageCars,GarageArea



#'전체 품질', 'GrLivArea', 'TotalBsmt'.SF'는 '세일프라이스'와 밀접한 관련이 있다. 체크!

#'가라지카'와 '가라지아레아'도 가장 강하게 상관관계가 있는 변수 중 하나이다. 그러나 지난번 서브포인트에서 논의했듯이, 차고지에 맞는 자동차의 수는 차고지의 결과물이다. '가라지카'와 '가라지아레아'는 쌍둥이 형제와 같다. 절대로 그들을 구별할 수 없을 것이다. 따라서 우리는 분석에 이러한 변수들 중 하나만 있으면 된다('SalePrice'와의 상관관계가 더 높기 때문에 'GarageCars'를 유지할 수 있다).

#'토탈BsmtSF'와 '1stFloor'도 쌍둥이 형제처럼 보인다. TotalBsmt는 계속 사용할 수 있다.SF'는 단지 우리의 첫 추측이 맞았다고만 말할 뿐이다('그래서...'를 다시 읽는다... 우리가 기대할 수 있는 것은 무엇인가)라고 말했다.

#'풀바스'? 정말?

#TotRmsAbvGrd와 GrLivArea, 다시 쌍둥이 형제. 체르노빌의 데이터 세트인가?

#아... 'YearBuilt'... 'YearBuilt'는 'SalePrice'와 약간 상관관계가 있는 것 같다. 솔직히 말해서, 나는 'YearBuilt'에 대해 생각하는 것이 두렵다. 왜냐하면 나는 이것을 올바르게 하기 위해 약간의 시계열 분석을 해야 한다는 생각이 들기 때문이다. 이거 숙제로 두고 갈게.
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df_train[cols], size = 2.5)

plt.show();





#비록 우리가 이미 주요 인물들을 알고 있지만, 이 거대한 산점도는 변수 관계에 대한 합리적인 생각을 제공한다.



#우리가 흥미롭게 여길 수 있는 수치 중 하나는 'TotalBsmt' 사이의 수치다.SF'와 '그라이브아레아'가 그것이다. 이 그림에서 우리는 점들이 거의 경계선처럼 작용하는 선형 선을 그리는 것을 볼 수 있다. 대부분의 점들이 그 선 아래에 머무른다는 것은 완전히 이치에 맞는다. 지하지역은 위의 지상생활공간과 같을 수 있지만, 지하생활공간보다 더 큰 지하공간은 기대되지 않는다(벙커를 사려고 하지 않는 한).



#'세일프라이스'와 '이어빌트'에 관한 줄거리도 우리를 생각하게 할 수 있다. '점 구름'의 바닥에서 우리는 거의 수줍은 지수 함수처럼 보이는 것을 본다. 우리는 또한 '점 구름'의 상한선에서도 이와 같은 경향을 볼 수 있다(더 창조적이 되라). 또한, 지난 해와 관련된 점들의 집합이 어떻게 이 한도를 넘는 경향이 있는지 주목하라(나는 단지 지금 가격이 더 빠르게 상승하고 있다고 말하고 싶었다).



#좋아, 일단 로르샤흐 테스트는 충분해. 누락된 데이터: 누락된 데이터로 이동합시다!

    

# GrLiveArea, TotalBsmtSF, yearBuilt
# TotalBsmtSF : 지하연면적, GrLivArea : 지상연면적, GarageArea : 주차면적, OverallQual : 주택의 질과 면적에 대한 점수 등을 데이터로 써보고 싶다.

# 그전에 누락 데이터는 없는지 확인하고 데이터 크린징해본다.

# 누락된 데이터에 대해 생각할 때 중요한 질문:



#누락된 데이터는 얼마나 널리 퍼졌는가?

#누락된 데이터가 랜덤한가, 아니면 패턴이 있는가?

#데이터 누락이 표본 크기를 줄이는 것을 의미할 수 있기 때문에 이러한 질문에 대한 해답은 실질적인 이유로 중요하다. 이것은 우리가 분석을 진행하지 못하게 할 수 있다. 더욱이 실질적인 관점에서 우리는 누락된 데이터 과정이 편향되지 않고 불편한 진실을 숨기지 않도록 해야 한다.

#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)

# 누락 데이터가 얼마인지 확인하여 사용할 데이터와 사용하지 않을 데이터 거르기

# 비율이 좀 있는것들은 항목자체를 제거

# 누락 데이터 건수가 작은 것들은 해당 건만 제거

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...
# 사용할 데이터에 대해 특이값을 확인해보고 크린징 해본다.

# 특이값 확인하기, SalePrice의 하위값, 상위값을 확인ㅍ하고 일반적인 데이터와 동떨어진 데이터를 찾아 크린징한다.

#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)



#여기서 일차적인 관심사는 관찰을 특이사항으로 정의하는 임계값을 설정하는 것이다. 그러기 위해서 데이터를 표준화하겠다. 이 맥락에서 데이터 표준화는 데이터 값을 평균 0, 표준 편차 1로 변환하는 것을 의미한다.

#로우 레인지 값은 비슷하고 0에서 그리 멀지 않다.

#하이 레인지 값은 0에서 멀고 7.20대 값은 정말 범위를 벗어났다.

#현재로서는, 우리는 이 가치들 중 어느 것도 특이사항으로 여기지 않을 것이지만, 우리는 이 두 가지 7. 10 가치에 주의해야 한다.
# saleprice - 연면적 관계 중에 동떨어진 데이터를 찾아보다.

# 아마도 농업용지로 판단된다. 건수가 적으므로 제거한다.

#bivariate analysis saleprice/grlivarea

var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#deleting points

#GrLivArea가 더 큰 두 개의 가치는 이상하게 보이고 그들은 군중을 따라가지 않는다. 우리는 왜 이런 일이 일어나는지 추측할 수 있다. 아마도 그들은 농업 지역을 언급하고 그것이 낮은 가격을 설명할 수 있을 것이다. 나는 이것에 대해서는 확신할 수 없지만 이 두 가지 점들이 전형적인 경우를 대표하는 것이 아니라고 확신한다. 그러므로, 우리는 그들을 특이사항으로 정의하고 삭제할 것이다.

#줄거리의 맨 위에 있는 두 가지 관찰은 우리가 조심해야 한다고 말한 7가지 관찰이다. 그들은 두 가지 특별한 경우처럼 보이지만, 그들은 그 추세를 따르고 있는 것 같다. 그런 이유로, 우리는 그들을 지킬 것이다.

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
# saleprice - 연면적 관계 중에 동떨어진 데이터를 찾아본다.

# 크게 떨어져 있는 데이터는 없어보이므로 전부 사용하기로 한다.

#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

# 일부 관찰(예: TotalBsmt)을 제거하고 싶은 유혹을 느낄 수 있다.SF > 3000) 그러나 나는 그럴 가치가 없다고 생각한다. 우린 그걸 가지고 살 수 있으니까 아무것도 하지 않을 거야.
# work

# 이 질문에 대한 해답은 다변량 분석을 위한 통계적 기초의 가정을 시험하는 데 있다. 

# 우리는 이미 약간의 데이터 정리를 했고 'SalePrice'에 대해 많은 것을 발견했다. 

# 이제 '세일프라이스'가 다변량 기법을 적용할 수 있는 통계적 가정을 어떻게 준수하는지 깊이 이해할 때가 되었다.



# 헤어 외 연구진(2013년)에 따르면 다음과 같은 네 가지 가정을 시험해야 한다.

# 다변량 분석 (mutivariate)

# https://www.amazon.com/Multivariate-Data-Analysis-Joseph-Hair/dp/9332536503/ref=as_sl_pc_tf_til?tag=pmarcelino-20&linkCode=w00&linkId=5e9109fa2213fef911dae80731a07a17&creativeASIN=9332536503



# 정규성 - 정규성에 대해 이야기할 때 우리가 의미하는 것은 데이터가 정규 분포처럼 보여야 한다는 것이다. 

# 이는 여러 통계 검정이 이것에 의존하기 때문에 중요하다(예: t-통계). 

# 이 연습에서는 'SalePrice'에 대한 일변량 정규성(제한적 접근법)을 점검할 것이다. 

# 일변량 정규성이 다변량 정규성을 보장하지는 않지만, 그것이 도움이 된다는 것을 기억하라. 

# 고려해야 할 또 다른 세부 사항은 큰 표본(>200개의 관측치)에서 정규성은 그러한 문제가 아니라는 것이다. 

# 그러나, 우리가 정규성을 해결한다면, 우리는 많은 다른 문제들(예: 이성질성)을 피하게 되므로, 이것이 우리가 이 분석을 하는 주된 이유다.



# 동질성 - 나는 단지 내가 그것을 올바르게 썼기를 바란다. 

# 동위성은 '종속 변수가 예측 변수의 범위에 걸쳐 동일한 수준의 차이를 보이는 가정'을 의미한다(Hair et al., 2013). 

# 독립 변수의 모든 값에서 오차 항이 동일해지기를 원하기 때문에 동격성이 바람직하다.



# 선형성- 선형성을 평가하는 가장 일반적인 방법은 산점도를 검사하고 선형 패턴을 찾는 것이다. 

# 패턴이 선형적이지 않다면, 데이터 변환을 탐구할 가치가 있을 것이다. 

# 하지만, 우리가 본 대부분의 산란 플롯은 선형 관계를 가지고 있는 것으로 보이기 때문에 우리는 이것을 시작하지 않을 것이다.



# 관련 오류의 부재 - 정의에서 제시한 바와 같이 관련 오류는 한 오류가 다른 오류와 상관될 때 발생한다. 

# 예를 들어, 하나의 긍정적인 오류가 체계적으로 부정적인 오류를 범한다면, 그것은 이 변수들 사이에 관계가 있다는 것을 의미한다. 

# 이것은 종종 시계열에서 발생하는데, 여기서 어떤 패턴들은 시간과 관련이 있다. 

# 우리 또한 이 일에 관여하지 않을 것이다. 하지만, 만약 여러분이 무언가를 감지한다면, 여러분이 받고 있는 효과를 설명할 수 있는 변수를 추가하려고 노력하라. 

# 그것은 관련 오류에 대한 가장 일반적인 해결책이다.





# 여기서 요점은 '세일프라이스'를 매우 희박하게 시험하는 것이다. 다음 사항에 주의하여 이 작업을 수행하십시오.



# 히스토그램 - 커토스 및 왜곡.

# 정규 확률도 - 데이터 분포는 정규 분포를 나타내는 대각선을 근접하게 따라야 한다.

#histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)



#좋아, '세일프라이스'는 정상적이지 않아. 그것은 '정점성'과 긍정적인 왜곡을 보여주며 대각선을 따르지 않는다.



#하지만 모든 것이 사라진 것은 아니다. 단순한 데이터 변환으로 문제를 해결할 수 있다. 

#이것은 통계책에서 배울 수 있는 놀라운 것들 중 하나이다: 긍정적인 왜곡의 경우, 로그 변환은 대개 잘 작동한다. 

#이것을 발견했을 때, 나는 호그와트의 한 학생이 새로운 시원한 주문을 발견하는 것처럼 느꼈다.



# stats.probplot : 커널 밀도(kernel density)는 커널이라는 함수를 겹치는 방법으로 히스토그램보다 부드러운 형태의 분포 곡선을 보여주는 방법이다.

# http://seaborn.pydata.org/generated/seaborn.distplot.html



# Q-Q

# 로그 변환을 통해 왜곡을
#applying log transformation

df_train['SalePrice'] = np.log(df_train['SalePrice'])



#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#data transformation

df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

#transformed histogram and normal probability plot

sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot

sns.distplot(df_train['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)



#좋아, 이제 우린 큰 보스를 상대하는 거야. 여기 뭐가 있지?



#일반적으로는, 비뚤어진 것을 나타내는 것.

#값이 0인 유의한 관측치 수(지하실이 없는 집)

#0 값은 로그 변환을 허용하지 않기 때문에 큰 문제.

#여기서 로그 변환을 적용하기 위해 지하실(이진 변수)의 유무효를 얻을 수 있는 변수를 만들겠다. 

#그런 다음 값이 0인 관측치를 무시하고 모든 관측치에 대한 로그 변환을 수행할 겁니다. 이렇게 하면 지하실이 있든 없든 간에 우리는 데이터를 변환할 수 있다.



#나는 이 방법이 옳은지 확신할 수 없다. 나한테는 딱 맞는 것 같았다. 그것이 바로 내가 '고위험 공학'이라고 부르는 것이다.
#create column for new variable (one is enough because it's a binary categorical feature)

#if area>0 it gets 1, for area==0 it gets 0

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)

df_train['HasBsmt'] = 0 

df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1



#transform data

df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])



#histogram and normal probability plot

sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

df_train
# 다중선형회귀가 가능하겠다고 판단

import pandas as pd

import statsmodels.formula.api as sm

from statsmodels.sandbox.regression.predstd import wls_prediction_std



# 회귀분석 수행

result = sm.ols(formula = 'SalePrice ~ GrLivArea + TotalBsmtSF + GarageArea + OverallQual', data = df_train).fit()



# 요약결과 출력

result.summary()



# OLS이해 - 보통최소제곱

# 실제값과 측정값의 최소제곱의 함이 최소가 되도록..

# https://terms.naver.com/entry.nhn?docId=3569970&cid=58944&categoryId=58970

 

# R squared 에 대한 이해

# https://jinchory.tistory.com/332



# 교제 4.2 다중선형회귀 p.156,158