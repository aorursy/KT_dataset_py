# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob # glob 모듈은 유닉스 셸이 사용하는 규칙에 따라 지정된 패턴과 일치하는 모든 경로명을 찾습니다
# Panda’s concat and append can do this for us. I’m going to use append in this example.
# The code snippet below will initialize a blank DataFrame then append all of the individual files into the all_data DataFrame.

all_data = pd.DataFrame()
for filename in glob.glob('/kaggle/input/koreandic/*.xls'):
    df = pd.read_excel(filename)
    all_data = all_data.append(df,ignore_index=True)
# examine the data

# all_data # print the first 30 and last 30 rows
# type(all_data) # DataFrame
# all_data.head() # print the first 5 rows
# all_data.tail() # print the last 5 rows
all_data.index # "the index" (aka "the labels")
# all_data.columns # column names (which is "an index")
# all_data.dtypes # data types of each column
all_data.shape # number of rows and columns
# all_data.values # underlying numpy array
all_data.info() # concise summary (includes memory usage as of pandas 0.15.0
# 전처리 (특수 문자 제거)
all_data['어휘'] = all_data['어휘'].str.replace(pat=r'[^ ㄱ-ㅣ가-힣]+', repl=r' ', regex=True)  # replace all special symbols to space

# print('추출 갯수: ',len(all_data))
# all_data.head()

# 전처리 (한 글자 제거)
mask_pre_short = all_data['어휘'].str.len() != 1
all_data = all_data.loc[mask_pre_short]
print('추출 갯수: ',len(all_data))
all_data.head()
mask11 = all_data['품사'] == '명사'
mask12 = all_data['품사'] == '대명사'
mask13 = all_data['품사'] == '의존 명사'
mask14 = all_data['품사'] == '감탄사'

# filter specific values(품사가 명사 및 대명사만 추출)
class_filter = all_data.loc[mask11 | mask12]
print('추출 갯수: ',len(class_filter))
class_filter.head()
# filter specific values(품사가 명사 및 대명사만 추출)
class_filter_01 = all_data.loc[mask11]
class_filter_02 = all_data.loc[mask12]
class_filter_03 = all_data.loc[mask11 | mask12]

print('전체 데이터 갯수: ',len(all_data))
print('명사 갯수: ',len(class_filter_01))
print('대명사 갯수: ',len(class_filter_02))
print('명사 및 대명사 갯수: ',len(class_filter_03))
print('')
print('품사별 갯수: ',all_data.groupby('품사').size())
mask21 = all_data['구성 단위'] == '단어'
mask22 = all_data['구성 단위'] == '구'

# filter specific values(구성 단위가 단어 및 구만 추출)
type_filter = all_data.loc[mask21 | mask22]
print('추출 갯수: ',len(type_filter))
type_filter.head()
# filter type values(구성 단위가 단어 또는 구만 추출)
type_filter_01 = all_data.loc[mask21]
type_filter_02 = all_data.loc[mask22]
type_filter_03 = all_data.loc[mask21 | mask22]



#print('전체 데이터 갯수: ',len(all_data))
print('단어 갯수: ',len(type_filter_01))
print('구 갯수: ',len(type_filter_02))
print('단어 구 갯수: ',len(type_filter_03))
print('')
print('구성 단위별 갯수: ',all_data.groupby('구성 단위').size())
mask31 = all_data['전문 분야'] == '『보건 일반』'
mask32 = all_data['전문 분야'] == '『의학』'
# mask33 = all_data['전문 분야'] == '한의'

# filter specific values(전문 분야가 보건 일반 및 의학만 추출)
specific_filter_01 = all_data.loc[mask31 | mask32]
print('추출 갯수: ',len(specific_filter_01))
specific_filter_01.head()
mask33 = all_data['전문 분야'].str.contains('일반') == True

# filter specific values(전문 분야가 보건 일반 및 의학만 추출)
specific_filter_02 = all_data.loc[mask31 | mask32 | mask33 ]
print('추출 갯수: ',len(specific_filter_02))
specific_filter_02.head()
# NaN이 하나라도 들어간 행 제외
mask34 = all_data['전문 분야'].notnull() == True

# filter specific values(전문 분야가 보건 일반 및 의학만 추출)
specific_filter_03 = all_data.loc[mask34]
print('추출 갯수: ',len(specific_filter_03))
specific_filter_03.head()
#print('전체 데이터 갯수: ',len(all_data))
#print('보건 일반 갯수: ',len(specific_filter_01))
#print('의학 갯수: ',len(specific_filter_02))
#print('보건 일반 및 의학 갯수: ',len(specific_filter_03))
#print('')
#print('전문 분야별 갯수: ',all_data.groupby('전문 분야').size())
# 명사/대명사, 단어, 보건 일반/의학
total_filter_01 = all_data.loc[(mask11 | mask12) & mask21  & (mask31 | mask32)]
# 명사/대명사, 단어, 보건 일반/의학/*일반
total_filter_02 = all_data.loc[(mask11 | mask12) & mask21  & (mask31 | mask32 | mask33)]
# 명사/대명사, 단어, 전문분야 전체
total_filter_03 = all_data.loc[(mask11 | mask12) & mask21  & mask34]

#print('전체 데이터 갯수: ',len(all_data))
print('필터 01: ',len(total_filter_01))
total_filter_01.head()

#print('전체 데이터 갯수: ',len(all_data))
print('필터 02: ',len(total_filter_02))
total_filter_02.head()

#print('전체 데이터 갯수: ',len(all_data))
print('필터 03: ',len(total_filter_03))
total_filter_03.head()
#total_filter_01.to_excel('keywords_01.xls')
#total_filter_02.to_excel('keywords_02.xls')
total_filter_03.to_excel('keywords_03.xlsx')