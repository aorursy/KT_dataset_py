# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd
!wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv

!wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv

!wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv
!ls
#read data and table cleaning process

conf_df = pd.read_csv('time_series_19-covid-Confirmed.csv')

deaths_df = pd.read_csv('time_series_19-covid-Deaths.csv')

recv_df = pd.read_csv('time_series_19-covid-Recovered.csv')



dates = ['1/22/20', '1/23/20', '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', 

         '1/29/20', '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', 

         '2/5/20', '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20', 

         '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20', '2/19/20',

         '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20', '2/25/20', '2/26/20',

         '2/27/20', '2/28/20', '2/29/20', '3/1/20', '3/2/20', '3/3/20', '3/4/20']



conf_df_long = conf_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Confirmed')



deaths_df_long = deaths_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Deaths')



recv_df_long = recv_df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], 

                            value_vars=dates, var_name='Date', value_name='Recovered')



full_table = pd.concat([conf_df_long, deaths_df_long['Deaths'], recv_df_long['Recovered']], 

                       axis=1, sort=False)

full_table.head()



# converting to proper data format

full_table['Date'] = pd.to_datetime(full_table['Date'])

full_table['Recovered'] = full_table['Recovered'].astype('int')



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values with 0 in columns ('Confirmed', 'Deaths', 'Recovered')

full_table[['Confirmed', 'Deaths', 'Recovered']] = full_table[['Confirmed', 'Deaths', 'Recovered']].fillna(0)

full_table[['Province/State']] = full_table[['Province/State']].fillna('NA')



# full table

full_table.head()
full_table['Country/Region'].unique() 
full_table["Country/Region"]= full_table["Country/Region"].replace({ 

    'China' : '중국', 'Thailand' : '태국', 'Japan' : '일본', 'South Korea' : '대한민국', 'Taiwan' : '대만', 'US' : '미국'

  , 'Macau' : '마카오', 'Hong Kong' : '홍콩', 'Singapore' : '싱가포르', 'Vietnam' : '베트남', 'France' : '프랑스'

  , 'Nepal' : '네팔', 'Malaysia' : '말레이시아', 'Canada' : '캐나다', 'Australia' : '호주', 'Cambodia' : '캄보디아'

  , 'Sri Lanka' : '일본', 'Germany' : '독일', 'Finland' : '핀란드', 'United Arab Emirates' : '아랍에미리트'

  , 'Philippines' : '필리핀', 'India' : '인도', 'Italy' : '일본', 'UK' : '영국', 'Russia' : '러시아', 'Sweden' : '스웨덴'

  , 'Spain' : '스페인', 'Belgium' : '벨기에', 'Egypt' : '이집트', 'Iran' : '이란', 'Others' : '기타운송수단'

  , 'Israel' : '이스라엘', 'Lebanon' : '레바논', 'Iraq' : '이라크', 'Oman' : '오만', 'Afghanistan' : '아프가니스탄'

  , 'Bahrain' : '바레인', 'Kuwait' : '쿠웨이트',  'Algeria': '알제리',

       'Croatia' : '크로아티아', 'Switzerland' : '스위스', 'Austria' : '오스트리아', 'Pakistan':'파키스탄', 'Brazil':'브라질',

       'Georgia':'조지아', 'Greece':'그리스', 'North Macedonia' : '설명북마케도니아', 'Norway':'노르웨이', 'Romania':'로마니아',

       'Denmark':'덴마크', 'Estonia':'에스토니아', 'Netherlands':'네덜란드', 'San Marino':'설명산마리노', 'Belarus':'벨라루스',

       'Iceland':'아이슬란드', 'Lithuania':'리투아니아', 'Mexico':'멕시코', 'New Zealand':'뉴질랜드', 'Nigeria':'나이지리아',

       'Ireland':'아일랜드', 'Luxembourg':'룩섬부르크', 'Monaco':'모나코', 'Qatar':'카타르', 'Ecuador':'에콰도르',

       'Azerbaijan':'아제르바이잔', 'Czech Republic':'체코', 'Armenia':'아르마니아', 'Dominican Republic':'도미니카공화국',

       'Indonesia':'인도네시아', 'Portugal':'포르투갈', 'Andorra':'안도라', 'Latvia':'라트비아', 'Morocco':'모로코',

       'Saudi Arabia':'사우디아라비아', 'Senegal':'세네갈', 'Argentina':'아르헨티나', 'Chile':'칠래', 'Jordan':'요르단',

       'Ukraine':'우크레이나', 'Saint Barthelemy':'설명생바르텔레미', 'Hungary':'헝가리', 'Faroe Islands':'페로 제도',

       'Gibraltar':'지브롤터', 'Liechtenstein':'설명리히텐슈타인', 'Poland':'폴란드', 'Tunisia':'투니시아'})

full_table['Country/Region'].unique() 
full_table['Province/State'].unique() 
full_table["Province/State"]= full_table["Province/State"].replace({

       'Anhui' : '안후이성', 'Beijing' : '베이징시', 'Chongqing' : '충칭시', 'Fujian' : '푸젠성'

     , 'Gansu' : '간쑤성', 'Guangdong' : '광둥성', 'Guangxi' : '광시 좡족 자치구', 'Guizhou' : '구이저우성'

     , 'Hainan' : '하이난성', 'Hebei' : '허베이성', 'Heilongjiang' : '헤이룽장성', 'Henan' : '허난성'

     , 'Hubei' : '후베이성', 'Hunan': '후난성', 'Inner Mongolia' : '내몽골 자치구', 'Jiangsu' : '장쑤성'

     , 'Jiangxi' : '장시성', 'Jilin' : '지린성',  'Liaoning' : '랴오닝성', 'Ningxia' : '닝샤 후이족 자치구'

     , 'Qinghai' : '칭하이성', 'Shaanxi' : '산시성', 'Shandong' :'산둥성','Shanghai' : '상하이시'

     , 'Shanxi' : '산시성', 'Sichuan' : '쓰촨성', 'Tianjin' : '톈진시', 'Tibet' : '티베트 자치구', 'Xinjiang' : '신장 위구르 자치구'

     , 'Yunnan' : '윈난성', 'Zhejiang' : '저장성', 'Taiwan' : '대만', 'Seattle, WA' : '시애틀, WA', 'Chicago, IL' : '시카고, IL'

     , 'Tempe, AZ' : '템피, 아리조나', 'Macau' : '마카오', 'Hong Kong' : '홍콩', 'Toronto, ON' : '토론토, ON'

     , 'British Columbia' : '브리티시컬럼비아 주', 'Orange, CA' : '오렌지, CA', 'Los Angeles, CA' : '로스앤젤레스, CA'

     , 'New South Wales' : '뉴사우스웨일스', 'Victoria' : '빅토리아', 'Queensland' : '퀸즐랜드 주', 'London, ON': '런던, ON'

     , 'Santa Clara, CA' : '샌타클래라, CA', 'South Australia' : '사우스오스트레일리아', 'Boston, MA' : '보스턴, MA'

     , 'San Benito, CA' : '샌 베니토, CA', 'Madison, WI' : '매디슨, WI', 'Diamond Princess cruise ship' : '다이아몬드 프린세스호'

     , 'San Diego County, CA' : '샌디에이고 군', 'San Antonio, TX' : '샌안토니오, TX' 

     , 'Omaha, NE (From Diamond Princess)' : '오마하, NE (다이아몬드 프린세스호)', 'Sacramento County, CA' : '새크라멘토 카운티, CA'

     , 'Travis, CA (From Diamond Princess)' : '트래비스 공군기지, CA (다이아몬드 프린세스호)'

     , 'From Diamond Princess' : '다이아몬드 프린세스호', 'Humboldt County, CA' : '훔볼트 카운티, CA'

     , 'Lackland, TX (From Diamond Princess)' : '렉랜드 공군기지, TX (다이아몬드 프린세스호)'

     , 'Unassigned Location (From Diamond Princess)' : '위치미정 (다이아몬드 프린세스호)', 

       ' Montreal, QC':'몬트리올, QC', 'Western Australia':'설명웨스턴오스트레일리아주',

       'Snohomish County, WA':'스노호미시 군,WA', 'Providence, RI':'설명프로비던스, RI', 'Tasmania':'설명태즈메이니아주',

       'Grafton County, NH':'그라프턴 카운티, NH', 'Hillsborough, FL':'힐즈버러 카운티, FL', 'New York City, NY':'뉴옥, NY',

       'Placer County, CA':'플레이서 카운티, CA', 'San Mateo, CA':'설명샌머테이오, CA', 'Sarasota, FL':'설명새러소타, FL',

       'Sonoma County, CA':'소노마 카운티, CA', 'Umatilla, OR':'우마틸라, OR', 'Fulton County, GA':'풀턴 카운티, GA',

       'Washington County, OR':'워싱턴 카운티, OR', ' Norfolk County, MA': '노퍽 카운티, MA', 'Berkeley, CA':'버클리, CA',

       'Maricopa County, AZ':'설명매리코파 군, AZ', 'Wake County, NC':'웨이크 카운티, NC', 'Westchester County, NY':'웨스트체스터 군, NY',

       'Orange County, CA':'오렌지 카운티, CA', 'Northern Territory':'노던 준주',

       'Contra Costa County, CA':'설명컨트라코스타 카운티, CA'})

full_table['Province/State'].unique()
full_table.rename(columns={'Province/State' : '행정구역', 'Country/Region' : '국가/지역', 'Lat' : '위도', 'Long' : '경도'

                   , 'Date' : '날짜','Confirmed' :'확진자', 'Deaths': '사망자', 'Recovered' :'회복자'}, inplace=True)

full_table.head()
full_table[(full_table['행정구역'] == '위치미정 (다이아몬드 프린세스호)') |  (full_table['행정구역'] == '위치미정 (다이아몬드 프린세스호)') 

        |  (full_table['행정구역'] == '렉랜드 공군기지, TX (다이아몬드 프린세스호)') |  (full_table['행정구역'] == '트래비스 공군기지, CA (다이아몬드 프린세스호)') 

        |  (full_table['행정구역'] == '오마하, NE (다이아몬드 프린세스호)')]
full_table.to_csv("COVID-19_Korean.csv",index = False)