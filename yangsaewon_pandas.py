import numpy as np

import pandas as pd
df = pd.DataFrame({'A':['A1', 'A2', 'A3'],

                  'B':['B1', 'B2', 'B3'],

                  'C':['C1', 'C2', 'C3'],

                  'D':['D1', 'D2', 'D3']},

                  index=[0, 1, 2])
df
Series_1 = pd.Series(['S1', 'S2', 'S3'], name='S')

Series_2 = pd.Series([0, 2, 5])

Series_3 = pd.Series([1, 3, 2])
pd.concat([df, Series_1], axis=0)
# 만약 series에 name이 있으면 자동으로 Column name으로 사용

pd.concat([df, Series_1], axis=1)
# ignore_index=True 하면 column name 무시

pd.concat([df, Series_1], axis=1, ignore_index=True)
# Series끼리 concat할 경우 keys를 활용하여 column names를 덮어씌울 수 있다

pd.concat([Series_1, Series_2, Series_3], axis=1, keys=['C1', 'C2', 'C3'])
Series_4 = pd.Series(['Z1', 'Z2', 'Z3', 'Z4'], index=['A', 'B', 'C', 'E'])

Series_4
# append를 활용하여 DataFrame에 Series를 합칠 수 있다.

df.append(Series_4, ignore_index=True)
df_left = pd.DataFrame({'KEY': ['K1', 'K2', 'K3', 'K4'],

                        'A': ['A1', 'A2', 'A3', 'A4'],

                        'B': ['B1', 'B2', 'B3', 'B4']})



df_right = pd.DataFrame({'KEY': ['K3', 'K4', 'K5', 'K6'],

                        'C': ['C1', 'C2', 'C3', 'C4'],

                        'D': ['D1', 'D2', 'D3', 'D4']})
df_left
df_right
# left outer join (sql)

pd.merge(df_left, df_right, on='KEY', how='left')
# right outer join (sql)

pd.merge(df_left, df_right, on='KEY', how='right')
# inner join (sql)

pd.merge(df_left, df_right, on='KEY', how='inner')
# full outer join (sql)

pd.merge(df_left, df_right, on='KEY', how='outer')
# indicator를 통해서 출처 정보 확인 가능

pd.merge(df_left, df_right, how='outer', on='KEY', indicator=True)
# boolean이 아니라 변수이름을 설정 할 수도 있다

pd.merge(df_left, df_right, how='outer', on='KEY', indicator='indicator_info')
df_left = pd.DataFrame({'KEY': ['K1', 'K2', 'K3', 'K4'],

                        'A': ['A1', 'A2', 'A3', 'A4'],

                        'B': ['B1', 'B2', 'B3', 'B4'], 

                        'C': ['B1', 'B2', 'B3', 'B4']})



df_right = pd.DataFrame({'KEY': ['K1', 'K2', 'K3', 'K4'],

                        'B': ['B1', 'B2', 'B3', 'B4'],

                        'C': ['C1', 'C2', 'C3', 'C4'],

                        'D': ['D1', 'D2', 'D3', 'D4']})
df_left

df_right
# 변수 이름이 중복될 경우 접미사를 붙이기도 한다. suffixes = ('_X', '_y')가 default

pd.merge(df_left, df_right, how='inner', on='KEY')
pd.merge(df_left, df_right, how='inner', on='KEY', suffixes=['_left', '_right'])
df_left = pd.DataFrame({'A': ['A1', 'A2', 'A3', 'A4'],

                        'B': ['B1', 'B2', 'B3', 'B4']},

                        index=['K1', 'K2', 'K3', 'K4'])
df_right = pd.DataFrame({'C': ['C1', 'C2', 'C3', 'C4'],

                        'D': ['D1', 'D2', 'D3', 'D4']},

                        index=['K2', 'K3', 'K4', 'K5'])
df_left
df_right
# merge를 사용한 방법

pd.merge(df_left, df_right, left_index=True, right_index=True, how='left')
# join을 사용한 방법

df_left.join(df_right, how='left')
# inner join도 가능하다

pd.merge(df_left, df_right, how='inner', left_index=True, right_index=True)
# outer join

pd.merge(df_left, df_right, how='outer', left_index=True, right_index=True)
df_left_2 = pd.DataFrame({'KEY': ['K1', 'K2', 'K3', 'K4'],

                       'A': ['A1', 'A2', 'A3', 'A4'],

                       'B': ['B1', 'B2', 'B3', 'B4']})

df_right_2 = pd.DataFrame({'C': ['A1', 'A2', 'A3', 'A4'],

                       'D': ['B1', 'B2', 'B3', 'B4']},

                       index=['K2', 'K3', 'K4', 'K5'])
df_left_2
df_right_2
# left_df의 key 사용, right_df의 index 사용하여 merge

pd.merge(df_left_2, df_right_2, left_on='KEY', right_index=True, how='left')
# inner join

pd.merge(df_left_2, df_right_2, left_on='KEY', right_index=True, how='inner')
# outer join

pd.merge(df_left_2, df_right_2, left_on='KEY', right_index=True, how='outer')