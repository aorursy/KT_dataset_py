%%time
#  このセル（jupyter でのプログラムの実行単位です。）は、ライブラリを読み込み、sqlite データと接続して、一時的なテーブルを作成します。
# pandas というデータを扱うライブラリを読み込みます。
# このプログラムでは、データ操作はＳＱＬを通じて行うので、pandas は、グラフを書くために使うのが主です。
import pandas as pd
import numpy as np
import pandas.io.sql as psql

# sqliteを読み込むライブラリ
import sqlite3

# 線形回帰　使う頻度はあまりないですが、とりあえず、いれておきます。
from sklearn import linear_model
clf = linear_model.LinearRegression()



# HTMLで表示する　エクセルにコピペするときに便利
from IPython.display import display, HTML

# markdown 用
from tabulate import tabulate

# 日時を扱う
from datetime import datetime as dt
import time

# グラフ
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
%matplotlib inline

# system 関係のライブラリ
import sys
# os の機能を使うライブラリ
import os
# 正規表現
import re

# json,yaml 形式を扱う
import json
import yaml

# 変数の状態を調べる
import inspect

# 文字コード
import codecs

# Web からデータを取得する
import requests

# 貿易統計のデータ
# http://www.customs.go.jp/toukei/info/tsdl_e.htm
# コード　輸出は日本語のみ
# https://www.customs.go.jp/toukei/sankou/code/code_e.htm 

# sqlite に show tables がないので補足するもの
show_tables = "select tbl_name from sqlite_master where type = 'table'"
# describe もないで、補完します。
desc = "PRAGMA table_info([{table}])"
# メモリで、sqlite を使います。kaggle のスクリプト上では、オンメモリでないと新規テーブルがつくれません
# プログラムの一行が長いときは　\　で改行します。
conn = \
    sqlite3.connect(':memory:')

# sql を実行するための変数です。
cursor = conn.cursor()

# 1997 年から、2019 年までの年ベースのデータです。テーブルは、year_from_1997 
# year_from_1997
attach = 'attach "../input/japan-trade-statistics/y_1997.db" as y_1997'
cursor.execute(attach)

# 2018 年の月別集計 テーブル名も ym_2018 
attach = 'attach "../input/japan-trade-statistics/ym_2018.db" as ym_2018'
cursor.execute(attach)

# 2019 年の月別集計 テーブル名も ym_2019
attach = 'attach "../input/japan-trade-statistics/ym_2019.db" as ym_2019'
cursor.execute(attach)

# 2020 年の月別集計 テーブル名も ym_2020
attach = 'attach "../input/japan-trade-statistics/ym_2020.db" as ym_2020'
cursor.execute(attach)

# hs code,country,HSコードです。使いやすいように pandas　に変更しておきます。
attach = 'attach "../input/japan-trade-statistics/codes.db" as code'
cursor.execute(attach)
# import hs,country code as pandas
tmpl = "{hs}_{lang}_df =  pd.read_sql('select * from code.{hs}_{lang}',conn)"
for hs in ['hs2','hs4','hs6','hs6','hs9']:
    for lang in ['jpn','eng']:
        exec(tmpl.format(hs=hs,lang=lang))        

# 国コードも pandas で扱えるようにします。
# country table: country_eng,country_jpn
country_eng_df = pd.read_sql('select * from code.country_eng',conn)
country_eng_df['Country']=country_eng_df['Country'].apply(str)
country_jpn_df = pd.read_sql('select * from code.country_jpn',conn)
country_jpn_df['Country']=country_jpn_df['Country'].apply(str)

# custom  table: code.custom 税関別のコードです
custom_df = pd.read_sql('select * from code.custom',conn)
attach = 'attach "../input/japan-trade-statistics/custom_from_2012.db" as custom_from'
cursor.execute(attach)
attach = 'attach "../input/custom-2016/custom_2018.db" as custom_2018'
cursor.execute(attach)
attach = 'attach "../input/custom-2016/custom_2019.db" as custom_2019'
cursor.execute(attach)

#attach = 'attach "../input/japan-trade-statistics/custom_2020.db" as custom_2020'
#cursor.execute(attach)

# 計算時間を節約するために、年のデータから、2019 年を切り出します。
# 最初のはエラー処理です。y_2019 というテーブルが存在すると、新規に y_2019 を作ろうとするとエラーになります。
# error の場合は、何もせず、次にすすみます。
try:
    cursor.execute('drop table y_2019')
except:
    pass

# これからが、SQl になります。複数行で書くことが多いのでsql という変数に複数行を代入します。
# 最後の [1:-1] は、一行目（改行で空白）と最後の行（これも改行だけで空白）をとりのぞくためです。
# 0 から始まるので、1 だと、２行目から最後の行のひとつ手前までです。
sql = """
create table y_2019 
as select * from year_from_1997
where Year = 2019
"""[1:-1]
# 上記の sql を実行しして、2019 年のデータをつくります。
cursor.execute(sql)
conn.commit()

# sql の説明です。
# create table テーブル名 : テーブルを新規作成　ここでは、y_2019 
# as select * from  テーブル名　: テーブル名(year_from_1997)からつくります。
# where Year = 2019 : 2019 年のデータを指定します。Year は、数値なので、2019 と書きます。


# https://www.customs.go.jp/toukei/srch/index.htm?M=01&P=1,1,,,,,,,,4,1,2019,0,0,0,2,020230100,,,,,,,,,,6,120,,,,,,,,,,,,,,,,,,,,,20

# graph 用の　color　https://matplotlib.org/examples/color/named_colors.htmlsql_sample 
# ym_2018_2020 2018-2020 の月別集計のデータをまとめます。
# 年＋月のカラム(ym)をつくります。
# 年は、整数、月は、文字列なので、型変換と文字列結合を行います。
# CAST(Year AS  str )||month as ym がその処理をしている部分です。
# 文字列結合が、|| なので違和感ありますが、しょうがないです。

try:
    cursor.execute('drop table ym_2018_2020')
except:
    pass

sql = """
create table ym_2018_2020
as select CAST(Year AS  str )||month as ym,* from ym_2018
"""[1:-1]
cursor.execute(sql)

sql = """
insert into  ym_2018_2020
 select CAST(Year AS  str )||month as ym,* from ym_2019
"""[1:-1]
cursor.execute(sql)

sql = """
insert into  ym_2018_2020
 select CAST(Year AS  str )||month as ym,* from ym_2020
"""[1:-1]
cursor.execute(sql)

conn.commit()
# 便利なクラス（sql 実行 + グラフ）ut.関数名で使います。
class util():
    def sql(self,sql):
        return(pd.read_sql(sql,conn))
 
    # グラフ作成 一系列のみ
    def g1(self,df,x,y,color='b'):
        plt.figure(figsize=(20, 10))

        ax = sns.lineplot(x=x,y=y,data=df,linewidth=7.0,color=color)
        # これは、x軸（時系列）の単位が省略されないようにする設定
        # 何もしないと、2000,2005,2010のように一年分がとばされてしまいます。
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) 
        
    # グラフ作成 2系列　輸出入　比較につかいます。輸出が青、輸入が赤
    def g2(self,df,x,y,hue,palette={1: "b", 2: "r"}):
        plt.figure(figsize=(20, 10))
        ax  = sns.lineplot(x=x,y=y,hue=hue,linewidth = 7.0,
             palette=palette,
             data=df)
        # 凡例の位置　２は左上
        ax.legend_._loc = 2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))    
        
    # 複数系列のグラフ
    def gx(self,df,x,y,hue,palette={}):
        plt.figure(figsize=(20, 10))
        if palette == {}:
            ax  = sns.lineplot(x=x,y=y,hue=hue,linewidth = 7.0,data=df)
        else:
            ax  = sns.lineplot(x=x,y=y,hue=hue,linewidth = 7.0,palette=palette,data=df)
        # 凡例の位置　２は左上
        ax.legend_._loc = 2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))    
        
    def bar(self,df,y,x,prefix='',color='b'):
        # 色見本
        #https://matplotlib.org/examples/color/named_colors.html
        # 色の意味　合計: gold     輸出: b (blue) 輸入: r ( red )　をつかっています。
        if len(prefix) > 0:
            df[y] = df[y].map(lambda x: 'hs' + str(x))
        ax = sns.barplot(y=y, x=x, data=df,color=color)
        plt.show()
        plt.close()


    # 輸出入コードのurl を表示する
    # 輸入 
    def hs_url(self,hs_code,exp_imp=2,yyyy_mm='2020_4'):
        # 輸出 https://www.customs.go.jp/yusyutu/index.htm
        # 輸入 https://www.customs.go.jp/tariff/index.htm
        hs = hs_code[0:2]
    
        if exp_imp == 1:
            ex = 'yusyutu'
        else:
            ex = 'tariff'
        
        tmpl = 'https://www.customs.go.jp/{ex}/{yyyy_mm}/data/print_j_{hs}.htm'
        print(tmpl.format(ex=ex,yyyy_mm=yyyy_mm,hs=hs))

    # db との接続 conn は、global 変数として使う 
    def hs_table_create(self,hs_code,tables=['y_2019','year_from_1997','ym_2018_2020']):
        
        if len(hs_code) not in (2,4,6,9):
            print(hs_code + ': 桁数がおかしいです。')
            return
        
        hs = 'hs' + str(len(hs_code))
        
        sql = """
        create table hs{hs_code}_{table}
        as select * from {table}
        where {hs} = '{hs_code}'
        """[1:-1]
        
        for table in tables:
            tg = 'drop table hs{hs_code}_{table}'.format(hs_code=hs_code,table=table)
            print(tg)
            try:
                cursor.execute(tg)
            except:
                pass
            cursor.execute(sql.format(hs=hs,hs_code=hs_code,table=table))

        conn.commit()
        
    def hs_name_get(self,hs_code):
        hs = len(hs_code)
        if hs not in (2,4,6,9):
            print('HS コードの長さがまちがっています。 ' + str(hs))
        hs = str(hs)
        print(hs_code)
        text = 'hs' + hs + '_eng_df.query(' +"'"+ 'hs' + hs + '=="' + hs_code + '"' + "')"
        df = eval(text)
        print(df['hs' + hs + '_name'].values[0])
        text = 'hs' + hs + '_jpn_df.query(' +"'"+ 'hs' + hs + '=="' + hs_code + '"' + "')"
        df = eval(text)
        print(df['hs' + hs + '_name'].values[0])

            
        

    #  国コード(複数) のデータを抽出　国コードは、文字列のはずだが、ときどきなる整数になるので注意
    def countries_table_create(self,countries=['105','304','103','106','601'],tables=['y_2019','year_from_1997','ym_2018_2020']):
        clist = "('" + "','".join(countries) + "')" 
        sql = """
        create table countries_{table}
        as select * from {table}
        where Country in {clist}
        """[1:-1]
        
        for table in tables:
            tg = 'drop table countries_{table}'.format(table=table)
            print(tg)
            try:
                cursor.execute(tg)
            except:
                pass
            cursor.execute(sql.format(clist=clist,table=table))

        conn.commit()
        
    # 国別折れ線グラフのときに、国与える色です。
    def national_colors(self):
        return ({'105': ['中国', 'gold'],
        '304': ['アメリカ', 'red'],
         '103': ['韓国', 'blue'],
         '106:': ['台湾', 'cyan'],
         '601:': ['オーストラリア', 'green'],
         '111:': ['タイ', 'violet'],
         '213:': ['ドイツ', 'lightgrey'],
         '110:': ['ベトナム', 'crimson'],
         '108:': ['香港', 'orangered'],
         '112:': ['シンガポール', 'aqua'],
         '147:': ['アラブ首長国連邦', 'black'],
         '137:': ['サウジ', 'darkgreen'],
         '118:': ['インドネシア', 'darkorange'],
         '113:': ['マレーシア', 'yellow'],
         '205:': ['イギリス', 'darkblue'],
         '224:': ['ロシア', 'pink'],
         '117:': ['フィリピン', 'olive'],
         '302:': ['カナダ', 'salmon'],
         '210:': ['フランス', 'indigo'],
         '305:': ['メキシコ', 'greenyellow']})
    
    # 虹の7色を割り当てる
    def rank_color(self,xlist):
        clist = ['red','ornage','yellow','green','blue','indigo','violet']
        palette = {xlist[i]:clist[i] for i in range(len(xlist))}
        return(palette)


ut = util()
hs_code = '230910'
ut.hs_table_create(hs_code)
# 輸出入を hs_code で、年 比較　
table = 'year_from_1997'
sql = """
select Year,exp_imp,sum(Value) as Value 
from hs{hs_code}_{table}
group by Year,exp_imp
"""[1:-1]
df = pd.read_sql(sql.format(hs_code=hs_code,table = table),conn)
ut.g2(df,'Year','Value','exp_imp')
# 輸出入を hs_code で、年月比較　2月にはおちていますが、3月にはあがっています。
table = 'ym_2018_2020'
sql = """
select ym,exp_imp,sum(Value) as Value 
from hs{hs_code}_{table}
group by ym,exp_imp
"""[1:-1]
df = pd.read_sql(sql.format(hs_code=hs_code,table = table),conn)
ut.g2(df,'ym','Value','exp_imp')

# HS 9 桁別を、最新年で集計

table = 'y_2019'

sql = """
select hs9,exp_imp,sum(Value) as Value 
from hs{hs_code}_{table}
group by hs9,exp_imp
order by exp_imp,Value desc
"""[1:-1]
df = pd.read_sql(sql.format(hs_code=hs_code ,table=table),conn)
ut.hs_url(hs_code)
df


# 230910091(気密容器にはっている) に限定して調べます。
# 輸入の国別ランキングを出します。
table = 'y_2019'

sql = """
select y.Country,Country_name,sum(Value) as Value 
from hs{hs_code}_{table} y,country_eng c
where hs9='230910091' and
exp_imp = 2 and
y.Country = c.Country 
group by y.Country
order by Value desc
"""[1:-1]
df = pd.read_sql(sql.format(hs_code=hs_code ,table=table),conn)



ut.bar(df,'Country_name','Value',color='r')
df.head(10)
#  HS 230910091 輸入 exp_imp = 2
country = '111' # タイ
table = 'year_from_1997'
sql = """
select Year,sum(Value) as Value
from hs{hs_code}_{table}
where 
hs9='230910091' and
Country = '{country}' and
exp_imp = 2 
group by Year
order by Value desc
"""[1:-1]
df = pd.read_sql(sql.format(hs_code=hs_code ,table=table,country=country),conn)
ut.g1(df,'Year','Value','r')

#   HS 230910091 輸入 exp_imp = 2
country = '111'
table = 'ym_2018_2020'
sql = """
select ym,sum(Value) as Value
from hs{hs_code}_{table}
where 
hs9='230910091' and
Country = '{country}' and
exp_imp = 2 
group by ym
order by Value desc
"""[1:-1]
df = pd.read_sql(sql.format(hs_code=hs_code ,table=table,country=country),conn)
ut.g1(df,'ym','Value','r')
