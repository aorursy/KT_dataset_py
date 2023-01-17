# 导入pandas、altair库
import pandas as pd
import altair as alt
# 读取数据集
ICU_csv_data = pd.read_csv("../input/xuehangicu/ICU.csv")
"""
Number-数据序号，ID-患者ID，Survive-1表示生存，0表示死亡，Age-年龄
AgeGroup-年龄阶层，Sex-性别，1表示男，0表示女，Infection-感染情况，1表示感染，0表示无
SysBP-心电监护，Pulse-脉搏，Emergency-紧急情况
"""
ICU_csv_data
# Survive生存排序，并增加索引index
Survive_sort=ICU_csv_data.sort_values(by='Survive')
Survive_sort['index'] = range(len(Survive_sort))
# Survive生存数据分布，鼠标悬停显示， 显示40人死亡，160人存活
alt.Chart(Survive_sort).mark_bar().encode(
    x=alt.X('Survive:Q', bin=alt.Bin(maxbins=2)),
    y='index:Q',
    color='Survive:N',
    tooltip=["Survive"]
)
# Age年龄数据分布，鼠标悬停显示信息
alt.Chart(ICU_csv_data).mark_point().encode(
x='Number',
y='Age',
color='Survive:N',
tooltip=["ID","Survive","Age"]
)
ICU_csv_data[ICU_csv_data.Age>60].shape[0]
# AgeGroup年龄分布数据，鼠标悬停显示信息
alt.Chart(ICU_csv_data).mark_point().encode(
x='Number',
y='AgeGroup',
color='AgeGroup:N',
tooltip=["ID","Survive","Age","AgeGroup"]
)
# 年龄排序，并增加索引in
Sex_sort=ICU_csv_data.sort_values(by='Sex')
Sex_sort['index'] = range(len(Sex_sort))
# Age年龄数据分布，鼠标悬停显示信息
alt.Chart(Sex_sort).mark_point().encode(
x='index',
y='Sex',
color='Sex',
tooltip=["ID","Survive","Age","AgeGroup","Sex"]
)
ICU_csv_data[ICU_csv_data.Sex==1].shape[0],ICU_csv_data[ICU_csv_data.Sex==0].shape[0]
alt.Chart(ICU_csv_data).mark_point().encode(
x='Number',
y='Age',
color='Infection:N',
tooltip=["ID","Survive","Age","Infection"]
)
Survive=ICU_csv_data[ICU_csv_data.Survive==0]
ICU_csv_data[ICU_csv_data.Infection==1].shape[0],ICU_csv_data[ICU_csv_data.Infection==0].shape[0],Survive[Survive.Infection==1].shape[0]
alt.Chart(ICU_csv_data).mark_point().encode(
x='Pulse',
y='SysBP',
color='Emergency:N',
tooltip=["ID","Survive","Age","Infection","Pulse","SysBP","Emergency"]
)
ICU_csv_data[ICU_csv_data.Pulse>140]
ICU_csv_data[ICU_csv_data.SysBP<90]