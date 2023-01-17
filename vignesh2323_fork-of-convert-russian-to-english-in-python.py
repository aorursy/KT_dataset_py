import pandas as pd # linear algebra
df=pd.read_csv("../input/items.csv")
symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
           u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA")
tr = {ord(a):ord(b) for a, b in zip(*symbols)}
def lang_convert(row):
    english_word=row.translate(tr)
    return english_word
df['item_name'] = df.apply(lambda row: lang_convert(row['item_name']), axis=1)
df.tail()