from os import listdir, mkdir

from os.path import join, exists, split, splitext
listdir('../input/')
TEXT_PATH ='../input'
#本次实验的五种类别

categories = ['acad', 'fic', 'mag', 'news', 'spok'] 
file_names = listdir(TEXT_PATH)
splitext(file_names[1])[0].split('_')
def parse_coca_filename(fn):

    _, category, year = splitext(fn)[0].split('_')

    return category, int(year)
parse_coca_filename(file_names[0])
SPLIT_YEAR = 2000
periods = ('before_'+str(SPLIT_YEAR), 'since_'+str(SPLIT_YEAR)) 
def get_text(fn):    

    with open(fn) as f:

        text = f.read()

    return text
import re





def clean_text(text):



    #过滤HTML标签

    html_filted = re.sub('</?\w+[^>]*?>', '', text)



    #过滤其他非字母符号

    other_filted = re.sub('[\W|\d]', '', html_filted)



    #将所有字母统一为小写

    cleaned = other_filted.lower()



    return cleaned
t = get_text(join(TEXT_PATH,file_names[0]))
cleaned = clean_text(t)
def save_corpus():

    data = {

        c: {

            'before_' + str(SPLIT_YEAR): '',

            'since_' + str(SPLIT_YEAR): ''

        }

        for c in categories

    }



    for file_name in file_names:

        fn = join(TEXT_PATH, file_name)

        text = get_text(fn)

        text = clean_text(text)

        category, year = parse_coca_filename(file_name)

        if year <= SPLIT_YEAR:

            data[category]['before_' + str(SPLIT_YEAR)] += text

        else:

            data[category]['since_' + str(SPLIT_YEAR)] += text



    for c in categories:

        if not exists(c):

            mkdir(c)

        for period in ('before_' + str(SPLIT_YEAR), 'since_' + str(SPLIT_YEAR)):

            des_fn = join(c, period + '.txt')

            with open(des_fn, 'w') as f:

                f.write(data[c][period])
save_corpus()
def read_corpus():

    data = {}

    for category in categories:

        data[category] = {}

        for file_name in listdir(category): 

            period = splitext(file_name)[0]

            with open(join(category,file_name)) as f:

                data[category][period] = f.read()

    return data
data = read_corpus()
import numpy as np

import pandas as pd
#从一个长字符串text中随机采样一段length长度的连续子串作为样本

def sample_from_text(text, length):

    start = np.random.randint(len(text) - length)

    return text[start:start + length]
s = data['acad']['before_2000']
len(s)
sample_from_text(s, 500)
OFFSET = 3  #替代算法中的偏移量

NUMBER_OF_CHARACTERS = 26 #进行替代时支持的字符类别总数, 这里考虑分别在大小写各26个英文字符中做偏移替代
#替代加密算法

def encrypt(text):

    res = ''

    for c in text:

        if 'A' <= c <= 'Z':

            res += chr(

                ord('A') + (ord(c) - ord('A') + OFFSET) % NUMBER_OF_CHARACTERS)

        elif 'a' <= c <= 'z':

            res += chr(

                ord('a') + (ord(c) - ord('a') + OFFSET) % NUMBER_OF_CHARACTERS)

        else:

            res += c

    return res
s = data['news']['before_2000'] 
s[:100]
text = encrypt(s)
text[:100]
from collections import Counter    



def decrypt(text):    

    most_common_letter = pd.Series(Counter(text)).idxmax()

    offset = ord(most_common_letter) - ord('e')

    res = ''

    for c in text:

        res += chr(ord('a') + (ord(c)- ord('a') - offset)%NUMBER_OF_CHARACTERS)

    return res
decode = decrypt(text)
decode[:100]
decode == s
SAMPLE_SIZE = 200

NUMBER_OF_SAMPLES = 200
#绘制有效性曲线所选择的文本长度（横轴坐标）

np.concatenate((np.arange(100,1000,100),np.arange(1000, 5000, 200), np.arange(5000,11000,1000)))
length_sequences = np.concatenate((np.arange(100,1000,100),np.arange(1000, 5000, 200), np.arange(5000,11000,1000)))

result = []



for category in categories:

    for period in periods: #遍历每一类别的每一时间段

        

        #遍历选择的文本长度

        for length in length_sequences: 

            

            #重复进行NUMBER_OF_SAMPLES次实验

            for n in range(NUMBER_OF_SAMPLES):

                count = 0



                #获得SAMPLE_SIZE个样本，对其加密后再解密，计算解密正确率

                for i in range(SAMPLE_SIZE):

                    sample = sample_from_text(data[category][period], length)

                   #cipher = encrypt(sample)

                   #decipher = decrypt(cipher)



                   #if sample == decipher:

                    if Counter(sample).most_common(1)[0][0] == 'e':

                        count += 1

                        

                accuracy = count / SAMPLE_SIZE

                result.append([category, period, length, accuracy])

                

result = pd.DataFrame(result, columns=['category', 'period', 'length', 'accuracy'])

            
result
import seaborn as sns



sns.set_style('ticks')
sns.relplot(data=result, x='length', y='accuracy', row='period', col='category', kind='line')
grid = sns.relplot(data=result, x='length', y='accuracy', col='period', hue='category', kind='line')

# grid.axes[0,1].set_xlim(0, 5000)

# grid.axes[0,1].set_xlim(0, 500)
sns.relplot(data=result, x='length', y='accuracy', style='period', hue='period',col='category',col_wrap=2, kind='line')