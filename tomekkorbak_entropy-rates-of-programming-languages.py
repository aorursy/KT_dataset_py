from collections import defaultdict, deque, Counter
import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.cloud import bigquery
import warnings
warnings.filterwarnings('ignore')

sns.set()
%matplotlib inline
client = bigquery.Client()
def get_files(language_extension):
    QUERY = ('''
    SELECT SPLIT(content, '\\n') AS line
    FROM `bigquery-public-data.github_repos.sample_contents`
    WHERE sample_path LIKE "%.{ext}"
    
    '''.format(ext=language_extension))
    query_job = client.query(QUERY)
    bigquery_iterator = iter(query_job.result(timeout=120))
    return bigquery_iterator
bigquery_iterator = get_files('py')
next(bigquery_iterator)[0] 
class MarkovModel(object):
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(Counter)
        self.freqs = Counter()
        self.buffer = deque(maxlen=self.n)
    
    def fit(self, stream):
        for token in stream:
            prefix = tuple(self.buffer)
            self.buffer.append(token)
            if len(prefix) == self.n:
                self.freqs[prefix] += 1
                self.model[prefix][token] += 1
    
    def entropy(self, prefix):
        prefix_freqs = self.model[prefix].values()
        normalization_factor = self.freqs[prefix]
        return -np.sum(f/normalization_factor * np.log2(f/normalization_factor) 
                       for f in prefix_freqs)
                
    def entropy_rate(self):
        normalization_factor = sum(self.freqs.values())
        unnormalized_rate = np.sum(self.freqs[prefix] * self.entropy(prefix) for prefix in self.freqs)
        try:
            return unnormalized_rate/normalization_factor
        except ZeroDivisionError:
            return 0
def tokenizer(text):
    return text  # char-level tokens

def iterator(bigquery_iterator, items=10000):
    counter = 0
    while(counter < items):
        counter += 1
        file = next(bigquery_iterator)[0]
        for line in file:
            if line == '':
                continue
            for token in tokenizer(line.lower().strip()):
                yield token
%%time
py_files = list(iterator(get_files('py')))
cpp_files = list(iterator(get_files('cpp')))
js_files = list(iterator(get_files('js')))

rates = pd.DataFrame(columns=['n', 'language', 'entropy_rate'])
for language, it in [('py', py_files), ('js', js_files), ('cpp', cpp_files)]:
    for n in range(5):
        model = MarkovModel(n=n)
        model.fit(stream=it)
        rate = model.entropy_rate()
        print(n, language, rate)
        rates.loc[len(rates)] = [n, language, rate]
rates
rates.n = np.float32(rates.n)
f, (ax1, ax2, ax3) = plt.subplots(3, sharey=True, figsize=(15,15))
ax1.set_ylim(0, 8)
sns.regplot(x='n', y='entropy_rate', data=rates[rates.language=='py'], ax=ax1, order=2).set_title("Python")
sns.regplot(x='n', y='entropy_rate', data=rates[rates.language=='cpp'], ax=ax2, order=2).set_title("C++")
sns.regplot(x='n', y='entropy_rate', data=rates[rates.language=='js'], ax=ax3, order=2).set_title("JavaScript")
plt.subplots_adjust(hspace=0.3)
