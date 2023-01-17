from functools import reduce
from lxml import etree
import numpy as np
import glob
from tqdm import tqdm as tqdm
import re
import os
SEC_FOLDER = '../input/files/files/'
HTML_FOLDER = '../input/html/html/'
DOC_FOLDER = '../input/docs/docs/'
BODY_FOLDER = '../input/body/body/'

for folder in [SEC_FOLDER, HTML_FOLDER, DOC_FOLDER, BODY_FOLDER]:
    if not os.path.isdir(folder):
        os.mkdir(folder)
filenames = list(map(lambda x: re.sub(SEC_FOLDER, '', x), glob.glob(SEC_FOLDER + '*.txt')))
class Doc:
    def __init__(self, filename):
        self.errors = False
        self.filename = filename
        self.load_attachments()
        
    def load_attachments(self):
        with open(SEC_FOLDER + self.filename, 'r') as f:
            file = f.read()
        header = re.search(r'<sec-header>.*?</sec-header>', file, re.DOTALL | re.IGNORECASE)
        if not header:
            header = re.search(r'<ims-header>.*?</ims-header>', file, re.DOTALL | re.IGNORECASE)
            if not header and not re.match(r'^<DOCUMENT>', file, flags= re.DOTALL | re.IGNORECASE | re.MULTILINE):
                raise Exception('Wrong file: ' + self.filename)
        if header:
            params = {}
            self.header = header.group()
            for line in self.header.split('\n'):
                split = line.split(':')
                if len(split) == 2 and split[0].strip() and split[1].strip():
                    params[split[0].strip()] = split[1].strip()
            self.params = {re.sub('[^a-z ]', '', k.lower()).replace(' ', '_'):v for (k, v) in params.items()}
            self.__dict__.update(self.params)
        self.docs = re.findall(r'<document>.*?</document>', file, re.DOTALL | re.IGNORECASE)
        self.parsed_docs = []
        for doc in self.docs:
            parsed = {}
            for line in doc.split('\n')[1:]:
                if line == "<TEXT>": break
                else:
                    try:
                        pair = re.search(r'^<(?P<k>[a-z]+)>(?P<v>.*)$', line.lower())
                        parsed[pair.group('k')] = pair.group('v')
                    except:
                        self.errors = True
                        print(line)
                        break
            body = re.search(r'<text>.*</text>', doc, re.IGNORECASE | re.DOTALL)
            html = re.search(r'<html>.*</html>', body.group(), re.IGNORECASE | re.DOTALL)
            try:
                if html:
                    parsed['doctype'] = 'html'
                    parsed['body'] = etree.HTML(html.group().replace('&', '&amp;').replace('<br>', ''))
                else:
                    parsed['doctype'] = 'xbrl'
                    parsed['body'] = etree.HTML(body.group())
            except Exception as err:
                print(body.group()[:100000])
                self.errors = True
                raise err
                break
            self.parsed_docs.append(parsed)
        self.parsed_docs = sorted(self.parsed_docs, key=lambda x: int(x['sequence']))
files = []
for a in tqdm(filenames[:10]):
    doc = Doc(a)
    if doc.errors:
        break
    else:
        files.append(doc)
#[[(i, j, y['doctype']) for j, y in enumerate(x.parsed_docs)] for i, x in enumerate(files)]
files[3].filename
print(files[-1].parsed_docs[0]['body'][1][906][0][0][0].text)
