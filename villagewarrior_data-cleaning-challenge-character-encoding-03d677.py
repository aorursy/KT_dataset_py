# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
# start with a string
before = "This is the euro symbol: €"

# check to see what datatype it is
type(before)
# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors = "replace")

# check the type
type(after)
# take a look at what the bytes look like
after
# convert it back to utf-8
print(after.decode("utf-8"))
# try to decode our bytes with the ascii encoding
try:
    print(after.decode("ascii"))
except Exception:
    print("'ascii' codec can't decode byte")
# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(
# Your turn! Try encoding and decoding different symbols to ASCII and
# see what happens. I'd recommend $, #, 你好 and नमस्ते but feel free to
# try other characters. What happens? When would this cause problems?

dollar_char_b = 'dollar $'
hash_char_b = 'hash #'
japanese = "你好"
hindi = "नमस्ते"

def encode_decode(str_):
    print('processing..: %s' %str_)
    dollar_char_a = str_.encode("ascii", errors= "replace")
    print("dollar char after: %s" %dollar_char_a)
    dollar_char_decode = dollar_char_a.decode("ascii")
    print("dollar char decoded: %s" %dollar_char_decode)
    print('-------------')

# call encode_decode for a list of strings defined above and return nothing
_ = [encode_decode(str_) for str_ in [dollar_char_b, hash_char_b, japanese, hindi]]

# try to read in a file not in UTF-8
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv")
# look at the first ten thousand bytes to guess the character encoding
with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)
# read in the file with the encoding detected by chardet
kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252')

# look at the first few lines
kickstarter_2016.head()
# Your Turn! Trying to read in this file gives you an error. Figure out
# what the correct encoding should be and read in the file. :)
try:
    police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
except Exception:
    print("exception while reading assuming UTF-8 encoding")
    with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))
    print(result)
    
# since we've figured out the encoding is Windows-1252
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("police_killings_utf8.csv")