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
print(before)
print(after)
# take a look at what the bytes look like
after
# convert it back to utf-8
print(after.decode("utf-8"))
# try to decode our bytes with the ascii encoding
print(after.decode("ascii"))
# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")
print(after)
# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been 
# replaced with the underlying byte string for the unknown character :(
# Your turn! Try encoding and decoding different symbols to ASCII and
# see what happens. I'd recommend $, #, 你好 and नमस्ते but feel free to
# try other characters. What happens? When would this cause problems?
dollar = "$"
dollar_ascii = dollar.encode("ascii", errors = "replace")
print(dollar_ascii)
print(dollar_ascii.decode("ascii"))
hashtag = "#"
hashtag_ascii = hashtag.encode("ascii", errors = "replace")
print(hashtag_ascii)
print(hashtag_ascii.decode("ascii"))
chinese = "你好"
chinese_ascii = chinese.encode("ascii", errors = "replace")
print(chinese_ascii)
print(chinese_ascii.decode("ascii"))
chinese = "你好"
chinese_utf = chinese.encode("utf-8", errors = "replace")
print(chinese_utf)
print(chinese_utf.decode("utf-8"))
hindi = "नमस्ते"
hindi_ascii = hindi.encode("ascii", errors = "replace")
print(hindi_ascii)
print(hindi_ascii.decode("ascii"))
hindi = "नमस्ते"
hindi_utf = hindi.encode("utf-8", errors = "replace")
print(hindi_utf)
print(hindi_utf.decode("utf-8"))
o = "ö"
o_ascii = o.encode("ascii", errors = "replace")
print(o_ascii)
print(o_ascii.decode("ascii"))
o = "ö"
o_utf = o.encode("utf-8", errors = "replace")
print(o_utf)
print(o_utf.decode("utf-8"))
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
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", "rb") as police:
    result = chardet.detect(police.read(50000))
    
print(result)
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding = "Windows-1252")
police_killings.head()
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("police_killings-utf8.csv")