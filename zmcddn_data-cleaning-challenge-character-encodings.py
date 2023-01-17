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
print(after.decode("ascii"))
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
test_string = '$, #, 你好 and नमस्ते'
test_string_ascii = test_string.encode("ascii", errors = "replace")
test_string_ascii_decode = test_string_ascii.decode("ascii", errors = "replace")
print('Original string: [{}], encoding is {}\nASCII enconded string: [{}], encoding is {}\nASCII decoded string: [{}], encoding is {}'.format(
    test_string, type(test_string), test_string_ascii, type(test_string_ascii), test_string_ascii_decode, type(test_string_ascii_decode)))
# Try with UTF-8 enconding
test_string_utf8 = test_string.encode("UTF-8")
test_string_utf8_decode = test_string_utf8.decode("utf-8")
print('Original string: [{}], encoding is {}\nASCII enconded string: [{}], encoding is {}\nASCII decoded string: [{}], encoding is {}'.format(
    test_string, type(test_string), test_string_utf8, type(test_string_utf8), test_string_utf8_decode, type(test_string_utf8_decode)))
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
# police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv")
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as pk_rawdata:
    result2 = chardet.detect(pk_rawdata.read(100000))

# check what the character encoding might be
print(result2)
police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Windows-1252')
police_killings.sample(5)
# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 
police_killings.to_csv("police-killings-201801-utf8.csv")
with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", 'rb') as percent_rawdata:
    percent_rawdata.seek(0,2)
    total_num = percent_rawdata.tell()
    print(total_num)
    percent_rawdata.seek(0,0)
    result3 = chardet.detect(percent_rawdata.read(total_num//10))
print(result3)
data_path = "../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv"

def check_encoding(data_path, aggressive=False, print_detail=False):
    '''
    A function to check the file encoding with different portion of 
    the file data, and return the most possible encoding
    data_path = the load path of the data
    aggressive = set True to check the full range of percentage, but will be slower
    print_detail = set True to display the checking details
    '''
    percentage = [0.1, 0.25, 0.5, 0.75, 0.9]
    encoding_info = {}
    result_encoding = []
    with open(data_path, 'rb') as raw_data:
        raw_data.seek(0, 2) # Shift pointer to the end of file 
        total_num = raw_data.tell()  # Check file size
        
        for percent in percentage:
            raw_data.seek(0,0)  # Shift pointer to the beginning of file
            result = chardet.detect(raw_data.read(int(total_num*percent)))
            encoding_info[percent] = [result['encoding'], result['confidence']]
            result_encoding.append(result['encoding'])
            if not aggressive:
                if percent >= 0.5:
                    break

    if print_detail:
        for i in encoding_info:
            print('Using {}% of data, find encoding to be [{}] with confidence of {}%'.format(
                i*100, encoding_info[i][0], encoding_info[i][1]*100))
    
    from collections import Counter
    result_counts = Counter(result_encoding)

    return max(result_counts, key=result_counts.get)
        
print(check_encoding(data_path, aggressive=True, print_detail=True))