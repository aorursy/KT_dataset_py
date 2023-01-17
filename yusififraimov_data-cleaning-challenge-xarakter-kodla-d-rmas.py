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

print(type(after)) #the type has changed from 'str' to 'bytes'



#Now let's play a little bit :)



dəyişən = 'Python variables can be named in Azerbaijani language!'

print (dəyişən) 



#dəyişən - variable
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



'''The problem with Chinese characters,for example, would be that they are not a part of standard

ASCII encodings as they have their own specific encodings whic cannot be interpreted if we just try to use 

the standard ASCII encoding

The problem with Azerbaijani characters is quite more interesting. Basically, if you have noticed some characters are exactly the same as in English...

'''



some_senetnce_in_Azerbaijani = 'Python mənim ən sevdiyim proqramlaşdırma dillərindən biridir ' #...However, there are unique characters like - ç,ə,ğ,ö,ü,ı which are not part of standard ASCII encodings...



encodedStr=some_senetnce_in_Azerbaijani.encode('ascii',errors='replace')



print(encodedStr)

print(encodedStr.decode("ascii"))



encodedStr=some_senetnce_in_Azerbaijani.encode('UTF-8',errors='replace')



print(encodedStr)

print(encodedStr.decode("UTF-8")) #...but they are part of UTF-8 encodings.









#As mentioned above, if try to encode Chinese characters with UTF-8 or even ASCII it won't work properly.



some_word_in_chinese = '小人'



encodedStr=some_word_in_chinese.encode('ascii',errors='replace')



print(encodedStr) #chinese symbols are not part of ASCII encoding

print(encodedStr.decode("ascii")) #so we receive '??' 



encodedStr=some_word_in_chinese.encode('UTF-8',errors='replace')



print(encodedStr) #but they are part of UTF-8 encoding

print(encodedStr.decode("UTF-8")) #there are a lot of articles on this topic. It is quite interesting how UTF-8 processes chinese symbols
# try to read in a file not in UTF-8

kickstarter_2016 = pd.read_csv("../input/kickstarter-projects/ks-projects-201612.csv", encoding='Windows-1252') 
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

'''Because the error message was about a particular character not being parsed,

I tried to google some parts of the file and found out some symbols are part of Latin-1 character set.



I just tried to use .encoding('Latin-1') and that have worked out :)



In this case though,the chardlet guess is wrong as it suggests that the character encoding is in ascii but actually

the file is encoded in Latin-1 encoding.

'''



with open("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",'rb') as data:

    result = chardet.detect(data.read(10000))



print(result)



police_killings = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv",encoding="Latin-1")

#There is no error occuring at this step so that is most probably the correct encoding :)

# save our file (will be saved as UTF-8 by default!)

kickstarter_2016.to_csv("ks-projects-201801-utf8.csv")
# Your turn! Save out a version of the police_killings dataset with UTF-8 encoding 



pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding='Latin-1').to_csv("PoliceKillingsUS-utf8.csv")