import re
string = "Python java c++ sql"
pattern = 'Python'
match = re.match(pattern,string)
match
match.group(0)
match.span()
string = "c Python java c++ sql"
pattern = 'c'
match2 = re.match(pattern,string)
match2.group(0)
string = "Python java c sql"
pattern = 'c'
search= re.search(pattern,string,re.IGNORECASE)
search.group(0)
string = "Python java c++ sql python "
pattern = 'Python'
search = re.search(pattern,string,re.IGNORECASE)
search.group(0)
string = "Python java c++ sql Python "
pattern ='Python'
findall = re.findall(pattern,string)
findall
string ="java is easy learn languages"
pattern ="python"
sub=re.sub("java",pattern,string)
sub
string ="java is easy learn languages"
split = re.split("\s", string)
split
string ="java is easy learn languages"
split = re.split("\s", string,2)
split
string = "name and regno:venkateshwaran-105"
compile = re.compile(r'[0-9][0-9][0-9]')
compile_result=compile.search(string)
compile_result.group()
a= "venkateshwaran-105"
v = re.findall("\d",a)
v
b= "venkateshwaran-105"
v = re.findall("\D",b)
v
c="venkatesh waran k"
v = re.findall("\S", c)
v
d= "venkateshwaran - 105"
v = re.findall("\w", d)
v
e="venkatesh waran k"
v=re.findall("k\Z", e)
v
f="venkatesh waran k"
v=re.findall("\Avenkatesh", f)
v
a="venkatesh waran k"
v = re.findall("v.......h", a)
v
a="venkatesh waran k venky vicky"
v= re.findall("ve*", a)
v
a="venkatesh waran k venky vicky"
v= re.findall("venkatesh | venky", a)
v
f="venkatesh waran k"
v=re.findall("^venkatesh", f)
v
e="venkatesh waran k"
v=re.findall("k$", e)
v
a="venkateshwaran +++ 16104105 "
pattern = re.compile('[a-zA-Z0-9.]+')
res=re.findall(pattern,a)
res="".join(res)
res
a="venkateshwaran 16104105 "
pattern = re.compile('[^0-9 ]')
res=re.findall(pattern,a)
res="".join(res)
res