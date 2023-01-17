import pandas 
null = pandas.read_csv('../input/null-and-replace/NULL and replace.csv')
null.head()
null.fillna(0)
null.fillna({
    'b1-value':3,
    'b2-value':200,
    'b3-value':5,
})
null.fillna(method = "ffill")
null.fillna(method = "bfill")
null.fillna(method = "bfill",axis ="columns")
null.fillna(method = "ffill",axis ="columns")
null.fillna(method = "ffill",axis ="rows",limit = 1)
null.interpolate()
null.dropna()
null.dropna(how="all")
null.dropna(thresh=3)
null.replace(['jan','feb','march','apiri','may','jun','july','aguage','sem','oct','nov','dec'],[1,2,3,4,5,6,7,8,9,10,11,12])
df = null.fillna(method = "ffill")
df.loc[15,'month'] = 'napiri'
df