# Import libs

from numpy import nan

import pandas as pd

import re



# Set width of display

print('display.width:', pd.get_option('display.width'))

pd.set_option('display.width', 120)

print('display.width:', pd.get_option('display.width'))



# Get data

moma = pd.read_csv('/kaggle/input/museum-of-modern-art-collection/Artworks.csv')

moma.head()
moma.info()
# Number of null values per column

moma_nan_cnt = moma[['Title', 'Artist', 'ArtistBio', 'Nationality', 'BeginDate', 'EndDate',

                     'Gender', 'Date', 'Department']].isnull().sum()



# Percentage of null values

moma_nan_pct = moma_nan_cnt / moma.shape[0] * 100



moma_nan = pd.DataFrame({'nan_count': moma_nan_cnt, 'nan_percentage': moma_nan_pct})

moma_nan
# Duplicated

print('Duplicated: {}'.format(moma.duplicated(subset='ObjectID').sum()))
artist_bio_pattern_org = r'(?:founded|established|est\.|active|formed)'



# Value examples

moma.loc[((moma['ArtistBio'].notnull())

          & (moma['ArtistBio'].str.contains(artist_bio_pattern_org, flags=re.I))

         ),

         ['Title', 'Artist', 'ConstituentID', 'ArtistBio',

          'Nationality', 'BeginDate', 'EndDate', 'Gender', 'Date'

         ]

        ].head(15)
# Value examples

moma['Gender'].value_counts(dropna=False).head(15)
gender_pattern_arr = r'(?:\((?:male|female)?\))'



# Value examples

moma.loc[((moma['Gender'].notnull())

          & (moma['Gender'].str.count(gender_pattern_arr, flags=re.I) > 1)

         )

        ]
# Value examples

moma.iloc[[101220, 63914, 131800, 136750], 0:20]
# Value examples

print(moma['BeginDate'].value_counts(dropna=False).sort_index().tail(40),

      moma['BeginDate'].value_counts(dropna=False).sort_index().head(40).index,

      sep='\n'

     )
# Value examples

print(moma['EndDate'].value_counts(dropna=False).sort_index().tail(40),

      moma['EndDate'].value_counts(dropna=False).sort_index().head(40).index,

      sep='\n'

     )
# Value examples

print(moma['Date'].value_counts(dropna=False).sort_index())
print('Before drop:', moma.columns, sep='\n', end='\n\n')



# List to drop

drop_cols = ['ConstituentID', 'Medium', 'Dimensions', 'CreditLine', 'AccessionNumber', 'DateAcquired',

             'Cataloged', 'ObjectID', 'URL', 'ThumbnailURL', 'Circumference (cm)',

             'Depth (cm)', 'Diameter (cm)', 'Height (cm)', 'Length (cm)', 'Weight (kg)', 'Width (cm)',

             'Seat Height (cm)', 'Duration (sec.)'

            ]



# Drop columns

moma.drop(drop_cols, axis=1, inplace=True)



print('After drop:', moma.columns, sep='\n')
print('Before rename:', moma.columns, sep='\n', end='\n\n')



# Convert to lower case

moma.columns = moma.columns.str.lower()



# Add underline

cols = {'artistbio':'artist_bio', 'begindate':'begin_date', 'enddate':'end_date'}

moma.rename(columns=cols, inplace=True)



print('After rename:', moma.columns, sep='\n')
artist_bio_pattern_drop = r'(?:founded|established|est\.|active|formed)'



# Test

artist_bio_test = pd.DataFrame(['(British, founded 1967)',

                                '(Italian, established 1969)',

                                '(est. 1933)',

                                '(American, active 1904–present)'

                               ], columns=['artist_bio'])

artist_bio_test['artist_bio_pattern_drop'] = artist_bio_test['artist_bio'].str.contains(artist_bio_pattern_drop, flags=re.I)

print(artist_bio_test, end='\n\n')



artist_bio_bool_drop = moma['artist_bio'].str.contains(artist_bio_pattern_drop, flags=re.I) # bool mask to drop

artist_bio_bool_drop.fillna(False, inplace=True) # do not drop artist_bio with NaN



# Number of valid (True) and invalid (False) rows

artist_bio_cnt = (~artist_bio_bool_drop).value_counts(dropna=False)



# Percentage of valid (True) and invalid (False) rows

artist_bio_pct = artist_bio_cnt * 100 / moma.shape[0]

artist_bio_pct = (~artist_bio_bool_drop).value_counts(dropna=False)



artist_bio_stat = pd.DataFrame({'valid_count': artist_bio_cnt, 'valid_percentage': artist_bio_pct})

artist_bio_stat
print('Before drop:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~artist_bio_bool_drop).value_counts(dropna=False), end='\n\n')



# Drop

artist_bio_drop = moma[artist_bio_bool_drop].index # rows to drop

moma.drop(index=artist_bio_drop, inplace=True)



print('After drop:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~(moma['artist_bio'].str.contains(artist_bio_pattern_drop, flags=re.I)

                           .fillna(False)

        )

      ).value_counts(dropna=False)

     )
print('Before dropna:')

print('total:', moma.shape[0]) # print the total number of rows before

print('NaNs:', moma['begin_date'].isna().sum(), end='\n\n') # print the number of NaNs before



moma.dropna(subset=['begin_date'], axis=0, inplace=True) # drop NaNs



print('After dropna:')

print('total:', moma.shape[0]) # print the total number of rows after

print('NaNs:', moma['begin_date'].isna().sum()) # print the number of NaNs after
begin_date_pattern = r'^\(([0-2]\d{3})\)$'



# Test

begin_date_test = pd.DataFrame(['(0)',

                                '(0)  (0)',

                                '(1885) (0)',

                                '(0) (1995)'

                                '(1895) (1847) (1900)',

                                '(1880)'

                         ], columns=['begin_date'])

begin_date_test['begin_date_pattern'] = (begin_date_test['begin_date'].str.replace(r'\s', '')

                                                                      .str.match(begin_date_pattern, flags=re.I)

                                        )

print(begin_date_test, end='\n\n')



# Valid rows

begin_date_bool_valid = moma['begin_date'].str.replace(r'\s', '').str.match(begin_date_pattern, flags=re.I)



# Number of valid (True) and invalid (False) rows

begin_date_cnt = begin_date_bool_valid.value_counts(dropna=False)



# Percentage of valid (True) and invalid (False) rows

begin_date_pct = begin_date_cnt * 100 / moma.shape[0]



begin_date_stat = pd.DataFrame({'valid_count': begin_date_cnt, 'valid_percentage': begin_date_pct})

begin_date_stat
print('Before drop:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print(begin_date_bool_valid.value_counts(dropna=False), end='\n\n')



# Drop

begin_date_drop = moma[~begin_date_bool_valid].index # rows to drop

moma.drop(index=begin_date_drop, inplace=True)



print('After drop:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((moma['begin_date'].str.replace(r'\s', '')

                         .str.match(begin_date_pattern, flags=re.I)

                         .value_counts(dropna=False)

      ), end='\n\n'

     )
# Extract the birth year

moma['begin_date_clean'] = (moma['begin_date'].str.replace(r'\s', '')

                                              .str.extract(begin_date_pattern, flags=re.I)

                                              .astype(int)

                           )

moma[['begin_date', 'begin_date_clean']].head(10) # check the values
print('NaNs:', moma['end_date'].isna().sum()) # print the number of NaNs
end_date_pattern = r'^\(([0-2]\d{3})\)$'



# Valid rows

end_date_bool_valid = moma['end_date'].str.replace(r'\s', '').str.match(end_date_pattern, flags=re.I)



# Number of valid (True) and invalid (False) rows

end_date_cnt = end_date_bool_valid.value_counts(dropna=False)



# Percentage of valid (True) and invalid (False) rows

end_date_pct = end_date_cnt * 100 / moma.shape[0]



end_date_stat = pd.DataFrame({'valid_count': end_date_cnt, 'valid_percentage': end_date_pct})

end_date_stat
# Inspect values

moma.loc[~end_date_bool_valid, 'end_date'].value_counts()
# Extract the death year

moma['end_date_clean'] = (moma.loc[end_date_bool_valid, 'end_date'].str.replace(r'\s', '')

                                                                   .str.extract(end_date_pattern, flags=re.I)

                         )

moma['end_date_clean'].fillna(0, inplace=True)

moma['end_date_clean'] = moma['end_date_clean'].astype(int)



# Number of valid (True) and invalid (False) rows

print(moma['end_date_clean'].notnull().value_counts())

moma[['end_date', 'end_date_clean']].head(6) # check the values
print('NaNs:', moma['gender'].isna().sum()) # print the number of NaNs
gender_pattern = r'^\((?P<gender>(?:male|female))\)$'



# Valid rows

gender_bool_valid = moma['gender'].str.replace(r'\s', '').str.match(gender_pattern, flags=re.I)



# Number of valid (True) and invalid (False) rows

gender_cnt = gender_bool_valid.value_counts(dropna=False)



# Percentage of valid (True) and invalid (False) rows

gender_pct = gender_cnt * 100 / moma.shape[0]



gender_stat = pd.DataFrame({'valid_count': gender_cnt, 'valid_percentage': gender_pct})

gender_stat
# Inspect values

moma.loc[~gender_bool_valid, 'gender'].value_counts()
# Extract the gender

moma['gender_clean'] = (moma.loc[gender_bool_valid, 'gender'].str.replace(r'\s', '')

                                                             .str.extract(gender_pattern, flags=re.I)['gender']

                                                             .str.lower()

                       )

# Number of valid (True) and invalid (False) rows

print(moma['gender_clean'].notnull().value_counts(), end='\n\n')

print(moma['gender_clean'].value_counts(dropna=False))

moma[['gender', 'gender_clean']].head() # check the values
print('NaNs:', moma['nationality'].isna().sum()) # print the number of NaNs
nationality_pattern = r'^\((?P<nationality>(.+))\)$'



# Valid rows

nationality_bool_valid = moma['nationality'].str.replace(r'\s', '').str.match(nationality_pattern, flags=re.I)



# Number of valid (True) and invalid (False) rows

nationality_cnt = nationality_bool_valid.value_counts(dropna=False)



# Percentage of valid (True) and invalid (False) rows

nationality_pct = nationality_cnt * 100 / moma.shape[0]



nationality_stat = pd.DataFrame({'valid_count': nationality_cnt, 'valid_percentage': nationality_pct})

nationality_stat
# Inspect values

moma.loc[~nationality_bool_valid, 'nationality'].value_counts()
# Extract the nationality

moma['nationality_clean'] = (moma.loc[nationality_bool_valid, 'nationality'].str.replace(r'\s', '')

                                     .str.extract(nationality_pattern, flags=re.I)['nationality']

                                     .str.lower()

                       )

# Number of valid (True) and invalid (False) rows

print(moma['nationality_clean'].notnull().value_counts(), end='\n\n')

print(moma['nationality_clean'].value_counts(dropna=False))

moma[['nationality', 'nationality_clean']].head() # check the values
print('Before dropna:')

print('total:', moma.shape[0]) # print the total number of rows before

print('NaNs:', moma['date'].isna().sum(), end='\n\n') # print the number of NaNs before



moma.dropna(subset=['date'], axis=0, inplace=True) # drop NaNs



print('After dropna:')

print('total:', moma.shape[0]) # print the total number of rows after

print('NaNs:', moma['date'].isna().sum()) # print the number of NaNs after
date_pattern_char_replace = {r'(?:\bc\.\s?|\(|\)|;|:)': '', # remove special chars

                             r'\s+': ' ', # reduce gaps

                             r'(?:\–|\/|\s\-\s)': '-' # set range character as hyphen

                            } # dictionary to replace



# Test

date_test = pd.DataFrame(['c. 1960s',

                          'c. 1964, printed 1992',

                          'c.1935-1945',

                          'c. 1983, signed 2007',

                          '(c. 1914-20)',

                          '1964, assembled c.1965',

                          '1927.  (Print executed c. 1925-1927).',

                          'published c. 1946',

                          '(1960s)',

                          '1973 (published 1974)',

                          'Published 1944 (Prints executed 1915-1930)',

                          '(September 29-October 24, 1967)',

                          '1965-66,  printed 1983',

                          '1968 - 1972',

                          '1947–49, published 1949',

                          'Dec. 9, 1954'

                         ], columns=['date'])

date_test['date_pattern_char_replace'] = date_test['date'].replace(regex=date_pattern_char_replace)

print(date_test, end='\n\n')



# Replace chars

print('Before replace:', moma['date'].tail(10), sep='\n', end='\n\n')

moma['date'] = moma['date'].replace(regex=date_pattern_char_replace) # replace

print('After replace:', moma['date'].tail(10), sep='\n')
date_pattern_drop_1 = r'([0-2]\d{3})'



date_bool_drop_1 = moma['date'].str.count(date_pattern_drop_1) == 0 # bool mask to drop



# Inspect values

print('moma values:', moma.loc[date_bool_drop_1, 'date'].value_counts(dropna=False), sep='\n', end='\n\n')

print('Matched: {}'.format(date_bool_drop_1.sum()))
print('Before drop 1:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~date_bool_drop_1).value_counts(dropna=False), end='\n\n')



# Drop

date_drop_1 = moma[date_bool_drop_1].index # rows to drop

moma.drop(index=date_drop_1, inplace=True)



print('After drop 1:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((moma['date'].str.count(date_pattern_drop_1) != 0).value_counts(dropna=False))
date_pattern_drop_2 = (r'^(?:early|late)?\s?[0-2]\d{3}\'?(?:s|\s?\?|s\?)'

                         '(?:(?:\-|\sor\s)(?:[0-2]\d)?\d{2}\'?(?:s|\s?\?|s\?))?$'

                      )



# Test

date_test = pd.DataFrame(['1915?',

                          '1860s?',

                          '1880 ?',

                          '1920s',

                          '1880s-90s',

                          '1960s-1970s',

                          '1920s or 1930s',

                          'late 1950s',

                          'early 1940s',

                          'Early 1970\'s'

                         ], columns=['date'])

date_test['date_pattern_drop_2'] = date_test['date'].str.contains(date_pattern_drop_2, flags=re.I)

print(date_test, end='\n\n')



date_bool_drop_2 = moma['date'].str.contains(date_pattern_drop_2, flags=re.I) # bool mask to drop



# Inspect values

pd.set_option('display.max_rows', 80) # increase the number of rows to display

print('moma values:', moma.loc[date_bool_drop_2, 'date'].value_counts(dropna=False), sep='\n', end='\n\n')

print('Matched: {}'.format(date_bool_drop_2.sum()))

pd.reset_option('display.max_rows') # reset the number of rows to display to default
print('Before drop 2:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~date_bool_drop_2).value_counts(dropna=False), end='\n\n')



# Drop

date_drop_2 = moma[date_bool_drop_2].index # rows to drop

moma.drop(index=date_drop_2, inplace=True)



print('After drop 2:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~moma['date'].str.contains(date_pattern_drop_2, flags=re.I)).value_counts(dropna=False))
date_pattern_drop_3 = r'^(?:before|after)\s?[0-2]\d{3}\s?\??$'



# Test

date_test = pd.DataFrame(['Before 1900',

                          'Before 1900?',

                          'After 1933',

                          'after 1891'

                         ], columns=['date'])

date_test['date_pattern_drop_3'] = date_test['date'].str.contains(date_pattern_drop_3, flags=re.I)

print(date_test, end='\n\n')



date_bool_drop_3 = moma['date'].str.contains(date_pattern_drop_3, flags=re.I) # bool mask to drop



# Inspect values

print('moma values:', moma.loc[date_bool_drop_3, 'date'].value_counts(dropna=False), sep='\n', end='\n\n')

print('Matched: {}'.format(date_bool_drop_3.sum()))
print('Before drop 3:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~date_bool_drop_3).value_counts(dropna=False), end='\n\n')



# Drop

date_drop_3 = moma[date_bool_drop_3].index # rows to drop

moma.drop(index=date_drop_3, inplace=True)



print('After drop 3:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~moma['date'].str.contains(date_pattern_drop_3, flags=re.I)).value_counts(dropna=False))
date_pattern_drop_4 = r'^[0-2]\d{3}\sor\s(?:before|after|earlier)\??$'



# Test

date_test = pd.DataFrame(['1898 or earlier',

                          '1898 or before?'

                         ], columns=['date'])

date_test['date_pattern_drop_4'] = date_test['date'].str.contains(date_pattern_drop_4, flags=re.I)

print(date_test, end='\n\n')



date_bool_drop_4 = moma['date'].str.contains(date_pattern_drop_4, flags=re.I) # bool mask to drop



# Inspect values

print('moma values:', moma.loc[date_bool_drop_4, 'date'].value_counts(dropna=False), sep='\n', end='\n\n')

print('Matched: {}'.format(date_bool_drop_4.sum()))
print('Before drop 4:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~date_bool_drop_4).value_counts(dropna=False), end='\n\n')



# Drop

date_drop_4 = moma[date_bool_drop_4].index # rows to drop

moma.drop(index=date_drop_4, inplace=True)



print('After drop 4:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~moma['date'].str.contains(date_pattern_drop_4, flags=re.I)).value_counts(dropna=False))
date_pattern_drop_5 = r'(?!.*prints executed.*)^(?:newspapers?\s)?(?:published.*)'



# Test

date_test = pd.DataFrame(['published 1965',

                          'Published 1946',

                          'published April 1898',

                          'newspaper published May 15-16, 1999',

                          'Published 1944 Prints executed 1915-1930' # must be False (we'll explore this later)

                         ], columns=['date'])

date_test['date_pattern_drop_5'] = date_test['date'].str.contains(date_pattern_drop_5, flags=re.I)

print(date_test, end='\n\n')



date_bool_drop_5 = moma['date'].str.contains(date_pattern_drop_5, flags=re.I) # bool mask to drop



# Inspect values

print('moma values:', moma.loc[date_bool_drop_5, 'date'].value_counts(dropna=False), sep='\n', end='\n\n')

print('Matched: {}'.format(date_bool_drop_5.sum()))
print('Before drop 5:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~date_bool_drop_5).value_counts(dropna=False), end='\n\n')



# Drop

date_drop_5 = moma[date_bool_drop_5].index # rows to drop

moma.drop(index=date_drop_5, inplace=True)



print('After drop 5:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~moma['date'].str.contains(date_pattern_drop_5, flags=re.I)).value_counts(dropna=False))
date_updated_replace = (r'(?:published|repainted\sin|printed\sin|printed|assembled|'

                         'realized|signed|reprinted|reconstructed|fabricated|'

                         'released|cast|arranged|manufactured)'

                       ) # keys to replace with placeholder

date_pattern_1 = (r'^(?P<year_1>[0-2]\d{3})'

                   '(?:\-(?P<year_2>(?:[0-2]\d)?\d{2}))?'

                   '(?:,?(?:\supdated|\-)\s?[0-2]\d{3}(?:\-(?:[0-2]\d)?\d{2})?)?s?\.?$'

                 )



# Test

date_test = pd.DataFrame(['1896',

                          '1941-1948',

                          '1969-70',

                          '1965 printed 2014',

                          '1964, printed 1992',

                          '2000-01, printed 2007',

                          '1973 published 1974',

                          '1941, published 1943',

                          '1975 Published 1976.',

                          '1947-49, published 1949',

                          '1918, published 1922-1923',

                          '1961, assembled 1964-65',

                          '1969, realized 1973',

                          '1983, signed 2007',

                          '1945, reprinted 1990',

                          '1961, reconstructed 1981',

                          '1963, fabricated 1975',

                          '1985, released 1990',

                          '1944, printed in 1967',

                          '1966 repainted in 1990',

                          '1950-52 manufactured 1955',

                          '1950-55-1980'

                         ], columns=['date'])

date_test[['year_1', 'year_2']] = (date_test['date'].str.replace(date_updated_replace, 'updated', flags=re.I)

                                                    .str.extract(date_pattern_1, flags=re.I)

                                  )

print(date_test, end='\n\n')



date_bool_1 = (moma['date'].str.replace(date_updated_replace, 'updated', flags=re.I)

                           .str.match(date_pattern_1, flags=re.I)

              ) # bool mask to extract the years



# Inspect values

print('moma values:', moma.loc[date_bool_1, 'date'].value_counts(dropna=False).tail(40), sep='\n', end='\n\n')

print('Matched: {}'.format(date_bool_1.sum()))
print('Before extract 1:')

# Total number of rows

print('total:', moma.shape[0])

# Number of rows matching the pattern (True) and the rest (False)

print(date_bool_1.value_counts(dropna=False), end='\n\n')



# Extract

moma.loc[date_bool_1, ['year_1', 'year_2']] = (moma.loc[date_bool_1, 'date']

                                                   .str.replace(date_updated_replace, 'updated', flags=re.I)

                                                   .str.extract(date_pattern_1, flags=re.I)

                                              )



# Inspect values

print('After extract 1:', moma[['date', 'year_1', 'year_2']].describe().loc[['count', 'unique']], sep='\n', end='\n\n')

moma.loc[date_bool_1, ['date', 'year_1', 'year_2']].head(8)
(moma.loc[moma['date'].str.contains(r'print executed', flags=re.I), 'classification']

     .value_counts()

)
# Value examples

(moma.loc[moma['date'].str.contains(r'print executed', flags=re.I)])
date_char_trim = r'[\.,]'

date_pattern_2 = (r'^(?:[0-2]\d{3})?(?:\-\d{2,4})?,?\s?'

                   '(?:originals?|drawings?|prints?|woodcuts?|sculpture?)?\s?executed\s(?:in\s)?'

                   '(?P<year_1>[0-2]\d{3})(?:\-(?P<year_2>(?:[0-2]\d)?\d{2}))?$'

                 )



# Test

date_test = pd.DataFrame(['1921 executed 1920',

                          '1922, executed 1920-21',

                          '1935 originals executed 1933-34',

                          '1935 drawings executed 1933-34',

                          '1922-23 original executed in 1922',

                          '1973-1974, executed 1973',

                          'Print executed 1936',

                          'Prints executed 1956',

                          '1950, print executed 1949-50',

                          '1972. Print executed 1971-1972.',

                          '1962. Print executed 1960.',

                          '1944. Print executed 1942.',

                          '1927. Print executed 1925-1927.',

                          '1963 Woodcuts executed 1907',

                          '1970. Sculpture executed 1968-1970.'

                         ], columns=['date'])

date_test[['year_1', 'year_2']] = (date_test['date'].str.replace(date_char_trim, '', flags=re.I)

                                                    .str.extract(date_pattern_2, flags=re.I)

                                  )

print(date_test, end='\n\n')



date_bool_2 = (moma['date'].str.replace(date_char_trim, '', flags=re.I)

                           .str.match(date_pattern_2, flags=re.I)

              ) # bool mask to extract the years



# Inspect values

print('moma values:', moma.loc[date_bool_2, 'date'].value_counts(dropna=False).tail(40), sep='\n', end='\n\n')

print('Matched: {}'.format(date_bool_2.sum()))
print('Before extract 2:')

# Total number of rows

print('total:', moma.shape[0])

# Number of rows matching the pattern (True) and the rest (False)

print(date_bool_2.value_counts(dropna=False), end='\n\n')



# Extract

moma.loc[date_bool_2, ['year_1', 'year_2']] = (moma.loc[date_bool_2, 'date'].str.replace(date_char_trim, '', flags=re.I)

                                                                            .str.extract(date_pattern_2, flags=re.I)

                                              )



# Inspect values

print('After extract 2:', moma[['date', 'year_1', 'year_2']].describe().loc[['count', 'unique']], sep='\n', end='\n\n')

moma.loc[date_bool_2, ['date', 'year_1', 'year_2']][120:130]
date_char_trim = r'[\.,]'

date_pattern_special = (r'^(?:\d{,2}\s)?(?:issy-les-moulineaux\ssummer|fontainebleau\ssummer|'

                         'summer|spring|winter|autumn|fall|january|february|march|'

                         'april|may|june|july|august|september|october|november|'

                         'december|decemer|dec|begun|late|early|'

                         'mars\s7\sh\smatin|mars\s8\sh\smatin|mars|'

                         'avril\s7\sh\smatin|avril|'

                         'paris\sjune\s-\sjuly|paris\searly|paris\swinter|paris\sspring|paris|'

                         'juin\s7\sh\smatin|juin|'

                         'mai\s8\sh\smatin|mai|'

                         'gallifa|juillet|kamakura|août|frankfurt|cannes|circa|hiver|bogotá|cuba|'

                         'berlin|meudon|jupiter\sisland|barcelona|cavalière|arles|germany|rome|'

                         'horta\sde\ssan\sjoan|collioure|new\syork|saint\srémy|issy-les-moulineaux)'

                       ) # special cases

date_pattern_3 = (r'^.*?(?P<year_1>[0-2]\d{3})'

                   '(?:\-(?:(?P<year_2_2>[0-2]\d)|.*(?P<year_2_4>[0-2]\d{3})))?$'

                 )



# Test

date_test = pd.DataFrame(['October 1977',

                          'August 15 1966',

                          'February 1, 1970',

                          'May 15, 1962.',

                          '11 July 1854',

                          'May-June 1991',

                          'May 13-19, 1970',

                          'May 2-10 1969',

                          'September 29-October 24, 1967',

                          'August 5, 1877-June 22, 1894',

                          'Dec. 9, 1954',

                          'Spring 1909',

                          'Early 1969',

                          'Late 1924-1925',

                          'Mars 1926',

                          'Mars, 7 h. matin, 1925',

                          'Mai 1926',

                          'Mai, 8 h. matin, 1925',

                          'Gallifa, 1956',

                          'Juillet 1921',

                          'Fontainebleau, summer 1921',

                          'Avril, 7 h. matin, 1925',

                          'Juin, 7 h. matin, 1925',

                          'Kamakura, 1952',

                          'Août 1924',

                          'Decemer 1888',

                          'Issy-les-Moulineaux, summer 1916',

                          'Paris, early 1899',

                          'Paris, June - July 1914',

                          'Paris, winter 1914-15',

                          'Paris, spring 1908',

                          'Frankfurt 1920',

                          'Cannes, 1958',

                          'Mars, 8 h. matin, 1925',

                          'circa 1980',

                          'Begun 1938',

                          'Berlin 1926',

                          'Meudon 1932',

                          'Jupiter Island 1992',

                          'Seasons of 1871, 1872 and 1873' # must be False (we'll explore this later)

                         ], columns=['date'])

date_test['date_pattern_special'] = (date_test['date'].str.replace(date_char_trim, '', flags=re.I)

                                                      .str.contains(date_pattern_special, flags=re.I)

                                    )

date_test[['year_1', 'year_2_2', 'year_2_4']] = (date_test['date'].str.replace(date_char_trim, '', flags=re.I)

                                                                  .str.extract(date_pattern_3, flags=re.I)

                                                )

date_test['year_2'] = date_test['year_2_2'].fillna(date_test['year_2_4'])

print(date_test, end='\n\n')



date_bool_3 = (moma['date'].str.replace(date_char_trim, '', flags=re.I)

                           .str.match(date_pattern_3, flags=re.I)

               & moma['year_1'].isnull() # among the remaining rows

               & moma['date'].str.replace(date_char_trim, '', flags=re.I)

                             .str.contains(date_pattern_special, flags=re.I) # among the rows with special cases

              ) # bool mask to extract the years



# Inspect values

print('moma values:', moma.loc[date_bool_3, 'date'].value_counts(dropna=False).tail(20), sep='\n\n', end='\n\n')

print('Matched: {}'.format(date_bool_3.sum()))
print('Before extract 3:')

# Total number of rows

print('total:', moma.shape[0])

# Number of rows matching the pattern (True) and the rest (False)

print(date_bool_3.value_counts(dropna=False), end='\n\n')



# Extract

moma.loc[date_bool_3, ['year_1', 'year_2_2', 'year_2_4']] = (moma.loc[date_bool_3, 'date']

                                                                 .str.replace(date_char_trim, '', flags=re.I)

                                                                 .str.extract(date_pattern_3, flags=re.I)

                                                            )

moma.loc[date_bool_3, 'year_2'] = moma.loc[date_bool_3, 'year_2_2'].fillna(moma.loc[date_bool_3, 'year_2_4'])



# Inspect values

print('After extract 3:', moma[['date', 'year_1', 'year_2']].describe().loc[['count', 'unique']], sep='\n', end='\n\n')

moma.loc[date_bool_3, ['date', 'year_1', 'year_2']].head(8)
date_bool_rest = ~(date_bool_1 

                    | date_bool_2

                    | date_bool_3

                  )

# Or another way

# date_bool_not_year_1 = moma['year_1'].isnull()



# Statistics for the rest rows

print('rest count: {}'.format(date_bool_rest.sum()))

print('rest percentage: {}'.format(round(date_bool_rest.sum()*100/moma.shape[0], 2)))

print('rest count unique: {}'.format(moma.loc[date_bool_rest, 'date'].value_counts(dropna=False).shape[0]))

print('total: {}'.format(moma.shape[0]), end='\n\n')



# Inspect values

print('moma values:',

      (moma.loc[date_bool_rest, 'date']

           .value_counts(dropna=False)

           .sort_values(ascending=False)

           .head(16)

      ),

      sep='\n'

     )
print('Before drop rest:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((~date_bool_rest).value_counts(dropna=False), end='\n\n')



# Drop

date_drop_rest = moma[date_bool_rest].index # rows to drop

moma.drop(index=date_drop_rest, inplace=True)



print('After drop rest:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print(moma['year_1'].notnull().value_counts(dropna=False))
# Fill in a two-digit year to four digits

year_2_bool_two = moma['year_2'].str.len() == 2



moma.loc[year_2_bool_two, 'year_2'] = (moma.loc[year_2_bool_two, 'year_1'].str[0:2] 

                                       + moma.loc[year_2_bool_two, 'year_2']

                                      )



# Inspect values

print(moma.loc[year_2_bool_two, ['year_1', 'year_2']].tail())



# Fill in NaN 'year_2' with 'year_1'

moma['year_2'].fillna(value=moma['year_1'], inplace=True)

# Cast years to int

moma[['year_1', 'year_2']] = moma[['year_1', 'year_2']].astype(int)

# Calculate date as average

moma['date_clean'] = round((moma['year_2'] + moma['year_1']) / 2)

moma['date_clean'] = moma['date_clean'].astype(int) # cast to int

# Calculate age

moma['age'] = moma['date_clean'] - moma['begin_date_clean']

moma['age'] = moma['age'].astype(int) # cast to int

# Inspect values

moma[['begin_date_clean', 'date_clean', 'age']].head(10)
moma_bool_invalid = ~(

                      (moma['year_1'] <= moma['year_2'])

                      & (moma['begin_date_clean'] < moma['year_1'])

                      & (((moma['end_date_clean'] >= moma['year_2']) & (moma['end_date_clean'] != 0))

                         | (moma['end_date_clean'] == 0)

                        )

                     )



# Statistics for the rest rows

print('invalid count: {}'.format(moma_bool_invalid.sum()))

print('invalid percentage: {}'.format(round(moma_bool_invalid.sum()*100/moma.shape[0], 2)))

print('total:', moma.shape[0], end='\n\n')



# Inspect values

print('moma invalid values:')

moma.loc[moma_bool_invalid]
print('Before drop invalid:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of the valid (True) and invalid (False) rows

print((~moma_bool_invalid).value_counts(dropna=False), end='\n\n')



# Drop

moma_drop_invalid = moma[moma_bool_invalid].index # rows to drop

moma.drop(index=moma_drop_invalid, inplace=True)



print('After drop invalid:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of valid (True) and invalid (False) rows

print((

       (moma['year_1'] <= moma['year_2'])

       & (moma['begin_date_clean'] < moma['year_1'])

       & (((moma['end_date_clean'] >= moma['year_2']) & (moma['end_date_clean'] != 0))

          | (moma['end_date_clean'] == 0)

         )

      ).value_counts(dropna=False)

     )
moma['age'].describe()
pd.set_option('display.max_rows', 85) # increase the number of rows to display



# Inspect values

print(moma.loc[moma['age'] > 90, 'artist'].value_counts())

moma[moma['age'] > 90].sort_values('age').tail(85)
# Inspect values

print(moma.loc[moma['age'] < 10, 'artist'].value_counts())

moma[moma['age'] < 10].sort_values('age')
artist_org = ['Hi Red Center', 'General Idea', 'Gorgona artists group', 

              'Grey Organisation', 'Grapus', 'Banana Equipment', 

              'Atelier Martine, Paris, France'

             ]

artist_bool_org = moma['artist'].isin(artist_org)



# Statistics for the rest rows

print('org count:', artist_bool_org.sum())

print('org percentage:', round(artist_bool_org.sum()*100/moma.shape[0], 2))

print('total:', moma.shape[0], end='\n\n')



# Inspect values

print('moma org values:')

moma.loc[artist_bool_org]
print('Before drop org:')

# Total number of rows before

print('total:', moma.shape[0])

# Number of the valid (True) and invalid (False) rows

print((~artist_bool_org).value_counts(dropna=False), end='\n\n')



# Drop

artist_drop_org = moma[artist_bool_org].index # rows to drop

moma.drop(index=artist_drop_org, inplace=True)



print('After drop org:')

# Total number of rows after

print('total:', moma.shape[0])

# Number of the valid (True) and invalid (False) rows

print((~(moma['artist'].isin(artist_org))).value_counts(dropna=False))
moma['age'].value_counts(dropna=False, normalize=True).head(20)*100
bins=[i for i in range(0, 110, 5)] # age groups

moma['age'].value_counts(dropna=False, bins=bins, normalize=True) * 100
# Import libs

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Turn on svg rendering

%config InlineBackend.figure_format = 'svg'



# Color palette for the blog

snark_palette = ['#e0675a', # red

                 '#5ca0af', # green

                 '#edde7e', # yellow

                 '#211c47' # dark blue

                ]
# Set the figure

sns.set(context='paper', style='ticks', palette=snark_palette,

        rc={'xtick.major.size': 4, 'ytick.left':False,

            'axes.spines.left': False, 'axes.spines.bottom': True,

            'axes.spines.right': False, 'axes.spines.top': False

           }

       )



# Create the plot

ax_age = sns.distplot(moma['age'], hist=True, rug=False)

ax_age.axvline(x=33, ymin=0, ymax=0.97, marker='x', linestyle=':', color=snark_palette[-1]) # 33 boundary



# Set some aesthetic params for the plot

ax_age.annotate('33', [35, 0.0325], c=snark_palette[-1]) # set label for the 33 boundary

ax_age.set_title('Amount of Artworks by Age', loc='right', pad=0, c=snark_palette[-1]) # set title of the plot

ax_age.set_xlabel('Age', c=snark_palette[-1]) # set label of x axis

ax_age.get_yaxis().set_visible(False) # hide y axis

ax_age.set_xticks([i for i in range(0, 110, 10)]) # set x ticks labels

ax_age.set_xlim([10, 100]) # set x axis range

ax_age.tick_params(axis='x', colors=snark_palette[-1]) # color x ticks

ax_age.spines['bottom'].set_color(snark_palette[-1]) # color x axis



# Save and plot

plt.savefig('plot.pic\plot.age.png', dpi=150)

plt.show()
# Women

moma.loc[(moma['gender_clean'] == 'female'), 'age'].value_counts(normalize=True, bins=bins).head(20) * 100
# Men

moma.loc[(moma['gender_clean'] == 'male'), 'age'].value_counts(normalize=True, bins=bins).head(20) * 100
# Set the figure

sns.set(context='paper', style='ticks', palette=snark_palette,

        rc={'xtick.major.size': 4, 'ytick.left':False,

            'axes.spines.left': False, 'axes.spines.bottom': True,

            'axes.spines.right': False, 'axes.spines.top': False

           }

       )



# Create the plot

f_ag, ax_ag = plt.subplots()

sns.distplot(moma.loc[moma['gender_clean'] == 'female', 'age'], hist=False, rug=False, label='female', ax=ax_ag)

sns.distplot(moma.loc[moma['gender_clean'] == 'male', 'age'], hist=False, rug=False, label='male', ax=ax_ag)



ax_ag.axvline(x=33, ymin=0, ymax=0.98, marker='x', linestyle=':', color=snark_palette[-1]) # 33 boundary



# Set some aesthetic params for the plot

ax_ag.annotate('33', [28, 0.0323], c=snark_palette[-1]) # set label for the 33 boundary

ax_ag.legend() # set legend

ax_ag.set_title('Amount of Artworks by Age: gender', loc='right', c=snark_palette[-1]) # set title of the plot

ax_ag.set_xlabel('Age', c=snark_palette[-1]) # set label of x axis

ax_ag.get_yaxis().set_visible(False) # hide y axis

ax_ag.set_xticks([i for i in range(0, 110, 10)]) # set x ticks labels

ax_ag.set_xlim([10, 110]) # set x axis range

ax_ag.tick_params(axis='x', colors=snark_palette[-1]) # color x ticks

ax_ag.spines['bottom'].set_color(snark_palette[-1]) # color x axis



# Save and plot

plt.savefig('plot.pic\plot.age.gender.png', dpi=150)

plt.show()
# Women

print('Total by gender:', moma['gender_clean'].value_counts(), sep='\n')



# 46 peack

moma.loc[(moma['gender_clean'] == 'female'), 'age'].value_counts().head(20)
# Top5 women in 46

women_46_top5 = (moma.loc[(moma['gender_clean'] == 'female') & (moma['age'] == 46), 'artist']).value_counts().head()



print(women_46_top5)

moma.loc[(moma['artist'].isin(women_46_top5.index)) & (moma['age'] == 46)]
# 90 peack

print(moma.loc[(moma['gender_clean'] == 'female') & (moma['age'].between(80, 90)), 'age'].value_counts().head(20))



# Top5 women in 88

women_88_top5 = (moma.loc[(moma['gender_clean'] == 'female') & (moma['age'] == 88), 'artist']).value_counts().head()



print(women_88_top5)

moma.loc[(moma['artist'].isin(women_88_top5.index)) & (moma['age'] == 88)]
nationality_top4 = moma['nationality_clean'].value_counts(normalize=False).head(4)

print(nationality_top4)
# Set the figure

sns.set(context='paper', style='ticks', palette=snark_palette,

        rc={'xtick.major.size': 4, 'ytick.left':False,

            'axes.spines.left': False, 'axes.spines.bottom': True,

            'axes.spines.right': False, 'axes.spines.top': False

           }

       )



# Create the plot

moma_nationality = moma.loc[moma['nationality_clean'].isin(nationality_top4.index), ['nationality_clean', 'age']] # data

g_an = sns.FacetGrid(moma_nationality, hue='nationality_clean')

g_an = g_an.map(sns.distplot, 'age', hist=False, rug=False)



g_an.ax.axvline(x=33, ymin=0, ymax=0.98, marker='x', linestyle=':', color=snark_palette[-1]) # 33 boundary



# Set some aesthetic params for the plot

g_an.fig.set_size_inches(6, 4)

g_an.ax.annotate('33', [28, 0.0415], c=snark_palette[-1]) # set label for the 33 boundary

g_an.ax.legend() # set legend

g_an.ax.set_title('Amount of Artworks by Age: nationality', loc='right', c=snark_palette[-1]) # set title of the plot

g_an.ax.set_xlabel('Age', c=snark_palette[-1]) # set label of x axis

g_an.ax.get_yaxis().set_visible(False) # hide y labels

g_an.despine(left=True) # hide y axis

g_an.ax.set_xticks([i for i in range(0, 110, 10)]) # set x ticks labels

g_an.ax.set_xlim([10, 110]) # set x axis range

g_an.ax.tick_params(axis='x', colors=snark_palette[-1]) # color x ticks

g_an.ax.spines['bottom'].set_color(snark_palette[-1]) # color x axis



# Save and plot

g_an.fig.subplots_adjust(bottom=0.125, top=0.88, left=0.125, right=0.9) # adjust for the post picture

g_an.savefig('plot.pic\plot.age.nationality.png', dpi=150, bbox_inches=None)

plt.show()
# Extract 

moma['century'] = ((moma['date_clean'] // 100) + 1).astype(int)



# Inspect values

moma[['date_clean', 'century']]
moma['century'].value_counts().sort_index(ascending=False)
# Set the figure

sns.set(context='paper', style='ticks', palette=snark_palette,

        rc={'xtick.major.size': 4, 'ytick.left':False,

            'axes.spines.left': False, 'axes.spines.bottom': True,

            'axes.spines.right': False, 'axes.spines.top': False

           }

       )



# Create the plot

moma_century = moma.loc[moma['century'].isin([19, 20, 21]), ['century', 'age']] # data

g_ac = sns.FacetGrid(moma_century, hue='century')

g_ac = g_ac.map(sns.distplot, 'age', hist=False, rug=False)



g_ac.ax.axvline(x=33, ymin=0, ymax=0.98, marker='x', linestyle=':', color=snark_palette[-1]) # 33 boundary



# Set some aesthetic params for the plot

g_ac.fig.set_size_inches(6, 4)

g_ac.ax.annotate('33', [28, 0.041], c=snark_palette[-1]) # set label for the 33 boundary

g_ac.ax.legend() # set legend

g_ac.ax.set_title('Amount of Artworks by Age: century', loc='right', c=snark_palette[-1]) # set title of the plot

g_ac.ax.set_xlabel('Age', c=snark_palette[-1]) # set label of x axis

g_ac.ax.get_yaxis().set_visible(False) # hide y labels

g_ac.despine(left=True) # hide y axis

g_ac.ax.set_xticks([i for i in range(0, 110, 10)]) # set x ticks labels

g_ac.ax.set_xlim([10, 110]) # set x axis range

g_ac.ax.tick_params(axis='x', colors=snark_palette[-1]) # color x ticks

g_ac.ax.spines['bottom'].set_color(snark_palette[-1]) # color x axis



# Save and plot

g_ac.fig.subplots_adjust(bottom=0.125, top=0.88, left=0.125, right=0.9) # adjust for post picture

g_ac.savefig('plot.pic\plot.age.century.png', dpi=150, bbox_inches=None)

plt.show()