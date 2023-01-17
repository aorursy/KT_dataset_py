import pandas as pd



#imported pandas module



"""

i downloaded datas from kaggle as csv file and i put that file in my pycharm work project folder to read it here!



i will also upload csv file here





we will work on youtube trend videos on this project to make understand pandas basics...



"""



# HOW TO READ DATA



# lets read out dataset first( you can see here " how to read datasets")



youtube = pd.read_csv("USVideos.csv")



print(youtube)



"""

          video_id  ...                                        description

0      2kyS6SvSYSE  ...  SHANTELL'S CHANNEL - https://www.youtube.com/s...

1      1ZAPwfrtAFY  ...  One year after the presidential election, John...

2      5qpjK5DgCt4  ...  WATCH MY PREVIOUS VIDEO ▶ \n\nSUBSCRIBE ► http...

3      puqaWrEC7tY  ...  Today we find out if Link is a Nickelback amat...

4      d380meD0W0M  ...  I know it's been a while since we did this sho...

...            ...  ...                                                ...

23357  pH7VfJDq7f4  ...  ...and other musings on thermal movement of la...

23358  hV-yHbbrKRA  ...  Visit Our Website! ▶ http://www.townsends.us/ ...

23359  CwKp6Xhy3_4  ...  Chris Young's Hangin' On from his #1 album Los...

23360  vQiiNGllGQo  ...  very wholesome stuff.\n\nThis video was taken ...

23361  2afSbqlp5HU  ...  The new marvel film Black Panther is set in th...



[23362 rows x 16 columns]



this csv file has 23362 rows and 16 columns as you can seee

we will work on this file.



"""



# HOW TO READ FIRST 5 DATAS





# if you wanna read first 10 data: you can use  " head() " function

# if you dont give "n=" parameter in the head function its gonna take automatically first 5 data!



print(youtube.head())



# we did not give n= parameter anything, if you dont its gonna take automatically first 5 data



"""

 video_id  ...                                        description

0  2kyS6SvSYSE  ...  SHANTELL'S CHANNEL - https://www.youtube.com/s...

1  1ZAPwfrtAFY  ...  One year after the presidential election, John...

2  5qpjK5DgCt4  ...  WATCH MY PREVIOUS VIDEO ▶ \n\nSUBSCRIBE ► http...

3  puqaWrEC7tY  ...  Today we find out if Link is a Nickelback amat...

4  d380meD0W0M  ...  I know it's been a while since we did this sho...



[5 rows x 16 columns]





"""



# HOW TO DELETE SOMETHING AND UPDATE AS DELETED:

# we can use "drop" function to delete



"""

for example if we want to delete this from dataset and update as deleted:



"thumbnail_link","video_id","trending_date","publish_time","thumbnail_link","comments_disabled","ratings_disabled","video_error_or_removed"



"""



youtube.drop(["thumbnail_link","video_id","trending_date","publish_time","thumbnail_link","comments_disabled","ratings_disabled","video_error_or_removed"],axis = 1,inplace = True)



# we choose axis=1 because we re deleting from column and also "inplace=True" this is for update dataset as deleted



print(youtube)



"""



0                     WE WANT TO TALK ABOUT OUR MARRIAGE  ...  SHANTELL'S CHANNEL - https://www.youtube.com/s...

1      The Trump Presidency: Last Week Tonight with J...  ...  One year after the presidential election, John...

2      Racist Superman | Rudy Mancuso, King Bach & Le...  ...  WATCH MY PREVIOUS VIDEO ▶ \n\nSUBSCRIBE ► http...

3                       Nickelback Lyrics: Real or Fake?  ...  Today we find out if Link is a Nickelback amat...

4                               I Dare You: GOING BALD!?  ...  I know it's been a while since we did this sho...

...                                                  ...  ...                                                ...

23357                                Why Bridges Move...  ...  ...and other musings on thermal movement of la...

23358                      Macaroni - A Recipe From 1784  ...  Visit Our Website! ▶ http://www.townsends.us/ ...

23359                           Chris Young - Hangin' On  ...  Chris Young's Hangin' On from his #1 album Los...

23360      Elderly man making sure his dog won't get wet  ...  very wholesome stuff.\n\nThis video was taken ...

23361         How to speak like Black Panther - BBC News  ...  The new marvel film Black Panther is set in th...



[23362 rows x 9 columns]



There are 9 columns after deleted other columns as you can see it was 16 columns before..



NOTE:

for columns you need to choose axis=1

for rows you need to choose axis=0

and if you wanna update your dataset after you change something on it you need write "inplace=True"



"""



# HOW TO FIND THERE ARE HOW MANY COLUMNS AND ROWS ON DATASET



#for rows(index)

print(len(youtube.index))



"""

23362



there are 23362 rows

"""

#for columns

print(len(youtube.columns))



"""

9



there are 9 columns 

"""



#HOW TO FIND THERE IS HOW MANY LIKE AND DISLIKE ON THIS DATASETS and FINDING AVARAGES( this is a youtube trending videos file as csv format)



# we need to take "likes" and dislikes columns for it and use "mean()" function for taking avarage



print(youtube["likes"].mean())

"""

45085.35972947522

the avarage for "likes"



"""



print(youtube["dislikes"].mean())

"""

2809.7094426847016

the avarage for dislikes



"""



# HOW TO FIND:WHATS THE NAME OF MAX VIEWED YOUTUBE VIDEOS ON THIS DATASET:

# we are gonna need to make filter for this



print(youtube[youtube["views"].max() == youtube["views"]]["title"].iloc[0])

# if you guys wanna make this look better you can take only name as " YouTube Rewind: The Shape of 2017 | #YouTubeRewind " with "iloc[]" function



# you can take any index with iloc[] function



"""



YouTube Rewind: The Shape of 2017 | #YouTubeRewind



this video is max viewed one on this youtube csv file





"""



# HOW TO FIND:WHATS THE NAME OF MINIMUM VIEWED YOUTUBE VIDEOS ON THIS DATASET:

## we are gonna need to make filter again here. we are taking "views" column and using "min()" function to find which video viewed min.



print(youtube[youtube["views"].min() == youtube["views"]]["title"].iloc[0])

# same method we used here to take "1 dead, others injured after Ky. school shooting" index with "iloc[]"



"""

1 dead, others injured after Ky. school shooting



this video is min viewed one on this youtube csv file



"""



# HOW TO FIND: AVARAGE OF COMMENTS BY CATEGORIES(category_id)



#so we need to use groupby method here to find avarage of comments by categories





# STEP1 here: we need to take category_id with groupby and then for finding avarege we can use "mean()" function



print(youtube.groupby("category_id").mean())





"""

                   views          likes      dislikes  comment_count

category_id                                                          

1            2.049144e+06   47593.034743   2027.794562    6003.651813

2            1.412712e+06    9989.032353    613.911765    1730.438235

10           2.555946e+06  120687.778075   4003.628045   10383.559418

15           5.538671e+05   16425.096121    389.399663    2474.376054

17           1.310862e+06   30259.176898   2066.235057    3256.199515

19           8.613965e+05    9193.538168    839.041985    1833.011450

20           1.025884e+06   36906.295082   2456.649180    7578.383607

22           1.075469e+06   36033.730117   1829.890244    4491.865854

23           1.162698e+06   55259.666330   1771.822548    5432.868554

24           1.354847e+06   38033.904071   5283.989217    7178.388176

25           3.327692e+05    4034.712638   1192.622968    1523.749869

26           6.562508e+05   26010.616689    894.296743    4992.165997

27           4.889018e+05   19307.393214    649.418164    2143.574850

28           9.361251e+05   25603.914692   1281.696005    3350.945159

29           3.462797e+06  306774.895833  68938.812500  100071.562500

43           1.274492e+05    3363.437500    184.875000    1340.375000



We took all of them here as you can see now we will take only comment_count because we need to find avarage of comments



"""



# STEP2 here: we will take also "comment_count"





print(youtube.groupby("category_id").mean()["comment_count"])



"""

category_id

1       6003.651813

2       1730.438235

10     10383.559418

15      2474.376054

17      3256.199515

19      1833.011450

20      7578.383607

22      4491.865854

23      5432.868554

24      7178.388176

25      1523.749869

26      4992.165997

27      2143.574850

28      3350.945159

29    100071.562500

43      1340.375000

Name: comment_count, dtype: float64





this output came as series if you wanna print out this as dataframe:



"""



# STEP3(optional): if you wanna print out this as dataframe:



print(youtube.groupby("category_id").mean()[["comment_count"]])



"""

category_id               

1              6003.651813

2              1730.438235

10            10383.559418

15             2474.376054

17             3256.199515

19             1833.011450

20             7578.383607

22             4491.865854

23             5432.868554

24             7178.388176

25             1523.749869

26             4992.165997

27             2143.574850

28             3350.945159

29           100071.562500

43             1340.375000



this output came as dataframe 



"""





# HOW TO FIND : THERE ARE HOW MANY VIDEOS IN CATEGORIES



# we need to use "value_count" function here





print(youtube["category_id"].value_counts())



"""

24    5379

10    3366

26    2241

23    1978

25    1907

22    1886

28    1477

1     1324

17    1238

27    1002

15     593

2      340

20     305

19     262

29      48

43      16

Name: category_id, dtype: int64





"""



#HOW TO FIND: ADDING NEW COLUMN WITH INCLUDING LENGHT OF TITLES



#adding in new column here(how to create new column and add in)

youtube["title_lenght"] = youtube["title"].apply(len)



print(youtube)



"""

                                                  title  ... title_lenght

0                     WE WANT TO TALK ABOUT OUR MARRIAGE  ...           34

1      The Trump Presidency: Last Week Tonight with J...  ...           62

2      Racist Superman | Rudy Mancuso, King Bach & Le...  ...           53

3                       Nickelback Lyrics: Real or Fake?  ...           32

4                               I Dare You: GOING BALD!?  ...           24

...                                                  ...  ...          ...

23357                                Why Bridges Move...  ...           19

23358                      Macaroni - A Recipe From 1784  ...           29

23359                           Chris Young - Hangin' On  ...           24

23360      Elderly man making sure his dog won't get wet  ...           45

23361         How to speak like Black Panther - BBC News  ...           42



[23362 rows x 10 columns]



new column added as "title_length" and including length of titles





"""





# HOW TO FIND: THERE ARE HOW MANY TAGS IN VIDEOS FIND IT AND THEN CREATE NEW COLUMN PUT IT IN THIS COLUMN



# first we need to create an function here



def countTag(tags):

    tagList = tags.split("|")

    return len(tagList)



#and here we are adding in new column

youtube["tag_count"] = youtube["tags"].apply(countTag)



print(youtube)



"""

                                                   title  ... tag_count

0                     WE WANT TO TALK ABOUT OUR MARRIAGE  ...         1

1      The Trump Presidency: Last Week Tonight with J...  ...         4

2      Racist Superman | Rudy Mancuso, King Bach & Le...  ...        23

3                       Nickelback Lyrics: Real or Fake?  ...        27

4                               I Dare You: GOING BALD!?  ...        14

...                                                  ...  ...       ...

23357                                Why Bridges Move...  ...         4

23358                      Macaroni - A Recipe From 1784  ...        31

23359                           Chris Young - Hangin' On  ...        25

23360      Elderly man making sure his dog won't get wet  ...        11

23361         How to speak like Black Panther - BBC News  ...        13



[23362 rows x 11 columns]



new column is here as you can see guys



"""



# now we have hard question as challenge if you guys understand this one it means you went long way



#HOW TO: SORTING VIDEOS FORM LARGE TO SMALL BY RATIGN(LIKES AND DISLIKES)

# look for "iteritems" it ill help you to understand better





def likesAnddislikes(likes, dislikes):

    likesList = list()

    dislikesList = list()

    for key, value in likes.iteritems():

        likesList.append(value)

    for key, value in dislikes.iteritems():

        dislikesList.append(value)



    likeanddislikeratio = list()

    for like, dislike in zip(likesList, dislikesList):

        if like + dislike == 0:

            likeanddislikeratio.append(0)

        else:

            likeanddislikeratio.append(like / (like + dislike))

    return likeanddislikeratio





youtube["likes_dislikes"] = likesAnddislikes(youtube["likes"], youtube["dislikes"])





youtube.sort_values(by = "likes_dislikes",ascending = False,inplace=True)



print(youtube)



"""

                                                   title  ... likes_dislikes

8694                                  Christmas Day 2000  ...            1.0

16100                                  Magic Highway USA  ...            1.0

921             #StarOnFox Exclusive Clip for TheYBF.com  ...            1.0

16369  First Responders Arrive To The Collapsed Floor...  ...            1.0

147    Improvising in the style of different classica...  ...            1.0

...                                                  ...  ...            ...

15575       Official XFL Announcement with Vince McMahon  ...            0.0

4752                      The New Snapchat in 60 Seconds  ...            0.0

16590                                    To Our Daughter  ...            0.0

15510   Carolyn's RV Life Spots A Fluffy Kitty In Blythe  ...            0.0

22020         GET READY WITH ME | Halo Beauty Launch Day  ...            0.0



[23362 rows x 12 columns]



likes_dislikes columnd added as you can see guys...



look for "iteritems()" how it works





"""





import pandas as pd

USvideos = pd.read_csv("../input/USvideos.csv")