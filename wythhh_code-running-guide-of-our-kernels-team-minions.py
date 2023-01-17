'''
Hi Kagglers,

It is our hornor to share our thoughts and codes with you. And we here provide a guide on running our code so that you would not be
driven mad by so many messy kernels we upload here(we are sorry for the inconvenience but we have so many files...).

We forgot to sed seed for some of our models during competition but we do set seed this time and you will get AUC of 0.97504 of 
private board using the code we provided.

There are two ways to run our code:
1. (Recommeded Way): Clone files from our github: https://github.com/liuxi94/WiDS_2018, and follow README.md to run code step by step. Keep in mind
that we don't upload train.csv and test.csv to our github beacuse of the size limit on uploading. Thus please put train.csv and 
test.csv to folder 'part1' and folder 'part2' before running the code.
2. Run on kaggle kernels: 
    Before running, please check the path of output file.
    A.Run kernel part 1.1:https://www.kaggle.com/wythhh/team-minions-part1-1. And it will produce two prediction files.
    B.Run kernel part 1.2:https://www.kaggle.com/liuxi94/team-minions-part-1-2. Code in part 1.2 needs to import a module, but we 
    don't know how to import that on kaggle kernel, thus we upload the moduel here:https://www.kaggle.com/liuxi94/team-minions-meanencoder
    This step will also produce two prediction files.
    C.(Optional) kernel part 2.1 actually does a feature selection and we upload the new dataset with selected feature to kernel part 2.2
    so that you dont have to run part 2.1.
    D.Run kernel part 2.2:https://www.kaggle.com/wythhh/team-minions-part2-2. And it will produce three prediction files.
    E.Use this script:https://www.kaggle.com/wythhh/team-minions-avg, to average the result from step A and step B
    F.Also use the AVG script to average ‘lgb_seed77_top150_grouping_no_elimination.csv’ and ‘lgb_seed99_top150_selected_grouping.csv’ from
    step D.
    G.Also use the AVG script to average the result from step F and the ‘lgb_seed88_top150_selected_grouping.csv’ from step D.
    H.Use the blend script:https://www.kaggle.com/wythhh/team-minions-blend, to get the final result. 'avg1.csv' in this script should be
    the result of step E and 'avg3.csv' in this script should be the result of step G.
That's all, thank you.
'''
