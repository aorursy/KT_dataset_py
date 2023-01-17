import pandas as pd

from pathlib import Path



pd.set_option("display.max_colwidth", 80)
root = Path("../input")

comp = pd.read_csv(root.joinpath("Competitions.csv"))

comp_tags = pd.read_csv(root.joinpath("CompetitionTags.csv"))

tags = pd.read_csv(root.joinpath("Tags.csv"))



featured_comp = comp[comp.HostSegmentTitle == "Featured"]

featured_comp.EnabledDate = pd.to_datetime(featured_comp.EnabledDate)



# reference: https://www.kaggle.com/sudalairajkumar/winning-solutions-of-kaggle-competitions

def get_comp_tags(comp_id):

    temp_df = comp_tags[comp_tags["CompetitionId"]==comp_id]

    temp_df = pd.merge(temp_df, tags, left_on="TagId", right_on="Id")

    tags_str = "Tags : "

    for ind, row in temp_df.iterrows():

        tags_str += row["Name"] + ", "

    return tags_str.strip(", ")



query = (featured_comp.EnabledDate >= "2018-01-01") & (featured_comp.EnabledDate <= "2018-12-31")

recently_comp = featured_comp[query]

recently_comp["tags"] = recently_comp.Id.apply(get_comp_tags)
recently_comp[["Slug", "tags"]]