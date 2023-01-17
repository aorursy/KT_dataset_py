import turicreate
sf = turicreate.SFrame('../input/basicml-lecture1/people_wiki.sframe')
sf
sf.num_rows()
sf[-1]['name']
sf[sf['name']=='Harpdog Brown']['text']
sf.sort('text', ascending=True)[0]['name']
