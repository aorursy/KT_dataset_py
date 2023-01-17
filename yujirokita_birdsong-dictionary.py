import os

import glob

import pandas as pd

from IPython.display import Audio, HTML
df = pd.read_csv('../input/birdsong-recognition/train.csv')

df = df[['ebird_code', 'species']].drop_duplicates('ebird_code').sort_values('ebird_code').reset_index(drop=True)

df['audio_src'] = [sorted(glob.glob(f'../input/birdsong-recognition/train_audio/{code}/*'), key=os.path.getsize)[0] for code in df.ebird_code]

df['image_src'] = [

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/160820341/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64807071/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/37758621/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/59858041/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/124706471/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70583881/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67453361/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64514531/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60412911/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60017891/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64829511/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64971291/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66120941/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65682561/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/39434211/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60329071/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/171534931/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68123441/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68123101/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63666541/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65764731/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/171505231/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68034931/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66038761/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70580971/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64801421/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60411301/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/171636111/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67447491/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64804711/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/32803411/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65681201/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67447361/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/59859171/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67362321/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67379741/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/169495571/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60394861/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67373991/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64973431/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68037151/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67456051/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/113813231/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/71534291/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64806111/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/162294151/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64896271/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/216531741/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68033861/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65761591/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67358881/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68041771/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60292801/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65533581/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68034441/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/71318281/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65610021/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64516241/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/59953191/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64913361/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68036991/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/59867271/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67283591/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/169902221/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65761191/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/59874471/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64972021/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66113881/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64803561/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/39398791/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70580641/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68122541/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63907721/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67364561/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63918061/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63910971/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/171540961/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63739541/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/171454191/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/170310831/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66117571/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60324921/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68930221/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70582301/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66115711/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66027281/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60397941/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/39407511/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/164000111/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63740061/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65684501/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67377581/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65681481/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/171452241/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65617351/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66035091/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67454961/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/120746601/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64972791/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70580031/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66115301/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65533461/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67356541/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64990931/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63893401/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/161342541/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64978511/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60322141/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60314201/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/170864441/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70610211/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60388671/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64836521/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63741611/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70691741/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67363301/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/62999411/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70199711/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68929201/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70628391/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/71319301/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67459811/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63910041/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64456461/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65055691/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68927561/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63736771/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63742431/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/59860711/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67385731/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68044081/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64809651/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65054631/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64973941/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67386591/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/39512491/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65617791/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64822761/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/171271101/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67272481/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/71378621/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64838631/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/37166641/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64984571/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64810681/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/164865521/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65685251/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63902961/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63000191/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64453201/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64798651/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60021841/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68035321/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/160652121/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67471171/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68054751/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60386921/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63667361/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60403261/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70774731/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63743751/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64518191/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63892991/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60312481/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63002141/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68280781/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68929651/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/170855871/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/120747051/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60320581/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64439011/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64892971/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65680391/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/170727571/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/160654851/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67452711/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63919931/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60408691/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67356151/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/94974311/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67276581/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64893901/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65763191/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64895071/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67284371/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68038671/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63911811/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68040881/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65053401/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64995071/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67282981/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63894341/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65763991/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/164867191/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/71533441/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60384771/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63744251/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/71316071/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63895181/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63733281/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66031271/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/39523701/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/60395561/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67474361/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63912421/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65615501/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68934031/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67375011/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/171509391/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/32805051/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64981751/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65681771/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67449631/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/40259851/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64809171/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64824531/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/37180721/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70780501/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67450281/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/59939451/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64834191/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66116661/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/164608031/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64833231/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64991441/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70582731/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/71545031/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64985151/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67460821/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68280061/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/59954771/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/63745741/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/59956021/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67469161/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64980841/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68279821/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65763751/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67472541/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65071971/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65683361/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67378741/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64827111/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67449081/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65617091/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68039391/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64978031/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70695751/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64980371/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66114531/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/38663041/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64829071/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65616101/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/68035611/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64914601/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65533521/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/70581631/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/170865181/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/71547131/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/160814931/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65051951/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/67376561/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/64802871/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/66119101/160',

    'https://test.cdn.download.ams.birds.cornell.edu/api/v1/asset/65760791/160'

]
html = "<table>"

for d in df.to_dict('records'):

    html += f"""

    <tr>

        <td>{d['ebird_code']}</td>

        <td>{d['species']}</td>

        <td><img src="{d['image_src']}" width="100"></td>

        <td><audio controls><source src="{Audio(d['audio_src']).src_attr()}" type="audio/wav"></audio></td>

    </tr>

    """

html += "</table>"



HTML(html)