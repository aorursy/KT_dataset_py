import pandas as pd



sub = pd.read_csv('../input/best-private-sub/effnet6_image_chrisgroupkfold_fastplatoschedule_cutoutshiftsrotates_customaugs_denseelu_mdropout_difflrs_distributed_bigbatch_nodupls_384res_extrenaldata1817_batchacum4_SWA_6ttamicroscope6randomttas_meanfolds.csv')

sub.to_csv('submission.csv', index=False)