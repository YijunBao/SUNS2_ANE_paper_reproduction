@REM Training pipeline of ANE on the CNMF-E dataset

@REM Before training ANE, run SUNS2 first (at least after pre-processing, until 
@REM the sub-folder 'SUNS_TUnCaT_{}\TUnCaT\alpha= 1.000' has all videos)


@REM blood_vessel_10Hz
@REM Randomly drop out neurons from GT set
matlab -nojvm -nodesktop -r "data_ind = 1; dropout_GTMasks_data_CNMFE_crop; exit"

@REM Calculate the weight of each frame in each block for GT data, 
@REM and propose putative missing neurons using clustering for GT data
matlab -nojvm -nodesktop -r "data_ind = 1; clustering_mm_GT_drop_data_CNMFE_crop; exit"

@REM Determine true or false neurons by matching to dropped out GT neurons
matlab -nojvm -nodesktop -r "data_ind = 1; match_GT_drop_data_CNMFE_crop; exit"

@REM Train CNN discriminator
python train_CNN_classifier_cv_drop_data_CNMFE_crop.py 0


@REM PFC4_15Hz
@REM Randomly drop out neurons from GT set
matlab -nojvm -nodesktop -r "data_ind = 2; dropout_GTMasks_data_CNMFE_crop; exit"

@REM Calculate the weight of each frame in each block for GT data, 
@REM and propose putative missing neurons using clustering for GT data
matlab -nojvm -nodesktop -r "data_ind = 2; clustering_mm_GT_drop_data_CNMFE_crop; exit"

@REM Determine true or false neurons by matching to dropped out GT neurons
matlab -nojvm -nodesktop -r "data_ind = 2; match_GT_drop_data_CNMFE_crop; exit"

@REM Train CNN discriminator
python train_CNN_classifier_cv_drop_data_CNMFE_crop.py 1


@REM bma22_epm
@REM Randomly drop out neurons from GT set
matlab -nojvm -nodesktop -r "data_ind = 3; dropout_GTMasks_data_CNMFE_crop; exit"

@REM Calculate the weight of each frame in each block for GT data, 
@REM and propose putative missing neurons using clustering for GT data
matlab -nojvm -nodesktop -r "data_ind = 3; clustering_mm_GT_drop_data_CNMFE_crop; exit"

@REM Determine true or false neurons by matching to dropped out GT neurons
matlab -nojvm -nodesktop -r "data_ind = 3; match_GT_drop_data_CNMFE_crop; exit"

@REM Train CNN discriminator
python train_CNN_classifier_cv_drop_data_CNMFE_crop.py 2


@REM CaMKII_120_TMT Exposure_5fps
@REM Randomly drop out neurons from GT set
matlab -nojvm -nodesktop -r "data_ind = 4; dropout_GTMasks_data_CNMFE_crop; exit"

@REM Calculate the weight of each frame in each block for GT data, 
@REM and propose putative missing neurons using clustering for GT data
matlab -nojvm -nodesktop -r "data_ind = 4; clustering_mm_GT_drop_data_CNMFE_crop; exit"

@REM Determine true or false neurons by matching to dropped out GT neurons
matlab -nojvm -nodesktop -r "data_ind = 4; match_GT_drop_data_CNMFE_crop; exit"

@REM Train CNN discriminator
python train_CNN_classifier_cv_drop_data_CNMFE_crop.py 3
