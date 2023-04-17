@REM Training pipeline of ANE on the CNMF-E dataset

@REM Before training ANE, run SUNS2 first (at least after pre-processing, until 
@REM the sub-folder 'complete_TUnCaT\TUnCaT\alpha= 1.000' has all videos)

@REM Randomly drop out neurons from GT set
matlab -nojvm -nodesktop -r "data_ind = 4; dropout_GTMasks_data_CNMFE; exit"

@REM Calculate the weight of each frame in each block for GT data, 
@REM and propose putative missing neurons using clustering for GT data
matlab -nojvm -nodesktop -r "data_ind = 4; pre_CNN_data_CNMFE_drop_GT_blockwise_weighted_sum_mm; exit"

@REM Determine true or false neurons by matching to dropped out GT neurons
matlab -nojvm -nodesktop -r "data_ind = 4; post_GUI_data_CNMFE_drop_GT_blockwise; exit"

@REM Train CNN discriminator
python train_CNN_classifier_data_CNMFE_cv_drop.py 3
