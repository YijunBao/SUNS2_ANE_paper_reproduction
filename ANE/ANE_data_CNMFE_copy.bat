@REM Traing pipeline of ANE on the CNMF-E dataset

@REM Before training ANE, run SUNS2 first (at least after pre-processing, until 
@REM the sub-folder 'complete_TUnCaT\TUnCaT\alpha= 1.000' has all videos)

@REM Randomly drop out neurons from GT set
@REM matlab -nojvm -nodesktop -r "data_ind = 1; dropout_GTMasks_data_CNMFE; exit"
@REM matlab -nojvm -nodesktop -r "data_ind = 1; pre_CNN_data_CNMFE_drop_GT_blockwise_weighted_sum_mm; exit"
@REM matlab -nojvm -nodesktop -r "data_ind = 1; post_GUI_data_CNMFE_drop_GT_blockwise; exit"

@REM matlab -nojvm -nodesktop -r "data_ind = 2; dropout_GTMasks_data_CNMFE; exit"
@REM matlab -nojvm -nodesktop -r "data_ind = 2; pre_CNN_data_CNMFE_drop_GT_blockwise_weighted_sum_mm; exit"
@REM matlab -nojvm -nodesktop -r "data_ind = 2; post_GUI_data_CNMFE_drop_GT_blockwise; exit"

@REM matlab -nojvm -nodesktop -r "data_ind = 3; dropout_GTMasks_data_CNMFE; exit"
@REM matlab -nojvm -nodesktop -r "data_ind = 3; pre_CNN_data_CNMFE_drop_GT_blockwise_weighted_sum_mm; exit"
@REM matlab -nojvm -nodesktop -r "data_ind = 3; post_GUI_data_CNMFE_drop_GT_blockwise; exit"

@REM Train CNN discriminator
@REM python train_CNN_classifier_data_CNMFE_blockwise_cv_drop.py 0
@REM python train_CNN_classifier_data_CNMFE_blockwise_cv_drop.py 1
@REM python train_CNN_classifier_data_CNMFE_blockwise_cv_drop.py 2

@REM matlab -nojvm -nodesktop -r "data_ind = 1; pre_CNN_data_CNMFE_crop_SUNS_blockwise_mm; exit"
@REM matlab -nojvm -nodesktop -r "data_ind = 2; pre_CNN_data_CNMFE_crop_SUNS_blockwise_mm; exit"
@REM matlab -nojvm -nodesktop -r "data_ind = 3; pre_CNN_data_CNMFE_crop_SUNS_blockwise_mm; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_data_CNMFE_blockwise_cv_drop.py 0 "complete_TUnCaT\\4816[1]th5\\output_masks"
python test_CNN_classifier_data_CNMFE_blockwise_cv_drop.py 1 "complete_TUnCaT\\4816[1]th4\\output_masks"
python test_CNN_classifier_data_CNMFE_blockwise_cv_drop.py 2 "complete_TUnCaT\\4816[1]th4\\output_masks"

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "data_ind = 1; dir_SUNS_sub = 'complete_TUnCaT\\4816[1]th5'; post_CNN_data_CNMFE_drop_blockwise_cv"
matlab -nojvm -nodesktop -r "data_ind = 2; dir_SUNS_sub = 'complete_TUnCaT\\4816[1]th4'; post_CNN_data_CNMFE_drop_blockwise_cv"
matlab -nojvm -nodesktop -r "data_ind = 3; dir_SUNS_sub = 'complete_TUnCaT\\4816[1]th4'; post_CNN_data_CNMFE_drop_blockwise_cv"
