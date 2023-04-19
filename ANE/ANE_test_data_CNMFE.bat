@REM Test pipeline of ANE on the CNMF-E dataset

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)


@REM blood_vessel_10Hz
@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "data_ind = 1; pre_CNN_data_CNMFE_crop_SUNS_blockwise_mm; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_data_CNMFE_cv_drop.py "complete_TUnCaT/4816[1]th5/output_masks" 0

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "data_ind = 1; dir_SUNS_sub = fullfile('complete_TUnCaT','4816[1]th5');  post_CNN_data_CNMFE_drop_blockwise_cv"


@REM PFC4_15Hz
@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "data_ind = 2; pre_CNN_data_CNMFE_crop_SUNS_blockwise_mm; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_data_CNMFE_cv_drop.py "complete_TUnCaT/4816[1]th4/output_masks" 1

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "data_ind = 2; dir_SUNS_sub = fullfile('complete_TUnCaT','4816[1]th4'); post_CNN_data_CNMFE_drop_blockwise_cv"


@REM bma22_epm
@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "data_ind = 3; pre_CNN_data_CNMFE_crop_SUNS_blockwise_mm; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_data_CNMFE_cv_drop.py "complete_TUnCaT/4816[1]th4/output_masks" 2

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "data_ind = 3; dir_SUNS_sub = fullfile('complete_TUnCaT','4816[1]th4'); post_CNN_data_CNMFE_drop_blockwise_cv"


@REM CaMKII_120_TMT Exposure_5fps
@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "data_ind = 4; pre_CNN_data_CNMFE_crop_SUNS_blockwise_mm; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_data_CNMFE_cv_drop.py "complete_TUnCaT/4816[1]th4/output_masks" 3

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "data_ind = 4; dir_SUNS_sub = fullfile('complete_TUnCaT','4816[1]th4'); post_CNN_data_CNMFE_drop_blockwise_cv"
