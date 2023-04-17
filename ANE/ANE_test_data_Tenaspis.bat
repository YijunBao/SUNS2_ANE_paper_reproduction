@REM Test pipeline of ANE on the Tenaspis dataset

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "dir_SUNS_sub = fullfile('complete_TUnCaT','4816[1]th5'); pre_CNN_data_TENASPIS_SUNS_blockwise_mm; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_data_TENASPIS_cv_drop.py "complete_TUnCaT/4816[1]th5/output_masks"

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "dir_SUNS_sub = fullfile('complete_TUnCaT','4816[1]th5'); post_CNN_data_TENASPIS_drop_blockwise_cv"
