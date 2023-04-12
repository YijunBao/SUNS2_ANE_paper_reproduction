@REM Test pipeline of ANE on the CNMF-E dataset

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r pre_CNN_data_CNMFE_crop_SUNS_blockwise_mm

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_data_CNMFE_blockwise_cv_drop.py 0.8exp(-8) "complete_TUnCaT\\4816[1]th4\\output_masks"

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r post_CNN_data_CNMFE_drop_blockwise_cv_res0_avg
