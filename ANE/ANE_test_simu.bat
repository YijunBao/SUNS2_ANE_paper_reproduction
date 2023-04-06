@REM Test pipeline of ANE on the simulated dataset

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r pre_CNN_simu_SUNS_blockwise_mm
@REM matlab -nojvm -nodesktop -r add_weights_data_CNMFE_crop_SUNS_blockwise
@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=5e+03,poisson=1
@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r post_CNN_simu_drop_blockwise_cv_res0_avg
