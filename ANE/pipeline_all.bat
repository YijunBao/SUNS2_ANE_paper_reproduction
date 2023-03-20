@REM Training pipeline
@REM Calculate the weight of each frame in each block for GT data, 
@REM and propose putative missing neurons using clustering for GT data
@REM Use GUI to manually determine whether each putative missing neuron is valid
matlab -nojvm -nodesktop -r pre_CNN_data_CNMFE_crop_GT_blockwise_weighted_sum
@REM Use GUI to label missing neurons
matlab -nojvm -nodesktop -r use_GUI_find_missing
@REM Manually move the GUI output to the GUI input folder
@REM Combine the GUI input and output to CNN input
matlab -nojvm -nodesktop -r post_GUI_data_CNMFE_crop_GT_blockwise
@REM Train CNN discriminator
python train_CNN_classifier_blockwise_cv.py classifier_res0 0 nomask 0 1

@REM Testing pipeline
@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r pre_CNN_data_CNMFE_crop_SUNS_blockwise
@REM matlab -nojvm -nodesktop -r add_weights_data_CNMFE_crop_SUNS_blockwise
@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_blockwise_cv.py classifier_res0 0 nomask 0 1
@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r post_CNN_data_CNMFE_crop_blockwise_cv

@REM Used functions
@REM Calculate the weight of each frame in each block for GT data
frame_weight_blockwise.m
@REM Propose putative missing neurons using clustering for test data
find_missing_multi_frame_blockwise_weighted_sum.m
@REM GUI to manually determine whether each putative missing neuron is valid
GUI_find_missing_4train_blockwise_weighted_sum.m
@REM Data augmentation
image_preprocessing_keras_shuffle_thb.py
@REM CNN discriminator 
CNN_classifier.py

@REM python "C:\Matlab Files\timer\timer_start_next.py"
@REM python "C:\Matlab Files\timer\timer_start_from_no_file.py"
@REM python "C:\Matlab Files\timer\timer_stop.py"

@REM shutdown -s -t 60
@REM shutdown -a
