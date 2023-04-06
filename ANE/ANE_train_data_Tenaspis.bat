@REM Training pipeline of ANE on the Tenaspis dataset

@REM Randomly drop out neurons from GT set
matlab -nojvm -nodesktop -r dropout_GTMasks_data_TENASPIS
@REM Calculate the weight of each frame in each block for GT data, 
@REM and propose putative missing neurons using clustering for GT data
@REM Use GUI to manually determine whether each putative missing neuron is valid
matlab -nojvm -nodesktop -r pre_CNN_data_TENASPIS_drop_GT_blockwise_weighted_sum_mm
@REM @REM Use GUI to label missing neurons
@REM matlab -nojvm -nodesktop -r use_GUI_find_missing
@REM Manually move the GUI output to the GUI input folder
@REM Combine the GUI input and output to CNN input
matlab -nojvm -nodesktop -r post_GUI_data_TENASPIS_drop_GT_blockwise
@REM Train CNN discriminator
python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
