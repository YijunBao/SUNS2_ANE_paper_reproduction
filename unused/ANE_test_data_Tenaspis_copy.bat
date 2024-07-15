@REM Test pipeline of ANE on the Tenaspis dataset

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "dir_SUNS_sub = fullfile('complete_TUnCaT_SF25','4816[1]th6'); clustering_mm_SUNS_data_TENASPIS_copy; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_cv_data_TENASPIS_copy.py "complete_TUnCaT_SF25/4816[1]th6/output_masks"

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "dir_SUNS_sub = fullfile('complete_TUnCaT_SF25','4816[1]th6'); post_CNN_cv_data_TENASPIS_copy; exit"

python "C:\Matlab Files\timer\timer_stop_2.py"