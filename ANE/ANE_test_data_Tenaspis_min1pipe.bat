@REM Test pipeline of ANE on the Tenaspis dataset initialized by the MIN1PIPE output

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "dir_sub_save = 'min1pipe/cv_save_20230111'; clustering_mm_SUNS_data_TENASPIS_min1pipe; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_cv_data_TENASPIS.py "min1pipe/cv_save_20230111"

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "dir_method = 'min1pipe'; save_date = '20230111'; post_CNN_cv_data_TENASPIS_min1pipe"
