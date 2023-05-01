@REM Test pipeline of ANE on the Tenaspis dataset initialized by the CNMF-E output

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "dir_sub_save = 'CNMFE/cv_save_20221221'; clustering_mm_SUNS_data_TENASPIS_min1pipe; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_cv_data_TENASPIS.py "CNMFE/cv_save_20221221"

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "dir_method = 'CNMFE'; save_date = '20221221'; post_CNN_cv_data_TENASPIS_min1pipe"
