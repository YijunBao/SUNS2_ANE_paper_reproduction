@REM Test pipeline of ANE on the CNMF-E dataset

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF50','4816[1]th4'); data_ind = 4; clustering_mm_SUNS_data_CNMFE_crop; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_cv_data_CNMFE_crop.py "SUNS_TUnCaT_SF50/4816[1]th4/output_masks" 3

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF50','4816[1]th4'); data_ind = 4; post_CNN_cv_data_CNMFE_crop"
