@REM Test pipeline of ANE on the simulated dataset

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
matlab -nojvm -nodesktop -r "dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF25','4816[1]th4'); data_name='lowBG=5e+03,poisson=1'; clustering_mm_SUNS_simu; exit"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_cv_simu.py "SUNS_TUnCaT_SF25/4816[1]th4/output_masks" lowBG=5e+03,poisson=1

@REM Add valid missing neurons to output neurons
matlab -nojvm -nodesktop -r "dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF25','4816[1]th4'); data_name='lowBG=5e+03,poisson=1'; post_CNN_cv_simu"
