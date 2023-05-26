@REM Training pipeline of ANE on the CNMF-E dataset

@REM Before training ANE, run SUNS2 first (at least after pre-processing, until 
@REM the sub-folder 'SUNS_TUnCaT_{}\TUnCaT\alpha= 1.000' has all videos)

@REM Randomly drop out neurons from GT set
python dropout_GTMasks_data_CNMFE_crop.py 3

@REM Calculate the weight of each frame in each block for GT data, 
@REM and propose putative missing neurons using clustering for GT data
@REM Determine true or false neurons by matching to dropped out GT neurons
python clustering_mm_GT_drop_data_CNMFE_crop.py 3

@REM Train CNN discriminator
python train_CNN_classifier_cv_drop_data_CNMFE_crop_py.py 3



@REM Test pipeline of ANE on the CNMF-E dataset

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
python clustering_mm_SUNS_data_CNMFE_crop.py "SUNS_TUnCaT_SF50/4816[1]th4/output_masks" 3

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_cv_data_CNMFE_crop_py.py "SUNS_TUnCaT_SF50/4816[1]th4/output_masks" 3

@REM Add valid missing neurons to output neurons
python post_CNN_cv_data_CNMFE_crop.py "SUNS_TUnCaT_SF50/4816[1]th4/output_masks" 3
