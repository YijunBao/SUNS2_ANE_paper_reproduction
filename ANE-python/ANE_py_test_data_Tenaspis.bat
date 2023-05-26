@REM Test pipeline of ANE on the Tenaspis dataset

@REM Before ANE, run SUNS2 first (or any other segmentation algorithm)

@REM Calculate the weight of each frame in each block for test data, 
@REM and propose putative missing neurons using clustering for test data
python clustering_mm_SUNS_data_TENASPIS.py "SUNS_TUnCaT_SF25/4816[1]th6/output_masks"

@REM Apply CNN discriminator to determine whether each putative missing neuron is valid
python test_CNN_classifier_cv_data_TENASPIS_py.py "SUNS_TUnCaT_SF25/4816[1]th6/output_masks"

@REM Add valid missing neurons to output neurons
python post_CNN_cv_data_TENASPIS.py "SUNS_TUnCaT_SF25/4816[1]th6/output_masks"


@REM python "C:\Matlab Files\timer\timer_stop.py"
