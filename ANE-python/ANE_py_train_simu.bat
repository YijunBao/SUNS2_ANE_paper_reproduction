@REM Traing pipeline of ANE on the simulated dataset

@REM Before training ANE, run SUNS2 first (at least after pre-processing, until 
@REM the sub-folder 'SUNS_TUnCaT_{}\TUnCaT\alpha= 1.000' has all videos)

@REM Randomly drop out neurons from GT set
python dropout_GTMasks_simu.py lowBG=5e+03,poisson=1

@REM Calculate the weight of each frame in each block for GT data, 
@REM and propose putative missing neurons using clustering for GT data
python clustering_mm_GT_drop_simu.py lowBG=5e+03,poisson=1

@REM Train CNN discriminator
python train_CNN_classifier_cv_drop_simu_py.py lowBG=5e+03,poisson=1
