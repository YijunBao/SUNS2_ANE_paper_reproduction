python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=5e+03,poisson=1

python test_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=5e+03,poisson=1


python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)

python test_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)

python test_CNN_classifier_data_TENASPIS_cv_min1pipe.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)

python test_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)

python train_CNN_classifier_data_TENASPIS_cv.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask

python test_CNN_classifier_blockwise_cv_folder_sub.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask

python train_CNN_classifier_blockwise_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-8)

python test_CNN_classifier_blockwise_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-8)

@REM @REM Used functions
@REM @REM Calculate the weight of each frame in each block for GT data
@REM frame_weight_blockwise.m
@REM @REM Propose putative missing neurons using clustering for test data
@REM find_missing_multi_frame_blockwise_weighted_sum.m
@REM @REM GUI to manually determine whether each putative missing neuron is valid
@REM GUI_find_missing_4train_blockwise_weighted_sum.m
@REM @REM Data augmentation
@REM image_preprocessing_keras_shuffle_thb.py
@REM @REM CNN discriminator 
@REM CNN_classifier.py
