@REM matlab -nojvm -nodesktop -r find_missing_weights_data_CNMFE_crop
@REM python test_CNN_classifier_crop.py
@REM matlab -nojvm -nodesktop -r find_missing_CNN_final_data_CNMFE_crop

@REM python "C:\Matlab Files\timer\timer_start_next.py"
@REM python "C:\Matlab Files\timer\timer_start_from_no_file.py"
@REM python "C:\Matlab Files\timer\timer_stop.py"

@REM shutdown -s -t 60
@REM shutdown -a

python train_CNN_classifier_data_TENASPIS_cv.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask
python train_CNN_classifier_data_TENASPIS_cv.py classifier_res0 0 mask 0 1 _weighted_sum_unmask
python train_CNN_classifier_data_TENASPIS_cv.py classifier_res0 0 mask 0 2 _weighted_sum_unmask
python train_CNN_classifier_data_TENASPIS_cv.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask
python train_CNN_classifier_data_TENASPIS_cv.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask
python train_CNN_classifier_data_TENASPIS_cv.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask
python train_CNN_classifier_data_TENASPIS_cv.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask

@REM python test_CNN_classifier_blockwise_cv_folder_sub.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask
@REM python test_CNN_classifier_blockwise_cv_folder_sub.py classifier_res0 0 mask 0 1 _weighted_sum_unmask
@REM python test_CNN_classifier_blockwise_cv_folder_sub.py classifier_res0 0 mask 0 2 _weighted_sum_unmask
@REM python test_CNN_classifier_blockwise_cv_folder_sub.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask
@REM python test_CNN_classifier_blockwise_cv_folder_sub.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask
@REM python test_CNN_classifier_blockwise_cv_folder_sub.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask
@REM python test_CNN_classifier_blockwise_cv_folder_sub.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask

@REM python train_CNN_classifier_blockwise_cv.py classifier_res1 0 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res2 0 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res3 0 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res4 0 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res5 0 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res6 0 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res7 0 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res8 0 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res9 0 nomask 0 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 nomask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 nomask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 nomask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 nomask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 nomask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 nomask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 nomask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 nomask 1 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res1 0 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res2 0 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res3 0 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res4 0 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res5 0 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res6 0 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res7 0 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res8 0 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res9 0 mask 0 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 mask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 mask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 mask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 mask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 mask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 mask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 mask 1 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res1 0 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res2 0 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res3 0 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res4 0 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res5 0 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res6 0 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res7 0 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res8 0 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res9 0 bmask 0 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 bmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 bmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 bmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 bmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 bmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 bmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 bmask 1 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res1 0 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res2 0 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res3 0 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res4 0 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res5 0 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res6 0 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res7 0 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res8 0 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res9 0 Xmask 0 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 Xmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 Xmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 Xmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 Xmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 Xmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 Xmask 1 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 Xmask 1 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res1 0 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res2 0 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res3 0 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res4 0 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res5 0 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res6 0 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res7 0 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res8 0 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res9 0 mask 0 2

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 mask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 mask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 mask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 mask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 mask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 mask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 mask 1 2

@REM python train_CNN_classifier_blockwise_cv.py classifier_res1 0 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res2 0 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res3 0 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res4 0 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res5 0 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res6 0 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res7 0 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res8 0 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res9 0 bmask 0 2

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 bmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 bmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 bmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 bmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 bmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 bmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 bmask 1 2

@REM python train_CNN_classifier_blockwise_cv.py classifier_res1 0 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res2 0 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res3 0 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res4 0 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res5 0 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res6 0 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res7 0 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res8 0 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res9 0 Xmask 0 2

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 Xmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 Xmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 Xmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 Xmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 Xmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 Xmask 1 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 Xmask 1 2

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 mask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 mask 0 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 bmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 bmask 0 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 Xmask 0 1
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 Xmask 0 1

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 mask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 mask 0 2

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 bmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 bmask 0 2

@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 2 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 3 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 4 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 5 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 6 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 7 Xmask 0 2
@REM python train_CNN_classifier_blockwise_cv.py classifier_res0 8 Xmask 0 2
