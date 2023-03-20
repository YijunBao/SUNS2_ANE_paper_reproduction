@REM matlab -nojvm -nodesktop -r find_missing_weights_data_CNMFE_crop
@REM python test_CNN_classifier_crop.py
@REM matlab -nojvm -nodesktop -r find_missing_CNN_final_data_CNMFE_crop

@REM python "C:\Matlab Files\timer\timer_start_next.py"
@REM python "C:\Matlab Files\timer\timer_start_from_no_file.py"
@REM python "C:\Matlab Files\timer\timer_stop.py"

@REM shutdown -s -t 60
@REM shutdown -a

python train_CNN_classifier_blockwise_cv.py classifier_res0 0 nomask 0 1
python test_CNN_classifier_blockwise_cv.py classifier_res0 0 nomask 0 1
