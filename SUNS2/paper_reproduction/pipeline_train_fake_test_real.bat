REM REM Training pipeline
REM python train_CNN_params_noSF_J115.py
REM python train_CNN_params_noSF_J123.py
REM python train_CNN_params_noSF_K53.py
REM python train_CNN_params_noSF_YST.py

REM python "C:\Matlab Files\timer\timer_start_next.py"
REM Run SUNS batch
REM python test_batch_noSF_CM.py
REM Run SUNS online
@REM python test_online_noSF_CM.py
REM Run SUNS batch
REM python test_batch_complete_CM.py
REM Run SUNS online
@REM python test_online_complete_CM.py
@REM python "C:\Matlab Files\timer\timer_stop.py"

@REM python unmix_crop.py 0 complete_TUnCaT
@REM python unmix_crop.py 0 CNMFE
@REM python unmix_crop.py 0 complete_FISSA
@REM python unmix_crop.py 0 min1pipe
@REM python unmix_crop.py 1 complete_TUnCaT
@REM python unmix_crop.py 1 complete_FISSA
@REM python unmix_crop.py 1 CNMFE
@REM python unmix_crop.py 1 min1pipe

@REM python train_CNN_params_vary_CNN_add_data_TENASPIS.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.01_rotate
@REM python test_batch_vary_CNN_add_data_TENASPIS.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.01_rotate
python test_batch_vary_CNN_data_TENASPIS_train_add.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.01_rotate

@REM python train_CNN_params_vary_CNN_add_data_TENASPIS.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.005_rotate
@REM python test_batch_vary_CNN_add_data_TENASPIS.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.005_rotate
python test_batch_vary_CNN_data_TENASPIS_train_add.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.005_rotate

@REM python train_CNN_params_vary_CNN_add_data_TENASPIS.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.02_rotate
@REM python test_batch_vary_CNN_add_data_TENASPIS.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.02_rotate
python test_batch_vary_CNN_data_TENASPIS_train_add.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.02_rotate

@REM python train_CNN_params_vary_CNN_add_data_TENASPIS.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.003_rotate
@REM python test_batch_vary_CNN_add_data_TENASPIS.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.003_rotate
python test_batch_vary_CNN_data_TENASPIS_train_add.py 3 4 [1] elu True 4816[1] 5 25 9 add_neurons_0.003_rotate
@REM python "C:\Matlab Files\timer\timer_stop.py"
