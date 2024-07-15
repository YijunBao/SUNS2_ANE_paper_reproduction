@REM Generate sparse GT masks
@REM python generate_sparse_GT.py "../../data/data_TENASPIS/added_refined_masks/GT Masks"

@REM Training
@REM python train_CNN_params_data_TENASPIS.py 5 25 7 TUnCaT

@REM Argument meanings (same for training and testing scripts):
@REM 1. SNR threshold
@REM 2. Spatial filter kernal size
@REM 3. The maximum video index that parameter optimization runs through
@REM 4. Unmixing algorithm (TUnCaT or FISSA)
@REM 5. Video index of the CNMF-E dataset
@REM python "C:\Matlab Files\timer\timer_start_next_2.py"

@REM Testing
@REM python test_batch_data_TENASPIS_copy.py 3 25 7 TUnCaT
@REM python test_batch_data_TENASPIS_copy.py 4 25 7 TUnCaT
@REM python test_batch_data_TENASPIS_copy.py 5 25 7 TUnCaT
@REM python test_batch_data_TENASPIS_original.py 6 25 7 TUnCaT
@REM python test_batch_data_TENASPIS_original.py 6 25 7 TUnCaT
@REM python test_batch_data_TENASPIS_original.py 5 25 7 TUnCaT
@REM python test_batch_data_TENASPIS_original.py 4 25 7 TUnCaT
@REM python test_batch_data_TENASPIS_original.py 3 25 7 TUnCaT

@REM python test_batch_data_TENASPIS_original.py 2 25 7 FISSA
@REM python test_batch_data_TENASPIS_original.py 3 25 7 FISSA
@REM python test_batch_data_TENASPIS_original.py 4 25 7 FISSA

@REM python train_CNN_params_data_TENASPIS_lowP.py 2 0 3 FISSA
@REM python train_CNN_params_data_TENASPIS_lowP.py 2 0 7 FISSA
@REM python test_batch_data_TENASPIS_copy.py 2 0 7 FISSA

@REM python train_CNN_params_data_TENASPIS_lowP.py 3 0 3 FISSA
@REM python train_CNN_params_data_TENASPIS_lowP.py 3 0 7 FISSA
@REM python test_batch_data_TENASPIS_copy.py 3 0 7 FISSA

@REM python train_CNN_params_data_TENASPIS_lowP.py 4 0 3 FISSA
@REM python train_CNN_params_data_TENASPIS_lowP.py 4 0 7 FISSA
@REM python test_batch_data_TENASPIS_copy.py 4 0 7 FISSA

python train_CNN_params_data_CNMFE_crop_lowP.py 2 0 3 FISSA 0
python test_batch_data_CNMFE_crop_copy.py 2 0 3 FISSA 0

python train_CNN_params_data_CNMFE_crop_lowP.py 3 0 3 FISSA 0
python test_batch_data_CNMFE_crop_copy.py 3 0 3 FISSA 0

python train_CNN_params_data_CNMFE_crop_lowP.py 4 0 3 FISSA 0
python test_batch_data_CNMFE_crop_copy.py 4 0 3 FISSA 0

python train_CNN_params_data_CNMFE_crop_lowP.py 2 0 3 FISSA 3
python test_batch_data_CNMFE_crop_copy.py 2 0 3 FISSA 3

python train_CNN_params_data_CNMFE_crop_lowP.py 3 0 3 FISSA 3
python test_batch_data_CNMFE_crop_copy.py 3 0 3 FISSA 3

python train_CNN_params_data_CNMFE_crop_lowP.py 4 0 3 FISSA 3
python test_batch_data_CNMFE_crop_copy.py 4 0 3 FISSA 3
