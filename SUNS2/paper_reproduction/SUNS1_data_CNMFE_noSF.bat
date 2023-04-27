@REM Generate sparse GT masks
python generate_sparse_GT.py "../../data/data_CNMFE/*/GT Masks"

@REM Training on blood_vessel_10Hz
python train_CNN_params_data_CNMFE_crop.py 2 0 3 FISSA 0

@REM Argument meanings (same for training and testing scripts):
@REM 1. SNR threshold
@REM 2. Spatial filter kernal size
@REM 3. The maximum video index that parameter optimization runs through
@REM 4. Unmixing algorithm (TUnCaT or FISSA)
@REM 5. Video index of the CNMF-E dataset

@REM Testing on blood_vessel_10Hz
python test_batch_data_CNMFE_crop.py 2 0 3 FISSA 0


@REM Training on PFC4_15Hz
python train_CNN_params_data_CNMFE_crop.py 2 0 3 FISSA 1

@REM Testing on PFC4_15Hz
python test_batch_data_CNMFE_crop.py 2 0 3 FISSA 1


@REM Training on bma22_epm
python train_CNN_params_data_CNMFE_crop.py 2 0 3 FISSA 2

@REM Testing on bma22_epm
python test_batch_data_CNMFE_crop.py 2 0 3 FISSA 2


@REM Training on CaMKII_120_TMT
python train_CNN_params_data_CNMFE_crop.py 2 0 3 FISSA 3

@REM Testing on CaMKII_120_TMT
python test_batch_data_CNMFE_crop.py 2 0 3 FISSA 3

