@REM Generate sparse GT masks
python generate_sparse_GT.py "../../data/data_CNMFE/CaMKII_120_TMT Exposure_5fps/GT Masks"

@REM Training on CaMKII_120_TMT
python train_CNN_data_CNMFE_crop.py 4 50 0 TUnCaT 3
python train_params_data_CNMFE_crop.py 4 50 3 TUnCaT 3

@REM Argument meanings (same for training and testing scripts):
@REM 1. SNR threshold
@REM 2. Spatial filter kernal size
@REM 3. The maximum video index that parameter optimization runs through
@REM 4. Unmixing algorithm (TUnCaT or FISSA)
@REM 5. Video index of the CNMF-E dataset

@REM Testing on CaMKII_120_TMT
python test_batch_data_CNMFE_crop.py 4 50 3 TUnCaT 3

