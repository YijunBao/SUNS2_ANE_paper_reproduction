@REM Generate sparse GT masks
python generate_sparse_GT.py "../../data/data_TENASPIS/added_refined_masks/GT Masks"

@REM Training
python train_CNN_data_TENASPIS.py 2 0 0 TUnCaT
python train_params_data_TENASPIS.py 2 0 3 TUnCaT
python train_params_data_TENASPIS.py 2 0 7 TUnCaT

@REM Argument meanings (same for training and testing scripts):
@REM 1. SNR threshold
@REM 2. Spatial filter kernal size
@REM 3. The maximum video index that parameter optimization runs through
@REM 4. Unmixing algorithm (TUnCaT or FISSA)
@REM 5. Video index of the CNMF-E dataset

@REM Testing
python test_batch_data_TENASPIS.py 2 0 7 TUnCaT
