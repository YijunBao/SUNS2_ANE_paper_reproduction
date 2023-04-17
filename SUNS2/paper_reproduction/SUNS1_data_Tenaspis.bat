@REM Generate sparse GT masks
@REM python generate_sparse_GT.py "../../data/data_TENASPIS/added_refined_masks/GT Masks"

@REM Training pipeline
python train_CNN_params_vary_CNN_data_TENASPIS.py 3 25 3 FISSA
python train_CNN_params_vary_CNN_data_TENASPIS_continue.py 3 25 7 FISSA

@REM Argument meanings (same for training and testing scripts):
@REM 1. SNR threshold
@REM 2. Spatial filter kernal size
@REM 3. The maximum video index that parameter optimization runs through
@REM 4. Unmixing algorithm (TUnCaT or FISSA)
@REM 5. Video index of the CNMF-E dataset

@REM Run SUNS batch
python test_batch_vary_CNN_data_TENASPIS.py 3 25 7 FISSA
