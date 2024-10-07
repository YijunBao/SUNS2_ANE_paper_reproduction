@REM Generate sparse GT masks
python generate_sparse_GT.py "../../data/data_TUnCaT/GT Masks"

@REM Training and Testing SUNS1
python train_CNN_params_data_TUnCaT.py 3 25 8 FISSA
python test_batch_data_TUnCaT.py 3 25 8 FISSA

@REM Argument meanings (same for training and testing scripts):
@REM 1. SNR threshold
@REM 2. Spatial filter kernal size
@REM 3. The maximum video index that parameter optimization runs through
@REM 4. Unmixing algorithm (TUnCaT or FISSA)
@REM 5. Video index of the CNMF-E dataset

@REM Testing TUnCaT after SUNS1
python TUnCaT_data_TUnCaT_tol_bin.py "SUNS_FISSA_SF25/4816[1]th3/output_masks" 1e-4 2
