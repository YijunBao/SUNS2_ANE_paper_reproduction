@REM Generate sparse GT masks
python generate_sparse_GT.py "../../data/data_TUnCaT/GT Masks"

@REM Training and Testing SUNS2
python train_CNN_params_data_TUnCaT.py 10 25 8 TUnCaT
python test_batch_data_TUnCaT.py 10 25 8 TUnCaT

@REM Argument meanings (same for training and testing scripts):
@REM 1. SNR threshold
@REM 2. Spatial filter kernal size
@REM 3. The maximum video index that parameter optimization runs through
@REM 4. Unmixing algorithm (TUnCaT or FISSA)
@REM 5. Video index of the CNMF-E dataset

@REM Testing TUnCaT after SUNS2
python TUnCaT_data_TUnCaT_tol_bin.py "SUNS_TUnCaT_SF25/4816[1]th10/output_masks" 1e-4 2

@REM Testing TUnCaT after SUNS2-ANE
python TUnCaT_after_ANE_data_TUnCaT_tol_bin.py "SUNS_TUnCaT_SF25/4816[1]th10/output_masks" 1e-4 2
