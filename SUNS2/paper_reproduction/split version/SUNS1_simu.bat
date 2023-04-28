@REM Generate sparse GT masks
@REM python generate_sparse_GT.py "../../data/data_simulation/*/GT Masks"

@REM Training
python train_CNN_simu.py 2 25 0 FISSA lowBG=5e+03,poisson=1
python train_params_simu.py 2 25 4 FISSA lowBG=5e+03,poisson=1
python train_params_simu.py 2 25 9 FISSA lowBG=5e+03,poisson=1

@REM Argument meanings (same for training and testing scripts):
@REM 1. SNR threshold
@REM 2. Spatial filter kernal size
@REM 3. The maximum video index that parameter optimization runs through
@REM 4. Unmixing algorithm (TUnCaT or FISSA)
@REM 5. Video index of the CNMF-E dataset

@REM Testing
python test_batch_simu.py 2 25 9 FISSA lowBG=5e+03,poisson=1
