REM Generate sparse GT masks
python generate_sparse_GT.py 'E:\\data_CNMFE\\CaMKII_120_TMT Exposure_5fps\\GT Masks'

REM Training pipeline
python train_CNN_params_vary_CNN_crop_CaMKII.py 3 4 [1] elu True 4816[1] 4 1 9

REM Run SUNS batch
python test_batch_vary_CNN_crop_CaMKII.py 3 4 [1] elu True 4816[1] 4 1 9
