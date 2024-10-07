@REM Tenaspis dataset

@REM Convert h5 video to mmap format
python h5ToMmap_data_Tenaspis.py

@REM Optimize parameters 
python runCaimanBatch_data_Tenaspis.py

@REM Run CNMF
python runCaimanBatch_data_Tenaspis_test.py

@REM Evaluate the result
matlab -nojvm -nodesktop -r "EvalPerformance_Batch_data_Tenaspis"
