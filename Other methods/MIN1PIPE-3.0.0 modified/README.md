This folder contains our modified code of MIN1PIPE, which processed our test datasets using the parameters optimized through directed evolution. First, run the initialization script (`min1pipe_init.m`) to set up the environment and cvx. 

Users can directly run the test scripts below using our optimized parameters:
* min1pipe_simu_direvl_test.m: Test the simulated videos using optimized parameters.
* min1pipe_simu_direvl_cv_test.m: Test the simulated videos using optimized parameters through cross-validation.
* min1pipe_data_TENASPIS_direvl_test.m: Test the Tenaspis videos using optimized parameters.
* min1pipe_data_TENASPIS_direvl_cv_test.m: Test the Tenaspis videos using optimized parameters through cross-validation.
* min1pipe_data_CNMFE_direvl_cv_test.m: Test the CNMF-E videos using optimized parameters through cross-validation.
* min1pipe_data_TUnCaT_direvl_test.m: Test the TUnCaT videos using optimized parameters.
* min1pipe_data_TUnCaT_direvl_cv_test.m: Test the TUnCaT videos using optimized parameters through cross-validation.
* min1pipe_data_TENASPIS_SNR_direvl_test.m: Test the Tenaspis SNR videos using optimized parameters.

When running the CNMF-E videos, specify the video name by setting `data_ind` in [1,2,3,4] accordingly. 

Users can also run our training scripts to optmize the parameters:
* min1pipe_par_simu_direvl_train.m: Optimize the tunable parameters of the simulated videos using directed evolution. 
* min1pipe_par_simu_direvl_cv_train.m: Optimize the tunable parameters of the simulated videos using directed evolution through cross-validation. 
* min1pipe_par_data_TENASPIS_direvl_train.m: Optimize the tunable parameters of the Tenaspis videos using directed evolution. 
* min1pipe_par_data_TENASPIS_direvl_cv_train.m: Optimize the tunable parameters of the Tenaspis videos using directed evolution through cross-validation. 
* min1pipe_data_CNMFE_direvl_train.m: Optimize the tunable parameters of the CNMF-E videos using directed evolution. 
* min1pipe_data_CNMFE_direvl_cv_train.m: Optimize the tunable parameters of the CNMF-E videos using directed evolution through cross-validation. 
* min1pipe_par_data_TUnCaT_direvl_train.m: Optimize the tunable parameters of the TUnCaT videos using directed evolution. 
* min1pipe_par_data_TUnCaT_direvl_cv_train.m: Optimize the tunable parameters of the TUnCaT videos using directed evolution through cross-validation. 
* min1pipe_par_data_TENASPIS_SNR_direvl_train.m: Optimize the tunable parameters of the Tenaspis SNR videos using directed evolution. 
