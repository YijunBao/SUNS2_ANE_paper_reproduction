This folder contains our modified code of CNMF-E, which processed our test datasets using the parameters optimized through directed evolution. First, run the initialization script (`cnmfe_setup.m`) to set up the environment and cvx. 

Users can directly run the test scripts below using our optimized parameters:
* CNMFE_simu_direvl_cv_test.m: Test the simulated videos using optimized parameters through cross-validation.
* CNMFE_data_TENASPIS_direvl_cv_test.m: Test the Tenaspis videos using optimized parameters through cross-validation..
* CNMFE_data_TENASPIS_direvl_test.m: Test the Tenaspis videos using optimized parameters.
* CNMFE_data_CNMFE_direvl_cv_test.m: Test the CNMF-E videos using optimized parameters through cross-validation.
* CNMFE_data_CNMFE_direvl_test.m: Test the CNMF-E videos using optimized parameters through cross-validation.

When running the CNMF-E videos, specify the video name by setting `data_ind` in [1,2,3,4] accordingly. 

Users can also run our training scripts to optmize the parameters:
* CNMFE_par_data_CNMFE_direvl_cv_train.m: Optimize the tunable parameters of the simulated videos using directed evolution through cross-validation. 
* CNMFE_par_data_CNMFE_direvl_train.m: Optimize the tunable parameters of the simulated videos using directed evolution. 
* CNMFE_par_data_TENASPIS_direvl_cv_train.m: Optimize the tunable parameters of the Tenaspis videos using directed evolution through cross-validation. 
* CNMFE_par_data_TENASPIS_direvl_train.m: Optimize the tunable parameters of the Tenaspis videos using directed evolution. 
* CNMFE_par_simu_direvl_cv_train.m: Optimize the tunable parameters of the CNMF-E videos using directed evolution through cross-validation. 
* CNMFE_par_simu_direvl_train.m: Optimize the tunable parameters of the CNMF-E videos using directed evolution. 

If you run the training script before the test script for a dataset, you should run the initialization script for that dataset before the training script to ensure parallel processing of multiple videos runs correctly:
* CNMFE_data_CNMFE_init_for_par.m: Initialize the training for CNMF-E dataset.
* CNMFE_simu_init_for_par.m: Initialize the training for simulated dataset.
* CNMFE_data_TENASPIS_init_for_par.m: Initialize the training for Tenaspis dataset.
