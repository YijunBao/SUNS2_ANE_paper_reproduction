# SUNS2_ANE_paper_reproduction
Code used to reproduce the results of the [SUNS2-ANE paper](https://doi.org/10.1038/s42003-024-06668-7).

There are totally 8 folders in this repository:
* `SUNS2` contains the python code of SUNS2.
* `ANE` contains the MATLAB and python code of ANE.
* `Manual labeling GUI` contains the MATLAB code of the Manual labeling GUI.
* `TUnCaT` contains the MATLAB and python code of evaluating the spatiotemporal accuracy of the TUnCaT dataset.
* `data` should contain our test videos in the four datasets, as well as our processing results of these videos. To save to time to download the code, we actually put the content in [figShare](https://doi.org/10.6084/m9.figshare.22304569), so this folder in GitHub is empty.
* `generate_data` contains the code to generate our test videos in the three datasets from source data or source code.
* `plot figures` contains the code to plot the figures in the paper.
* `Other methods` contains the code of other methods used in the paper to compare with our method.
* * `MIN1PIPE-3.0.0 modified` contains our modified code of [MIN1PIPE](https://github.com/JinghaoLu/MIN1PIPE) v3.0.0. We ran it on our test datasets using the parameters optimized through directed evolution and compared the results with SUNS2-ANE.
* * `CNMF_E-1.1.2 modified` contains our modified code of [CNMF-E](https://github.com/zhoupc/CNMF_E) v1.1.2. We ran it on our test datasets using the parameters optimized through directed evolution and compared the results with SUNS2-ANE.
* * `EXTRACT` contains our modified code of [EXTRACT](https://github.com/schnitzer-lab/EXTRACT-public). We ran it on our test datasets and compared the results with SUNS2-ANE.
* * `caiman-Batch` contains our modified code of [CaImAn](https://github.com/flatironinstitute/CaImAn) v1.6.4. We ran it on our Tenaspis dataset and compared the results with SUNS2-ANE. We did not use the CNN classifier so that it is more consistent with the original CNMF.
* * `DeepWonder` contains our modified code of [DeepWonder](https://github.com/
yuanlong-o/Deep_widefield_cal_inferece) v0.3. We ran it on our Tenaspis and CNMF-E datasets and compared the results with SUNS2-ANE. We used the CNN model trained from the original paper. We spatially scaled our test videos so that the average neuron areas in the unit of pixels equal the average neuron area of the DeepWonder segmentation of the widefield video in that paper (368 pixels).

Among these folders, the first three folders are the major novelty of the paper. The remaining code are used to reproduce the results in the paper. To run the full paper reproduction, please download our datasets from [figShare](https://doi.org/10.6084/m9.figshare.22304569), and put the videos and masks to the corresponding folders. Our processing results are in the same figShare link. 

# SUNS2
Shallow UNet Neuron Segmentation (SUNS) was a python software developed in our [previous work](https://doi.org/10.1038/s42256-021-00342-x) for two-photon calcium imaging videos. We extended the original version (SUNS1) to process one-photon videos (SUNS2) by applying spatial homomorphic filtering in pre-processing and replacing FISSA by TUnCaT. The folder `paper_reproduction` contains the code to reproduce the paper results, and the remaining folders contain the code of SUNS2. Because we also ran SUNS1 in the paper for comparison, the installation and paper reproduction code also include necessary files to run SUNS1. 

We compiled a series of code running in sequence to Windows bat files, including training and testing SUNS1 or SUNS2 on each dataset:
* SUNS2_demo.bat: Train and test SUNS2 on the demo dataset.
* SUNS2_simu.bat: Train and test SUNS2 on the simulated dataset.
* SUNS2_data_Tenaspis.bat: Train and test SUNS2 on the Tenaspis dataset.
* SUNS2_data_Tenaspis_noSF.bat: Train and test SUNS2 without spatial filtering on the Tenaspis dataset.
* SUNS2_data_CNMFE.bat: Train and test SUNS2 on the CNMF-E dataset.
* SUNS2_data_CNMFE_noSF.bat: Train and test SUNS2 without spatial filtering on the CNMF-E dataset.
* SUNS2_data_TUnCaT.bat: Train and test SUNS2 on the TUnCaT dataset.
* SUNS1_simu.bat: Train and test SUNS1 on the simulated dataset.
* SUNS1_data_Tenaspis.bat: Train and test SUNS1 on the Tenaspis dataset.
* SUNS1_data_Tenaspis_noSF.bat: Train and test SUNS1 without spatial filtering on the Tenaspis dataset.
* SUNS1_data_CNMFE.bat: Train and test SUNS1 on the CNMF-E dataset.
* SUNS1_data_CNMFE_noSF.bat: Train and test SUNS1 without spatial filtering on the CNMF-E dataset.
* SUNS1_data_TUnCaT.bat: Train and test SUNS1 on the TUnCaT dataset.
Users can skip the training steps and directly run the testing steps using the training results that we provided. If the training get hung after finishing CNN training and starting parameter optimization (i.e., you see `Using thresh_pmap=` but the CPU utilization dropped to zero), please go to the folder `split_version` and run the corresponding scripts instead.

# ANE
Additional neuron extraction (ANE) is a supplementary algorithm developed in MATLAB and python to locate small or dim neurons missed by SUNS. It started from the output masks of SUNS2, and used hierarchical clustering and CNN discrimination to extract additional neurons. They python portions of the program can be run in the SUNS environment installed before. 

We compiled a series of code running in sequence to Windows bat files, including training and testing ANE on each dataset:
* ANE_train_demo.bat: Train ANE on the demo dataset.
* ANE_train_simu.bat: Train ANE on the simulated dataset.
* ANE_train_data_Tenaspis.bat: Train ANE on the Tenaspis dataset.
* ANE_train_data_CNMFE.bat: Train ANE on the CNMF-E dataset.
* ANE_train_data_TUnCaT.bat: Train ANE on the TUnCaT dataset.
* ANE_test_demo.bat: Test ANE on the demo dataset.
* ANE_test_simu.bat: Test ANE on the simulated dataset.
* ANE_test_data_Tenaspis.bat: Test ANE on the Tenaspis dataset.
* ANE_test_data_CNMFE.bat: Test ANE on the CNMF-E dataset.
* ANE_test_data_TUnCaT.bat: Test ANE on the TUnCaT dataset.
Users can skip the training steps and directly run the testing steps using the training results that we provided. But it is possible that the saved training results cannot be correctly loaded to programs run in a different environment. It is the best practice to train and test in the same environment. 

# Manual labeling GUI
We developed three MATLAB GUIs to semi-automatically label neurons and refine their shapes, so that they can be used as ground truth neurons for training and evaluation. 

We compiled a series of code running in sequence to Windows bat files, including manual labeling on the Tenaspis and the CNMF-E dataset:
* pipeline_labeling_data_Tenaspis.m: manual labeling on the Tenaspis dataset.
* pipeline_labeling_data_CNMFE_full.m: manual labeling on the full videos of the CNMF-E dataset.

# Software requirement and installation
We used python and MATLAB to implement and run our code. We tested our code on python 3.7.8 (Anaconda) and MATLAB R2019a on a Windows 10 computer (AMD 1920X CPU, 128 GB RAM, NVIDIA Titan RTX GPU). Please refer to the readme of SUNS2 for installation of the custom python package. 

# Test datasets
The above code ran on four datasets, which can be accessed in [figShare](https://doi.org/10.6084/m9.figshare.22304569). They should be downloaded and put under the folder `data`. For easier validation, we included the cropped dorsal striatum video from the CNMF-E dataset in this repository as demo videos, which are in the folder `data/data_CNMFE/blood_vessel_10Hz`. 

The expected segmentation result of SUNS2-ANE on the video “blood_vessel_10Hz_part21” should look like Fig. 3A of our paper. The average processing time of the four videos on our test computer when only running the test script is 5 s using SUNS2 and 40 s using ANE. 

Of course, you can modify the demo scripts to process other videos. You need to set the folders of the videos and GT masks, and change some parameters in the python scripts to direct to your videos. The videos should be in h5 format. If you don’t have GT masks for training, you can use the Manual labeling GUI to create them. 

# Create data from source
We used three experimental datasets (Tenaspis, CNMF-E, and TUnCaT) and one simulated dataset to test our algorithm, and compared with existing algorithms. The folder `generate_data` contains the code to generate the Tenaspis, CNMF-E, and simulated videos in the three datasets from source data or source code. The TUnCaT dataset is the same as our [previous work](https://doi.org/10.3389/fnins.2021.797421), but we updated the GT masks using our manual labeling GUI. 

## Simulated videos
The sub-folder `simulation_1p` contains the code to simulate our videos. The code is modified from [the simulation code used in the CNMF-E paper](https://github.com/zhoupc/eLife_submission). The main script is `sim_data_10_randBG_corr_noise.m`. We changed the parameter `scale_lowBG` within [1e3, 5e3] to set the scale of the slowly varying background as low and high. We changed the parameter `scale_noise` within [0.1, 0.3, 1] to set the scale of the random noise as high, medium, and low. 

The simulation results will be saved in the folder `data/data_simulation/lowBG={scale_lowBG},poisson={scale_noise}`. The videos are saved as `sim_{n}.h5`. The ground truth masks are saved as `GT Masks/FinalMasks_sim_{n}.mat`. Some other information are saved in `GT info/GT_sim_{n}.mat`. 

## Tenaspis videos
The sub-folder `crop data_Tenaspis` contains the code to crop the eight Tenaspis videos used in our paper from motion-corrected videos. 

## CNMF-E videos
The sub-folder `crop data_CNMFE` contains the code to crop the four CNMF-E videos used in our paper into sub-videos. 
