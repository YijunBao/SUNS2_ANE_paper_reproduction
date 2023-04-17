# SUNS2_ANE_paper_reproduction
 Code used to reproduce the results of the SUNS2-ANE paper.

There are totally 8 folders in this repository:
* `SUNS2` contains the python code of SUNS2.
* `ANE` contains the MATLAB and python code of ANE.
* `Manual labeling GUI` contains the MATLAB code of the Manual labeling GUI.
* `MIN1PIPE-3.0.0 modified` contains the code of MIN1PIPE v3.0.0. We ran it on our test datasets and compared the results with SUNS2-ANE.
* `CNMF_E-1.1.2 modified` contains the code of CNMF_E v1.1.2. We ran it on our test datasets and compared the results with SUNS2-ANE.
* `simulation_1p` contains the code to simulate test videos with various background and noise levels.
* `crop data_CNMFE` contains the code to crop the sub-videos from the full videos of the CNMF-E dataset.
* `plot figures` contains the code to plot the figures in the paper.
Among these folders, the first three folders are the original code we developed, and are the major novelty of the paper. The remaining code are used to reproduce the results in the paper. To run the full paper reproduction, please download our datasets from [figShare](https://doi.org/10.6084/m9.figshare.22304569), and put the videos and masks to the corresponding folders.

# SUNS2
Shallow UNet Neuron Segmentation (SUNS) was a python software developed in our [previous work](https://doi.org/10.1038/s42256-021-00342-x) for two-photon calcium imaging videos. We extended the original version (SUNS1) to process one-photon videos (SUNS2) by applying spatial homomorphic filtering in pre-processing and replacing FISSA by TUnCaT. The folder `paper_reproduction` contains the code to reproduce the paper results, and the remaining folders contain the code of SUNS2. Because we also ran SUNS1 in the paper for comparison, the installation and paper reproduction code also include necessary files to run SUNS1. 

We compiled a series of code running in sequence to Windows bat files, including training and testing SUNS1 or SUNS2 on each dataset:
* SUNS2_simu.bat: Trainand test SUNS2 on the simulated dataset.
* SUNS2_data_Tenaspis.bat: Trainand test SUNS2 on the Tenaspis dataset.
* SUNS2_data_Tenaspis_noSF.bat: Trainand test SUNS2 without spatial filtering on the Tenaspis dataset.
* SUNS2_data_CNMFE.bat: Train and test SUNS2 on the CNMF-E dataset.
* SUNS2_data_CNMFE_noSF.bat: Train and test SUNS2 without spatial filtering on the CNMF-E dataset.
* SUNS1_simu.bat: Trainand test SUNS1 on the simulated dataset.
* SUNS1_data_Tenaspis.bat: Trainand test SUNS1 on the Tenaspis dataset.
* SUNS1_data_Tenaspis_noSF.bat: Trainand test SUNS1 without spatial filtering on the Tenaspis dataset.
* SUNS1_data_CNMFE.bat: Train and test SUNS1 on the CNMF-E dataset.
* SUNS1_data_CNMFE_noSF.bat: Train and test SUNS1 without spatial filtering on the CNMF-E dataset.
Users can skip the training steps and directly run the testing steps using the training results that we provided. 

# ANE
Additional neuron extraction (ANE) is a supplementary algorithm developed in MATLAB and python to locate small or dim neurons missed by SUNS. It started from the output masks of SUNS2, and used hierarchical clustering and CNN discrimination to extract additional neurons. 

We compiled a series of code running in sequence to Windows bat files, including training and testing ANE on each dataset:
* ANE_train_demo.bat: Train ANE on the demo dataset.
* ANE_train_simu.bat: Train ANE on the simulated dataset.
* ANE_train_data_Tenaspis.bat: Train ANE on the Tenaspis dataset.
* ANE_train_data_CNMFE.bat: Train ANE on the CNMF-E dataset.
* ANE_test_demo.bat: Test ANE on the demo dataset.
* ANE_test_simu.bat: Test ANE on the simulated dataset.
* ANE_test_data_Tenaspis.bat: Test ANE on the Tenaspis dataset.
* ANE_test_data_CNMFE.bat: Test ANE on the CNMF-E dataset.
Users can skip the training steps and directly run the testing steps using the training results that we provided. 

# Manual labeling GUI
We developed three MATLAB GUIs to semi-automatically label neurons and refine their shapes, so that they can be used as ground truth neurons for training and evaluation. 

We compiled a series of code running in sequence to Windows bat files, including manual labeling on the Tenaspis and the CNMF-E dataset:
* pipeline_labeling_Tenaspis.bat: manual labeling on the Tenaspis dataset.
* pipeline_labeling_CNMFE_full.bat: manual labeling on the full videos of the CNMF-E dataset.

# Software requirement and installation
We used python and MATLAB to implement and run our code. We tested our code on python 3.7.8 (Anaconda) and MATLAB R2019a on a Windows 10 computer (AMD 1920X CPU, 128 GB RAM, NVIDIA Titan RTX GPU). Please refer to the readme of SUNS2 for installation of the custom python package. 

# Test datasets
The above code ran on three datasets, which can be accessed in [figShare](https://doi.org/10.6084/m9.figshare.22304569). They should be downloaded and put under the folder `data`. For easier validation, we included the cropped BNST video from the CNMF-E dataset in this repository as demo videos, which are in the folder `data/data_CNMFE/CaMKII_120_TMT Exposure_5fps`. 

The expected segmentation result of SUNS2-ANE on the video “CaMKII_120_TMT Exposure_5fps_part22” should look like Fig. 4A of our paper. The average processing time of the four videos on our test computer when only running the test script is 3.6 s using SUNS2 and 20.0 s using ANE. 

Of course, you can modify the demo scripts to process other videos. You need to set the folders of the videos and GT masks, and change some parameters in the python scripts to direct to your videos. The videos should be in h5 format. If you don’t have GT masks for training, you can use the Manual labeling GUI to create them. 
