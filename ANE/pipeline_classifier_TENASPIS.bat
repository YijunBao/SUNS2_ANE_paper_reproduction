@REM matlab -nojvm -nodesktop -r find_missing_weights_data_CNMFE_crop
@REM python test_CNN_classifier_crop.py
@REM matlab -nojvm -nodesktop -r find_missing_CNN_final_data_CNMFE_crop

@REM python "C:\Matlab Files\timer\timer_start_next.py"
@REM python "C:\Matlab Files\timer\timer_start_from_no_file.py"
@REM python "C:\Matlab Files\timer\timer_stop.py"
@REM python "C:\Matlab Files\timer\time_waite.py"

@REM shutdown -s -t 60
@REM shutdown -a

@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.01
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.01
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.01
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.01
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.01
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.01
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.01

@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.005
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.005
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.005
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.005
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.005
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.005
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.005

@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.02
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.02
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.02
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.02
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.02
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.02
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.02

@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.003
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.003
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.003
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.003
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.003
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.003
@REM python train_CNN_classifier_data_TENASPIS_cv_addfake.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.003

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python test_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python test_CNN_classifier_data_TENASPIS_GT_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.8exp(-10)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.8exp(-10)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.8exp(-10)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.8exp(-10)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.8exp(-10)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-10)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-10)

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.8exp(-20)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.8exp(-20)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.8exp(-20)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.8exp(-20)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.8exp(-20)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-20)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-20)

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res1 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res2 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res3 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res4 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res5 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res6 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res7 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res8 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res9 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 2 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 3 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 4 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 5 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 6 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 7 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 8 Xmask 0 2 _weighted_sum_unmask 0.8exp(-15)

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 2 Xmask 1 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 3 Xmask 1 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 4 Xmask 1 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 5 Xmask 1 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 6 Xmask 1 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 7 Xmask 1 2 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 8 Xmask 1 2 _weighted_sum_unmask 0.8exp(-15)

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 2 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 3 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 4 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 5 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 6 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 7 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 8 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 2 Xmask 1 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 3 Xmask 1 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 4 Xmask 1 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 5 Xmask 1 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 6 Xmask 1 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 7 Xmask 1 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 8 Xmask 1 1 _weighted_sum_unmask 0.8exp(-15)

@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res1 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res2 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res3 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res4 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res5 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res6 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res7 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res8 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)
@REM python train_CNN_classifier_data_TENASPIS_cv_drop.py classifier_res9 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-15)

python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-10) lowBG=5e+03,poisson=1
python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=5e+03,poisson=1
python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-5) lowBG=5e+03,poisson=1
python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-3) lowBG=5e+03,poisson=1

python test_CNN_classifier_simu_GT_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-3) lowBG=5e+03,poisson=1
python test_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-3) lowBG=5e+03,poisson=1
python test_CNN_classifier_simu_GT_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-10) lowBG=5e+03,poisson=1
python test_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-10) lowBG=5e+03,poisson=1
python test_CNN_classifier_simu_GT_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=5e+03,poisson=1
python test_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=5e+03,poisson=1
python test_CNN_classifier_simu_GT_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-5) lowBG=5e+03,poisson=1
python test_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-5) lowBG=5e+03,poisson=1

@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.8exp(-10) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.8exp(-10) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.8exp(-10) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.8exp(-10) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.8exp(-10) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-10) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-10) lowBG=1e+03,poisson=1

@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.8exp(-8) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.8exp(-8) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-8) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-8) lowBG=1e+03,poisson=1

@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.8exp(-5) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.8exp(-5) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.8exp(-5) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.8exp(-5) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.8exp(-5) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-5) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-5) lowBG=1e+03,poisson=1

@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 nomask 0 1 _weighted_sum_unmask 0.8exp(-3) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 mask 0 1 _weighted_sum_unmask 0.8exp(-3) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 mask 0 2 _weighted_sum_unmask 0.8exp(-3) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 bmask 0 1 _weighted_sum_unmask 0.8exp(-3) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 bmask 0 2 _weighted_sum_unmask 0.8exp(-3) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 1 _weighted_sum_unmask 0.8exp(-3) lowBG=1e+03,poisson=1
@REM python train_CNN_classifier_simu_cv_drop.py classifier_res0 0 Xmask 0 2 _weighted_sum_unmask 0.8exp(-3) lowBG=1e+03,poisson=1

@REM shutdown -s -t 60
@REM shutdown -a
