%% pipeline of missing finder trained on added GT
% pre_CNN_data_CNMFE_crop_GT_blockwise;
% use_GUI_find_missing;
% post_GUI_data_CNMFE_crop_GT_blockwise;

pre_CNN_data_CNMFE_add_GT_blockwise_weighted_sum;
post_GUI_data_CNMFE_add_GT_blockwise;

% pre_CNN_data_CNMFE_crop_SUNS_blockwise;
pre_CNN_data_CNMFE_add_SUNS_blockwise;

%% pipeline of missing finder trained on dropped out neurons
dropout_GTMasks_data_TENASPIS;
pre_CNN_data_TENASPIS_drop_GT_blockwise_weighted_sum_mm;
post_GUI_data_TENASPIS_drop_GT_blockwise;

post_CNN_data_TENASPIS_drop_blockwise_GT_res0_avg;
post_CNN_data_TENASPIS_drop_blockwise_cv_res0_avg;

%% pipeline of missing finder trained on dropped out neurons
dropout_GTMasks_simu;
pre_CNN_simu_drop_GT_blockwise_weighted_sum_mm;
post_GUI_simu_drop_GT_blockwise;

pre_CNN_simu_SUNS_blockwise_mm;
post_CNN_simu_drop_blockwise_GT_res0_avg;
post_CNN_simu_drop_blockwise_cv_res0_avg;

%% pipeline of missing finder trained on dropped out neurons
dropout_GTMasks_data_CNMFE;
pre_CNN_data_CNMFE_drop_GT_blockwise_weighted_sum_mm;
post_GUI_data_CNMFE_drop_GT_blockwise;

post_CNN_data_CNMFE_drop_blockwise_GT_res0_avg;
post_CNN_data_CNMFE_drop_blockwise_cv_res0_avg;

%%
addpath('C:\Matlab Files\timer');
% timer_start_next;
% dropout_GTMasks_data_TENASPIS;
% try
%     pre_CNN_data_TENASPIS_GT_blockwise_weighted_sum_mm;
% end
try
    pre_CNN_data_TENASPIS_drop_GT_blockwise_weighted_sum_mm;
    post_GUI_data_TENASPIS_drop_GT_blockwise;
end
timer_stop;

%% pipeline for adding and refining masks
ManualLabeling;
% Copy "Results\Add_{name}.mat" to "GT Masks original\FinalMasks_{name}.mat"
pre_CNN_data_CNMFE_full_GT_blockwise_weighted_sum_mm;
use_GUI_find_missing_data_CNMFE_full;
% Rename "GT Masks original\add_new_blockwise_weighted_sum_unmask\masks_processed().mat" as "{name}_processed.mat"
post_GUI_data_CNMFE_full_GT_blockwise;
calculate_traces_bgtraces_SNR_data_CNMFE_full;
mask_correction_GUI_data_CNMFE_full;
mask_correction_GUI_final_data_CNMFE_full;
% manually merge neurons using "merge_masks.m", and save as "FinalMasks_{name}_merge.mat"
crop_videos;

calculate_traces_bgtraces_data_CNMFE;
temporal_filter_data_CNMFE;

%%
dir_parent='E:\simulation_CNMFE_corr_noise\lowBG=5e+03,poisson=0.3\';
dir_SUNS = fullfile(dir_parent, 'complete_FISSA\4816[1]th2'); % 4 v1
% dir_parent='D:\data_TENASPIS\original_masks\';
% dir_SUNS = fullfile(dir_parent, 'complete_TUnCaT_SF25\4816[1]th4'); % 4 v1
dir_masks = fullfile(dir_SUNS, 'output_masks');
sub_folder = 'add_new_blockwise_weighted_sum_unmask'; % _weighted_sum_expanded_edge_unmask
dir_add_new = fullfile(dir_masks, sub_folder);
load(fullfile(dir_add_new,'trained dropout 0.8exp(-5)\avg_Xmask_0.5\classifier_res0_0+1 frames\eval.mat'));
% Table_time = [list_time(:,end)];
% Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
% Table_scores = [list_Recall, list_Precision, list_F1];
% Table_scores_ext=[Table_scores;nanmean(Table_scores,1);nanstd(Table_scores,1,1)];
Table_time = [list_Recall, list_Precision, list_F1, list_time(:,end)];
Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];

