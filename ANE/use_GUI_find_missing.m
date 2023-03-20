
%% load saved data for GUI
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
data_ind = 3;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
dir_parent=fullfile('E:\data_CNMFE\',[data_name]); % ,'_original_masks'
dir_video = dir_parent; 
% dir_SUNS = fullfile(dir_parent, 'complete_TUnCaT'); % 4 v1
dir_masks = fullfile(dir_parent, 'GT Masks');
dir_add_new = fullfile(dir_masks, 'add_new_blockwise_weighted_sum_unmask');

eid = 4;
Exp_ID = list_Exp_ID{eid};
load(fullfile(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),...
    'list_weight','list_weight_trace','list_weight_frame',...
    'sum_edges','traces_raw','video','masks');
load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
    'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
    'images_added_crop', 'patch_locations'); % ,'time_weights','list_valid'
folder = dir_add_new;
GUI_find_missing_4train_blockwise_weighted_sum_unmask(video, folder, masks, patch_locations,...
images_added_crop, masks_added_crop, added_frames, added_weights);
