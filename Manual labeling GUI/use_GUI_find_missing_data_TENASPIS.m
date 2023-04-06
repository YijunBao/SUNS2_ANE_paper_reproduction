%%
global gui;
global txtFntSz;
global video;
global masks;
global area;
% global r_bg_ratio;
global comx;
global comy;
global list_weight;
global list_weight_trace;
global list_weight_frame;
global list_neighbors; 
% global list_mask_update; 
% global masks_update; 
global edges;
global sum_edges;
global traces_raw;
global traces_out_exclude;
global traces_bg_exclude;
global r_bg_ext;
% global 

%% load saved data for GUI
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
dir_parent='D:\data_TENASPIS\original_masks\';
dir_video = dir_parent; 
% dir_SUNS = fullfile(dir_parent, 'complete_TUnCaT'); % 4 v1
dir_masks = fullfile(dir_parent, 'GT Masks');
dir_add_new = fullfile(dir_masks, 'add_new_blockwise_weighted_sum_unmask');

eid = 8;
Exp_ID = list_Exp_ID{eid};
load(fullfile(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),...
    'list_weight','list_weight_trace','list_weight_frame',...
    'sum_edges','traces_raw','video','masks');
load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
    'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
    'images_added_crop', 'patch_locations'); % ,'time_weights','list_valid'
folder = dir_add_new;
GUI_find_missing_4train_blockwise_weighted_sum_unmask(video, folder, masks, patch_locations,...
images_added_crop, masks_added_crop, added_frames, added_weights); % , update_result
