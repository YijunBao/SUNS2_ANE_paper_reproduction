addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');
%%
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
data_ind = 2;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
% list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
rate_hz = [10,15,7.5,5]; % frame rate of each video
list_avg_radius = [5,6,0,0];
r_bg_ratio = 3;
leng = r_bg_ratio*list_avg_radius(data_ind);

th_IoU_split = 0.5;
thj_inclass = 0.4;
thj = 0.7;
meth_baseline='median'; % {'median','median_mean','median_median'}
meth_sigma='quantile-based std'; % {'std','mode_Burr','median_std','std_back','median-based std'}

% vid=2;
% Exp_ID = list_Exp_ID{vid};

%% Load traces and ROIs
% folder of the GT Masks
dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
dir_video = dir_parent; 
for th_SNR = 3:5
dir_SUNS = fullfile(dir_parent, ['complete_TUnCaT\4816[1]th',num2str(th_SNR)]); % 4 v1
dir_masks = fullfile(dir_SUNS, 'output_masks');
% dir_masks = fullfile(dir_parent, 'GT Masks');
dir_add_new = fullfile(dir_masks, 'add_new_blockwise_weighted_sum');
fs = rate_hz(data_ind);
% folder = ['.\Result_',data_name];
if ~ exist(dir_add_new,'dir')
    mkdir(dir_add_new);
end
time_weights = zeros(num_Exp,1);

% eid = 4;
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks');
    masks=permute(logical(Masks),[3,2,1]);

%     load(fullfile(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),...
%         'list_weight','list_weight_trace','list_weight_frame',...
%         'sum_edges','traces_raw','video','masks');
    load(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
        'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
        'images_added_crop', 'patch_locations','time_weights'); % ,'list_valid'
    
    %% Calculate the neighboring neurons
    N = length(added_frames);
    list_neighbors = cell(1,N);
    masks_sum = sum(masks,3);
    for n = 1:N
        locations = patch_locations(n,:);
        list_neighbors{n} = masks_sum(locations(1):locations(2),locations(3):locations(4));
    end
    masks_neighbors_crop = cat(3,list_neighbors{:});

    save(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
        'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
        'images_added_crop', 'patch_locations','time_weights','masks_neighbors_crop'); % ,'list_valid'

    %%
%     FinalMasks = cat(3,masks,masks_added_full);
%     save(fullfile(dir_masks,['FinalMasks_',Exp_ID,'_added_blockwise.mat']),'FinalMasks');
end
end
%%
disp('Finished this step');