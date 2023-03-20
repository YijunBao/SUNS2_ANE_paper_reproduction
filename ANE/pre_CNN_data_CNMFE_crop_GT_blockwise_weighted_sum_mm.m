addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');
%%
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
data_ind = 3;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
% list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
rate_hz = [10,15,7.5,5]; % frame rate of each video
list_avg_radius = [5,6,8,0];
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
dir_parent=fullfile('E:\data_CNMFE\',[data_name]); % ,'_original_masks'
dir_video = dir_parent; 
dir_SUNS = fullfile(dir_parent, 'complete_TUnCaT'); % 4 v1
dir_masks = fullfile(dir_parent, 'GT Masks');
dir_add_new = fullfile(dir_masks, 'add_new_blockwise_weighted_sum_unmask');
fs = rate_hz(data_ind);
% folder = ['.\Result_',data_name];
if ~ exist(dir_add_new,'dir')
    mkdir(dir_add_new);
end
time_weights = zeros(num_Exp,1);

% eid = 4;
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    masks=logical(FinalMasks); % permute(logical(Masks),[3,2,1]);
    fname=fullfile(dir_video,[Exp_ID,'.h5']);
    video_raw = h5read(fname, '/mov');
%     video_raw = mov;

%     tic;
    video_sf =homo_filt(video_raw, 50);
    [mu, sigma] = SNR_normalization_video(video_sf,meth_sigma,meth_baseline);
    video_SNR = (video_sf-mu)./sigma;

    video_SNR = imgaussfilt(video_SNR); % ,1
%     save(fullfile(dir_trace,['SNR video ',Exp_ID,'.mat']),'video_SNR');
%     fname=fullfile(dir_trace,['SNR video ', Exp_ID,'.mat']);
%     load(fname, 'video_SNR');

    % max_SNR = max(video_SNR,[],3);
    [Lx,Ly,T] = size(video_SNR);
    npatchx = ceil(Lx/leng)-1;
    npatchy = ceil(Ly/leng)-1;
    num_avg = max(60, min(90, ceil(T*0.01)));

    %% Create memory mapping files for SNR video and masks
    fileName='video_SNR.dat';
    if exist('mm','var')
        clear mm;
    end
    if exist(fileName,'file')
        delete(fileName);
    end
    fileID=fopen(fileName,'w');
    video_class=class(video_SNR);
    fwrite(fileID,video_SNR,video_class);
    fclose(fileID);
    mm = memmapfile(fileName,'Format',{video_class,[Lx,Ly,T],'video'}, 'Repeat', 1);
    mm2 = memmapfile(fileName,'Format',{video_class,[Lx*Ly,T],'video'}, 'Repeat', 1);
%     max(max(max(mm.Data.video)));
    
    %%
    traces_raw=generate_traces_from_masks_mm(mm2,masks);
%     traces_bg_exclude=generate_bgtraces_from_masks_exclude(video_SNR,masks);
%     traces_out_exclude=generate_outtraces_from_masks_exclude(video_SNR,masks);
%     save(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude','traces_out_exclude');
%     load(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude','traces_out_exclude');
    % load('.\Result_PFC4_15Hz\masks_added(1--237).mat','update_result')
    % list_added_manual = update_result.list_added;

    %%
    video = video_SNR;
    [list_weight,list_weight_trace,list_weight_frame,sum_edges]...
    = frame_weight_blockwise_mm(mm, traces_raw, masks, leng);

    %%
    area = squeeze(sum(sum(masks,1),2));
    avg_area = median(area);
%     avg_radius = sqrt(mean(area)/pi);
%     r_bg = avg_radius*r_bg_ratio;
%     r_bg_ext = round(list_avg_radius(data_ind) * (r_bg_ratio+1));
% %     r_bg_ext = 24;
%     masks_sum = sum(masks,3);
    [list_added_full, list_added_crop, list_added_images_crop,...
    list_added_frames, list_added_weights, list_locations] = deal(cell(npatchx,npatchy)); % nlist_list_valid, 

    %%
    parfor ix = 1:npatchx
    for iy = 1:npatchy
        xmin = min(Lx-2*leng+1, (ix-1)*leng+1);
        xmax = min(Lx, (ix+1)*leng);
        ymin = min(Ly-2*leng+1, (iy-1)*leng+1);
        ymax = min(Ly, (iy+1)*leng);
        weight = list_weight{ix,iy};
        [image_new_crop, mask_new_crop, mask_new_full, select_frames_class, select_weight_calss] = ...
        find_missing_blockwise_weighted_sum_unmask_mm(mm, masks, xmin, xmax, ymin, ymax, ...
        weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split);
        list_added_images_crop{ix,iy} = image_new_crop;
        list_added_crop{ix,iy} = mask_new_crop;
        list_added_full{ix,iy} = mask_new_full;
        list_added_frames{ix,iy} = select_frames_class;
        list_added_weights{ix,iy} = select_weight_calss;
        n_class = size(image_new_crop,3);
        list_locations{ix,iy} = [xmin, xmax, ymin, ymax].*ones(n_class,1);

%         mask_new_full_2 = reshape(mask_new_full,Lx*Ly,n_class);
%         mask_valid_full = list_added_manual{nn};
%         mask_valid_full_2 = reshape(mask_valid_full,Lx*Ly,[]);
%         [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(mask_valid_full_2,mask_new_full_2,0.1);
%         list_valid = any(m>0,1);
%         list_list_valid{nn} = list_valid;
    end    
    end    

    %% Save Masks
    masks_added_full = cat(3,list_added_full{:});
    images_added_crop = cat(3,list_added_images_crop{:});
    masks_added_crop = cat(3,list_added_crop{:});
    patch_locations = cat(1,list_locations{:});
    added_frames = horzcat(list_added_frames{:});
    added_weights = horzcat(list_added_weights{:});
%     time_weights = toc;
%     list_valid = cell2mat(list_list_valid);
    save(fullfile(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),...
        'list_weight','list_weight_trace','list_weight_frame',...
        'sum_edges','traces_raw','video','masks');
    save(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
        'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
        'images_added_crop', 'patch_locations'); % ,'time_weights','list_valid'
    
    %% use GUI to label 
%     folder = ['.\Result_',Exp_ID];
%     GUI_find_missing_4train_blockwise_weighted_sum(video, folder, masks, patch_locations,...
%     images_added_crop, masks_added_crop, added_frames, added_weights);
%     GUI_find_missing_4train_blockwise_weighted_sum(video, folder, masks, patch_locations,...
%     images_added_crop, masks_added_crop, added_frames, added_weights, update_result);
    %%
%     load(fullfile(folder,'Result_PFC4_15Hz_part11\masks_added(1--29).mat'), 'update_result');
%     list_valid = cell2mat(update_result.list_valid);
%     masks_added_full = cell2mat(reshape(update_result.list_added,1,1,[]));
%     list_avg_frame = cell2mat(reshape(update_result.list_avg_frame,1,1,[]));
%     list_mask_update = cell2mat(reshape(update_result.list_mask_update,1,1,[]));
%     
%     save(fullfile(folder,[Exp_ID,'_added_auto.mat']), ...
%         'masks_added_full','masks_added_crop','images_added_crop','list_valid');
end
%%
disp('Finished this step');