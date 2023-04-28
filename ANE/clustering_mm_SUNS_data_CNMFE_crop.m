% data_ind = 1; dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF25','4816[1]th5');  
% data_ind = 2; dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF25','4816[1]th4');  
% data_ind = 3; dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF25','4816[1]th4');  
% data_ind = 4; dir_SUNS_sub = fullfile('SUNS_TUnCaT_SF50','4816[1]th4');  
addpath(genpath('.'))
gcp;

%%
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
rate_hz = [10,15,7.5,5]; % frame rate of each video
list_avg_radius = [5,6,8,14];
r_bg_ratio = 3;
leng = r_bg_ratio*list_avg_radius(data_ind);

th_IoU_split = 0.5;
thj_inclass = 0.4;
thj = 0.7;
meth_baseline='median'; % {'median','median_mean','median_median'}
meth_sigma='quantile-based std'; % {'std','mode_Burr','median_std','std_back','median-based std'}
sub_added = '';

%% Load traces and ROIs
% folder of the GT Masks
dir_parent=fullfile('..','data','data_CNMFE',[data_name,sub_added]);
dir_video = dir_parent; 
dir_SUNS = fullfile(dir_parent, dir_SUNS_sub);
dir_masks = fullfile(dir_SUNS, 'output_masks');
% dir_masks = fullfile(dir_parent, 'GT Masks');
dir_add_new = fullfile(dir_masks, 'add_new_blockwise');
% fs = rate_hz(data_ind);
if ~ exist(dir_add_new,'dir')
    mkdir(dir_add_new);
end
time_weights = zeros(num_Exp,1);

for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_masks,['Output_Masks_',Exp_ID,'.mat']),'Masks');
    masks=permute(logical(Masks),[3,2,1]);
%     load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
%     masks=logical(FinalMasks);
    fname=fullfile(dir_video,[Exp_ID,'.h5']);
    video_raw = h5read(fname, '/mov');

    tic;
    video_sf =homo_filt(video_raw, 50);
    [mu, sigma] = SNR_normalization_video(video_sf,meth_sigma,meth_baseline);
    video_SNR = (video_sf-mu)./sigma;

    video_SNR = imgaussfilt(video_SNR); % ,1

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
    
    %%
    traces_raw=generate_traces_from_masks_mm(mm2,masks);

    %%
    video = video_SNR;
    [list_weight,list_weight_trace,list_weight_frame,sum_edges]...
        = frame_weight_blockwise_mm(mm, traces_raw, masks, leng);

    %%
    area = squeeze(sum(sum(masks,1),2));
    avg_area = median(area);
    [list_added_full, list_added_crop, list_added_images_crop,...
        list_added_frames, list_added_weights, list_locations] = deal(cell(npatchx,npatchy));

    %%
    parfor ix = 1:npatchx
    for iy = 1:npatchy
        xmin = min(Lx-2*leng+1, (ix-1)*leng+1);
        xmax = min(Lx, (ix+1)*leng);
        ymin = min(Ly-2*leng+1, (iy-1)*leng+1);
        ymax = min(Ly, (iy+1)*leng);
        weight = list_weight{ix,iy};
        [image_new_crop, mask_new_crop, mask_new_full, select_frames_class, select_weight_calss] = ...
            find_missing_blockwise_mm(mm, masks, xmin, xmax, ymin, ymax, ...
            weight, num_avg, avg_area, thj, thj_inclass, th_IoU_split);
        list_added_images_crop{ix,iy} = image_new_crop;
        list_added_crop{ix,iy} = mask_new_crop;
        list_added_full{ix,iy} = mask_new_full;
        list_added_frames{ix,iy} = select_frames_class;
        list_added_weights{ix,iy} = select_weight_calss;
        n_class = size(image_new_crop,3);
        list_locations{ix,iy} = [xmin, xmax, ymin, ymax].*ones(n_class,1);
    end    
    end    

    %%
    masks_added_full = cat(3,list_added_full{:});
    images_added_crop = cat(3,list_added_images_crop{:});
    masks_added_crop = cat(3,list_added_crop{:});
    patch_locations = cat(1,list_locations{:});
    added_frames = horzcat(list_added_frames{:});
    added_weights = horzcat(list_added_weights{:});
    
    %% Calculate the neighboring neurons
    N = length(added_frames);
    list_neighbors = cell(1,N);
    masks_sum = sum(masks,3);
    for n = 1:N
        locations = patch_locations(n,:);
        list_neighbors{n} = masks_sum(locations(1):locations(2),locations(3):locations(4));
    end
    masks_neighbors_crop = cat(3,list_neighbors{:});
    time_weights = toc;

    %% Save Masks
    % save(fullfile(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),...
    %     'list_weight','list_weight_trace','list_weight_frame',...
    %     'sum_edges','traces_raw','video','masks');
    save(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
        'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
        'images_added_crop', 'patch_locations','time_weights','masks_neighbors_crop');
end
%%
clear mm;
delete(fileName);
disp('Finished this step');
