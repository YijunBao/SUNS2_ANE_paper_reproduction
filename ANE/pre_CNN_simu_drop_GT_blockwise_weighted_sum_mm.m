addpath(genpath('.'))
gcp;

%%
scale_lowBG = 1e3;
scale_noise = 0.3;
results_folder = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
list_patch_dims = [253,316]; 
num_Exp = 10;

list_data_names={results_folder};
rate_hz = 10; % frame rate of each video
radius = 6;
data_ind = 1;
data_name = list_data_names{data_ind};
path_name = fullfile('E:\simulation_CNMFE_corr_noise',data_name);
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

%%
avg_radius = radius; % original_masks
% avg_radius = 9; % added_refined_masks
r_bg_ratio = 3;
leng = r_bg_ratio*avg_radius;

th_IoU_split = 0.5;
thj_inclass = 0.4;
thj = 0.7;
meth_baseline='median'; % {'median','median_mean','median_median'}
meth_sigma='quantile-based std'; % {'std','mode_Burr','median_std','std_back','median-based std'}
% vid=2;
% Exp_ID = list_Exp_ID{vid};

d0 = 0.8;
for lam = [3,5,8,10] % 
%% Load traces and ROIs
% folder of the GT Masks
dir_parent=path_name;
dir_video = dir_parent; 
% dir_SUNS = fullfile(dir_parent, 'complete_TUnCaT'); % 4 v1
dir_masks = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));
dir_add_new = fullfile(dir_masks, 'add_new_blockwise_weighted_sum_unmask');
% fs = rate_hz;
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
    % save(fullfile(dir_add_new,[Exp_ID,'_weights_blockwise.mat']),...
    %     'list_weight','list_weight_trace','list_weight_frame',...
    %     'sum_edges','traces_raw','video','masks');
    save(fullfile(dir_add_new,[Exp_ID,'_added_auto_blockwise.mat']), ...
        'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
        'images_added_crop', 'patch_locations'); % ,'time_weights','list_valid'
end
end
%%
clear mm;
delete(fileName);
disp('Finished this step');
