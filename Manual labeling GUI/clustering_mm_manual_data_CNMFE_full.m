gcp;
addpath(genpath('.'))
addpath(genpath(fullfile('..','ANE')))

%%
% name of the videos
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
num_Exp = length(list_Exp_ID);
rate_hz = 20; % frame rate of each video

th_IoU_split = 0.5;
thj_inclass = 0.4;
thj = 0.7;
meth_baseline='median'; % {'median','median_mean','median_median'}
meth_sigma='quantile-based std'; % {'std','mode_Burr','median_std','std_back','median-based std'}

dir_video=fullfile('E:','data_CNMFE');
% fs = rate_hz;
time_weights = zeros(num_Exp,1);

eid = 4;
Exp_ID = list_Exp_ID{eid};
DirSave = ['Results_',Exp_ID];
dir_manual_merged = fullfile(DirSave,'manual_draw_remove_overlap');
dir_added = fullfile(DirSave, 'add_new_blockwise');
if ~ exist(dir_added,'dir')
    mkdir(dir_added);
end

% load(fullfile(dir_manual_draw,['Added_',Exp_ID,'.mat']),'FinalMasks');
load(fullfile(dir_manual_merged,['Manual_',Exp_ID,'_nonoverlap.mat']),'FinalMasks');
masks=logical(FinalMasks); % permute(logical(Masks),[3,2,1]);
area = squeeze(sum(sum(masks,1),2));
avg_area = median(area);
avg_radius = round(sqrt(avg_area/pi));
r_bg_ratio = 3;
leng = r_bg_ratio*avg_radius;

load(fullfile(dir_video, [Exp_ID,'.mat']),'Y','Ysiz'); %
video_raw = Y;

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

%% Save Masks
masks_added_full = cat(3,list_added_full{:});
images_added_crop = cat(3,list_added_images_crop{:});
masks_added_crop = cat(3,list_added_crop{:});
patch_locations = cat(1,list_locations{:});
added_frames = horzcat(list_added_frames{:});
added_weights = horzcat(list_added_weights{:});
%     time_weights = toc;
save(fullfile(dir_added,[Exp_ID,'_weights_blockwise.mat']),...
    'list_weight','list_weight_trace','list_weight_frame',...
    'sum_edges','traces_raw','video','masks');
save(fullfile(dir_added,[Exp_ID,'_added_auto_blockwise.mat']), ...
    'added_frames','added_weights', 'masks_added_full','masks_added_crop',...
    'images_added_crop', 'patch_locations'); % ,'time_weights','list_valid'

%%
clear mm;
delete(fileName);
disp('Finished this step');
