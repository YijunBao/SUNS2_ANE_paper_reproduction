clear;
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
data_ind = 3;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
rate_hz = [10,15,7.5,5]; % frame rate of each video
% folder of the raw video
dir_parent=fullfile('E:\data_CNMFE\',data_name);
dir_video = dir_parent;

%%
for vid= 1:length(list_Exp_ID)
    Exp_ID = list_Exp_ID{vid};
    dir_masks = fullfile(dir_parent,'added_blockwise\GT Masks');
    dir_trace = fullfile(dir_parent,'SNR traces');
    if ~exist(dir_trace,'dir')
        mkdir(dir_trace);
    end

    % Load video
    tic;
    video_raw = h5read(fullfile(dir_video,[Exp_ID,'.h5']),'/mov');
    [Lx,Ly,T]=size(video_raw);
    clear Y;
    toc;

    % Load ground truth masks
    load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    ROIs=logical(FinalMasks);
    clear FinalMasks;
            
    %%
    meth_baseline='median'; % {'median','median_mean','median_median'}
    meth_sigma='quantile-based std'; % {'std','mode_Burr','median_std','std_back','median-based std'}
    % filter_tempolate = h5read(fullfile(dir_parent,Exp_ID,[Exp_ID,'_spike_tempolate.h5'],'/filter_tempolate');

    tic;
    video_sf =homo_filt(video_raw, 50);
    toc;
    clear video_raw;

    % [video_SNR, F0, mu, sigma] = Possion_noise_based_filter(video_sf,exp(1)-1,1,isBGremove,meth_F0,meth_sigma,window_half,true);
    tic;
    [mu, sigma] = SNR_normalization_video(video_sf,meth_sigma,meth_baseline);
    video_SNR = (video_sf-mu)./sigma;
    toc;
    clear video_sf;
    
    video_SNR = imgaussfilt(video_SNR); % ,1
    save(fullfile(dir_trace,['SNR video ',Exp_ID,'.mat']),'video_SNR');

    %%
    tic; 
    traces_raw=generate_traces_from_masks(video_SNR,ROIs);
    toc;
    traces_bg_exclude=generate_bgtraces_from_masks_exclude(video_SNR,ROIs);
    toc;
    traces_out_exclude=generate_outtraces_from_masks_exclude(video_SNR,ROIs);
    toc;

    save(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude','traces_out_exclude');
end
