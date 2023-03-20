clear;
% folder of the raw video
dir_parent='E:\data_CNMFE\';
% name of the videos
list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};

%%
for vid= 1 % 1:length(list_Exp_ID)
    Exp_ID = list_Exp_ID{vid};
    dir_masks = fullfile(dir_parent,Exp_ID,'GT Masks');
    dir_trace = fullfile(dir_parent,Exp_ID,'traces');
    if ~exist(dir_trace,'dir')
        mkdir(dir_trace);
    end

    for xpart = 1:2
        for ypart = 1:2
            Exp_ID_part = sprintf('%s_part%d%d',Exp_ID,xpart,ypart);
            % Load video
            tic;
            fname=fullfile(dir_parent,Exp_ID,[Exp_ID_part,'.h5']);
            video_raw=h5read(fname, '/mov');
            toc;

            % Load ground truth masks
            load(fullfile(dir_masks,['FinalMasks_',Exp_ID_part,'.mat']),'FinalMasks');
            ROIs=logical(FinalMasks);
            clear FinalMasks;
            
           %%
            tic; 
            traces_raw=generate_traces_from_masks(video_raw,ROIs);
            toc;
            traces_bg_exclude=generate_bgtraces_from_masks_exclude(video_raw,ROIs);
            toc;

            save(fullfile(dir_trace,['raw and bg traces ',Exp_ID_part,'.mat']),'traces_raw','traces_bg_exclude');
        end
    end
end
