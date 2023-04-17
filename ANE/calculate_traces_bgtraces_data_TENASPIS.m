clear;
% folder of the raw video
dir_parent='D:\data_TENASPIS\added_refined_masks\';
% name of the videos
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp_ID = length(list_Exp_ID);
list_avg_area = zeros(1,num_Exp_ID);

%%
for vid= 1:length(list_Exp_ID)
    Exp_ID = list_Exp_ID{vid};
    dir_masks = fullfile(dir_parent,'GT Masks');
    dir_trace = fullfile(dir_parent,'traces');
    if ~exist(dir_trace,'dir')
        mkdir(dir_trace);
    end

%     % Load video
%     tic;
%     fname=fullfile(dir_parent,[Exp_ID,'.h5']);
%     video_raw=h5read(fname, '/mov');
%     toc;

    % Load ground truth masks
    load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    ROIs=logical(FinalMasks);
    clear FinalMasks;
    area = sum(sum(ROIs,1),2);
    list_avg_area(vid) = mean(area);

   %%
%     tic; 
%     traces_raw=generate_traces_from_masks(video_raw,ROIs);
%     toc;
%     traces_bg_exclude=generate_bgtraces_from_masks_exclude(video_raw,ROIs);
%     toc;
% 
%     save(fullfile(dir_trace,['raw and bg traces ',Exp_ID,'.mat']),'traces_raw','traces_bg_exclude');
end
mean(list_avg_area)