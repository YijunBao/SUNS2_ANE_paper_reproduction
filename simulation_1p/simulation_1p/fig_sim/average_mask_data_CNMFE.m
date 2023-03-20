%%
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
rate_hz = [10,15,7.5,5]; % frame rate of each video
list_avg_radius = [5,6,0,0];

%%
data_ind = 1;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);

%%
dir_video=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
dir_masks = fullfile(dir_video, 'GT Masks');
dir_traces = fullfile(dir_video, 'complete_TUnCaT\TUnCaT\alpha= 1.000');
pct = 1; 

%%
for eid = 1:num_Exp
    %% Load video and masks
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    masks=logical(FinalMasks); % permute(logical(Masks),[3,2,1]);
    fname=fullfile(dir_video,[Exp_ID,'.h5']);
    video_raw = h5read(fname, '/mov');
    [d1,d2,T] = size(video_raw);
    load(fullfile(dir_traces, [Exp_ID,'.mat']),'traces_nmfdemix');
    
