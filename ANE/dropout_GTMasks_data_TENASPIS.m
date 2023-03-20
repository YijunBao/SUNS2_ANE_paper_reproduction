addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');
%%
% name of the videos
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
% list_Exp_ID={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
rate_hz = 20; % frame rate of each video
avg_radius = 10;
r_bg_ratio = 3;
leng = r_bg_ratio*avg_radius;
d0 = 0.8;

%% Load traces and ROIs
% folder of the GT Masks
for lam = [5,10,15,20] % 20 % 
dir_parent='D:\data_TENASPIS\added_refined_masks\';
% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_added_blockwise_weighted_sum_unmask']);
dir_video = dir_parent; 
dir_masks = fullfile(dir_parent, 'GT Masks');
dir_traces_raw=fullfile(dir_video,'complete_TUnCaT_SF25\TUnCaT\raw');
dir_traces_unmix=fullfile(dir_video,'complete_TUnCaT_SF25\TUnCaT\alpha= 1.000');
% folder = ['.\Result_',data_name];
doesunmix = 1;

dir_add_new = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));
if ~ exist(dir_add_new,'dir')
    mkdir(dir_add_new);
end
list_keep = cell(1,num_Exp);

% eid = 4;
for eid = 1:num_Exp
    %% Calculate PSNR
    Exp_ID = list_Exp_ID{eid};
    load(fullfile(dir_masks,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    masks=logical(FinalMasks);
    if doesunmix
        load(fullfile(dir_traces_unmix,[Exp_ID,'.mat']),'traces_nmfdemix'); % raw_traces
        traces = traces_nmfdemix';
        addon = 'unmix ';
    else
        load(fullfile(dir_traces_raw,[Exp_ID,'.mat']),'traces','bgtraces'); % raw_traces
        traces = traces - bgtraces;
        traces = traces';
        addon = 'nounmix ';
    end
    [med, sigma] = SNR_normalization(traces,'quantile-based std');
    SNR = (traces - med)./sigma;
    PSNR = max(SNR,[],2)';
    num = length(PSNR);
    
    %% Drop out neurons, with the probability proportional to exp(-PSNR)
    prob = d0.*exp(-PSNR/lam);
    keep = rand(1,num)>=prob; 
    DroppedMasks = FinalMasks(:,:,~keep);
    FinalMasks = FinalMasks(:,:,keep);
    list_keep{eid} = keep;

    %% Save Masks
    save(fullfile(dir_add_new,['FinalMasks_',Exp_ID,'.mat']),'FinalMasks');
    save(fullfile(dir_add_new,['DroppedMasks_',Exp_ID,'.mat']),'DroppedMasks');
end
%% Save list_keep
num_total = cellfun(@length, list_keep);
num_keep = cellfun(@sum, list_keep);
num_drop = num_total-num_keep;
drop_ratio = sum(num_drop)/sum(num_total)
save(fullfile(dir_add_new,'list_keep.mat'),'list_keep','num_total','num_keep','num_drop');
end
disp('Finished this step');
