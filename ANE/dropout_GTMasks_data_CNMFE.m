%%
% name of the videos
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
data_ind = 4;
data_name = list_data_names{data_ind};
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
rate_hz = [10,15,7.5,5]; % frame rate of each video
list_avg_radius = [5,6,8,14];
list_lam = [15,5,8,8];
r_bg_ratio = 3;
leng = r_bg_ratio*list_avg_radius(data_ind);
d0 = 0.8;
sub_added = '';

%% Load traces and ROIs
lam = list_lam(data_ind);
% dir_parent=fullfile('E:\data_CNMFE\',[data_name,'_original_masks']);
dir_parent=fullfile('E:\data_CNMFE\',[data_name,sub_added]);
dir_video = dir_parent; 
dir_masks = fullfile(dir_parent, 'GT Masks');
dir_traces_raw=fullfile(dir_video,'complete_TUnCaT\TUnCaT\raw');
dir_traces_unmix=fullfile(dir_video,'complete_TUnCaT\TUnCaT\alpha= 1.000');
% folder = ['.\Result_',data_name];
doesunmix = 1;

dir_add_new = fullfile(dir_parent, sprintf('GT Masks dropout %gexp(-%g)',d0,lam));
if ~ exist(dir_add_new,'dir')
    mkdir(dir_add_new);
end
list_keep = cell(1,num_Exp);

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
% disp('Finished this step');

