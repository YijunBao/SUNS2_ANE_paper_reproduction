%%
% name of the videos
list_Exp_ID = { 'c25_59_228','c27_12_326','c28_83_210',...
                'c25_163_267','c27_114_176','c28_161_149',...
                'c25_123_348','c27_122_121','c28_163_244'};
num_Exp = length(list_Exp_ID);
lam = 15;
d0 = 0.8;

%% Load traces and ROIs
dir_parent='E:\1photon-small\added_refined_masks\';
dir_video = dir_parent; 
dir_masks = fullfile(dir_parent, 'GT Masks');
dir_traces_raw=fullfile(dir_video,'SUNS_TUnCaT_SF25','TUnCaT','raw');
dir_traces_unmix=fullfile(dir_video,'SUNS_TUnCaT_SF25','TUnCaT','alpha= 1.000');
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
