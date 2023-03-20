addpath('C:\Matlab Files\SUNS-1p\1p-CNMFE');
%%
scale_lowBG = 5e3;
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

%% Load traces and ROIs
d0 = 0.8;
% folder of the GT Masks
for lam = [3,5,8,10] % 20 % 
dir_parent=path_name;
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
