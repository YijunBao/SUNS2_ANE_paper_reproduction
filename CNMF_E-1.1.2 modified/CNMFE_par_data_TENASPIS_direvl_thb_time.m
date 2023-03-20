%% clear the workspace and select data
warning off;
% gcp;
% if isempty(gcp('nocreate'))
%     parpool;
% end
addpath(genpath('C:\Other methods\CNMF_E-1.1.2'));
% addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'));
addpath(genpath('C:\Matlab Files\missing_finder'));
clear; clc; close all;  

%% 
patch_dims = [160,160]; 
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
rate_hz = 20; % frame rate of each video
radius = 9;
data_name = 'TENASPIS';
path_name = 'D:\data_TENASPIS\added_refined_masks';
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

dir_save = fullfile(path_name,'CNMFE');
if ~ exist(dir_save,'dir')
    mkdir(dir_save);
end
load_date = '20221220';

%% Set range of parameters to optimize over
gSiz = 2*radius;
list_params.rbg = [1.5, 1.8, 2]; % 1.5; % 
list_params.nk = [1, 2, 3]; % 3; % 
list_params.rdmin = [2, 2.5, 3, 4]; % 3; % 
% list_params.min_corr = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98];
% list_params.min_pnr = 4:4:24;
list_params.min_corr = 0.1:0.1:0.9;
list_params.min_pnr = 1:9;
list_params.merge_thr = 0.1:0.1:0.9;
list_params.merge_thr_spatial = 0.1:0.1:0.9;
list_params.merge_thr_temporal = 0.1:0.1:0.9;
list_th_binary = [0.2, 0.3, 0.4, 0.5];
name_params = fieldnames(list_params); 
range_params = struct2cell(list_params); 
num_param_names = length(range_params);
num_params = cellfun(@length, range_params);
num_thb = length(list_th_binary);
n_round = 2;
list_seq = cell(n_round,1);

%%
load(['eval_',data_name,'_thb history ',load_date,'.mat'],'history');
history_time = padarray(history,[0,1],'post');
used_time = zeros(num_Exp,1);

for h = 1:size(history,1)
    current_history = history(h,:);
    best_param = current_history(1:end-5);
    rbg = best_param(1);
    nk = best_param(2);
    rdmin = best_param(3);
    min_corr = best_param(4);
    min_pnr = best_param(5);
    merge_thr = best_param(6);
    mts = best_param(7);
    mtt = best_param(8);

    dir_sub = sprintf('gSiz=%d,rbg=%0.1f,nk=%d,rdmin=%0.1f,mc=%0.2f,mp=%d,mt=%0.2f,mts=%0.2f,mtt=%0.2f',...
        gSiz,rbg,nk,rdmin,min_corr,min_pnr,merge_thr,mts,mtt);
    %%
    for eid = 1:num_Exp
        Exp_ID = list_Exp_ID{eid};
        try
            saved_result = load(fullfile(dir_save,dir_sub,[Exp_ID,'_time.mat']),'process_time');
            temp_used_time = seconds(saved_result.process_time{4}-saved_result.process_time{2});
        catch
            temp_used_time = nan;
        end
        used_time(eid) = temp_used_time;
    end
    history_time(h,end) = max(used_time);
    disp(history_time(h,:));
end
%%
% Table_time = [best_Recall, best_Precision, best_F1, best_time];
% Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
%%
save(['eval_',data_name,'_thb history ',load_date,' time.mat'],'history_time');
