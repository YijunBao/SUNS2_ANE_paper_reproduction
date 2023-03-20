%% clear the workspace and select data
warning off;
% gcp;
% if isempty(gcp('nocreate'))
%     parpool;
% end
addpath(genpath('C:\Other methods\CNMF_E-1.1.2'));
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'));
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
save_date = '20221221'; % num2str(yyyymmdd(datetime));

%% pre-load the data to memory
load_date = '20221220';

for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    video = h5read(fullfile(path_name,[Exp_ID,'.h5']),'/mov');
    clear video;
end

%% Set range of parameters to optimize over
gSiz = 2 * radius; % 12;
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

for cv = 1:5 % Exp_ID
    % list_seq = cell(n_round,1);
    % load(['eval_',data_name,'_thb history ',load_date,'.mat'],'list_seq');
    load(['.\eval_',data_name,'_thb history ',save_date,' cv', num2str(cv),' 2round.mat'],...
        'list_params','history','best_history','list_seq');
    % 'best_Recall','best_Precision','best_F1','best_time','best_ind_param','best_thb','ind_param',

    %%
    seq = list_seq{1};
    nt = size(history,1);
    list_param_vary = zeros(nt,1);
    b = 1;
    end_best_history = best_history(1,:);
    end_history = history(1,:);
    end_param = end_history(1:end-5);
    end_F1 = end_history(end-1);
    for t = 2:size(history,1)
        current_history = history(t,:);
        current_param = current_history(1:end-5);
        current_F1 = current_history(end-1);
        ind_param_vary = find(current_param - end_param);
        list_param_vary(t) = ind_param_vary;
        if current_F1 > end_F1
            end_history = current_history;
            end_param = current_param;
            end_F1 = current_F1;
            b=b+1;
            end_best_history = best_history(b,:);
            if any(end_best_history ~= end_history)
                error('Mismatch');
            end
        end
    end

    ind_param_last1 = list_seq{1}(end);
    ind_param_first2 = list_seq{2}(1);
    if ind_param_last1 ~= ind_param_first2
        opt_last1 = find(list_param_vary == ind_param_last1);
        opt_first2 = find(list_param_vary == ind_param_first2);
        round1_last = opt_last1(find(opt_last1 < opt_first2(end),1,'last'));
    else
        disp('Warning');
        opt_last1 = find(list_param_vary == ind_param_last1);
        history_temp = history(opt_last1,:);
        for h = 2:length(opt_last1)
            if ~all(any(history_temp(1:h-1,:)-history_temp(h,:),2))
                round1_last = opt_last1(h-1);
                break;
            end
        end
    end

    history = history(1:round1_last,:);
    [best_F1, best_t] = max(history(:,end-1));
    best_history = best_history(best_history(:,end-1)<=best_F1,:);

    %%
    save(['.\eval_',data_name,'_thb history ',save_date,' cv', num2str(cv),'.mat'],...
    'list_params','history','best_history','list_seq');
end