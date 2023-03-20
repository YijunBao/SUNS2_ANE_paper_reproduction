%%% demo of the full MIN1PIPE %%%
clear
warning off
gcp;
min1pipe_init;
addpath('C:\Matlab Files\neuron_post');
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))

%% Set data and folder
patch_dims = [160,160]; 
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
rate_hz = 20; % frame rate of each video
radius = 10;
data_name = 'TENASPIS';
% path_name = 'D:\data_TENASPIS\added_refined_masks';
path_name = 'D:\data_TENASPIS\original_masks';
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

dir_save = fullfile(path_name,'min1pipe');
saved_date = '20230214';

%% session-specific parameter initialization %% 
Fsi = rate_hz; % 20;
Fsi_new = Fsi; %%% no temporal downsampling %%%
spatialr = 1; %%% no spatial downsampling %%%
% se = 5; % 3.6; %%% structure element for background removal %%%
ismc = false; % true; %%% run movement correction %%%
flag = 1; %%% use auto seeds selection; 2 if manual %%%
isvis = false; % true; % %% do visualize %%%
ifpost = false; %%% set true if want to see post-process %%%

%% Set range of parameters to optimize over
gSiz = 2 * radius; % 12;
% list_params.pix_select_sigthres = 0.8; % [0.1:0.05:0.95, 0.98]; % 
% list_params.pix_select_corrthres = [0.1:0.05:0.95, 0.98]; % 0.6;
% list_params.merge_roi_corrthres = [0.1:0.05:0.95, 0.98]; % 0.9;
% list_th_binary = [0.2, 0.3, 0.4, 0.5];
% name_params = fieldnames(list_params); 
% range_params = struct2cell(list_params); 
% num_param_names = length(range_params);
% num_params = cellfun(@length, range_params);
% num_thb = length(list_th_binary);
% n_round = 3;
[Recall, Precision, F1, used_time] = deal(zeros(num_Exp,1));
Table_time = cell(num_Exp,1);

%%
for cv = 1:num_Exp
    load(['eval_',data_name,'_thb history original_masks ',saved_date,'.mat'],'best_Recall','best_Precision','best_F1',...
        'best_time','best_ind_param','best_thb','ind_param','list_params','history','best_history'); % ,' cv', num2str(cv)
%     ind_param = best_ind_param;
%     best_param = cellfun(@(x,y) x(y), range_params,num2cell(best_ind_param));
    end_history = best_history(end,:);
    best_param = end_history(1:end-5);
    pix_select_sigthres = best_param(1);
    pix_select_corrthres = best_param(2);
    merge_roi_corrthres = best_param(3);
    dt = best_param(4);
    kappa = best_param(5);
    se = best_param(6);
    dir_sub = sprintf('pss=%0.2f_psc=%0.2f_mrc=%0.2f_dt=%0.2f_kappa=%0.2f_se=%d',...
        pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres, dt, kappa, se);
    dir_save_sub = fullfile(dir_save,dir_sub);
    folder = fullfile('min1pipe',dir_sub);

    for eid = cv % trains % 1:num_Exp
        Exp_ID = list_Exp_ID{eid};
        filename = [Exp_ID,'.h5'];
        %% main program %%
        fname = fullfile(dir_save_sub,[Exp_ID,'_data_processed.mat']);
        if true % ~exist(fname,'file') % 
            [fname, frawname, fregname] = min1pipe_h5_vary(path_name, folder, filename, ...
                Fsi, Fsi_new, spatialr, se, ismc, flag,...
                pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres, dt, kappa);
            delete([fname(1:end-18),'reg.mat']);
            delete([fname(1:end-18),'reg_post.mat']);
            delete([fname(1:end-18),'frame_all.mat']);
        end

        %% Calculate accuracy
        load(fname)
%         for tid = 1:num_thb
        th_binary = end_history(end-4);
        % load([dir_GT,'FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
        % load([dir_GT,'FinalMasks_',Exp_ID,'_sparse.mat'],'GTMasks_2')
        % [pixh, pixw, nGT] = size(FinalMasks);
        roib = roifn>th_binary*max(roifn,[],1); %;%
        Masks3 = reshape(full(roib), pixh, pixw, size(roib,2));
        save(fullfile(dir_save,dir_sub,[Exp_ID,'_Masks_',num2str(th_binary),'.mat']),'Masks3');
        % [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,roifn,0.5);
        [Recall(cv), Precision(cv), F1(cv)] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
        used_time(cv) = seconds(process_time{2}-process_time{1});
        Table_time{cv} = [end_history(1:end-4),Recall(cv), Precision(cv), F1(cv),used_time(cv),end_history(end-3:end)];
    end
end
%%
Table_time = cell2mat(Table_time);
Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
save(['eval_',data_name,'_thb ',saved_date,' original_masks test.mat'],'Table_time_ext');
