%%% demo of the full MIN1PIPE %%%
clear
warning off
gcp;
min1pipe_init;
addpath('C:\Matlab Files\neuron_post');
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))

%% Set data and folder
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_patch_dims = [120,120; 80,80; 88,88; 192,240]; 
rate_hz = [10,15,7.5,5]; % frame rate of each video
radius = [5,6,6,6];

data_ind = 1;
data_name = list_data_names{data_ind};
path_name = fullfile('E:\data_CNMFE',data_name);
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

dir_save = fullfile(path_name,'min1pipe');
saved_date = '20220522';

%% session-specific parameter initialization %% 
Fsi = rate_hz(data_ind); % 20;
Fsi_new = Fsi; %%% no temporal downsampling %%%
spatialr = 1; %%% no spatial downsampling %%%
se = 5; % 3.6; %%% structure element for background removal %%%
ismc = false; % true; %%% run movement correction %%%
flag = 1; %%% use auto seeds selection; 2 if manual %%%
isvis = false; % true; % %% do visualize %%%
ifpost = false; %%% set true if want to see post-process %%%

%% Set range of parameters to optimize over
gSiz = 12;
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
    load(['eval_',data_name,'_thb history ',saved_date,' cv', num2str(cv),'.mat'],'best_Recall','best_Precision','best_F1',...
        'best_time','best_ind_param','best_thb','ind_param','list_params','history','best_history');
%     ind_param = best_ind_param;
%     best_param = cellfun(@(x,y) x(y), range_params,num2cell(best_ind_param));
    end_history = best_history(end,:);
    best_param = end_history(1:end-5);
    pix_select_sigthres = best_param(1);
    pix_select_corrthres = best_param(2);
    merge_roi_corrthres = best_param(3);
    dir_sub = sprintf('pss=%0.2f_psc=%0.2f_mrc=%0.2f',...
        pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres);

    for eid = cv % trains % 1:num_Exp
        Exp_ID = list_Exp_ID{eid};
        filename = [Exp_ID,'.h5'];
        %% main program %%
        fname = fullfile(dir_save,dir_sub,[Exp_ID,'_data_processed.mat']);
        if ~exist(fname,'file')
            [fname, frawname, fregname] = min1pipe_h5_vary(path_name, filename, ...
                Fsi, Fsi_new, spatialr, se, ismc, flag,...
                pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres);
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
save(['eval_',data_name,'_thb ',saved_date,' cv.mat'],'Table_time_ext');
