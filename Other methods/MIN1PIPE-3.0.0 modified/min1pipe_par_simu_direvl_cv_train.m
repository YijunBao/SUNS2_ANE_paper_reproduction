%%% demo of the full MIN1PIPE %%%
clear
warning off
gcp;
% To disable parallel processing, change "parfor" to "for", and change "min1pipe_h5_vary_nopar" to "min1pipe_h5_vary"

min1pipe_init;
addpath(genpath('../ANE'))

%% Set data and folder
scale_lowBG = 5e3;
scale_noise = 1;
data_name = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
list_patch_dims = [253,316]; 
rate_hz = 10; % frame rate of each video
radius = 6;

path_name = fullfile('../data/data_simulation',data_name);
num_Exp = 10;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

dir_save = fullfile(path_name,'min1pipe');
if ~ exist(dir_save,'dir')
    mkdir(dir_save);
end
save_date = '20230218';
% save_date = num2str(yyyymmdd(datetime));
load(fullfile(dir_save,['eval_',data_name,'_thb history ',save_date,'.mat']),'list_seq');

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
list_params.pix_select_sigthres = [0.1:0.1:0.9, 0.95, 0.98]; % 0.8; % 
list_params.pix_select_corrthres = [0.1:0.1:0.9, 0.95, 0.98]; % 0.6;
list_params.merge_roi_corrthres = [0.1:0.1:0.9, 0.95, 0.98]; % 0.9;
list_params.anidenoise_dt = 0.05:0.05:0.4; % 0.25; # 1/7;
list_params.anidenoise_kappa = [0.1:0.1:0.9, 0.95, 0.98]; % 0.5;
list_params.se = 2:15; % 2:8; % 5;
list_th_binary = [0.2, 0.3, 0.4, 0.5];
name_params = fieldnames(list_params); 
range_params = struct2cell(list_params); 
num_param_names = length(range_params);
num_params = cellfun(@length, range_params);
num_thb = length(list_th_binary);

%%
n_round = 2;
% list_seq = cell(n_round,1);
% for r = 1:n_round
%     if r==1
%         seq = [6     1     5     3     2     4];
%     else
%         seq = [num_param_names, randperm(num_param_names-1)];
%     %     seq = randperm(num_param_names);
%     end
%     list_seq{r} = seq;
% end

%%
for cv = 1:num_Exp
    trains = [1:cv-1, cv+1:num_Exp]; 
    %% Initialize variable parameters
    % init_ind_param = round(num_params/2);
    % init_ind_param = [1, 11, 17]'; 
    init_ind_param = [8, 6, 9, 3, 5, 4]'; 
    ind_param = init_ind_param;
    temp_param = cellfun(@(x,y) x(y), range_params,num2cell(ind_param));

    best_ind_param = init_ind_param;
    [best_Recall, best_Precision, best_F1, best_time,...
        Recall, Precision, F1, used_time] = deal(zeros(num_Exp,num_thb));
    best_thb = list_th_binary(1);
    history = zeros(num_param_names+4,0);
    best_history = zeros(num_param_names+4,0);

    %%
    for r = 1:n_round
        seq = list_seq{r};
        for p = 1:num_param_names
            ind_param = best_ind_param;
            ind_vary = seq(p);
            current_ind = ind_param(ind_vary);
            current_params = range_params{ind_vary};
            ascend = (current_ind+1):num_params(ind_vary);
            decend = (current_ind-1):-1:1;
            if mean(F1) == 0
                list_test_ind = {current_ind, ascend, decend};
            else
                list_test_ind = {ascend, decend};
            end
            correct_direction = false;
            
            for direction = 1:length(list_test_ind)
                for test_ind = list_test_ind{direction}
    %                 if (test_ind > 3) && (test_ind ~= list_test_ind{direction}) && ~correct_direction
    %                     test_ind = list_test_ind{direction};
    %                 end
                    ind_param(ind_vary) = test_ind;
                    temp_param = cellfun(@(x,y) x(y), range_params,num2cell(ind_param));
    %                 temp_param(ind_vary) = range_params{ind_vary}(test_ind);
                    disp(temp_param);
                    pix_select_sigthres = temp_param(1);
                    pix_select_corrthres = temp_param(2);
                    merge_roi_corrthres = temp_param(3);
                    dt = temp_param(4);
                    kappa = temp_param(5);
                    se = temp_param(6);
                    dir_sub = sprintf('pss=%0.2f_psc=%0.2f_mrc=%0.2f_dt=%0.2f_kappa=%0.2f_se=%d',...
                        pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres, dt, kappa, se);
                    dir_save_sub = fullfile(dir_save,dir_sub);
                    folder = fullfile('min1pipe',dir_sub);
                    if ~exist(dir_save_sub,'dir')
                        mkdir(dir_save_sub)
                    end

                    parfor eid = 1:num_Exp
                        [temp_Recall, temp_Precision, temp_F1, temp_used_time] = deal(zeros(1,num_thb));
                        Exp_ID = list_Exp_ID{eid};
                        filename = [Exp_ID,'.h5'];
                        %% main program %%
                        fname = fullfile(dir_save_sub,[Exp_ID,'_data_processed.mat']);
                        if ~exist(fname,'file')
                            try
%                                 [fname, frawname, fregname] = min1pipe_h5_vary(path_name, folder, filename, ...
                                [fname, frawname, fregname] = min1pipe_h5_vary_nopar(path_name, folder, filename, ...
                                    Fsi, Fsi_new, spatialr, se, ismc, flag,...
                                    pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres, dt, kappa);
                                delete([fname(1:end-18),'reg.mat']);
                                delete([fname(1:end-18),'reg_post.mat']);
                                delete([fname(1:end-18),'frame_all.mat']);
                            catch
                                has_error = true;
                                parsave1(fname,has_error,'has_error');
    %                             clear has_error;
    %                             pause;
                            end
                        end
                        
                        %% Calculate accuracy
                        saved_result = load(fname);
                        for tid = 1:num_thb
                            if ~isfield(saved_result,'has_error')
                                thb = list_th_binary(tid);
                                % load([dir_GT,'FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
                                % load([dir_GT,'FinalMasks_',Exp_ID,'_sparse.mat'],'GTMasks_2')
                                % [pixh, pixw, nGT] = size(FinalMasks);
                                roi3 = reshape(full(saved_result.roifn), saved_result.pixh, saved_result.pixw, []);
                                Masks3 = threshold_Masks(roi3, thb); %;%
    %                             roib = saved_result.roifn>th_binary*max(saved_result.roifn,[],1); %;%
    %                             Masks3 = reshape(full(roib), saved_result.pixh, saved_result.pixw, size(roib,2));
                                parsave1(fullfile(dir_save,dir_sub,[Exp_ID,'_Masks_',num2str(thb),'.mat']),Masks3,'Masks3');
                                % [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,roifn,0.5);
                                [temp_Recall(tid), temp_Precision(tid), temp_F1(tid)] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
                                temp_used_time(tid) = seconds(saved_result.process_time{2}-saved_result.process_time{1});
                            end
                        end
                        Recall(eid,:) = temp_Recall;
                        Precision(eid,:) = temp_Precision;
                        F1(eid,:) = temp_F1;
                        used_time(eid,:) = temp_used_time;
                        disp([temp_Recall', temp_Precision', temp_F1', temp_used_time'])
    %                     save(['eval_',data_name,'.mat'],'list_Recall','list_Precision','list_F1','list_time',...
    %                         'list_pss','list_psc','list_mrc','list_thb')
                    end

                    %% evaluate the mean F1 using this parameter set
                    F1(isnan(F1)) = 0;
                    mean_F1 = mean(F1(trains,:),1);
                    mean_best_F1 = mean(best_F1(trains,:),1);
                    [max_mean_F1, ind_thb] = max(mean_F1);
                    [max_mean_best_F1, ind_thb_best] = max(mean_best_F1);
                    
                    thb = list_th_binary(ind_thb);
                    Table_time = [Recall(:,ind_thb), Precision(:,ind_thb), F1(:,ind_thb), used_time(:,ind_thb)];
                    new_history = [temp_param', thb, mean(Table_time(trains,:))];
                    history = [history; new_history];
                    
                    if max_mean_F1 > max_mean_best_F1
                        if max_mean_best_F1 > 0
                            correct_direction = true;
                        end
                        best_Recall = Recall;
                        best_Precision = Precision;
                        best_F1 = F1;
                        best_time = used_time;
                        best_ind_param = ind_param;
                        best_ind_thb = ind_thb;
                        best_thb = list_th_binary(ind_thb);
    %                     Table_time = [best_Recall(:,ind_thb), best_Precision(:,ind_thb), best_F1(:,ind_thb), best_time(:,ind_thb)];
                        best_history = [best_history; new_history];
                    end
                    save(fullfile(dir_save,['eval_',data_name,'_thb history cv', num2str(cv),'.mat']),...
                        'list_seq','best_Recall','best_Precision','best_F1','best_time','best_ind_param',...
                        'best_thb','ind_param','list_params','history','best_history')
                    if max_mean_F1 < max_mean_best_F1
                        break
                    end
                end
                if correct_direction
                    break
                end
            end
        end
    end
    %%
    movefile(fullfile(dir_save,['eval_',data_name,'_thb history cv', num2str(cv),'.mat']), ...
        fullfile(dir_save,['eval_',data_name,'_thb history ',save_date,' cv', num2str(cv),'.mat']));
end

%% Generate a summary table
list_Table = cell(num_Exp,1);
for cv = 1:num_Exp
    trains = [1:cv-1, cv+1:num_Exp]; 
    load(fullfile(dir_save,['eval_',data_name,'_thb history ',save_date,' cv', num2str(cv),'.mat']),...
        'list_seq','best_Recall','best_Precision','best_F1','best_time','best_ind_param',...
        'best_thb','ind_param','list_params','history','best_history')
    % best_history = history;
    end_history = best_history(end,:);
    best_param = end_history(1:end-4);
    thb = end_history(end-4);
    list_th_binary = [0.2, 0.3, 0.4, 0.5];
    best_ind_thb = find(list_th_binary == thb);
    Table_test = [best_Recall(cv,best_ind_thb), best_Precision(cv,best_ind_thb), best_F1(cv,best_ind_thb), best_time(cv,best_ind_thb)];
    Table_train = [best_Recall(trains,best_ind_thb), best_Precision(trains,best_ind_thb), best_F1(trains,best_ind_thb), best_time(trains,best_ind_thb)];
    Table_train = mean(Table_train);
    list_Table{cv} = [best_param, Table_test, Table_train];
end
Table = cell2mat(list_Table);
Table_ext=[Table;nanmean(Table,1);nanstd(Table,1,1)];
save(fullfile(dir_save,['eval_',data_name,'_thb ',save_date,' cv.mat']),'Table_ext');
