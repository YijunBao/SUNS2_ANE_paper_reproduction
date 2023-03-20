%%% demo of the full MIN1PIPE %%%
clear
warning off
gcp;
min1pipe_init;
addpath('C:\Matlab Files\neuron_post');
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))

%% Set data and folder
list_data_names={'noise1'};
list_patch_dims = [253,316]; 
rate_hz = 10; % frame rate of each video
radius = 6;

data_ind = 1;
data_name = list_data_names{data_ind};
path_name = fullfile('E:\simulation_CNMFE',data_name);
num_Exp = 10;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

dir_save = fullfile(path_name,'min1pipe');
if ~ exist(dir_save,'dir')
    mkdir(dir_save);
end

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
list_params.pix_select_sigthres = [0.1:0.1:0.9, 0.95]; % 0.8; % 
list_params.pix_select_corrthres = [0.1:0.05:0.95, 0.98]; % 0.6;
list_params.merge_roi_corrthres = [0.1:0.05:0.95, 0.98]; % 0.9;
list_th_binary = [0.2, 0.3, 0.4, 0.5];
name_params = fieldnames(list_params); 
range_params = struct2cell(list_params); 
num_param_names = length(range_params);
num_params = cellfun(@length, range_params);
num_thb = length(list_th_binary);
n_round = 3;

%% Initialize variable parameters
% init_ind_param = round(num_params/2);
% init_ind_param = [1, 11, 17]'; 
init_ind_param = [8, 11, 17]'; 
ind_param = init_ind_param;
temp_param = cellfun(@(x,y) x(y), range_params,num2cell(ind_param));

best_ind_param = init_ind_param;
[best_Recall, best_Precision, best_F1, best_used_time, ...
    Recall, Precision, F1, used_time] = deal(zeros(num_Exp,num_thb));
history = zeros(num_param_names+4,0);
best_history = zeros(num_param_names+4,0);

%%
for r = 1:n_round
%     seq = randperm(num_param_names);
    seq = 1:num_param_names;
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
                dir_sub = sprintf('pss=%0.2f_psc=%0.2f_mrc=%0.2f',...
                    pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres);

                for eid = 1:num_Exp
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
                    for tid = 1:num_thb
                        th_binary = list_th_binary(tid);
                        % load([dir_GT,'FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
                        % load([dir_GT,'FinalMasks_',Exp_ID,'_sparse.mat'],'GTMasks_2')
                        % [pixh, pixw, nGT] = size(FinalMasks);
                        roib = roifn>th_binary*max(roifn,[],1); %;%
                        Masks3 = reshape(full(roib), pixh, pixw, size(roib,2));
                        save(fullfile(dir_save,dir_sub,[Exp_ID,'_Masks_',num2str(th_binary),'.mat']),'Masks3');
                        % [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,roifn,0.5);
                        [Recall(eid,tid), Precision(eid,tid), F1(eid,tid)] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
                        used_time(eid,tid) = seconds(process_time{2}-process_time{1});
                    end
                    disp([Recall(eid,:)', Precision(eid,:)', F1(eid,:)', used_time(eid,:)'])
%                     save(['eval_',data_name,'.mat'],'list_Recall','list_Precision','list_F1','list_time',...
%                         'list_pss','list_psc','list_mrc','list_thb')
                end

                %% evaluate the mean F1 using this parameter set
                mean_F1 = mean(F1,1);
                mean_best_F1 = mean(best_F1,1);
                [max_mean_F1, ind_thb] = max(mean_F1);
                [max_mean_best_F1, ind_thb_best] = max(mean_best_F1);
                
                thb = list_th_binary(ind_thb);
                Table_time = [Recall(:,ind_thb), Precision(:,ind_thb), F1(:,ind_thb), used_time(:,ind_thb)];
                new_history = [temp_param', thb, mean(Table_time)];
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
                save(['eval_',data_name,'_thb history.mat'],'best_Recall','best_Precision','best_F1',...
                    'best_time','best_ind_param','best_thb','ind_param','list_params','history','best_history')
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
movefile(['eval_',data_name,'_thb history.mat'], ['eval_',data_name,'_thb history ',num2str(yyyymmdd(datetime)),'.mat']);

%%
% best_history = history;
end_history = best_history(end,:);
best_param = end_history(1:end-4);
th_binary = end_history(end-4);
list_th_binary = [0.2, 0.3, 0.4, 0.5];
best_ind_thb = find(list_th_binary == th_binary);
Table_eval = [best_Recall(:,best_ind_thb), best_Precision(:,best_ind_thb), best_F1(:,best_ind_thb), best_time(:,best_ind_thb)];
Table = [repmat(best_param,[num_Exp,1]), Table_eval];
Table_ext=[Table;nanmean(Table,1);nanstd(Table,1,1)];
