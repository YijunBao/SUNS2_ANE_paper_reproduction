%% clear the workspace and select data
warning off;
gcp;
% if isempty(gcp('nocreate'))
%     parpool;
% end
addpath(genpath('C:\Other methods\CNMF_E-1.1.2'));
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'));
addpath(genpath('C:\Matlab Files\missing_finder'));
clear; clc; close all;  


%% 
scale_lowBG = 5e3;
scale_noise = 1;
results_folder = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
list_data_names={results_folder};
list_patch_dims = [253,316]; 
rate_hz = 10; % frame rate of each video
radius = 6;

data_ind = 1;
data_name = list_data_names{data_ind};
path_name = fullfile('E:\simulation_CNMFE_corr_noise',data_name);
num_Exp = 10;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

dir_save = fullfile(path_name,'CNMFE');
if ~ exist(dir_save,'dir')
    mkdir(dir_save);
end
% save_date = num2str(yyyymmdd(datetime));
save_date = '20230217';
load(['eval_',data_name,'_thb history ',save_date,'.mat'],'list_seq');

%% pre-load the data to memory
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    video = h5read(fullfile(path_name,[Exp_ID,'.h5']),'/mov');
    clear video;
end

%% Set range of parameters to optimize over
gSiz = 12;
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

%% Other fixed parameters. 
% -------------------------    COMPUTATION    -------------------------  %
pars_envs = struct('memory_size_to_use', 120, ...   % GB, memory space you allow to use in MATLAB
    'memory_size_per_patch', 10, ...   % GB, space for loading data within one patch
    'patch_dims', list_patch_dims(data_ind,:));  %GB, patch size

% -------------------------      SPATIAL      -------------------------  %
ssub = 1;           % spatial downsampling factor
with_dendrites = true;   % with dendrites or not
spatial_constraints = struct('connected', true, 'circular', false);  % you can include following constraints: 'circular'
spatial_algorithm = 'hals_thresh';

% -------------------------      TEMPORAL     -------------------------  %
Fs = rate_hz(data_ind);             % frame rate
tsub = 1;           % temporal downsampling factor
deconv_options = struct('type', 'ar1', ... % model of the calcium traces. {'ar1', 'ar2'}
    'method', 'foopsi', ... % method for running deconvolution {'foopsi', 'constrained', 'thresholded'}
    'smin', -5, ...         % minimum spike size. When the value is negative, the actual threshold is abs(smin)*noise level
    'optimize_pars', true, ...  % optimize AR coefficients
    'optimize_b', true, ...% optimize the baseline);
    'max_tau', 100);    % maximum decay time (unit: frame);

% when changed, try some integers smaller than total_frame/(Fs*30)
detrend_method = 'spline';  % compute the local minimum as an estimation of trend.

% -------------------------     BACKGROUND    -------------------------  %
bg_model = 'ring';  % model of the background {'ring', 'svd'(default), 'nmf'}
nb = 1;             % number of background sources for each patch (only be used in SVD and NMF model)
%otherwise, it's just the width of the overlapping area
num_neighbors = []; % number of neighbors for each neuron
bg_ssub = 2;        % downsample background for a faster speed 

% -------------------------      MERGING      -------------------------  %
show_merge = false;  % if true, manually verify the merging step
method_dist = 'max';   % method for computing neuron distances {'mean', 'max'}

% -------------------------  INITIALIZATION   -------------------------  %
% K = [];             % maximum number of neurons per patch. when K=[], take as many as possible.
bd = 0;             % number of rows/columns to be ignored in the boundary (mainly for motion corrected data)
frame_range = [];   % when [], uses all frames
save_initialization = false;    % save the initialization procedure as a video.
use_parallel = true;    % use parallel computation for parallel computing
show_init = false;    % true;   % show initialization results
choose_params = false;    % true; % manually choose parameters
center_psf = true;  % set the value as true when the background fluctuation is large (usually 1p data)
% set the value as false when the background fluctuation is small (2p)
use_prev = false; % use previous initialization.

% -------------------------  Residual   -------------------------  %
min_corr_res = 0.7; % 0.7;
min_pnr_res = 6; % 6;
seed_method_res = 'auto';  % method for initializing neurons from the residual

% ----------------------  WITH MANUAL INTERVENTION  --------------------  %
% with_manual_intervention = false;

% -------------------------  FINAL RESULTS   -------------------------  %
% save_demixed = true;    % save the demixed file or not
kt = 3;                 % frame intervals

%%
n_round = 2;
% list_seq = cell(n_round,1);
% for r = 1:n_round
%     seq = randperm(num_param_names);
%     if r==1
%         seq = [4     5     7     2     3     8     1     6];
%     else
%         seq = randperm(num_param_names);
%         seq = seq(seq~=5 & seq~=4);
%         seq = [4,5,seq];
%     end
%     list_seq{r} = seq;
% end
    
%%
for cv = 1:num_Exp
    trains = [1:cv-1, cv+1:num_Exp]; 
    %% Initialize variable parameters
    % init_ind_param = round(num_params/2);
    init_ind_param = [1, 3, 2, 1, 2, 6, 8, 4]'; % noisy
%     init_ind_param = [1, 3, 2, 3, 2, 6, 8, 4]'; % clean
    ind_param = init_ind_param;
    temp_param = cellfun(@(x,y) x(y), range_params,num2cell(ind_param));

    best_ind_param = init_ind_param;
    [best_Recall, best_Precision, best_F1, best_time, ...
        Recall, Precision, F1, used_time] = deal(zeros(num_Exp,num_thb));
    history = zeros(num_param_names+4,0);
    best_history = zeros(num_param_names+4,0);
    best_thb = list_th_binary(1);

    %%
    for r = 1:n_round
    %     seq = randperm(num_param_names);
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
                    ind_param(ind_vary) = test_ind;
                    temp_param = cellfun(@(x,y) x(y), range_params,num2cell(ind_param));
    %                 temp_param(ind_vary) = range_params{ind_vary}(test_ind);
                    disp(temp_param);
                    rbg = temp_param(1);
                    nk = temp_param(2);
                    rdmin = temp_param(3);
                    min_corr = temp_param(4);
                    min_pnr = temp_param(5);
                    merge_thr = temp_param(6);
                    mts = temp_param(7);
                    mtt = temp_param(8);

                    dir_sub = sprintf('gSiz=%d,rbg=%0.1f,nk=%d,rdmin=%0.1f,mc=%0.2f,mp=%d,mt=%0.2f,mts=%0.2f,mtt=%0.2f',...
                        gSiz,rbg,nk,rdmin,min_corr,min_pnr,merge_thr,mts,mtt);
                    if ~ exist(fullfile(dir_save,dir_sub),'dir')
                        mkdir(fullfile(dir_save,dir_sub));
                    end

                    %%
                    parfor eid = 1:num_Exp
                        [temp_Recall, temp_Precision, temp_F1, temp_used_time] = deal(zeros(1,num_thb));
                        Exp_ID = list_Exp_ID{eid};
                        nam = fullfile(path_name,[Exp_ID,'.h5']);
                        %% choose data
                        neuron = Sources2D();
                        nam = neuron.select_data(nam);  %if nam is [], then select data interactively

                        if with_dendrites
                            % determine the search locations by dilating the current neuron shapes
                            updateA_search_method = 'dilate';  
                            updateA_bSiz = 5;
                            updateA_dist = neuron.options.dist;
                        else
                            % determine the search locations by selecting a round area
                            updateA_search_method = 'ellipse'; %#ok<UNRCH>
                            updateA_dist = 5;
                            updateA_bSiz = neuron.options.dist;
                        end

                        if ~exist(fullfile(dir_save,dir_sub,[Exp_ID,'_result.mat']),'file')
                            %% variable parameters
            %                 gSiz = list_gSiz; % 7;           % pixel, neuron diameter
                            gSig = round(gSiz/4); % 2;           % pixel, gaussian width of a gaussian kernel for filtering the data. 0 means no filtering
                            min_pixel = gSig^2;      % minimum number of nonzero pixels for each neuron
                            ring_radius = round(rbg*gSiz); % 1.5* % when the ring model used, it is the radius of the ring used in the background model.
                            dmin = gSiz/rdmin; % gSiz/3;       % minimum distances between two neurons. it is used together with merge_thr
                            dmin_only = dmin/2;  % merge neurons if their distances are smaller than dmin_only.
                        %     merge_thr = 0.65; % 0.65;     % thresholds for merging neurons; [spatial overlap ratio, temporal correlation of calcium traces, spike correlation]
                            merge_thr_spatial = [mts, mtt, -inf];  % merge components with highly correlated spatial shapes (corr=0.8) and small temporal correlations (corr=0.1)
            %                 nk = list_nk; % 3;             % detrending the slow fluctuation. usually 1 is fine (no detrending)
                        %     min_corr = 0.8; % 0.8;     % minimum local correlation for a seeding pixel
                    %         min_pnr = 8; % 8;       % minimum peak-to-noise ratio for a seeding pixel

                            %% -------------------------    UPDATE ALL    -------------------------  %
                            neuron.updateParams('gSig', gSig, ...       % -------- spatial --------
                                'gSiz', gSiz, ...
                                'ring_radius', ring_radius, ...
                                'ssub', ssub, ...
                                'search_method', updateA_search_method, ...
                                'bSiz', updateA_bSiz, ...
                                'dist', updateA_bSiz, ...
                                'spatial_constraints', spatial_constraints, ...
                                'spatial_algorithm', spatial_algorithm, ...
                                'tsub', tsub, ...                       % -------- temporal --------
                                'deconv_options', deconv_options, ...
                                'nk', nk, ...
                                'detrend_method', detrend_method, ...
                                'background_model', bg_model, ...       % -------- background --------
                                'nb', nb, ...
                                'ring_radius', ring_radius, ...
                                'num_neighbors', num_neighbors, ...
                                'bg_ssub', bg_ssub, ...
                                'merge_thr', merge_thr, ...             % -------- merging ---------
                                'dmin', dmin, ...
                                'method_dist', method_dist, ...
                                'min_corr', min_corr, ...               % ----- initialization -----
                                'min_pnr', min_pnr, ...
                                'min_pixel', min_pixel, ...
                                'bd', bd, ...
                                'center_psf', center_psf);
                            neuron.Fs = Fs;
                            update_sn = true;

                            %% distribute data and be ready to run source extraction
                            process_time = cell(1,4);
                            process_time{1} = datetime;
                            neuron.getReady(pars_envs);
                            process_time{2} = datetime;

                            %% initialize neurons from the video data within a selected temporal range
    %                         if choose_params
    %                             % change parameters for optimized initialization
    %                             [gSig, gSiz, ring_radius, min_corr, min_pnr] = neuron.set_parameters();
    %                         end

                            K = [];
                            [center, Cn, PNR] = neuron.initComponents_parallel(K, frame_range, save_initialization, use_parallel, use_prev); % use_prev
                            neuron.compactSpatial();
                            if show_init
                                figure();
                                ax_init= axes();
                                imagesc(Cn, [0, 1]); colormap gray;
                                hold on;
                                plot(center(:, 2), center(:, 1), '.r', 'markersize', 10);
                            end

                            try
                                %% estimate the background components
                                neuron.update_background_parallel(use_parallel);
                                neuron_init = neuron.copy();
                            end
                            process_time{3} = datetime;

                            if ~isempty(neuron.A)
                                %%  merge neurons and update spatial/temporal components
                                neuron.merge_neurons_dist_corr(show_merge);
                                neuron.merge_high_corr(show_merge, merge_thr_spatial);

                                %% update spatial components

                                %% pick neurons from the residual
                                [center_res, Cn_res, PNR_res] =neuron.initComponents_residual_parallel([], save_initialization, use_parallel, min_corr_res, min_pnr_res, seed_method_res);
                                if show_init
                                    axes(ax_init);
                                    plot(center_res(:, 2), center_res(:, 1), '.g', 'markersize', 10);
                                end
                                neuron_init_res = neuron.copy();

                                try
                                    %% udpate spatial&temporal components, delete false positives and merge neurons
                                    % update spatial
                                    if update_sn
                                        neuron.update_spatial_parallel(use_parallel, true);
                                        udpate_sn = false;
                                    else
                                        neuron.update_spatial_parallel(use_parallel);
                                    end
                                    % merge neurons based on correlations 
                                    neuron.merge_high_corr(show_merge, merge_thr_spatial);

                                    for m=1:2
                                        % update temporal
                                        neuron.update_temporal_parallel(use_parallel);

                                        % delete bad neurons
                                        neuron.remove_false_positives();

                                        % merge neurons based on temporal correlation + distances 
                                        neuron.merge_neurons_dist_corr(show_merge);
                                    end

                                    %% add a manual intervention and run the whole procedure for a second time
                                    neuron.options.spatial_algorithm = 'nnls';
        %                             if with_manual_intervention
        %                                 show_merge = true;
        %                                 neuron.orderROIs('snr');   % order neurons in different ways {'snr', 'decay_time', 'mean', 'circularity'}
        %                                 neuron.viewNeurons([], neuron.C_raw);
        % 
        %                                 % merge closeby neurons
        %                                 neuron.merge_close_neighbors(true, dmin_only);
        % 
        %                                 % delete neurons
        %                                 tags = neuron.tag_neurons_parallel();  % find neurons with fewer nonzero pixels than min_pixel and silent calcium transients
        %                                 ids = find(tags>0); 
        %                                 if ~isempty(ids)
        %                                     neuron.viewNeurons(ids, neuron.C_raw);
        %                                 end
        %                             end
                                    %% run more iterations
                                    neuron.update_background_parallel(use_parallel);
                                    neuron.update_spatial_parallel(use_parallel);
                                    neuron.update_temporal_parallel(use_parallel);

                                    K = size(neuron.A,2);
                                    tags = neuron.tag_neurons_parallel();  % find neurons with fewer nonzero pixels than min_pixel and silent calcium transients
                                    neuron.remove_false_positives();
                                    neuron.merge_neurons_dist_corr(show_merge);
                                    neuron.merge_high_corr(show_merge, merge_thr_spatial);

                                    if K~=size(neuron.A,2)
                                        neuron.update_spatial_parallel(use_parallel);
                                        neuron.update_temporal_parallel(use_parallel);
                                        neuron.remove_false_positives();
                                    end
                                end
                            end
                            process_time{4} = datetime;

                            %% save the workspace for future analysis
                            neuron.orderROIs('snr');
    %                         cnmfe_path = neuron.save_workspace();
                            fprintf('------------- SAVE THE WHOLE WORKSPACE ----------\n\n');
                            neuron.compress_results();
                            cnmfe_path = fullfile(dir_save,dir_sub,[Exp_ID,'_result.mat']);
    %                         cnmfe_path = fullfile(neuron.P.log_folder,  [strrep(get_date(), ' ', '_'), '.mat']);
                            log_file = neuron.P.log_file; 
                            parsave1(fullfile(dir_save,dir_sub,[Exp_ID,'_time.mat']),process_time,'process_time');
                            parsave1(cnmfe_path, neuron, 'neuron'); 
                            fclose('all');
                            try
                                fp = fopen(log_file, 'a');
                                fprintf(fp, '\n--------%s--------\n[%s]\bSave the current workspace into file \n\t%s\n\n', get_date(), get_minute(), cnmfe_path);
                                fprintf('The current workspace has been saved into file \n\t%s\n\n', cnmfe_path);
                                fp.close();
                            end

                            %% evaluate spatial segmentation accuracy
                            A = neuron.A;
                            A3 = neuron.reshape(A, 2);
                            for tid = 1:num_thb
                                thb = list_th_binary(tid);
                                Masks3 = threshold_Masks(A3, thb); %;%
%                                 Ab = A>th_binary*max(A,[],1); %;% 0.5
%                                 Masks3 = neuron.reshape(Ab, 2); 
                            %     Masks3 = permute(Masks3,[2,1,3]);
                                [temp_Recall(tid), temp_Precision(tid), temp_F1(tid)] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
                                if isnan(temp_F1(tid))
                                    temp_F1(tid) = 0;
                                end
                                temp_used_time(tid) = seconds(process_time{4}-process_time{2});
                                parsave1(fullfile(dir_save,dir_sub,[Exp_ID,'_Masks_',num2str(thb),'.mat']),Masks3,'Masks3');
                            end
                            Recall(eid,:) = temp_Recall;
                            Precision(eid,:) = temp_Precision;
                            F1(eid,:) = temp_F1;
                            used_time(eid,:) = temp_used_time;
                            disp([temp_Recall', temp_Precision', temp_F1', temp_used_time'])

                        else
                    %         load(fullfile(dir_save,dir_sub,[Exp_ID,'_Masks.mat']),'Masks3');
                            saved_result = load(fullfile(dir_save,dir_sub,[Exp_ID,'_result.mat']),'neuron');
                            neuron = saved_result.neuron;
                            A = neuron.A;
                            A3 = neuron.reshape(A, 2);
                            for tid = 1:num_thb
                                thb = list_th_binary(tid);
                                Masks3 = threshold_Masks(A3, thb); %;%
%                                 Ab = A>thb*max(A,[],1); %;%
%                                 Masks3 = neuron.reshape(Ab, 2); 
                                [temp_Recall(tid), temp_Precision(tid), temp_F1(tid)] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
                                if isnan(temp_F1(tid))
                                    temp_F1(tid) = 0;
                                end
                                try
                                    saved_result = load(fullfile(dir_save,dir_sub,[Exp_ID,'_time.mat']),'process_time');
                                    temp_used_time(tid) = seconds(saved_result.process_time{4}-saved_result.process_time{2});
                                catch
                                    temp_used_time(tid) = nan;
                                end
                            end
                            Recall(eid,:) = temp_Recall;
                            Precision(eid,:) = temp_Precision;
                            F1(eid,:) = temp_F1;
                            used_time(eid,:) = temp_used_time;
                            disp([temp_Recall', temp_Precision', temp_F1', temp_used_time'])
                        end

                        %% show neuron contours
                        % Coor = neuron.show_contours(0.6);
                        %% save neurons shapes
                        % neuron.save_neurons();
                    end

                    %% evaluate the mean F1 using this parameter set
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
                    save(['eval_',data_name,'_thb history cv', num2str(cv),'.mat'],'list_seq',...
                        'best_Recall','best_Precision','best_F1','best_time','best_ind_param',...
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
    % Table_time = [best_Recall, best_Precision, best_F1, best_time];
    % Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
    %%
    movefile(['eval_',data_name,'_thb history cv', num2str(cv),'.mat'], ...
        ['eval_',data_name,'_thb history ',save_date,' cv', num2str(cv),'.mat']);
end

%% Generate a summary table
list_Table = cell(num_Exp,1);
for cv = 1:num_Exp
    trains = [1:cv-1, cv+1:num_Exp]; 
    load(['.\eval_',data_name,'_thb history ',save_date,' cv', num2str(cv),'.mat'],'list_seq',...
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
save(['eval_',data_name,'_thb ',save_date,' cv.mat'],'Table_ext');
