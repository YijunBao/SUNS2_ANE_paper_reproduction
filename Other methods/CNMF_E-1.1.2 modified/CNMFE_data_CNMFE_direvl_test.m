%% clear the workspace and select data
warning off;
gcp;
addpath(genpath('.'))
addpath(genpath('../ANE'))
clear; clc; close all;  


%% 
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_patch_dims = [120,120; 80,80; 88,88; 192,240]; 
rate_hz = [10,15,15,5]; % frame rate of each video
radius = [5,6,8,14];
sub_added = '';
% sub_added = '_original_masks';
% sub_added = '_added_blockwise_weighted_sum_unmask';

data_ind = 4;
data_name = list_data_names{data_ind};
path_name = fullfile('../data/data_CNMFE',data_name);
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
num_Exp = length(list_Exp_ID);
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

dir_save = fullfile(path_name,'CNMFE');
if ~ exist(dir_save,'dir')
    mkdir(dir_save);
end

switch data_ind 
    case 1
        save_date = '20221216';
    case 2
        save_date = '20221215';
    case 3
        save_date = '20221218';
    case 4
        save_date = '20221215';
end

%% Set range of parameters to optimize over
gSiz = 2 * radius(data_ind); % 12;
% list_params.rbg = [1.5, 1.8, 2]; % 1.5; % 
% list_params.nk = [1, 2, 3]; % 3; % 
% list_params.rdmin = [2, 2.5, 3, 4]; % 3; % 
% list_params.min_corr = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98];
% list_params.min_pnr = 4:4:24;
% list_params.merge_thr = 0.1:0.1:0.9;
% list_params.merge_thr_spatial = 0.1:0.1:0.9;
% list_params.merge_thr_temporal = 0.1:0.1:0.9;
% list_th_binary = [0.2, 0.3, 0.4, 0.5];
% name_params = fieldnames(list_params); 
% range_params = struct2cell(list_params); 
% num_param_names = length(range_params);
% num_params = cellfun(@length, range_params);
% num_thb = length(list_th_binary);
% n_round = 2;
% list_seq = cell(n_round,1);

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
[Recall, Precision, F1, used_time] = deal(zeros(num_Exp,1));
Table_time = cell(num_Exp,1);

for cv = 1:num_Exp
    load(fullfile(dir_save,['eval_',data_name,'_thb history ',save_date,'.mat']),...
        'best_Recall','best_Precision','best_F1',...
        'best_time','best_ind_param','best_thb','ind_param','list_params','history','best_history') % ' cv', num2str(cv),
    end_history = best_history(end,:);
    best_param = end_history(1:end-5);
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
    if ~ exist(fullfile(dir_save,dir_sub),'dir')
        mkdir(fullfile(dir_save,dir_sub));
    end

    %%
    for eid = cv % 1:num_Exp
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

        if true % ~exist(fullfile(dir_save,dir_sub,[Exp_ID,'_result.mat']),'file')
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
            cnmfe_path = neuron.save_workspace();

            %% evaluate spatial segmentation accuracy
            A = neuron.A;
            A3 = neuron.reshape(A, 2);
            thb = end_history(end-4);
            Masks3 = threshold_Masks(A3, thb); %;%
            % Ab = A>thb*max(A,[],1); %;% 0.5
            % Masks3 = neuron.reshape(Ab, 2); 
        %     Masks3 = permute(Masks3,[2,1,3]);
            [Recall(cv), Precision(cv), F1(cv)] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
            used_time(cv) = seconds(process_time{4}-process_time{2});
            save(fullfile(dir_save,dir_sub,[Exp_ID,'_Masks_',num2str(thb),'.mat']),'Masks3');

            %% move the final results
            movefile(cnmfe_path, fullfile(dir_save,dir_sub,[Exp_ID,'_result.mat']));
            save(fullfile(dir_save,dir_sub,[Exp_ID,'_time.mat']),'process_time');
            fclose('all');

        else
    %         load(fullfile(dir_save,dir_sub,[Exp_ID,'_Masks.mat']),'Masks3');
            load(fullfile(dir_save,dir_sub,[Exp_ID,'_result.mat']),'neuron');
            load(fullfile(dir_save,dir_sub,[Exp_ID,'_time.mat']),'process_time');
            A = neuron.A;
            A3 = neuron.reshape(A, 2);
            thb = end_history(end-4);
            Masks3 = threshold_Masks(A3, thb); %;%
            % Ab = A>thb*max(A,[],1); %;% 0.5
            % Masks3 = neuron.reshape(Ab, 2); 
        %     Masks3 = permute(Masks3,[2,1,3]);
            [Recall(cv), Precision(cv), F1(cv)] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
            used_time(cv) = seconds(process_time{4}-process_time{2});
        end
        Table_time{cv} = [end_history(1:end-4),Recall(cv), Precision(cv), F1(cv),used_time(cv),end_history(end-3:end)];
    end
end
%%
Table_time = cell2mat(Table_time);
Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
save(fullfile(dir_save,['eval_',data_name,'_thb ',save_date,' test.mat']),'Table_time_ext');
