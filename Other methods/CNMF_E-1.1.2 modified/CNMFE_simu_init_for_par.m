%% clear the workspace and select data
warning off;
gcp;
addpath(genpath('.'))
addpath(genpath('../ANE'))
clear; clc; close all;  


%% 
scale_lowBG = 5e3;
scale_noise = 1;
data_name = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
patch_dims = [253,316]; 
num_Exp = 10;

rate_hz = 10; % frame rate of each video
radius = 6;
path_name = fullfile('../data/data_simulation',data_name);
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);
dir_GT = fullfile(path_name,'GT Masks'); % FinalMasks_

dir_save = fullfile(path_name,'CNMFE');
if ~ exist(dir_save,'dir')
    mkdir(dir_save);
end

%% Set range of parameters to optimize over
gSiz = 12;
list_params.rbg = [1.5, 1.8, 2]; % 1.5; % 
list_params.nk = [1, 2, 3]; % 3; % 
list_params.rdmin = [2, 2.5, 3, 4]; % 3; % 
list_params.min_corr = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98];
list_params.min_pnr = 4:4:24;
list_params.merge_thr = 0.1:0.1:0.9;
list_params.merge_thr_spatial = 0.1:0.1:0.9;
list_params.merge_thr_temporal = 0.1:0.1:0.9;
list_th_binary = [0.2, 0.3, 0.4, 0.5];
name_params = fieldnames(list_params); 
range_params = struct2cell(list_params); 
num_param_names = length(range_params);
num_params = cellfun(@length, range_params);
num_thb = length(list_th_binary);
n_round = 1; % 3;
list_seq = cell(n_round,1);

%% Other fixed parameters. 
% -------------------------    COMPUTATION    -------------------------  %
pars_envs = struct('memory_size_to_use', 120, ...   % GB, memory space you allow to use in MATLAB
    'memory_size_per_patch', 10, ...   % GB, space for loading data within one patch
    'patch_dims', patch_dims);  %GB, patch size

% -------------------------      SPATIAL      -------------------------  %
ssub = 1;           % spatial downsampling factor
with_dendrites = true;   % with dendrites or not
spatial_constraints = struct('connected', true, 'circular', false);  % you can include following constraints: 'circular'
spatial_algorithm = 'hals_thresh';

% -------------------------      TEMPORAL     -------------------------  %
Fs = rate_hz;             % frame rate
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

%% Initialize variable parameters
% init_ind_param = round(num_params/2);
    init_ind_param = [1, 3, 2, 3, 2, 6, 8, 4]'; % bv
%     init_ind_param = [1, 3, 2, 7, 4, 6, 8, 4]'; % PCF
ind_param = init_ind_param;
temp_param = cellfun(@(x,y) x(y), range_params,num2cell(ind_param));

best_ind_param = init_ind_param;
[best_Recall, best_Precision, best_F1, best_time, ...
    Recall, Precision, F1, used_time] = deal(zeros(num_Exp,num_thb));
history = zeros(num_param_names+4,0);
best_history = zeros(num_param_names+4,0);

%%
for r = 1%:n_round
    seq = randperm(num_param_names);
%     if r==1
%         seq =[6     3     7     8     5     1     2     4];
%     else
%         seq = randperm(num_param_names);
%     end
    list_seq{r} = seq;
    for p = 1%:num_param_names
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
        
        for direction = 1%:length(list_test_ind)
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
%                 if ~ exist(fullfile(dir_save,dir_sub),'dir')
%                     mkdir(fullfile(dir_save,dir_sub));
%                 end

                %%
                for eid = 1:num_Exp
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
                        neuron.getReady(pars_envs);
                    end
                end
                break
            end
        end
    end
end
