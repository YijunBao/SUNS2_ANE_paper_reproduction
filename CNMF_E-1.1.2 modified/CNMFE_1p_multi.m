%% clear the workspace and select data
warning off;
gcp;
addpath(genpath('C:\Other methods\CNMF_E-1.1.2'));
% addpath(genpath('.\'));
clear; clc; close all;  

%% 
path_name = 'E:\OnePhoton videos\full videos\';
list_Exp_ID = {'c25_1NRE','c27_NN','c28_1NRE1NLE'};
num_Exp = length(list_Exp_ID);
dir_GT = [path_name,'GT Masks\']; % FinalMasks_

dir_save = [path_name,'CNMFE\'];
if ~ exist(dir_save,'dir')
    mkdir(dir_save);
end

% list_nk = 3; % [1,3]; % 
% list_min_corr = 0.8; % [0.8, 0.9, 0.95]; % 
% list_min_pnr = 8; % [4,8,12]; % 
% list_merge_thr = 0.6; %[0.1, 0.2, 0.4]; % 
% list_merge_thr_spatial = 0.8; %[0.1, 0.2, 0.4]; % 
% list_merge_thr_temporal = 0.4; % [0.1, 0.2, 0.4]; % 

list_nk = 3; % [1,3]; % 
list_gSiz = 7; %[6,7,8,9]; % 
list_min_corr = 0.9; % [0.8, 0.9, 0.95]; % 
list_min_pnr = 8; % [4,8,12]; % 
list_rdmin = 3; % [2,3,4]; % 
list_merge_thr = 0.6; % [0.4, 0.6, 0.8]; % 
list_merge_thr_spatial = 0.2; %[0.1, 0.2, 0.4]; % 
list_merge_thr_temporal = 0.2; % [0.1, 0.2, 0.4]; % 

% list_nk = 3; % [1,3];
% list_min_corr = [0.6, 0.8, 0.9]; % 0.9
% list_min_pnr = 8; % [4,8,12]; % [6, 8, 10];
% list_merge_thr = [0.4, 0.6, 0.8]; % 0.4
% list_merge_thr_spatial = [0.6, 0.8, 0.9]; % 0.6 
% list_merge_thr_temporal = [0.2, 0.4, 0.6]; % 0.2

num_gSiz = length(list_gSiz);
num_rdmin = length(list_rdmin);
num_nk = length(list_nk);
num_mc = length(list_min_corr);
num_mp = length(list_min_pnr);
num_mt = length(list_merge_thr);
num_mts = length(list_merge_thr_spatial);
num_mtt = length(list_merge_thr_temporal);

% list_Recall = zeros(num_Exp, num_nk, num_mc, num_mp, num_mt, num_mts, num_mtt);
% list_Precision = zeros(num_Exp, num_nk, num_mc, num_mp, num_mt, num_mts, num_mtt);
% list_F1 = zeros(num_Exp, num_nk, num_mc, num_mp, num_mt, num_mts, num_mtt);
% list_time = zeros(num_Exp, num_nk, num_mc, num_mp, num_mt, num_mts, num_mtt);
list_Recall = zeros(num_Exp, num_rdmin, num_mc, num_mp, num_mt, num_mts, num_mtt);
list_Precision = zeros(num_Exp, num_rdmin, num_mc, num_mp, num_mt, num_mts, num_mtt);
list_F1 = zeros(num_Exp, num_rdmin, num_mc, num_mp, num_mt, num_mts, num_mtt);
list_time = zeros(num_Exp, num_rdmin, num_mc, num_mp, num_mt, num_mts, num_mtt);

%%
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    nam = [path_name,Exp_ID,'.h5'];
    %% choose data
    neuron = Sources2D();
    % nam = get_fullname('./data_1p.tif');          % this demo data is very small, here we just use it as an example
%     nam = get_fullname('E:\1photon-small\c25_59_228.h5');
    nam = neuron.select_data(nam);  %if nam is [], then select data interactively

%     for nk_id = 1:num_nk
%         nk = list_nk(nk_id);
%     for nk_id = 1:num_gSiz
%         gSiz = list_gSiz(nk_id);
    for nk_id = 1:num_rdmin
        rdmin = list_rdmin(nk_id);
    for mc_id = 1:num_mc
        min_corr = list_min_corr(mc_id);
    for mp_id = 1:num_mp
        min_pnr = list_min_pnr(mp_id);
    for mt_id = 1:num_mt
        merge_thr = list_merge_thr(mt_id);
    for mts_id = 1:num_mts
        mts = list_merge_thr_spatial(mts_id);
    for mtt_id = 1:num_mtt
        mtt = list_merge_thr_temporal(mtt_id);
%     dir_sub = sprintf('nk=%0.2f,mc=%0.2f,mp=%0.2f,mt=%0.2f,mts=%0.2f,mtt=%0.2f',...
%         nk,min_corr,min_pnr,merge_thr,mts,mtt);
    dir_sub = sprintf('rdmin=%0.2f,mc=%0.2f,mp=%0.2f,mt=%0.2f,mts=%0.2f,mtt=%0.2f',...
        rdmin,min_corr,min_pnr,merge_thr,mts,mtt);
    if ~ exist([dir_save,dir_sub],'dir')
        mkdir([dir_save,dir_sub]);
    end
    
    if ~exist([dir_save,dir_sub,'\',Exp_ID,'_Masks.mat'],'file')
        %% parameters
        % -------------------------    COMPUTATION    -------------------------  %
        pars_envs = struct('memory_size_to_use', 120, ...   % GB, memory space you allow to use in MATLAB
            'memory_size_per_patch', 10, ...   % GB, space for loading data within one patch
            'patch_dims', [64, 64]);  %GB, patch size

        % -------------------------      SPATIAL      -------------------------  %
        gSiz = 7;           % pixel, neuron diameter
        gSig = round(gSiz/4); % 2;           % pixel, gaussian width of a gaussian kernel for filtering the data. 0 means no filtering
        ssub = 1;           % spatial downsampling factor
        with_dendrites = true;   % with dendrites or not
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
        spatial_constraints = struct('connected', true, 'circular', false);  % you can include following constraints: 'circular'
        spatial_algorithm = 'hals_thresh';

        % -------------------------      TEMPORAL     -------------------------  %
        Fs = 20;             % frame rate
        tsub = 1;           % temporal downsampling factor
        deconv_options = struct('type', 'ar1', ... % model of the calcium traces. {'ar1', 'ar2'}
            'method', 'foopsi', ... % method for running deconvolution {'foopsi', 'constrained', 'thresholded'}
            'smin', -5, ...         % minimum spike size. When the value is negative, the actual threshold is abs(smin)*noise level
            'optimize_pars', true, ...  % optimize AR coefficients
            'optimize_b', true, ...% optimize the baseline);
            'max_tau', 100);    % maximum decay time (unit: frame);

        nk = 3;             % detrending the slow fluctuation. usually 1 is fine (no detrending)
        % when changed, try some integers smaller than total_frame/(Fs*30)
        detrend_method = 'spline';  % compute the local minimum as an estimation of trend.

        % -------------------------     BACKGROUND    -------------------------  %
        bg_model = 'ring';  % model of the background {'ring', 'svd'(default), 'nmf'}
        nb = 1;             % number of background sources for each patch (only be used in SVD and NMF model)
        ring_radius = round(1.5*gSiz);  % when the ring model used, it is the radius of the ring used in the background model.
        %otherwise, it's just the width of the overlapping area
        num_neighbors = []; % number of neighbors for each neuron
        bg_ssub = 2;        % downsample background for a faster speed 

        % -------------------------      MERGING      -------------------------  %
        show_merge = false;  % if true, manually verify the merging step
    %     merge_thr = 0.65; % 0.65;     % thresholds for merging neurons; [spatial overlap ratio, temporal correlation of calcium traces, spike correlation]
        method_dist = 'max';   % method for computing neuron distances {'mean', 'max'}
        dmin = gSiz/rdmin; % gSiz/3;       % minimum distances between two neurons. it is used together with merge_thr
        dmin_only = dmin/2;  % merge neurons if their distances are smaller than dmin_only.
    %     merge_thr_spatial = [0.8, 0.4, -inf];  % merge components with highly correlated spatial shapes (corr=0.8) and small temporal correlations (corr=0.1)
        merge_thr_spatial = [mts, mtt, -inf];  % merge components with highly correlated spatial shapes (corr=0.8) and small temporal correlations (corr=0.1)

        % -------------------------  INITIALIZATION   -------------------------  %
        K = [];             % maximum number of neurons per patch. when K=[], take as many as possible.
    %     min_corr = 0.8; % 0.8;     % minimum local correlation for a seeding pixel
%         min_pnr = 8; % 8;       % minimum peak-to-noise ratio for a seeding pixel
        min_pixel = gSig^2;      % minimum number of nonzero pixels for each neuron
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
%         update_sn = true;

        % ----------------------  WITH MANUAL INTERVENTION  --------------------  %
        with_manual_intervention = false;

        % -------------------------  FINAL RESULTS   -------------------------  %
        save_demixed = true;    % save the demixed file or not
        kt = 3;                 % frame intervals

        % -------------------------    UPDATE ALL    -------------------------  %
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
        process_time{1} = datetime;
        neuron.getReady(pars_envs);
        process_time{2} = datetime;

        %% initialize neurons from the video data within a selected temporal range
        if choose_params
            % change parameters for optimized initialization
            [gSig, gSiz, ring_radius, min_corr, min_pnr] = neuron.set_parameters();
        end

        [center, Cn, PNR] = neuron.initComponents_parallel(K, frame_range, save_initialization, use_parallel, use_prev); % use_prev
        neuron.compactSpatial();
        if show_init
            figure();
            ax_init= axes();
            imagesc(Cn, [0, 1]); colormap gray;
            hold on;
            plot(center(:, 2), center(:, 1), '.r', 'markersize', 10);
        end

        %% estimate the background components
        neuron.update_background_parallel(use_parallel);
        neuron_init = neuron.copy();
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
            if with_manual_intervention
                show_merge = true;
                neuron.orderROIs('snr');   % order neurons in different ways {'snr', 'decay_time', 'mean', 'circularity'}
                neuron.viewNeurons([], neuron.C_raw);

                % merge closeby neurons
                neuron.merge_close_neighbors(true, dmin_only);

                % delete neurons
                tags = neuron.tag_neurons_parallel();  % find neurons with fewer nonzero pixels than min_pixel and silent calcium transients
                ids = find(tags>0); 
                if ~isempty(ids)
                    neuron.viewNeurons(ids, neuron.C_raw);
                end
            end
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
        process_time{4} = datetime;

        %% save the workspace for future analysis
        neuron.orderROIs('snr');
        cnmfe_path = neuron.save_workspace();

        %% evaluate spatial segmentation accuracy
        A = neuron.A;
        Ab = A>0.5*max(A,[],1); %;%
        Masks3 = neuron.reshape(Ab, 2); 
    %     Masks3 = permute(Masks3,[2,1,3]);
        [Recall, Precision, F1] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
        used_time = seconds(process_time{4}-process_time{2});
        list_Recall(eid,nk_id,mc_id,mp_id,mt_id,mts_id,mtt_id) = Recall;
        list_Precision(eid,nk_id,mc_id,mp_id,mt_id,mts_id,mtt_id) = Precision;
        list_F1(eid,nk_id,mc_id,mp_id,mt_id,mts_id,mtt_id) = F1;
        list_time(eid,nk_id,mc_id,mp_id,mt_id,mts_id,mtt_id) = used_time;
        disp([Recall, Precision, F1, used_time])

        %% move the final results
%         t=split(string(datetime),'-');
%         child = [path_name,Exp_ID,'_source_extraction\'];
%         saved_files = dir([child,'\frames*\LOGS*\*',t{2},'*.mat']);
%         datenum = [saved_files.datenum];
%         [val,ind] = max(datenum);
%         saved_file = saved_files(ind);
%         movefile([saved_file.folder,'\',saved_file.name], [dir_save,dir_sub,'\',Exp_ID,'_result.mat']);
        movefile(cnmfe_path, fullfile(dir_save,dir_sub,[Exp_ID,'_result.mat']));
        save([dir_save,dir_sub,'\',Exp_ID,'_Masks.mat'],'Masks3');
        save([dir_save,dir_sub,'\',Exp_ID,'_time.mat'],'process_time');

    else
        load([dir_save,dir_sub,'\',Exp_ID,'_Masks.mat'],'Masks3');
        load([dir_save,dir_sub,'\',Exp_ID,'_time.mat'],'process_time');
        [Recall, Precision, F1] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
        used_time = seconds(process_time{4}-process_time{2});
        list_Recall(eid,nk_id,mc_id,mp_id,mt_id,mts_id,mtt_id) = Recall;
        list_Precision(eid,nk_id,mc_id,mp_id,mt_id,mts_id,mtt_id) = Precision;
        list_F1(eid,nk_id,mc_id,mp_id,mt_id,mts_id,mtt_id) = F1;
        list_time(eid,nk_id,mc_id,mp_id,mt_id,mts_id,mtt_id) = used_time;
        disp([Recall, Precision, F1, used_time])
    end
    %% show neuron contours
    % Coor = neuron.show_contours(0.6);

    %% create a video for displaying the
    % amp_ac = 140;
    % range_ac = 5+[0, amp_ac];
    % multi_factor = 10;
    % range_Y = 1300+[0, amp_ac*multi_factor];
    % 
    % avi_filename = neuron.show_demixed_video(save_demixed, kt, [], amp_ac, range_ac, range_Y, multi_factor);

    %% save neurons shapes
    % neuron.save_neurons();
    save('eval_1p_small.mat','list_Recall','list_Precision','list_F1','list_time','list_gSiz','list_rdmin',...
        'list_nk','list_merge_thr','list_merge_thr_spatial','list_merge_thr_temporal','list_min_corr','list_min_pnr')
    end
    end
    end
    end
    end
    end
end

% save('eval_1p_small.mat','list_Recall','list_Precision','list_F1','list_time')
Table_time = [list_Recall, list_Precision, list_F1, list_time];
Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
