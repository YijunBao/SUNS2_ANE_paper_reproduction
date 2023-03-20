function [file_name_to_save, filename_raw, filename_reg] = min1pipe_h5_vary(...
    path_name, filename, Fsi, Fsi_new, spatialr, se, ismc, flag,...
    pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres)
% main_processing
%   need to decide whether to use parallel computing
%   Fsi: raw sampling rate
%   Fsi_new: in use sampling rate
%   spatialr: spatial downsampling factor
%   Jinghao Lu 06/10/2016

    %% configure paths %%
%     min1pipe_init;
    
    %% initialize parameters %%
    if nargin < 3 || isempty(Fsi)
        defpar = default_parameters;
        Fsi = defpar.Fsi;
    end
    
    if nargin < 4 || isempty(Fsi_new)
        defpar = default_parameters;
        Fsi_new = defpar.Fsi_new;
    end
    
    if nargin < 5 || isempty(spatialr)
        defpar = default_parameters;
        spatialr = defpar.spatialr;
    end
    
    if nargin < 6 || isempty(se)
        defpar = default_parameters;
        se = defpar.neuron_size;
    end
    
    if nargin < 7 || isempty(ismc)
        ismc = true;
    end
    
    if nargin < 8 || isempty(flag)
        flag = 1;
    end
    
    if nargin < 9 || isempty(pix_select_sigthres)
        pix_select_sigthres = 0.8;
    end
    
    if nargin < 10 || isempty(pix_select_corrthres)
        pix_select_corrthres = 0.6;
    end
    
    if nargin < 11 || isempty(merge_roi_corrthres)
        merge_roi_corrthres = 0.9;
    end
    
    folder = sprintf('min1pipe\\pss=%0.2f_psc=%0.2f_mrc=%0.2f\\',...
        pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres);
    if ~exist(fullfile(path_name,folder),'dir')
        mkdir(fullfile(path_name,folder))
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%% parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% user defined parameters %%%                                     %%%
    Params.Fsi = Fsi;                                                   %%%
    Params.Fsi_new = Fsi_new;                                           %%%
    Params.spatialr = spatialr;                                         %%%
    Params.neuron_size = se; %%% half neuron size; 9 for Inscopix and 5 %%%
                            %%% for UCLA, with 0.5 spatialr separately  %%%
                                                                        %%%
    %%% fixed parameters (change not recommanded) %%%                   %%%
    Params.anidenoise_iter = 4;                   %%% denoise iteration %%%
    Params.anidenoise_dt = 1/7;                   %%% denoise step size %%%
    Params.anidenoise_kappa = 0.5;       %%% denoise gradient threshold %%%
    Params.anidenoise_opt = 1;                %%% denoise kernel choice %%%
    Params.anidenoise_ispara = 1;             %%% if parallel (denoise) %%%   
    Params.bg_remove_ispara = 1;    %%% if parallel (backgrond removal) %%%
    Params.mc_scl = 0.004;      %%% movement correction threshold scale %%%
    Params.mc_sigma_x = 5;  %%% movement correction spatial uncertainty %%%
    Params.mc_sigma_f = 10;    %%% movement correction fluid reg weight %%%
    Params.mc_sigma_d = 1; %%% movement correction diffusion reg weight %%%
    Params.pix_select_sigthres = pix_select_sigthres; % 0.8;     %%% seeds select signal level %%%
    Params.pix_select_corrthres = pix_select_corrthres; % 0.6; %%% merge correlation threshold1 %%%
    Params.refine_roi_ispara = 1;          %%% if parallel (refine roi) %%%
    Params.merge_roi_corrthres = merge_roi_corrthres; % 0.9;  %%% merge correlation threshold2 %%% 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% get dataset info %%
%     [path_name, file_base, file_fmt] = data_info;
    filename_split = split(filename,'.');
    file_base = filename_split(1);
    file_fmt = filename_split(2);
    process_time = cell(3,length(file_base));
    
    hpipe = tic;
    for i = 1: length(file_base)
        
        %%% judge whether do the processing %%%
        filecur = fullfile(path_name, folder, [file_base{i}, '_data_processed.mat']);
        msg = 'Redo the analysis? (y/n)';
        overwrite_flag = true; % judge_file(filecur, msg);
        
        if overwrite_flag
            %% data cat %%
            %%% --------- 1st section ---------- %%%
            Fsi = Params.Fsi;
            Fsi_new = Params.Fsi_new;
            spatialr = Params.spatialr;
            [m, filename_raw, imaxn, imeanf, pixh, pixw, nf, imx1, imn1] = data_h5(path_name, file_base{i}, file_fmt{i}, folder, Fsi, Fsi_new, spatialr);
            process_time{1,i} = datetime;
            
            %% neural enhancing batch version %%
            %%% --------- 2nd section ---------- %%%
            filename_reg = fullfile(path_name, folder, [file_base{i}, '_reg.mat']);
            [m, imaxy1, overwrite_flag, imx2, imn2, ibmax, ibmin] = neural_enhance(m, filename_reg, Params);
            
            %% neural enhancing postprocess %%
            if overwrite_flag
                nflag = 1;
                m = noise_suppress(m, imaxy1, Fsi_new, nflag);
            end
            
            %% movement correction %%
            if overwrite_flag
                if ismc
                    pixs = min(pixh, pixw);
                    Params.mc_pixs = pixs;
                    Fsi_new = Params.Fsi_new;
                    scl = Params.neuron_size / (7 * pixs);
                    sigma_x = Params.mc_sigma_x;
                    sigma_f = Params.mc_sigma_f;
                    sigma_d = Params.mc_sigma_d;
                    se = Params.neuron_size;
                    [m, corr_score, raw_score, scl, imaxy] = frame_reg(m, imaxy1, se, Fsi_new, pixs, scl, sigma_x, sigma_f, sigma_d);
                    Params.mc_scl = scl; %%% update latest scl %%%
                    
%                     file_name_to_save = [path_name, file_base{i}, '_data_processed.mat'];
%                     if exist(file_name_to_save, 'file')
%                         delete(file_name_to_save)
%                     end
                    save(m.Properties.Source, 'corr_score', 'raw_score', '-v7.3', '-append');
                else
                    %%% spatiotemporal stabilization %%%
                    m = frame_stab(m);
                    imaxy = max(m.reg,[],3);
                end
            end
            
            %% movement correction postprocess %%
            %%% --------- 3rd section ---------- %%%
            nflag = 2;
            filename_reg_post = fullfile(path_name, folder, [file_base{i}, '_reg_post.mat']);
            m = noise_suppress(m, imaxy, Fsi_new, nflag, filename_reg_post);
            
            %% get rough roi domain %%
            mask = dominant_patch(imaxy);
            
            %% parameter init %%
            [P, options] = par_init(m);
            
            %% select pixel %%
            [sigrf, roirf, seedsupdt, bgrf, bgfrf, datasmthf1, cutofff1, pkcutofff1] = iter_seeds_select(m, mask, Params, P, options, flag);
            
            %% merge roi %%
            corrthres = Params.merge_roi_corrthres;
            [roimrg, sigmrg, seedsmrg, datasmthf2, cutofff2, pkcutofff2] = merge_roi(m, roirf, sigrf, seedsupdt, imaxy, datasmthf1, cutofff1, pkcutofff1, corrthres);
    
%             %% 2nd step clean seeds %%
%             sz = Params.neuron_size;
%             [roic, sigc, seedsc, datasmthc, cutoffc, pkcutoffc] = final_seeds_select(m, roimrg, sigmrg, seedsmrg, datasmthf2, cutofff2, pkcutofff2, sz, imax);
            
            %% refine roi again %%
            noise = P.sn;
            Puse.p = 0;
            Puse.options = options;
            Puse.noise = noise;
            ispara = Params.refine_roi_ispara;
            [roifn1, sigfn1, seedsfn1, datasmthfn1, cutofffn1, pkcutofffn1] = refine_roi(m, sigmrg, bgfrf, roimrg, seedsmrg, Puse.noise, datasmthf2, cutofff2, pkcutofff2, ispara);
            [bgfn, bgffn] = bg_update(m, roifn1, sigfn1);
                         
            %% refine sig again %%
            Puse.p = 2; %%% 2nd ar model used %%%
            Puse.options.p = 2;
            Puse.options.temporal_iter = 1;
            [sigfn1, bgffn, roifn1, seedsfn1, datasmthfn1, cutofffn1, pkcutofffn1] = refine_sig(m, roifn1, bgfn, sigfn1, bgffn, seedsfn1, datasmthfn1, cutofffn1, pkcutofffn1, Puse.p, Puse.options);
                        
            %% final clean seeds %%
            sz = Params.neuron_size;
            [roifn, sigfn, seedsfn, datasmthfn, cutofffn, pkcutofffn] = final_seeds_select(m, roifn1, sigfn1, seedsfn1, datasmthfn1, cutofffn1, pkcutofffn1, sz, imaxy);
            process_time{2,i} = datetime;

            %% final trace clean %%
            tflag = 2;
            sigfn = trace_clean(sigfn, Fsi_new, tflag);
                        
            %% final refine sig %%
            [sigfn, spkfn] = pure_refine_sig(sigfn, Puse.options);
            
            %% final clean outputs %%
            sigfn = max(roifn, [], 1)' .* sigfn;
            roifn = roifn ./ max(roifn, [], 1);
%             dff = compute_dff(sigfn, bgfn, bgffn, seedsfn);

            %%% estimate df/f %%%
            imcur = imaxy1;
            imref = imaxy;
            [img, sx, sy] = logdemons_unit(imref, imcur);
            ibuse = ibmax - ibmin;
            for ii = 1: length(sx)
                ibuse = iminterpolate(ibuse, sx{ii}, sy{ii});
            end
            
            x = (imx1 - imn1) * (imx2 - imn2) + imn1;
            roifnt = roifn;
            roifnt = roifnt ./ sum(roifnt, 1);
            bguse1 = ibuse(:)' * roifnt;
            bguse2 = min(sigfn, [], 2) * x;
            bguse = bguse1(:) * (imx1 - imn1) + bguse2(:);
            dff = double(full((sigfn - min(sigfn, [], 2)) * x ./ bguse));
            process_time{3,i} = datetime;
            
            %% save data %%
            stype = parse_type(class(m.reg(1, 1, 1)));
            nsize = pixh * pixw * nf * stype; %%% size of single %%%
            nbatch = batch_compute(nsize);
            ebatch = ceil(nf / nbatch);
            idbatch = [1: ebatch: nf, nf + 1];
            nbatch = length(idbatch) - 1;
            imax = zeros(pixh, pixw);
            for j = 1: nbatch
                tmp = m.reg(1: pixh, 1: pixw, idbatch(j): idbatch(j + 1) - 1);
                imax = max(cat(3, max(tmp, [], 3), imax), [], 3);
            end
            
            file_name_to_save = fullfile(path_name, folder, [file_base{i}, '_data_processed.mat']);
            if exist(file_name_to_save, 'file')
%                 if ismc
%                     load(file_name_to_save, 'raw_score', 'corr_score')
%                 end
                delete(file_name_to_save)
            end
            
            if ismc
                try
                    load(m.Properties.Source, 'raw_score', 'corr_score')
                    save(file_name_to_save, 'roifn', 'sigfn', 'dff', 'seedsfn', 'spkfn', 'bgfn', 'bgffn', 'imax', 'pixh', 'pixw', 'corr_score', 'raw_score', 'Params', '-v7.3');
                catch
                    save(file_name_to_save, 'roifn', 'sigfn', 'dff', 'seedsfn', 'spkfn', 'bgfn', 'bgffn', 'imax', 'pixh', 'pixw', 'Params', '-v7.3');
                end
            else
                save(file_name_to_save, 'roifn', 'sigfn', 'dff', 'seedsfn', 'spkfn', 'bgfn', 'bgffn', 'imax', 'pixh', 'pixw', 'Params', '-v7.3');
            end
            
            save(file_name_to_save, 'imaxn', 'imaxy', 'imeanf', 'process_time', '-append');
            time1 = toc(hpipe);
            disp(['Done all, total time: ', num2str(time1), ' seconds'])
        else
            filename_raw = fullfile(path_name, folder, [file_base{i}, '_frame_all.mat']);
            filename_reg = fullfile(path_name, folder, [file_base{i}, '_reg.mat']);
            file_name_to_save = filecur;
            
            time1 = toc(hpipe);
            disp(['Done all, total time: ', num2str(time1), ' seconds'])
        end
    end
end
