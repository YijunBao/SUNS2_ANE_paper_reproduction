function [m, filename, imaxf, imeanf, pixh, pixw, nf, imx1, imn1] = data_h5(path_name, file_base, file_fmt, folder, Fsi, Fsi_new, ratio, ttype)
% Concatinate data pieces from raw 4GB-tif chunks,
%   new tiff files,
%   or avi files from UCLA miniscope
%   parallel or serial version supported
%   Jinghao Lu 01/16/2016

    hcat = tic;
    %% initialization %%
    %%% initialize parameters %%%
    if nargin < 4 || isempty(folder)
        folder = 'min1pipe\';
    end

    if nargin < 5 || isempty(Fsi)
        defpar = default_parameters;
        Fsi = defpar.Fsi;
    end

    if nargin < 6 || isempty(Fsi_new)
        defpar = default_parameters;
        Fsi_new = defpar.Fsi_new;
    end

    if nargin < 7 || isempty(ratio)
        defpar = default_parameters;
        ratio = defpar.spatialr;
    end
    
    if nargin < 8 || isempty(ttype)
        defpar = default_parameters;
        ttype = defpar.ttype;
    end
        
    if ~contains(file_fmt, 'mat')
        video_name = [file_base,'.',file_fmt];
        filename = fullfile(path_name, folder, [file_base, '_frame_all.mat']);
        msg = 'Overwrite raw .mat file (data)? (y/n)';
        overwrite_flag = true; % judge_file(filename, msg);
        if overwrite_flag
            if exist(filename, 'file')
                delete(filename);
            end
            
            %% get info %%
            disp('Begin collecting datasets info')
            info = h5info(fullfile(path_name,video_name));
            data_name = info.Datasets.Name;
            data_shape = info.Datasets.Dataspace.Size;
            nf = data_shape(3);
            pixho = data_shape(1);
            pixwo = data_shape(2);
            pixh = round(pixho * ratio);
            pixw = round(pixwo * ratio);
%             temp = h5read([path_name,video_name],['/',data_name],[1,1,1],[1,1,1]);
%             dtype = class(temp);

            time = toc(hcat);
            disp(['Done collecting datasets info, time: ', num2str(time)])
            
            %% initialization to get frames %%
            ds = Fsi / Fsi_new;
            f_use = 1: ds: nf;
            nf = length(f_use);
            
            %%% clear some variables %%%
            clear info
            
            %% batch configuration %%
            %%% parameters %%%
%             stype = parse_type(ttype);
%             nsize = pixh * pixw * nf * stype; %%% size of single %%%
            nbatch = 1; % batch_compute(nsize);
            ebatch = ceil(nf / nbatch);
            
            %%% extract batch-wise frame info %%%
            idbatch = [1: ebatch: nf, nf + 1];
%             nbatch = length(idbatch) - 1;
        
            %% collect data %%
            disp('Begin data cat')
            frame_all = single(h5read(fullfile(path_name,video_name),['/',data_name],[1,1,1],[inf,inf,inf],[1,1,ds]));
            if ratio ~=1
                frame_all = reshape(frame_all,[pixh, pixw, nf]);
            end
            imaxf = max(frame_all, [], 3);
            iminf = min(frame_all, [], 3);
            imeanf = mean(frame_all, 3);
            savef(filename, 2, 'frame_all')
            
            %%% normalize batch version %%%
            imx1 = max(imaxf(:));
            imn1 = min(iminf(:));
            m = normalize_batch(filename, 'frame_all', imx1, imn1, idbatch);
            save(fullfile(path_name, folder, [file_base, '_supporting.mat']), 'imx1', 'imn1')
            
        else %%% get outputs from the saved data file %%%
            load(fullfile(path_name, folder, [file_base, '_supporting.mat']), 'imx1', 'imn1')
            m = matfile(filename);
            [pixh, pixw, nf] = size(m, 'frame_all');
            imaxf = zeros(pixh, pixw);
            imeanf = zeros(pixh, pixw);
            stype = parse_type(ttype);
            nsize = pixh * pixw * nf * stype; %%% size of single %%%
            nbatch = batch_compute(nsize);
            ebatch = ceil(nf / nbatch);
            idbatch = [1: ebatch: nf, nf + 1];
            nbatch = length(idbatch) - 1;
            for i = 1: nbatch
                tmp = m.frame_all(1: pixh, 1: pixw, idbatch(i): idbatch(i + 1) - 1);
                imaxf = max(cat(3, max(tmp, [], 3), imaxf), [], 3);
                imeanf = (imeanf * (idbatch(i) - idbatch(1)) + sum(tmp, 3)) / (idbatch(i + 1) - idbatch(1));
            end
        end
    else %%% get .mat format %%%
        %%% get file info %%%
        fname = fullfile(path_name, [file_base, '.mat']);
        mm = matfile(fname);
        vnames = who(mm);
        eval(['dtype = class(mm.', vnames{1}, '(:, :, 1));'])
        [pixh, pixw, nff] = size(mm, vnames{1}); %%% assume only one variable %%%
        tratio = Fsi / Fsi_new;
        idd = false(1, nff);
        idd(1: tratio: nff) = true;
        nf = sum(idd);
        stype = parse_type(ttype);
        nsize = pixh * pixw * nff * stype; %%% size of single %%%
        nbatch = 1; % batch_compute(nsize);
        ebatch = ceil(nff / nbatch);
        
        %%% collect batch-wise frames %%%
        idbatch = [1: ebatch: nff, nff + 1];
        disp('Begin data cat')
        filename = fullfile(path_name, folder, [file_base, '_frame_all.mat']);
        msg = 'Overwrite raw .mat file (data)? (y/n)';
        overwrite_flag = true; % judge_file(filename, msg);
        
        %%% save data to the .mat file %%%
        imaxf = zeros(pixh, pixw);
        iminf = zeros(pixh, pixw);
        imeanf = zeros(pixh, pixw);
        idbatchn = ones(size(idbatch));
        if overwrite_flag
            if exist(filename, 'file')
                delete(filename);
            end
            for ib = 1: nbatch
                iddt = idd(idbatch(ib): idbatch(ib + 1) - 1);
                eval(['tmp = mm.', vnames{1}, '(1: pixh, 1: pixw, idbatch(ib): idbatch(ib + 1) - 1);'])
                frame_all = tmp(:, :, iddt);
                eval(['frame_all = ', ttype, '(frame_all);'])
                imaxf = max(cat(3, max(frame_all, [], 3), imaxf), [], 3);
                iminf = min(cat(3, min(frame_all, [], 3), iminf), [], 3);
                imeanf = (imeanf * (idbatch(ib) - idbatch(1)) + sum(frame_all, 3)) / (idbatch(ib + 1) - idbatch(1));
                savef(filename, 2, 'frame_all')
                idbatchn(ib + 1) = idbatchn(ib) + sum(iddt);
            end
            
            %%% normalize %%%
            imx1 = max(imaxf(:));
            imn1 = min(iminf(:));
            idbatch = idbatchn;
            m = normalize_batch(filename, 'frame_all', imx1, imn1, idbatch);
            save(fullfile(path_name, folder, [file_base, '_supporting.mat']), 'imx1', 'imn1')
        else
            load(fullfile(path_name, folder, [file_base, '_supporting.mat']), 'imx1', 'imn1')
            m = matfile(filename);
            imaxf = zeros(pixh, pixw);
            for i = 1: nbatch
                tmp = m.frame_all(1: pixh, 1: pixw, idbatch(i): idbatch(i + 1) - 1);
                imaxf = max(cat(3, max(tmp, [], 3), imaxf), [], 3);
                imeanf = (imeanf * (idbatch(i) - idbatch(1)) + sum(tmp, 3)) / (idbatch(i + 1) - idbatch(1));
            end
        end
    end
    
    time = toc(hcat);
    disp(['Done data cat, time: ', num2str(time)])
end



