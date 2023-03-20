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

% default
% list_pss = 0.8;
% list_psc = 0.6;
% list_mrc = 0.9;
% list_thb = 0.2;

% list_pss = [0.7, 0.8, 0.9]; % 0.8; % 
% list_psc = [0.6, 0.7, 0.8, 0.9]; % 0.4, 
% list_mrc = [0.5, 0.6, 0.7, 0.8]; % 0.3, , 0.9, 0.95
list_pss = 0.8; % [0.2, 0.4, 0.6, 0.8, 0.9];
list_psc = [0.8, 0.85, 0.9]; % 0.4;% 
list_mrc = [0.8, 0.85, 0.9, 0.95];
list_thb = 0.2; % [0.2, 0.5]; % 

num_pss = length(list_pss);
num_psc = length(list_psc);
num_mrc = length(list_mrc);
num_thb = length(list_thb);

list_Recall = zeros(num_Exp,num_pss,num_psc,num_mrc,num_thb);
list_Precision = zeros(num_Exp,num_pss,num_psc,num_mrc,num_thb);
list_F1 = zeros(num_Exp,num_pss,num_psc,num_mrc,num_thb);
list_time = zeros(num_Exp,num_pss,num_psc,num_mrc,num_thb);

%%
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    filename = [Exp_ID,'.h5'];
    for pss_id = 1:num_pss
        pix_select_sigthres = list_pss(pss_id);
    for psc_id = 1:num_psc
        pix_select_corrthres = list_psc(psc_id);
    for mrc_id = 1:num_mrc
        merge_roi_corrthres = list_mrc(mrc_id);
%% main program %%
folder = sprintf('min1pipe\\pss=%0.2f_psc=%0.2f_mrc=%0.2f',...
    pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres);
fname = fullfile(path_name,folder,[Exp_ID,'_data_processed.mat']);
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
for thb_id = 1:num_thb
    th_binary = list_thb(thb_id);
    % load([dir_GT,'FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
    % load([dir_GT,'FinalMasks_',Exp_ID,'_sparse.mat'],'GTMasks_2')
    % [pixh, pixw, nGT] = size(FinalMasks);
    roib = roifn>th_binary*max(roifn,[],1); %;%
    Masks3 = reshape(full(roib), pixh, pixw, size(roib,2));
    % [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,roifn,0.5);
    [Recall, Precision, F1] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
    used_time = seconds(process_time{2}-process_time{1});
    list_Recall(eid,pss_id,psc_id,mrc_id,thb_id) = Recall;
    list_Precision(eid,pss_id,psc_id,mrc_id,thb_id) = Precision;
    list_F1(eid,pss_id,psc_id,mrc_id,thb_id) = F1;
    list_time(eid,pss_id,psc_id,mrc_id,thb_id) = used_time;
    disp([Recall, Precision, F1, used_time])
end
save(['eval_',data_name,'.mat'],'list_Recall','list_Precision','list_F1','list_time',...
    'list_pss','list_psc','list_mrc','list_thb')

%% plot some images %%
if isvis
    figure('Position',get(0,'ScreenSize'))
    clf
    %%% raw max %%%
    subplot(2, 3, 1, 'align')
    imagesc(imaxn)
    axis square
    title('Raw')

    %%% neural enhanced before movement correction %%%
    subplot(2, 3, 2, 'align')
    imagesc(imaxy)
    axis square
    title('Before MC')

    %%% neural enhanced after movement correction %%%
    subplot(2, 3, 3, 'align')
    imagesc(imax)
    axis square
    title('After MC')

    %%% contour %%%
    subplot(2, 3, 4, 'align')
    plot_contour(roifn, sigfn, seedsfn, imax, pixh, pixw)
    axis square

    %%% movement measurement %%%
    subplot(2, 3, 5, 'align')
    axis off
    if ismc
        plot(raw_score); hold on; plot(corr_score); hold off;
        axis square
        title('MC Scores')
    else
        title('MC skipped')
    end

    %%% all identified traces %%%
    subplot(2, 3, 6, 'align')
    sigt = sigfn;
    for i = 1: size(sigt, 1)
        sigt(i, :) = normalize(sigt(i, :));
    end
    plot((sigt + (1: size(sigt, 1))')')
    axis tight
    axis square
    title('Traces')
    
    saveas(gcf,[path_name,'min1pipe\',Exp_ID,'.png']);

    %% make a movie %%
    % load(fname)
    % mraw = matfile(frawname);
    % mreg = matfile(fregname);
    % id = find(fname == filesep, 1, 'last');
    % fmovie = [fname(1: id), 'demo_vid.avi'];
    % v = VideoWriter(fmovie);
    % v.FrameRate = Fsi_new;
    % v.Quality = 100;
    % open(v)
    % 
    % %%% compute batch %%%
    % ttype = class(mraw.frame_all(1, 1, 1));
    % stype = parse_type(ttype);
    % dss = 2;
    % dst = 2;
    % nf = size(sigfn, 2);
    % nsize = pixh * pixw * nf * stype * 6 / (dss ^ 2); %%% size of single %%%
    % nbatch = batch_compute(nsize);
    % ebatch = ceil(nf / nbatch);
    % idbatch = [1: ebatch: nf, nf + 1];
    % nbatch = length(idbatch) - 1;
    % 
    % %%% make movie %%%
    % figure(2)
    % set(gcf, 'Units', 'normalized', 'position', [0.5, 0.1, 0.4, 0.2])
    % for ii = 1: nbatch
    %     dataraw = mraw.frame_all(1: dss: pixh, 1: dss: pixw, idbatch(ii): idbatch(ii + 1) - 1);
    %     datareg = mreg.reg(1: dss: pixh, 1: dss: pixw, idbatch(ii): idbatch(ii + 1) - 1);
    %     datar = reshape(roifn * sigfn(:, idbatch(ii): idbatch(ii + 1) - 1), pixh, pixw, []);
    %     datar = datar(1: dss: end, 1: dss: end, :);
    %     for i = 1: dst: size(dataraw, 3)
    %         clf
    %         subplot(1, 3, 1, 'align')
    %         imagesc(dataraw(:, :, i + idbatch(ii) - 1), [0, 1])
    %         axis off
    %         axis square
    %         title('Raw')
    %         
    %         subplot(1, 3, 2, 'align')
    %         imagesc(datareg(:, :, i + idbatch(ii) - 1), [0, 1])
    %         axis off
    %         axis square
    %         title('After MC')
    %         
    %         subplot(1, 3, 3, 'align')
    %         imagesc(datar(:, :, i + idbatch(ii) - 1), [0, 1])
    %         axis off
    %         axis square
    %         title('Processed')
    %         
    %         suptitle(['Frame #', num2str(i)])
    %         
    %         movtmp = getframe(gcf);
    %         writeVideo(v, movtmp);
    %     end
    % end
    % close(v)
end
    end
    end
    end

%% post-process %%
if ifpost
    real_neuron_select
end
end

Table_time = [list_Recall, list_Precision, list_F1, list_time];
Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
