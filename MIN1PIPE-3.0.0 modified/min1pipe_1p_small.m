%%% demo of the full MIN1PIPE %%%
clear
gcp;
min1pipe_init;
addpath('C:\Matlab Files\neuron_post');
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))

%% Set data and folder
path_name='E:\1photon-small\';
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};
num_Exp = length(list_Exp_ID);
dir_GT = 'E:\1photon-small\GT Masks\'; % FinalMasks_

%% session-specific parameter initialization %% 
Fsi = 20;
Fsi_new = 20; %%% no temporal downsampling %%%
spatialr = 1; %%% no spatial downsampling %%%
se = 4; % 3.6; %%% structure element for background removal %%%
ismc = false; % true; %%% run movement correction %%%
flag = 1; %%% use auto seeds selection; 2 if manual %%%
isvis = false; % true; % %% do visualize %%%
ifpost = false; %%% set true if want to see post-process %%%

list_Recall = zeros(num_Exp,1);
list_Precision = zeros(num_Exp,1);
list_F1 = zeros(num_Exp,1);
list_time = zeros(num_Exp,1);

%%
for eid = 1:num_Exp
    Exp_ID = list_Exp_ID{eid};
    filename = [Exp_ID,'.h5'];
%% main program %%
fname = [path_name,'min1pipe\',Exp_ID,'_data_processed.mat'];
% [fname, frawname, fregname] = min1pipe_h5(path_name, filename, Fsi, Fsi_new, spatialr, se, ismc, flag);

%% Calculate accuracy
load(fname)
% load([dir_GT,'FinalMasks_',Exp_ID,'.mat'],'FinalMasks')
% load([dir_GT,'FinalMasks_',Exp_ID,'_sparse.mat'],'GTMasks_2')
% [pixh, pixw, nGT] = size(FinalMasks);
roib = roifn>0.5*max(roifn,[],1); %;%
Masks3 = reshape(full(roib), pixh, pixw, size(roib,2));
% [Recall, Precision, F1, m] = GetPerformance_Jaccard_2(GTMasks_2,roifn,0.5);
[list_Recall(eid), list_Precision(eid), list_F1(eid)] = GetPerformance_Jaccard(dir_GT,Exp_ID,Masks3,0.5);
disp([list_Recall(eid), list_Precision(eid), list_F1(eid)])
list_time(eid) = seconds(process_time{2}-process_time{1});

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

%% post-process %%
if ifpost
    real_neuron_select
end
end

save('eval_1p_small.mat','list_Recall','list_Precision','list_F1','list_time')
Table_time = [list_Recall, list_Precision, list_F1, list_time];
Table_time_ext=[Table_time;nanmean(Table_time,1);nanstd(Table_time,1,1)];
