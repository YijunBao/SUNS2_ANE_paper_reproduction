color=[  0    0.4470    0.7410
    0.8500    0.3250    0.0980
    0.9290    0.6940    0.1250
    0.4940    0.1840    0.5560
    0.4660    0.6740    0.1880
    0.3010    0.7450    0.9330
    0.6350    0.0780    0.1840];
% green = [0.1,0.9,0.1]; % color(5,:); %
% red = [0.9,0.1,0.1]; % color(7,:); %
% blue = [0.1,0.8,0.9]; % color(6,:); %
yellow = [0.8,0.8,0.0]; % color(3,:); %
magenta = [0.9,0.3,0.9]; % color(4,:); %
green = [0.0,0.65,0.0]; % color(5,:); %
red = [0.8,0.0,0.0]; % color(7,:); %
blue = [0.0,0.6,0.8]; % color(6,:); %
colors_multi = [color; distinguishable_colors(16)];

save_figures = true;
mag=1;
% mag_kernel = ones(mag,mag,'uint8');
SNR_range = [2,10];
addpath(genpath('C:\Matlab Files\missing_finder'))
alpha = 0.8;

%% neurons and masks frame
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
list_title = {'upper left', 'upper right', 'lower left', 'lower right'};
rate_hz = [10,15,15,5]; % frame rate of each video
radius = [5,6,8,14];
data_ind = 3;
data_name = list_data_names{data_ind};
dir_video = fullfile('E:\data_CNMFE',data_name);
dir_video_raw = dir_video;
% varname = '/mov';
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);
% varname = '/network_input';
% dir_video_raw = fullfile(dir_video, 'SNR video');
dir_GT_masks = fullfile(dir_video,'GT Masks');

%%
dir_MIN1PIPE = 'C:\Other methods\MIN1PIPE-3.0.0';
dir_CNMFE = 'C:\Other methods\CNMF_E-1.1.2';
switch data_ind
    case 1
        saved_date_MIN1PIPE = '20221216';
        saved_date_CNMFE = '20221216';
        SUNS_FISSA_sub = 'complete_FISSA_SF25\4816[1]th3';
        SUNS_TUnCaT_sub = 'complete_TUnCaT_SF25\4816[1]th5';
        MF_sub = 'trained dropout 0.8exp(-15)\avg_Xmask_0.5\classifier_res0_0+1 frames';
    case 2
        saved_date_MIN1PIPE = '20221216';
        saved_date_CNMFE = '20221215';
        SUNS_FISSA_sub = 'complete_FISSA_SF25\4816[1]th2';
        SUNS_TUnCaT_sub = 'complete_TUnCaT_SF25\4816[1]th4';
        MF_sub = 'trained dropout 0.8exp(-5)\avg_Xmask_0.5\classifier_res0_0+1 frames';
    case 3
        saved_date_MIN1PIPE = '20221219';
        saved_date_CNMFE = '20221218';
        SUNS_FISSA_sub = 'complete_FISSA_SF25\4816[1]th2';
        SUNS_TUnCaT_sub = 'complete_TUnCaT_SF25\4816[1]th4';
        MF_sub = 'trained dropout 0.8exp(-8)\avg_Xmask_0.5\classifier_res0_0+1 frames';
    case 4
        saved_date_MIN1PIPE = '20221215';
        saved_date_CNMFE = '20221215';
        SUNS_FISSA_sub = 'complete_FISSA_SF50\4816[1]th2';
        SUNS_TUnCaT_sub = 'complete_TUnCaT_SF50\4816[1]th4';
        MF_sub = 'trained dropout 0.8exp(-8)\avg_Xmask_0.5\classifier_res0_0+1 frames';
end
SNR_sub = fileparts(SUNS_TUnCaT_sub);
dir_video_SNR = fullfile(dir_video,SNR_sub,'network_input');

%%
for k=1:4
% clear video_SNR video_raw
Exp_ID = list_Exp_ID{k};
% load(['ABO mat\SNR_max\SNR_max_',Exp_ID,'.mat'],'SNR_max');
% SNR_max=SNR_max';
% load(['ABO mat\raw_max\raw_max_',Exp_ID,'.mat'],'raw_max');
% raw_max=raw_max';
% unmixed_traces = h5read([dir_traces,Exp_ID,'.h5'],'/unmixed_traces'); % raw_traces
video_raw = h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),'/mov'); % raw_traces
raw_max = max(video_raw,[],3);
video_SNR = h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),'/network_input'); % raw_traces
SNR_max = max(video_SNR,[],3);
[Lx,Ly] = size(raw_max);
SNR_max = SNR_max(1:Lx,1:Ly);

load(fullfile(dir_GT_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
GT_Masks = logical(FinalMasks);
edge_GT_Masks = 0*FinalMasks;
for nn = 1:size(FinalMasks,3)
    edge_GT_Masks(:,:,nn) = edge(FinalMasks(:,:,nn));
end
% FinalMasks = permute(FinalMasks,[2,1,3]);
GT_Masks_sum = sum(GT_Masks,3);
edge_GT_Masks_sum = sum(edge_GT_Masks,3);

% magnify
% mag=4;
mag_kernel = ones(mag,mag,class(SNR_max));
mag_kernel_bool = logical(mag_kernel);
SNR_max_mag = kron(SNR_max,mag_kernel);
[Lxm,Lym] = size(SNR_max_mag);
GT_Masks_sum_mag=kron(GT_Masks_sum,mag_kernel_bool);
edge_GT_Masks_sum_mag=kron(edge_GT_Masks_sum,mag_kernel_bool);

xrange=1:Lxm; yrange=1:Lym;
% xrange = 334:487; yrange=132:487;
% xrange = 1:487; yrange=1:487;
crop_png=[86,64,length(yrange),length(xrange)];


%% SUNS + missing finder
dir_output_mask_SUNS = fullfile(dir_video,SUNS_TUnCaT_sub,'output_masks');
dir_output_mask = fullfile(dir_output_mask_SUNS,'add_new_blockwise_weighted_sum_unmask\',MF_sub);
% load([dir_output_mask, Exp_ID, '.mat'], 'Masks_2');
% Masks = reshape(full(Masks_2'),487,487,[]);

load(fullfile(dir_output_mask_SUNS, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
num_SUNS = size(Masks_SUNS,3);
edge_Masks_SUNS = 0*Masks_SUNS;
for nn = 1:num_SUNS
    edge_Masks_SUNS(:,:,nn) = edge(Masks_SUNS(:,:,nn));
end

load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '_added.mat']), 'Masks');
Masks_SUNS_MF = Masks;
num_SUNS_MF = size(Masks_SUNS_MF,3);
num_MF = num_SUNS_MF - num_SUNS;
Masks_MF = Masks_SUNS_MF(:,:,num_SUNS+1:num_SUNS_MF);
edge_Masks_MF = 0*Masks_MF;
for nn = 1:num_MF
    edge_Masks_MF(:,:,nn) = edge(Masks_MF(:,:,nn));
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS_MF,0.5);
F1(isnan(F1)) = 0;
% Masks = permute(Masks,[2,1,3]);
Masks_SUNS_sum = sum(Masks_SUNS,3);
edge_Masks_SUNS_sum = sum(edge_Masks_SUNS,3);
Masks_SUNS_sum_mag=kron(Masks_SUNS_sum,mag_kernel_bool);
edge_Masks_SUNS_sum_mag=kron(edge_Masks_SUNS_sum,mag_kernel_bool);
Masks_MF_sum = sum(Masks_MF,3);
edge_Masks_MF_sum = sum(edge_Masks_MF,3);
Masks_MF_sum_mag=kron(Masks_MF_sum,mag_kernel_bool);
edge_Masks_MF_sum_mag=kron(edge_Masks_MF_sum,mag_kernel_bool);

TP_2 = sum(m,1)>0;
TP_22 = sum(m,2)>0;
FP_2 = sum(m,1)==0;
FN_2 = sum(m,2)==0;
masks_TP = sum(Masks_SUNS_MF(:,:,TP_2),3);
masks_FP = sum(Masks_SUNS_MF(:,:,FP_2),3);
masks_FN = sum(GT_Masks(:,:,FN_2),3);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,1)
figure('Position',[50,450,400,300],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_SUNS_sum_mag(xrange,yrange), 'Color', colors_multi(7,:));
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange,yrange)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(1,:),1,1,3);
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(5,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_SUNS_sum_mag(xrange,yrange)));  
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(6,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_MF_sum_mag(xrange,yrange)));  

title(sprintf('SUNS + missing finder, F1 = %1.2f',F1),'FontSize',12);
h=colorbar;
set(get(h,'Label'),'String','Peak SNR');
set(h,'FontSize',12);

% rectangle('Position',rect5,'EdgeColor',color(7,:),'LineWidth',2);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    saveas(gcf,[data_name, ' Masks SUNS_MF ',list_title{k},'.png']);
    % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
% cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
% if save_figures
%     imwrite(permute(cdata,[2,1,3]),'Fig3A SUNS.tif');
% end


%% SUNS TUnCaT
dir_output_mask = fullfile(dir_video,SUNS_TUnCaT_sub,'output_masks');
% load([dir_output_mask, Exp_ID, '.mat'], 'Masks_2');
% Masks = reshape(full(Masks_2'),487,487,[]);
load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
edge_Masks_SUNS = 0*Masks_SUNS;
for nn = 1:size(Masks_SUNS,3)
    edge_Masks_SUNS(:,:,nn) = edge(Masks_SUNS(:,:,nn));
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS,0.5);
% Masks = permute(Masks,[2,1,3]);
Masks_SUNS_sum = sum(Masks_SUNS,3);
edge_Masks_SUNS_sum = sum(edge_Masks_SUNS,3);
Masks_SUNS_sum_mag=kron(Masks_SUNS_sum,mag_kernel_bool);
edge_Masks_SUNS_sum_mag=kron(edge_Masks_SUNS_sum,mag_kernel_bool);

TP_2 = sum(m,1)>0;
TP_22 = sum(m,2)>0;
FP_2 = sum(m,1)==0;
FN_2 = sum(m,2)==0;
masks_TP = sum(Masks_SUNS(:,:,TP_2),3);
masks_FP = sum(Masks_SUNS(:,:,FP_2),3);
masks_FN = sum(GT_Masks(:,:,FN_2),3);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,1)
figure('Position',[50,50,400,300],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_SUNS_sum_mag(xrange,yrange), 'Color', colors_multi(7,:));
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange,yrange)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(1,:),1,1,3);
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(5,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_SUNS_sum_mag(xrange,yrange)));  

title(sprintf('SUNS TUnCaT, F1 = %1.2f',F1),'FontSize',12);
h=colorbar;
set(get(h,'Label'),'String','Peak SNR');
set(h,'FontSize',12);

% rectangle('Position',rect5,'EdgeColor',color(7,:),'LineWidth',2);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    saveas(gcf,[data_name, ' Masks SUNS_TUnCaT ',list_title{k},'.png']);
    % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
% cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
% if save_figures
%     imwrite(permute(cdata,[2,1,3]),'Fig3A SUNS.tif');
% end


%% SUNS FISSA
dir_output_mask = fullfile(dir_video,SUNS_FISSA_sub,'output_masks');
% load([dir_output_mask, Exp_ID, '.mat'], 'Masks_2');
% Masks = reshape(full(Masks_2'),487,487,[]);
load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
edge_Masks_SUNS = 0*Masks_SUNS;
for nn = 1:size(Masks_SUNS,3)
    edge_Masks_SUNS(:,:,nn) = edge(Masks_SUNS(:,:,nn));
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS,0.5);
% Masks = permute(Masks,[2,1,3]);
Masks_SUNS_sum = sum(Masks_SUNS,3);
edge_Masks_SUNS_sum = sum(edge_Masks_SUNS,3);
Masks_SUNS_sum_mag=kron(Masks_SUNS_sum,mag_kernel_bool);
edge_Masks_SUNS_sum_mag=kron(edge_Masks_SUNS_sum,mag_kernel_bool);

TP_2 = sum(m,1)>0;
TP_22 = sum(m,2)>0;
FP_2 = sum(m,1)==0;
FN_2 = sum(m,2)==0;
masks_TP = sum(Masks_SUNS(:,:,TP_2),3);
masks_FP = sum(Masks_SUNS(:,:,FP_2),3);
masks_FN = sum(GT_Masks(:,:,FN_2),3);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,1)
figure('Position',[450,50,400,300],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_SUNS_sum_mag(xrange,yrange), 'Color', colors_multi(7,:));
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange,yrange)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(4,:),1,1,3);
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(4,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_SUNS_sum_mag(xrange,yrange)));  

title(sprintf('SUNS FISSA, F1 = %1.2f',F1),'FontSize',12);
h=colorbar;
set(get(h,'Label'),'String','Peak SNR');
set(h,'FontSize',12);

% rectangle('Position',rect5,'EdgeColor',color(7,:),'LineWidth',2);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    saveas(gcf,[data_name, ' Masks SUNS_FISSA ',list_title{k},'.png']);
    % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
% cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
% if save_figures
%     imwrite(permute(cdata,[2,1,3]),'Fig3A SUNS.tif');
% end


%% MIN1PIPE
% load(fullfile(dir_MIN1PIPE,['eval_',data_name,'_thb history ',saved_date,' cv', num2str(cv),'.mat']),...
%     'best_Recall','best_Precision','best_F1','best_time','best_ind_param',...
%     'best_thb','ind_param','list_params','history','best_history');
% end_history = best_history(end,:);
% best_param = end_history(1:end-5);
% pix_select_sigthres = best_param(1);
% pix_select_corrthres = best_param(2);
% merge_roi_corrthres = best_param(3);
% dt = best_param(4);
% kappa = best_param(5);
% se = best_param(6);
load(fullfile(dir_MIN1PIPE,['eval_',data_name,'_thb ',saved_date_MIN1PIPE,' cv.mat']),'Table_time_ext');
best_param = Table_time_ext(k,1:end-4);
pix_select_sigthres = best_param(1);
pix_select_corrthres = best_param(2);
merge_roi_corrthres = best_param(3);
dt = best_param(4);
kappa = best_param(5);
se = best_param(6);
thb = best_param(7);

dir_sub = sprintf('pss=%0.2f_psc=%0.2f_mrc=%0.2f_dt=%0.2f_kappa=%0.2f_se=%d',...
    pix_select_sigthres, pix_select_corrthres, merge_roi_corrthres, dt, kappa, se);
dir_output_mask = fullfile(dir_video,'min1pipe',dir_sub);
load(fullfile(dir_output_mask, [Exp_ID, '_Masks_',num2str(thb),'.mat']), 'Masks3');
% load(fullfile(dir_output_mask, [Exp_ID, '_data_processed.mat']), 'roifn');
% Masks3 = roifn>0.2*max(roifn,[],1); %;%
Masks_min1 = reshape(Masks3,Lx,Ly,[]);
edge_Masks_min1 = 0*Masks_min1;
for nn = 1:size(Masks_min1,3)
    edge_Masks_min1(:,:,nn) = edge(Masks_min1(:,:,nn));
end

% Masks_min1 = permute(Masks_min1,[2,1,3]);
[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_min1,0.5);
Masks_min1_sum = sum(Masks_min1,3);
edge_Masks_min1_sum = sum(edge_Masks_min1,3);
Masks_min1_sum_mag=kron(Masks_min1_sum,mag_kernel_bool);
edge_Masks_min1_sum_mag=kron(edge_Masks_min1_sum,mag_kernel_bool);

TP_4 = sum(m,1)>0;
TP_42 = sum(m,2)>0;
FP_4 = sum(m,1)==0;
FN_4 = sum(m,2)==0;
masks_TP = sum(Masks_min1(:,:,TP_4),3);
masks_FP = sum(Masks_min1(:,:,FP_4),3);
masks_FN = sum(GT_Masks(:,:,FN_4),3);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,4)
figure('Position',[850,50,400,300],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_min1_sum_mag(xrange,yrange), 'Color', color(4,:));
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange,yrange)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(17,:),1,1,3);
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(1,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_min1_sum_mag(xrange,yrange)));  

title(sprintf('MIN1PIPE, F1 = %1.2f',F1),'FontSize',12);
h=colorbar;
set(get(h,'Label'),'String','Peak SNR');
set(h,'FontSize',12);

if save_figures
    saveas(gcf,[data_name, ' Masks min1pipe ',list_title{k},'.png']);
    % saveas(gcf,['figure 2\',Exp_ID,' CaImAn Batch h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
% cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
% if save_figures
%     imwrite(permute(cdata,[2,1,3]),'Fig3B CaImAn.tif');
% end


%% CNMF-E
% load(['.\eval_',data_name,'_thb history ',save_date,'.mat'],'best_Recall','best_Precision','best_F1',...
%     'best_time','best_ind_param','best_thb','ind_param','list_params','history','best_history') % ' cv', num2str(cv),
% end_history = best_history(end,:);
% thb = end_history(end-4);
% best_param = end_history(1:end-5);
% rbg = best_param(1);
% nk = best_param(2);
% rdmin = best_param(3);
% min_corr = best_param(4);
% min_pnr = best_param(5);
% merge_thr = best_param(6);
% mts = best_param(7);
% mtt = best_param(8);

load(fullfile(dir_CNMFE,['eval_',data_name,'_thb ',saved_date_CNMFE,' cv.mat']),'Table_time_ext');
best_param = Table_time_ext(k,1:end-4);
gSiz = 2 * radius(data_ind); % 12;
rbg = best_param(1);
nk = best_param(2);
rdmin = best_param(3);
min_corr = best_param(4);
min_pnr = best_param(5);
merge_thr = best_param(6);
mts = best_param(7);
mtt = best_param(8);
thb = best_param(9);

dir_sub = sprintf('gSiz=%d,rbg=%0.1f,nk=%d,rdmin=%0.1f,mc=%0.2f,mp=%d,mt=%0.2f,mts=%0.2f,mtt=%0.2f',...
    gSiz,rbg,nk,rdmin,min_corr,min_pnr,merge_thr,mts,mtt);
dir_output_mask = fullfile(dir_video,'CNMFE',dir_sub);
% load(fullfile(dir_output_mask, [Exp_ID, '_Masks_',num2str(thb),'.mat']), 'Masks3');
load(fullfile(dir_output_mask, [Exp_ID, '_result.mat']), 'neuron');
A = neuron.A;
A3 = neuron.reshape(A, 2);
Masks_cnmfe = threshold_Masks(A3, thb); %;%
% Ab = A>thb*max(A,[],1); %;% 0.5
% Masks3 = neuron.reshape(Ab, 2); 
% Masks_cnmfe = reshape(Masks3,Lx,Ly,[]);
edge_Masks_cnmfe = 0*Masks_cnmfe;
for nn = 1:size(Masks_cnmfe,3)
    edge_Masks_cnmfe(:,:,nn) = edge(Masks_cnmfe(:,:,nn));
end

% Masks_cnmfe = permute(Masks_cnmfe,[2,1,3]);
[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_cnmfe,0.5);
Masks_cnmfe_sum = sum(Masks_cnmfe,3);
edge_Masks_cnmfe_sum = sum(edge_Masks_cnmfe,3);
Masks_cnmfe_sum_mag=kron(Masks_cnmfe_sum,mag_kernel_bool);
edge_Masks_cnmfe_sum_mag=kron(edge_Masks_cnmfe_sum,mag_kernel_bool);

TP_5 = sum(m,1)>0;
TP_52 = sum(m,2)>0;
FP_5 = sum(m,1)==0;
FN_5 = sum(m,2)==0;
masks_TP = sum(Masks_cnmfe(:,:,TP_5),3);
masks_FP = sum(Masks_cnmfe(:,:,FP_5),3);
masks_FN = sum(GT_Masks(:,:,FN_5),3);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,4)
figure('Position',[1250,50,400,300],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_min1_sum_mag(xrange,yrange), 'Color', color(4,:));
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange,yrange)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(20,:),1,1,3);
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(2,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_cnmfe_sum_mag(xrange,yrange)));  

title(sprintf('CNMF-E, F1 = %1.2f',F1),'FontSize',12);
h=colorbar;
set(get(h,'Label'),'String','Peak SNR');
set(h,'FontSize',12);

if save_figures
    saveas(gcf,[data_name, ' Masks CNMF-E ',list_title{k},'.png']);
    % saveas(gcf,['figure 2\',Exp_ID,' CaImAn Batch h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
% cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
% if save_figures
%     imwrite(permute(cdata,[2,1,3]),'Fig3B CaImAn.tif');
% end

%%
% figure('Position',[1250,750,500,300],'Color','w');
% imshow(SNR_max(xrange,yrange),SNR_range); axis('image'); colormap gray;
% xticklabels({}); yticklabels({});
% h=colorbar;
% set(get(h,'Label'),'String','Peak SNR','FontName','Arial');
% set(h,'FontSize',12);
% if save_figures
%     saveas(gcf,'colorbar_SNR.svg');
% %     save(['trace\',Exp_ID,' N',num2str(N_neuron),' trace.mat'],'trace_N');
% end
end