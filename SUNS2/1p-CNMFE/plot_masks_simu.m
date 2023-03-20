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
SNR_range = [2,5]; % [2,10]; % []; % 
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))
alpha = 0.8;

%% neurons and masks frame
scale_lowBG = 5e3;
scale_noise = 1;
results_folder = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
list_data_names={results_folder};
data_ind = 1;
data_name = list_data_names{data_ind};
dir_video = fullfile('E:\simulation_CNMFE_corr_noise',data_name);
dir_video_raw = dir_video;
% varname = '/mov';
dir_video_SNR = fullfile(dir_video,'complete_TUnCaT\network_input\');
num_Exp = 10;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);
list_title = list_Exp_ID;
% varname = '/network_input';
% dir_video_raw = fullfile(dir_video, 'SNR video');
dir_GT_masks = fullfile(dir_video,'GT Masks');

k=7;
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
mag=2;
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

%% improved SUNS 
dir_output_mask = fullfile(dir_video,'complete_TUnCaT\4816[1]th4\output_masks');
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
figure('Position',[50,10,700,500],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
% imagesc(SNR_max_mag(xrange,yrange)); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_SUNS_sum_mag(xrange,yrange), 'Color', colors_multi(7,:));
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange,yrange)));  
alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(6,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_SUNS_sum_mag(xrange,yrange)));  

title(sprintf('SUNS TUnCaT, F1 = %1.4f',F1),'FontSize',12);
h=colorbar;
set(get(h,'Label'),'String','Peak SNR');
set(h,'FontSize',12);

% rectangle('Position',rect5,'EdgeColor',color(7,:),'LineWidth',2);
rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

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


%% original SUNS
dir_output_mask = fullfile(dir_video,'complete_FISSA\4816[1]th2\output_masks');
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
figure('Position',[50,510,700,500],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
% imagesc(SNR_max_mag(xrange,yrange)); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_SUNS_sum_mag(xrange,yrange), 'Color', colors_multi(7,:));
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange,yrange)));  
alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(9,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_SUNS_sum_mag(xrange,yrange)));  

title(sprintf('SUNS FISSA, F1 = %1.4f',F1),'FontSize',12);
h=colorbar;
set(get(h,'Label'),'String','Peak SNR');
set(h,'FontSize',12);

% rectangle('Position',rect5,'EdgeColor',color(7,:),'LineWidth',2);
rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

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
dir_output_mask = fullfile(dir_video,'min1pipe\pss=0.80_psc=0.60_mrc=0.40_dt=0.05_kappa=0.60_se=6');
% dir_output_mask = 'D:\ABO\20 percent bin 5\CaImAn-Batch\Masks\';
load(fullfile(dir_output_mask, [Exp_ID, '_data_processed.mat']), 'roifn');
roib = roifn>0.3*max(roifn,[],1); %;%
Masks_min1 = reshape(roib,Lx,Ly,[]);
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
figure('Position',[750,10,700,500],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
% imagesc(SNR_max_mag(xrange,yrange)); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_min1_sum_mag(xrange,yrange), 'Color', color(4,:));
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange,yrange)));  
alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(17,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_min1_sum_mag(xrange,yrange)));  

title(sprintf('MIN1PIPE, F1 = %1.4f',F1),'FontSize',12);
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
dir_output_mask = fullfile(dir_video,'CNMFE\gSiz=12,rbg=1.5,nk=3,rdmin=2.0,mc=0.60,mp=2,mt=0.60,mts=0.30,mtt=0.40');
% dir_output_mask = 'D:\ABO\20 percent bin 5\CaImAn-Batch\Masks\';
load(fullfile(dir_output_mask, [Exp_ID, '_Masks_0.4.mat']), 'Masks3');
Masks3 = Masks3>0.2*max(Masks3,[],1); %;%
Masks_cnmfe = reshape(Masks3,Lx,Ly,[]);
edge_Masks_cnmfe = 0*Masks_cnmfe;
for nn = 1:size(Masks_cnmfe,3)
    edge_Masks_cnmfe(:,:,nn) = edge(Masks_cnmfe(:,:,nn));
end

% Masks_cnmfe = permute(Masks_cnmfe,[2,1,3]);
try
    [Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_cnmfe,0.5);
catch
    Recall = 0; Precision = 0; F1 = 0; m = []; 
end
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
figure('Position',[750,510,700,500],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
% imagesc(SNR_max_mag(xrange,yrange)); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_min1_sum_mag(xrange,yrange), 'Color', color(4,:));
alphaImg = ones(Lx*mag,Ly*mag).*reshape(color(3,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange,yrange)));  
alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(20,:),1,1,3);
image(alphaImg,'Alphadata',alpha*(edge_Masks_cnmfe_sum_mag(xrange,yrange)));  

title(sprintf('CNMF-E, F1 = %1.4f',F1),'FontSize',12);
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
