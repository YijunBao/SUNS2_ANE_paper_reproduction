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
colors_multi = distinguishable_colors(16);

save_figures = true;
SNR_range = [2,10];
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))

%% neurons and masks frame
dir_video = 'E:\1photon-small\';
dir_video_raw = dir_video;
dir_sub= 'complete_TUnCaT\';
dir_video_SNR = [dir_video,dir_sub,'network_input\'];
dir_GT_masks = fullfile(dir_video,'GT Masks');
list_Exp_ID = {'c25_59_228','c27_12_326','c28_83_210',...
    'c25_163_267','c27_114_176','c28_161_149',...
    'c25_123_348','c27_122_121','c28_163_244'};

% dir_video='E:\OnePhoton videos\full videos\';
% dir_video_raw = dir_video;
% dir_video_SNR = [dir_video,'complete_TUnCaT\network_input\'];
% dir_GT_masks = fullfile(dir_video,'GT Masks');
% list_Exp_ID = {'c25_1NRE','c27_NN','c28_1NRE1NLE'};

% spike_type = 'ABO';
% % DirData = 'D:\ABO\';
% % dir_raw = 'D:\ABO\20 percent\';
% % dir_traces=dir_video;
% dir_traces=['..\results\',spike_type,'\unmixed traces\'];

for k=8%1:9
% clear video_SNR video_raw
Exp_ID = list_Exp_ID{k};
% load(['ABO mat\SNR_max\SNR_max_',Exp_ID,'.mat'],'SNR_max');
% SNR_max=SNR_max';
% load(['ABO mat\raw_max\raw_max_',Exp_ID,'.mat'],'raw_max');
% raw_max=raw_max';
% unmixed_traces = h5read([dir_traces,Exp_ID,'.h5'],'/unmixed_traces'); % raw_traces
video_raw = h5read([dir_video_raw,Exp_ID,'.h5'],'/mov',[1,1,1],[inf,inf,1]); % raw_traces
raw_max = max(video_raw,[],3);
video_SNR = h5read([dir_video_SNR,Exp_ID,'.h5'],'/network_input'); % raw_traces
SNR_max = max(video_SNR,[],3);
[Lx,Ly] = size(raw_max);
SNR_max = SNR_max(1:Lx,1:Ly);

load([dir_GT_masks, '\FinalMasks_', Exp_ID, '.mat'], 'FinalMasks');
GT_Masks = logical(FinalMasks);
% FinalMasks = permute(FinalMasks,[2,1,3]);
GT_Masks_sum = sum(GT_Masks,3);

%% magnify
mag=2;
mag_kernel = ones(mag,mag,class(SNR_max));
mag_kernel_bool = logical(mag_kernel);
SNR_max_mag = kron(SNR_max,mag_kernel);
[Lxm,Lym] = size(SNR_max_mag);
GT_Masks_sum_mag=kron(GT_Masks_sum,mag_kernel_bool);

xrange=1:Lxm; yrange=1:Lym;
% xrange = 334:487; yrange=132:487;
% xrange = 1:487; yrange=1:487;
crop_png=[86,64,length(yrange),length(xrange)];

%% SUNS 
dir_output_mask = [dir_video,dir_sub,'test_CNN\4816[1]th6\output_masks\Output_Masks_'];
% load([dir_output_mask, Exp_ID, '.mat'], 'Masks_2');
% Masks = reshape(full(Masks_2'),487,487,[]);
load([dir_output_mask, Exp_ID, '.mat'], 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS,0.5);
% Masks = permute(Masks,[2,1,3]);
Masks_SUNS_sum = sum(Masks_SUNS,3);
% Masks_SUNS_sum_mag=kron(Masks_SUNS_sum,mag_kernel_bool);
Masks_SUNS_mag=zeros(Lxm,Lym,size(Masks_SUNS,3));
for n=1:size(Masks_SUNS,3)
    Masks_SUNS_mag(:,:,n) = kron(Masks_SUNS(:,:,n),mag_kernel_bool);
end
GT_Masks_mag=zeros(Lxm,Lym,size(GT_Masks,3));
for n=1:size(GT_Masks,3)
    GT_Masks_mag(:,:,n) = kron(GT_Masks(:,:,n),mag_kernel_bool);
end

TP_2 = find(sum(m,1)>0);
TP_22 = find(sum(m,2)>0);
FP_2 = find(sum(m,1)==0);
FN_2 = find(sum(m,2)==0);
% masks_TP = sum(Masks_SUNS(:,:,TP_2),3);
% masks_FP = sum(Masks_SUNS(:,:,FP_2),3);
% masks_FN = sum(GT_Masks(:,:,FN_2),3);
% masks_TP_mag=kron(masks_TP,mag_kernel_bool);
% masks_FP_mag=kron(masks_FP,mag_kernel_bool);
% masks_FN_mag=kron(masks_FN,mag_kernel_bool);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,1)
figure('Position',[50,50,400,300],'Color','w');
%     imshow(raw_max,[0,1024]);
imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP_mag(xrange,yrange), 'Color', green);
% contour(masks_FP_mag(xrange,yrange), 'Color', red);
% contour(masks_FN_mag(xrange,yrange), 'Color', blue);
for n=1:length(TP_2)
    contour(Masks_SUNS_mag(xrange,yrange,TP_2(n)), 'Color', green);
end
for n=1:length(FP_2)
    contour(Masks_SUNS_mag(xrange,yrange,FP_2(n)), 'Color', red);
end
for n=1:length(FN_2)
    contour(GT_Masks_mag(xrange,yrange,FN_2(n)), 'Color', blue);
end
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_SUNS_sum_mag(xrange,yrange), 'Color', colors_multi(7,:));

title(sprintf('SUNS, F1 = %1.2f',F1),'FontSize',12);
h=colorbar;
set(get(h,'Label'),'String','Peak SNR');
set(h,'FontSize',12);

% % rectangle('Position',rect1,'EdgeColor',yellow,'LineWidth',2);
% % rectangle('Position',rect2,'EdgeColor',yellow,'LineWidth',2);
% rectangle('Position',rect3,'EdgeColor',color(7,:),'LineWidth',2);
% % rectangle('Position',rect4,'EdgeColor',yellow,'LineWidth',2);
% rectangle('Position',rect5,'EdgeColor',color(7,:),'LineWidth',2);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    saveas(gcf, [Exp_ID,' Masks 3 SUNS_TUnCaT 2-10.png']);
    % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
end
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
% dir_output_mask = [dir_video,'\min1pipe\'];
% % dir_output_mask = 'D:\ABO\20 percent bin 5\CaImAn-Batch\Masks\';
% load([dir_output_mask, Exp_ID, '_data_processed.mat'], 'roifn');
% roib = roifn>0.5*max(roifn,[],1); %;%
% Masks_min1 = reshape(roib,Lx,Ly,[]);
% % Masks_min1 = permute(Masks_min1,[2,1,3]);
% [Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_min1,0.5);
% Masks_min1_sum = sum(Masks_min1,3);
% Masks_min1_sum_mag=kron(Masks_min1_sum,mag_kernel_bool);
% 
% TP_4 = sum(m,1)>0;
% TP_42 = sum(m,2)>0;
% FP_4 = sum(m,1)==0;
% FN_4 = sum(m,2)==0;
% masks_TP = sum(Masks_min1(:,:,TP_4),3);
% masks_FP = sum(Masks_min1(:,:,FP_4),3);
% masks_FN = sum(GT_Masks(:,:,FN_4),3);
%     
% % Style 2: Three colors
% % figure(98)
% % subplot(2,3,4)
% figure('Position',[650,50,400,300],'Color','w');
% %     imshow(raw_max,[0,1024]);
% imagesc(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
% xticklabels({}); yticklabels({});
% hold on;
% % contour(masks_TP(xrange,yrange), 'Color', green);
% % contour(masks_FP(xrange,yrange), 'Color', red);
% % contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'Color', color(3,:));
% contour(Masks_min1_sum_mag(xrange,yrange), 'Color', color(4,:));
% 
% title(sprintf('MIN1PIPE, F1 = %1.2f',F1),'FontSize',12);
% h=colorbar;
% set(get(h,'Label'),'String','Peak SNR');
% set(h,'FontSize',12);
% 
% if save_figures
%     saveas(gcf,'Masks min1pipe.png');
%     % saveas(gcf,['figure 2\',Exp_ID,' CaImAn Batch h.svg']);
% end

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
