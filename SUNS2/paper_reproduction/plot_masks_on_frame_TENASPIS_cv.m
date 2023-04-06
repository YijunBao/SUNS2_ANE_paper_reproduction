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
color_box = [0,0,0.8]; % color(7,:); % 
% color_box = colors_multi(15,:,:); % color(7,:); % 

save_figures = true;
image_only = true;
show_zoom = true;
% mag_kernel = ones(mag,mag,'uint8');
addpath(genpath('C:\Matlab Files\missing_finder'))
alpha = 0.8;

%% neurons and masks frame
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
list_title = list_Exp_ID;
rate_hz = 20; % frame rate of each video
radius = 9;
data_name = 'TENASPIS';
dir_video = 'D:\data_TENASPIS\added_refined_masks\';
dir_video_raw = dir_video;
% varname = '/mov';
dir_video_SNR = fullfile(dir_video,'complete_TUnCaT_SF25\network_input\');
dir_traces = fullfile(dir_video,'complete_TUnCaT_SF25\TUnCaT\alpha= 1.000\');
% varname = '/network_input';
% dir_video_raw = fullfile(dir_video, 'SNR video');
dir_GT_masks = fullfile(dir_video,'GT Masks');
dir_MIN1PIPE = 'C:\Other methods\MIN1PIPE-3.0.0';
dir_CNMFE = 'C:\Other methods\CNMF_E-1.1.2';
% saved_date_MIN1PIPE = '20221216';
% saved_date_CNMFE = '20221220';
saved_date_MIN1PIPE = '20230111 cv 2round';
saved_date_CNMFE = '20221221 cv 2round';
mag=1;
mag_crop=4;
%%
for k=1%:num_Exp_ID
% clear video_SNR video_raw
Exp_ID = list_Exp_ID{k};
load(['TENASPIS mat\SNR_max\SNR_max_',Exp_ID,'.mat'],'SNR_max');
% SNR_max=SNR_max';
load(['TENASPIS mat\raw_max\raw_max_',Exp_ID,'.mat'],'raw_max');
% raw_max=raw_max';
load(fullfile(dir_traces,[Exp_ID,'.mat']),'traces_nmfdemix'); % raw_traces
unmixed_traces = traces_nmfdemix;
% unmixed_traces = h5read([dir_traces,Exp_ID,'.h5'],'/unmixed_traces'); % raw_traces
% video_raw = h5read(fullfile(dir_video_raw,[Exp_ID,'.h5']),'/mov'); % raw_traces
% raw_max = max(video_raw,[],3);
% video_SNR = h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),'/network_input'); % raw_traces
% SNR_max = max(video_SNR,[],3);
[Lx,Ly] = size(raw_max);
SNR_max = SNR_max(1:Lx,1:Ly);

load(fullfile(dir_GT_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
GT_Masks = logical(FinalMasks);
NGT = size(GT_Masks,3);
% edge_GT_Masks = 0*FinalMasks;
% for nn = 1:size(FinalMasks,3)
%     edge_GT_Masks(:,:,nn) = edge(FinalMasks(:,:,nn));
% end
% FinalMasks = permute(FinalMasks,[2,1,3]);
% GT_Masks_sum = sum(GT_Masks,3);
% edge_GT_Masks_sum = sum(edge_GT_Masks,3);

% magnify
% mag=4;
mag_kernel = ones(mag,mag,class(SNR_max));
mag_kernel_uint8 = ones(mag_crop,mag_crop,'uint8');
mag_kernel_bool = logical(mag_kernel);
SNR_max_mag = kron(SNR_max,mag_kernel);
[Lxm,Lym] = size(SNR_max_mag);
% GT_Masks_sum_mag=kron(GT_Masks_sum,mag_kernel_bool);
% edge_GT_Masks_sum_mag=kron(edge_GT_Masks_sum,mag_kernel_bool);
if mag > 1
    GT_Masks_mag=zeros(Lxm,Lym,NGT,'logical');
    for n = 1:NGT
        GT_Masks_mag(:,:,n) = kron(GT_Masks(:,:,n),mag_kernel_bool);
    end
else
    GT_Masks_mag = GT_Masks;
end

% xrange=1:Lx; yrange=1:Ly;
% xrange=(Lx/3+1):Lx*2/3; yrange=1:Ly/3;
% xrange=(Lx/3+1):Lx*2/3; yrange=(Ly/3+1):Ly*2/3;
xrange=(Lx*2/3+1):Lx; yrange=(Ly*2/3+1):Ly;
% xrange=410:460; yrange=395:435;
% xrange=340:375; yrange=340:380;
Lxc = length(xrange); Lyc = length(yrange); 
xrange_mag=((xrange(1)-1)*mag+1):(xrange(end)*mag); 
yrange_mag=((yrange(1)-1)*mag+1):(yrange(end)*mag);
Lxc_mag = length(xrange_mag); Lyc_mag = length(yrange_mag); 
crop_png=[86,64,Lxc,Lyc];
N_neuron1 = 282; % 378; % 
N_neuron2 = 436; % 
N_neuron3 = 666; % 
rect1=[395,419,30,40];
rect2=[342,348,20,30];
rect3=[362,343,20,30];
rect1_sub=rect1 - [yrange(1)-1,xrange(1)-1,0,0];
rect2_sub=rect2 - [yrange(1)-1,xrange(1)-1,0,0];
rect3_sub=rect3 - [yrange(1)-1,xrange(1)-1,0,0];
SNR_range = [2,14]; % [0,5]; % [0,10]; % 

save_folder = sprintf('figures_%d-%d,%d-%d crop',xrange(1),xrange(end),yrange(1),yrange(end));
save_folder = ['.\',save_folder,'\'];
if ~exist(save_folder,'dir')
    mkdir(save_folder)
end

%%
% yrange_zoom = rect1(1):rect1(1)+rect1(3);
% xrange_zoom = rect1(2):rect1(2)+rect1(4);
% COM1 = rect1(1:2) + rect1(3:4)/2;
% COM1 = COM1 + [xrange(1), yrange(1)]-1;
% N_neuron = find(FinalMasks(COM1(1),COM1(2),:));
trace_N = unmixed_traces(:,N_neuron1);
PSNR = max(trace_N);
high = find(trace_N > min(10,PSNR-1));
% start = [rect1(2)+xrange(1)-1, rect1(1)+yrange(1)-1, 1];
start = [rect1(2), rect1(1), 1];
count = [rect1(4)+1, rect1(3)+1, 1];
num_high = length(high);
SNR_high = zeros(count(1),count(2),num_high,'single');
for it = 1:num_high
    t = high(it);
    start_t = [start(1),start(2),t];
    SNR_high(:,:,it) = h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),'/network_input',start_t, count); % SNR
end
% SNR_high = SNR_full(:,:,high);
SNR_high_mean = mean(SNR_high,3);
% SNR_max(start(1):start(1)+count(1)-1,start(2):start(2)+count(2)-1,:) = SNR_high_mean;

figure('Position',[100,650,300,150],'Color','w');
plot(trace_N,'Color',color(3,:),'LineWidth',2);
ylim([-1,4]);
hold on;
plot(high,trace_N(high),'Color',color(7,:),'LineWidth',2);
pos1=1800;
pos2=2.5;
plot(pos1+[0,200],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+1*[0,1],'k','LineWidth',2);
% text(pos1-20,pos2+0.5,{'SNR','1'},'HorizontalAlignment','right','FontSize',14); % ,'rotation',90
text(pos1-100,pos2+0.5,'SNR=1','HorizontalAlignment','center','rotation',90,'FontSize',14); % 
text(pos1,pos2-0.4,'10 s','FontSize',14); % ,'rotation',0
if save_figures
    saveas(gcf,[save_folder, data_name,' ',list_title{k}, ' trace ',num2str(N_neuron1),'.png']);
    saveas(gcf,[save_folder, data_name,' ',list_title{k}, ' trace ',num2str(N_neuron1),'.emf']);
end

%%
% yrange_zoom = rect2(1):rect2(1)+rect2(3);
% xrange_zoom = rect2(2):rect2(2)+rect2(4);
% COM1 = rect1(1:2) + rect1(3:4)/2;
% COM1 = COM1 + [xrange(1), yrange(1)]-1;
% N_neuron = find(FinalMasks(COM1(1),COM1(2),:));
trace_N = unmixed_traces(:,N_neuron2);
PSNR = max(trace_N);
high = find(trace_N > min(10,PSNR-1));
% start = [rect1(2)+xrange(1)-1, rect1(1)+yrange(1)-1, 1];
start = [rect2(2), rect2(1), 1];
count = [rect2(4)+1, rect2(3)+1, 1];
num_high = length(high);
SNR_high = zeros(count(1),count(2),num_high,'single');
for it = 1:num_high
    t = high(it);
    start_t = [start(1),start(2),t];
    SNR_high(:,:,it) = h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),'/network_input',start_t, count); % SNR
end
% SNR_high = SNR_full(:,:,high);
SNR_high_mean = mean(SNR_high,3);
% SNR_max(start(1):start(1)+count(1)-1,start(2):start(2)+count(2)-1,:) = SNR_high_mean;

figure('Position',[500,650,300,150],'Color','w');
plot(trace_N,'Color',color(3,:),'LineWidth',2);
ylim([-1,4]);
hold on;
plot(high,trace_N(high),'Color',color(7,:),'LineWidth',2);
pos1=1300;
pos2=2.5;
plot(pos1+[0,200],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+1*[0,1],'k','LineWidth',2);
% text(pos1-20,pos2+0.5,{'SNR','1'},'HorizontalAlignment','right','FontSize',14); % ,'rotation',90
text(pos1-100,pos2+0.5,'SNR=1','HorizontalAlignment','center','rotation',90,'FontSize',14); % 
text(pos1,pos2-0.4,'10 s','FontSize',14); % ,'rotation',0
if save_figures
    saveas(gcf,[save_folder, data_name,' ',list_title{k}, ' trace ',num2str(N_neuron2),'.png']);
    saveas(gcf,[save_folder, data_name,' ',list_title{k}, ' trace ',num2str(N_neuron2),'.emf']);
end

%%
% yrange_zoom = rect2(1):rect2(1)+rect2(3);
% xrange_zoom = rect2(2):rect2(2)+rect2(4);
% COM1 = rect1(1:2) + rect1(3:4)/2;
% COM1 = COM1 + [xrange(1), yrange(1)]-1;
% N_neuron = find(FinalMasks(COM1(1),COM1(2),:));
trace_N = unmixed_traces(:,N_neuron3);
PSNR = max(trace_N);
high = find(trace_N > min(10,PSNR-1));
% start = [rect1(2)+xrange(1)-1, rect1(1)+yrange(1)-1, 1];
start = [rect3(2), rect3(1), 1];
count = [rect3(4)+1, rect3(3)+1, 1];
num_high = length(high);
SNR_high = zeros(count(1),count(2),num_high,'single');
for it = 1:num_high
    t = high(it);
    start_t = [start(1),start(2),t];
    SNR_high(:,:,it) = h5read(fullfile(dir_video_SNR,[Exp_ID,'.h5']),'/network_input',start_t, count); % SNR
end
% SNR_high = SNR_full(:,:,high);
SNR_high_mean = mean(SNR_high,3);
% SNR_max(start(1):start(1)+count(1)-1,start(2):start(2)+count(2)-1,:) = SNR_high_mean;

figure('Position',[900,650,300,150],'Color','w');
plot(trace_N,'Color',color(3,:),'LineWidth',2);
ylim([-1,4]);
hold on;
plot(high,trace_N(high),'Color',color(7,:),'LineWidth',2);
pos1=1300;
pos2=2.5;
plot(pos1+[0,200],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+1*[0,1],'k','LineWidth',2);
% text(pos1-20,pos2+0.5,{'SNR','1'},'HorizontalAlignment','right','FontSize',14); % ,'rotation',90
text(pos1-100,pos2+0.5,'SNR=1','HorizontalAlignment','center','rotation',90,'FontSize',14); % 
text(pos1,pos2-0.4,'10 s','FontSize',14); % ,'rotation',0
if save_figures
    saveas(gcf,[save_folder, data_name,' ',list_title{k}, ' trace ',num2str(N_neuron3),'.png']);
    saveas(gcf,[save_folder, data_name,' ',list_title{k}, ' trace ',num2str(N_neuron3),'.emf']);
end

%%
SNR_max_mag = kron(SNR_max,mag_kernel);

%% SUNS + missing finder
dir_output_mask_SUNS = fullfile(dir_video,'complete_TUnCaT_SF25\4816[1]th5\output_masks');
dir_output_mask = fullfile(dir_output_mask_SUNS,'add_new_blockwise_weighted_sum_unmask\trained dropout 0.8exp(-15)\avg_Xmask_0.5\classifier_res0_0+1 frames');
% load([dir_output_mask, Exp_ID, '.mat'], 'Masks_2');
% Masks = reshape(full(Masks_2'),487,487,[]);

load(fullfile(dir_output_mask_SUNS, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
N_SUNS = size(Masks_SUNS,3);
% edge_Masks_SUNS = 0*Masks_SUNS;
% for nn = 1:num_SUNS
%     edge_Masks_SUNS(:,:,nn) = edge(Masks_SUNS(:,:,nn));
% end

load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '_added.mat']), 'Masks');
Masks_SUNS_MF = Masks;
N_SUNS_MF = size(Masks_SUNS_MF,3);
N_MF = N_SUNS_MF - N_SUNS;
Masks_MF = Masks_SUNS_MF(:,:,N_SUNS+1:N_SUNS_MF);
% edge_Masks_MF = 0*Masks_MF;
% for nn = 1:num_MF
%     edge_Masks_MF(:,:,nn) = edge(Masks_MF(:,:,nn));
% end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS_MF,0.5);
% Masks = permute(Masks,[2,1,3]);
% Masks_SUNS_sum = sum(Masks_SUNS,3);
% edge_Masks_SUNS_sum = sum(edge_Masks_SUNS,3);
% Masks_SUNS_sum_mag=kron(Masks_SUNS_sum,mag_kernel_bool);
% edge_Masks_SUNS_sum_mag=kron(edge_Masks_SUNS_sum,mag_kernel_bool);
% Masks_MF_sum = sum(Masks_MF,3);
% edge_Masks_MF_sum = sum(edge_Masks_MF,3);
% Masks_MF_sum_mag=kron(Masks_MF_sum,mag_kernel_bool);
% edge_Masks_MF_sum_mag=kron(edge_Masks_MF_sum,mag_kernel_bool);
if mag > 1
    Masks_SUNS_mag=zeros(Lxm,Lym,N_SUNS,'logical');
    Masks_MF_mag=zeros(Lxm,Lym,N_MF,'logical');
    for n = 1:N_SUNS
        Masks_SUNS_mag(:,:,n) = kron(Masks_SUNS(:,:,n),mag_kernel_bool);
        Masks_MF_mag(:,:,n) = kron(Masks_MF(:,:,n),mag_kernel_bool);
    end
else
    Masks_SUNS_mag = Masks_SUNS;
    Masks_MF_mag = Masks_MF;
end

TP_1 = sum(m,1)>0;
TP_12 = sum(m,2)>0;
FP_1 = sum(m,1)==0;
FN_1 = sum(m,2)==0;
masks_TP = sum(Masks_SUNS_MF(:,:,TP_1),3);
masks_FP = sum(Masks_SUNS_MF(:,:,FP_1),3);
masks_FN = sum(GT_Masks(:,:,FN_1),3);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,1)
figure('Position',[50,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange_mag,yrange_mag), 'EdgeColor',color(3,:),'LineWidth',0.5);
% contour(Masks_SUNS_sum_mag(xrange_mag,yrange_mag), 'EdgeColor',color(5,:),'LineWidth',0.5);
% contour(Masks_MF_sum_mag(xrange_mag,yrange_mag), 'EdgeColor',color(6,:),'LineWidth',0.5);
for n = 1:NGT
    temp = GT_Masks_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_SUNS
    temp = Masks_SUNS_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(5,:),'LineWidth',0.5);
    end
end
for n = 1:N_MF
    temp = Masks_MF_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(6,:),'LineWidth',0.5);
    end
end
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(3,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange_mag,yrange_mag)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(1,:),1,1,3);
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(5,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_Masks_SUNS_sum_mag(xrange_mag,yrange_mag)));  
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(6,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_Masks_MF_sum_mag(xrange_mag,yrange_mag)));  

if ~image_only
    title(sprintf('%s, SUNS + missing finder, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',[140,3,16,4],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,[save_folder, data_name, ' Masks SUNS_MF ',list_title{k},' ',mat2str(SNR_range),'.tif']);
    else
        saveas(gcf,[save_folder, data_name, ' Masks SUNS_MF ',list_title{k},' ',mat2str(SNR_range),'.png']);
    end
    % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
img_all=getframe(gcf);
cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
zoom1 = cdata(rect1_sub(2)+1:rect1_sub(2)+rect1_sub(4)-1,rect1_sub(1)+1:rect1_sub(1)+rect1_sub(3)-1,:);
% zoom1(2:5,end-1,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end

zoom2 = cdata(rect2_sub(2)+1:rect2_sub(2)+rect2_sub(4)-1,rect2_sub(1)+1:rect2_sub(1)+rect2_sub(3)-1,:);
% zoom2(3:4,3:10,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop,size(zoom2,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel_uint8);
end

zoom3 = cdata(rect3_sub(2)+1:rect3_sub(2)+rect3_sub(4)-1,rect3_sub(1)+1:rect3_sub(1)+rect3_sub(3)-1,:);
% zoom3(3:4,3:10,:) = 255;
zoom3_mag=zeros(size(zoom3,1)*mag_crop,size(zoom3,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom3_mag(:,:,kc)=kron(zoom3(:,:,kc),mag_kernel_uint8);
end

if save_figures
    imwrite(zoom1_mag,fullfile(save_folder,[data_name,' Masks SUNS_MF ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom2_mag,fullfile(save_folder,[data_name,' Masks SUNS_MF ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom3_mag,fullfile(save_folder,[data_name,' Masks SUNS_MF ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
%     imwrite(cdata,[save_folder, data_name,' Masks SUNS_MF ',list_title{k},'.tif']);
end


%% SUNS TUnCaT
dir_output_mask = fullfile(dir_video,'complete_TUnCaT_SF25\4816[1]th5\output_masks');
% load([dir_output_mask, Exp_ID, '.mat'], 'Masks_2');
% Masks = reshape(full(Masks_2'),487,487,[]);
load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
N_SUNS = size(Masks_SUNS,3);
% edge_Masks_SUNS = 0*Masks_SUNS;
% for nn = 1:size(Masks_SUNS,3)
%     edge_Masks_SUNS(:,:,nn) = edge(Masks_SUNS(:,:,nn));
% end
if mag > 1
    Masks_SUNS_mag=zeros(Lxm,Lym,N_SUNS,'logical');
    for n = 1:N_SUNS
        Masks_SUNS_mag(:,:,n) = kron(Masks_SUNS(:,:,n),mag_kernel_bool);
    end
else
    Masks_SUNS_mag = Masks_SUNS;
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS,0.5);
% Masks = permute(Masks,[2,1,3]);
% Masks_SUNS_sum = sum(Masks_SUNS,3);
% edge_Masks_SUNS_sum = sum(edge_Masks_SUNS,3);
% Masks_SUNS_sum_mag=kron(Masks_SUNS_sum,mag_kernel_bool);
% edge_Masks_SUNS_sum_mag=kron(edge_Masks_SUNS_sum,mag_kernel_bool);

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
figure('Position',[400,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'EdgeColor',color(3,:),'LineWidth',0.5);
% contour(Masks_SUNS_sum_mag(xrange,yrange), 'EdgeColor',color(5,:),'LineWidth',0.5);
for n = 1:NGT
    temp = GT_Masks_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_SUNS
    temp = Masks_SUNS_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(5,:),'LineWidth',0.5);
    end
end
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(3,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange_mag,yrange_mag)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(1,:),1,1,3);
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(5,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_Masks_SUNS_sum_mag(xrange_mag,yrange_mag)));  

if ~image_only
    title(sprintf('%s, SUNS TUnCaT, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',1);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,[save_folder, data_name, ' Masks SUNS_TUnCaT ',list_title{k},' ',mat2str(SNR_range),'.tif']);
    else
        saveas(gcf,[save_folder, data_name, ' Masks SUNS_TUnCaT ',list_title{k},' ',mat2str(SNR_range),'.png']);
    end
    % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
img_all=getframe(gcf);
cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
zoom1 = cdata(rect1_sub(2)+1:rect1_sub(2)+rect1_sub(4)-1,rect1_sub(1)+1:rect1_sub(1)+rect1_sub(3)-1,:);
zoom1(3:4,3:10,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end

zoom2 = cdata(rect2_sub(2)+1:rect2_sub(2)+rect2_sub(4)-1,rect2_sub(1)+1:rect2_sub(1)+rect2_sub(3)-1,:);
zoom2(3:4,3:10,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop,size(zoom2,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel_uint8);
end

zoom3 = cdata(rect3_sub(2)+1:rect3_sub(2)+rect3_sub(4)-1,rect3_sub(1)+1:rect3_sub(1)+rect3_sub(3)-1,:);
zoom3(3:4,3:10,:) = 255;
zoom3_mag=zeros(size(zoom3,1)*mag_crop,size(zoom3,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom3_mag(:,:,kc)=kron(zoom3(:,:,kc),mag_kernel_uint8);
end

if save_figures
    imwrite(zoom1_mag,fullfile(save_folder,[data_name,' Masks SUNS_TUnCaT ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom2_mag,fullfile(save_folder,[data_name,' Masks SUNS_TUnCaT ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom3_mag,fullfile(save_folder,[data_name,' Masks SUNS_TUnCaT ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
%     imwrite(cdata,[save_folder, data_name,' Masks SUNS_TUnCaT ',list_title{k},'.tif']);
end


%% SUNS FISSA
dir_output_mask = fullfile(dir_video,'complete_FISSA_SF25\4816[1]th3\output_masks');
% load([dir_output_mask, Exp_ID, '.mat'], 'Masks_2');
% Masks = reshape(full(Masks_2'),487,487,[]);
load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
N_SUNS = size(Masks_SUNS,3);
% edge_Masks_SUNS = 0*Masks_SUNS;
% for nn = 1:size(Masks_SUNS,3)
%     edge_Masks_SUNS(:,:,nn) = edge(Masks_SUNS(:,:,nn));
% end
if mag > 1
    Masks_SUNS_mag=zeros(Lxm,Lym,N_SUNS,'logical');
    for n = 1:N_SUNS
        Masks_SUNS_mag(:,:,n) = kron(Masks_SUNS(:,:,n),mag_kernel_bool);
    end
else
    Masks_SUNS_mag = Masks_SUNS;
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS,0.5);
% Masks = permute(Masks,[2,1,3]);
% Masks_SUNS_sum = sum(Masks_SUNS,3);
% edge_Masks_SUNS_sum = sum(edge_Masks_SUNS,3);
% Masks_SUNS_sum_mag=kron(Masks_SUNS_sum,mag_kernel_bool);
% edge_Masks_SUNS_sum_mag=kron(edge_Masks_SUNS_sum,mag_kernel_bool);

TP_3 = sum(m,1)>0;
TP_32 = sum(m,2)>0;
FP_3 = sum(m,1)==0;
FN_3 = sum(m,2)==0;
masks_TP = sum(Masks_SUNS(:,:,TP_3),3);
masks_FP = sum(Masks_SUNS(:,:,FP_3),3);
masks_FN = sum(GT_Masks(:,:,FN_3),3);
    
% Style 2: Three colors
% figure(98)
% subplot(2,3,1)
figure('Position',[750,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'EdgeColor',color(3,:),'LineWidth',0.5);
% contour(Masks_SUNS_sum_mag(xrange,yrange), 'EdgeColor',color(4,:),'LineWidth',0.5);
for n = 1:NGT
    temp = GT_Masks_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_SUNS
    temp = Masks_SUNS_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(4,:),'LineWidth',0.5);
    end
end
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(3,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange_mag,yrange_mag)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(4,:),1,1,3);
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(4,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_Masks_SUNS_sum_mag(xrange_mag,yrange_mag)));  

if ~image_only
    title(sprintf('%s, SUNS FISSA, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',1);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,[save_folder, data_name, ' Masks SUNS_FISSA ',list_title{k},' ',mat2str(SNR_range),'.tif']);
    else
        saveas(gcf,[save_folder, data_name, ' Masks SUNS_FISSA ',list_title{k},' ',mat2str(SNR_range),'.png']);
    end
    % saveas(gcf,['figure 2\',Exp_ID,' SUNS noSF h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
img_all=getframe(gcf);
cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
zoom1 = cdata(rect1_sub(2)+1:rect1_sub(2)+rect1_sub(4)-1,rect1_sub(1)+1:rect1_sub(1)+rect1_sub(3)-1,:);
% zoom1(2:5,end-1,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end

zoom2 = cdata(rect2_sub(2)+1:rect2_sub(2)+rect2_sub(4)-1,rect2_sub(1)+1:rect2_sub(1)+rect2_sub(3)-1,:);
% zoom1(2:5,end-1,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop,size(zoom2,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel_uint8);
end

zoom3 = cdata(rect3_sub(2)+1:rect3_sub(2)+rect3_sub(4)-1,rect3_sub(1)+1:rect3_sub(1)+rect3_sub(3)-1,:);
% zoom3(3:4,3:10,:) = 255;
zoom3_mag=zeros(size(zoom3,1)*mag_crop,size(zoom3,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom3_mag(:,:,kc)=kron(zoom3(:,:,kc),mag_kernel_uint8);
end


if save_figures
    imwrite(zoom1_mag,fullfile(save_folder,[data_name,' Masks SUNS_FISSA ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom2_mag,fullfile(save_folder,[data_name,' Masks SUNS_FISSA ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom3_mag,fullfile(save_folder,[data_name,' Masks SUNS_FISSA ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
%     imwrite(cdata,[save_folder, data_name,' Masks SUNS_FISSA ',list_title{k},'.tif']);
end


%% MIN1PIPE
% load(fullfile(dir_MIN1PIPE,['eval_',data_name,'_thb ',saved_date_MIN1PIPE,'.mat']),'Table_time_ext');
% best_param = Table_time_ext(k,1:end-4);
% load(fullfile(dir_MIN1PIPE,['eval_',data_name,'_thb history ',saved_date_MIN1PIPE,'.mat']),'best_history');
% best_param = best_history(end,1:end-4);
load(fullfile(dir_MIN1PIPE,['eval_',data_name,'_thb ',saved_date_MIN1PIPE,'.mat']),'Table_time_ext');
best_param = Table_time_ext(k,1:end-8);
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
% dir_output_mask = fullfile(dir_video,'min1pipe\pss=0.80_psc=0.80_mrc=0.70');
% dir_output_mask = fullfile(dir_video,'min1pipe\pss=0.80_psc=0.70_mrc=0.50_dt=0.15_kappa=0.40_se=8');
load(fullfile(dir_output_mask, [Exp_ID, '_data_processed.mat']), 'roifn');
roi3 = reshape(full(roifn), Lx,Ly, []);
Masks_min1 = threshold_Masks(roi3, thb); %;%
N_min1 = size(Masks_min1,3);
% roib = roifn>0.2*max(roifn,[],1); %;%
% Masks_min1 = reshape(roib,Lx,Ly,[]);
% edge_Masks_min1 = 0*Masks_min1;
% for nn = 1:size(Masks_min1,3)
%     edge_Masks_min1(:,:,nn) = edge(Masks_min1(:,:,nn));
% end
if mag > 1
    Masks_SUNS_mag=zeros(Lxm,Lym,N_min1,'logical');
    for n = 1:N_min1
        Masks_min1_mag(:,:,n) = kron(Masks_min1(:,:,n),mag_kernel_bool);
    end
else
    Masks_min1_mag = Masks_min1;
end

% Masks_min1 = permute(Masks_min1,[2,1,3]);
[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_min1,0.5);
% Masks_min1_sum = sum(Masks_min1,3);
% edge_Masks_min1_sum = sum(edge_Masks_min1,3);
% Masks_min1_sum_mag=kron(Masks_min1_sum,mag_kernel_bool);
% edge_Masks_min1_sum_mag=kron(edge_Masks_min1_sum,mag_kernel_bool);

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
figure('Position',[1100,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range);
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'EdgeColor',color(3,:),'LineWidth',0.5);
% contour(Masks_min1_sum_mag(xrange,yrange), 'EdgeColor',color(1,:),'LineWidth',0.5);
for n = 1:NGT
    temp = GT_Masks_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_min1
    temp = Masks_min1_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(1,:),'LineWidth',0.5);
    end
end
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(3,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange_mag,yrange_mag)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(17,:),1,1,3);
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(1,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_Masks_min1_sum_mag(xrange_mag,yrange_mag)));  

if ~image_only
    title(sprintf('%s, MIN1PIPE, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',1);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,[save_folder, data_name, ' Masks min1pipe ',list_title{k},' ',mat2str(SNR_range),'.tif']);
    else
        saveas(gcf,[save_folder, data_name, ' Masks min1pipe ',list_title{k},' ',mat2str(SNR_range),'.png']);
    end
    % saveas(gcf,['figure 2\',Exp_ID,' CaImAn Batch h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
img_all=getframe(gcf);
cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
zoom1 = cdata(rect1_sub(2)+1:rect1_sub(2)+rect1_sub(4)-1,rect1_sub(1)+1:rect1_sub(1)+rect1_sub(3)-1,:);
% zoom1(2:5,end-1,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end

zoom2 = cdata(rect2_sub(2)+1:rect2_sub(2)+rect2_sub(4)-1,rect2_sub(1)+1:rect2_sub(1)+rect2_sub(3)-1,:);
% zoom1(2:5,end-1,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop,size(zoom2,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel_uint8);
end

zoom3 = cdata(rect3_sub(2)+1:rect3_sub(2)+rect3_sub(4)-1,rect3_sub(1)+1:rect3_sub(1)+rect3_sub(3)-1,:);
% zoom3(3:4,3:10,:) = 255;
zoom3_mag=zeros(size(zoom3,1)*mag_crop,size(zoom3,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom3_mag(:,:,kc)=kron(zoom3(:,:,kc),mag_kernel_uint8);
end

if save_figures
    imwrite(zoom1_mag,fullfile(save_folder,[data_name,' Masks min1pipe ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom2_mag,fullfile(save_folder,[data_name,' Masks min1pipe ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom3_mag,fullfile(save_folder,[data_name,' Masks min1pipe ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
%     imwrite(cdata,[save_folder, data_name,' Masks min1pipe ',list_title{k},'.tif']);
end


%% CNMF-E
% load(fullfile(dir_CNMFE,['eval_',data_name,'_thb ',saved_date_CNMFE,' cv.mat']),'Table_time_ext');
% best_param = Table_time_ext(k,1:end-4);
% load(fullfile(dir_CNMFE,['eval_',data_name,'_thb history ',saved_date_CNMFE,'.mat']),'best_history');
% best_param = best_history(end,1:end-4);
load(fullfile(dir_CNMFE,['eval_',data_name,'_thb ',saved_date_CNMFE,'.mat']),'Table_time_ext');
best_param = Table_time_ext(k,1:end-8);
gSiz = 2 * radius; % 12;
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
N_cnmfe = size(Masks_cnmfe,3);
% dir_output_mask = fullfile(dir_video,'CNMFE\gSiz=18,rbg=1.5,nk=2,rdmin=2.0,mc=0.20,mp=2,mt=0.20,mts=0.50,mtt=0.10');
% load(fullfile(dir_output_mask, [Exp_ID, '_Masks_0.4.mat']), 'Masks3');
% Masks3 = Masks3>0.2*max(Masks3,[],1); %;%
% Masks_cnmfe = reshape(Masks3,Lx,Ly,[]);
% edge_Masks_cnmfe = 0*Masks_cnmfe;
% for nn = 1:size(Masks_cnmfe,3)
%     edge_Masks_cnmfe(:,:,nn) = edge(Masks_cnmfe(:,:,nn));
% end
if mag > 1
    Masks_cnmfe_mag=zeros(Lxm,Lym,N_cnmfe,'logical');
    for n = 1:N_cnmfe
        Masks_cnmfe_mag(:,:,n) = kron(Masks_cnmfe(:,:,n),mag_kernel_bool);
    end
else
    Masks_cnmfe_mag = Masks_cnmfe;
end

% Masks_cnmfe = permute(Masks_cnmfe,[2,1,3]);
[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_cnmfe,0.5);
% Masks_cnmfe_sum = sum(Masks_cnmfe,3);
% edge_Masks_cnmfe_sum = sum(edge_Masks_cnmfe,3);
% Masks_cnmfe_sum_mag=kron(Masks_cnmfe_sum,mag_kernel_bool);
% edge_Masks_cnmfe_sum_mag=kron(edge_Masks_cnmfe_sum,mag_kernel_bool);

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
figure('Position',[1450,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range);
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;
% contour(masks_TP(xrange,yrange), 'Color', green);
% contour(masks_FP(xrange,yrange), 'Color', red);
% contour(masks_FN(xrange,yrange), 'Color', blue);
% contour(GT_Masks_sum_mag(xrange,yrange), 'EdgeColor',color(3,:),'LineWidth',0.5);
% contour(Masks_cnmfe_sum_mag(xrange,yrange), 'EdgeColor',color(2,:),'LineWidth',0.5);
for n = 1:NGT
    temp = GT_Masks_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_cnmfe
    temp = Masks_cnmfe_mag(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(2,:),'LineWidth',0.5);
    end
end
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(3,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_GT_Masks_sum_mag(xrange_mag,yrange_mag)));  
% alphaImg = ones(Lx*mag,Ly*mag).*reshape(colors_multi(20,:),1,1,3);
% alphaImg = ones(Lxc_mag,Lyc_mag).*reshape(color(2,:),1,1,3);
% image(alphaImg,'Alphadata',alpha*(edge_Masks_cnmfe_sum_mag(xrange_mag,yrange_mag)));  

if ~image_only
    title(sprintf('%s, CNMF-E, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',1);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',1);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,[save_folder, data_name, ' Masks CNMF-E ',list_title{k},' ',mat2str(SNR_range),'.tif']);
    else
        saveas(gcf,[save_folder, data_name, ' Masks CNMF-E ',list_title{k},' ',mat2str(SNR_range),'.png']);
    end
    % saveas(gcf,['figure 2\',Exp_ID,' CaImAn Batch h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
img_all=getframe(gcf);
cdata=img_all.cdata;
% % cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
zoom1 = cdata(rect1_sub(2)+1:rect1_sub(2)+rect1_sub(4)-1,rect1_sub(1)+1:rect1_sub(1)+rect1_sub(3)-1,:);
% zoom1(2:5,end-1,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end

zoom2 = cdata(rect2_sub(2)+1:rect2_sub(2)+rect2_sub(4)-1,rect2_sub(1)+1:rect2_sub(1)+rect2_sub(3)-1,:);
% zoom1(2:5,end-1,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop,size(zoom2,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel_uint8);
end

zoom3 = cdata(rect3_sub(2)+1:rect3_sub(2)+rect3_sub(4)-1,rect3_sub(1)+1:rect3_sub(1)+rect3_sub(3)-1,:);
% zoom3(3:4,3:10,:) = 255;
zoom3_mag=zeros(size(zoom3,1)*mag_crop,size(zoom3,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom3_mag(:,:,kc)=kron(zoom3(:,:,kc),mag_kernel_uint8);
end

if save_figures
    imwrite(zoom1_mag,fullfile(save_folder,[data_name,' Masks CNMF-E ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom2_mag,fullfile(save_folder,[data_name,' Masks CNMF-E ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(zoom3_mag,fullfile(save_folder,[data_name,' Masks CNMF-E ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
%     imwrite(cdata,[save_folder, data_name,' Masks CNMF-E ',list_title{k},'.tif']);
end

%% Find the best place to present
figure('Position',[1250,750,500,300],'Color','w');
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
if ~image_only
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR','FontName','Arial');
    set(h,'FontSize',12);
end
% if save_figures
%     saveas(gcf,['colorbar_SNR ',mat2str(SNR_range),'.svg']);
%     saveas(gcf,['colorbar_SNR ',mat2str(SNR_range),'.emf']);
% %     save(['trace\',Exp_ID,' N',num2str(N_neuron),' trace.mat'],'trace_N');
% end

%% Find the best place to present
hold on;
% orange = [0.8500    0.50    0.0980];
good = (TP_22 & ~TP_42 & ~TP_52); %  & ~TP_42 & ~TP_52
Masks_good = sum(FinalMasks(:,:,good),3);
Masks_good_mag=kron(Masks_good,mag_kernel_bool);
contour(Masks_good_mag(xrange_mag,yrange_mag), 'Color', red);

better = (TP_12 & ~TP_22 & ~TP_42 & ~TP_52); %  & ~TP_42 & ~TP_52
Masks_better = sum(FinalMasks(:,:,better),3);
Masks_better_mag=kron(Masks_better,mag_kernel_bool);
contour(Masks_better_mag(xrange_mag,yrange_mag), 'Color', blue);
saveas(gcf,[save_folder, data_name, ' Masks difference ',list_title{k},' ',mat2str(SNR_range),'.png']);

end