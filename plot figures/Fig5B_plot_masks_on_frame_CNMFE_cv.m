addpath(genpath('../ANE'))
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
alpha = 0.8;

%% neurons and masks frame
list_data_names={'blood_vessel_10Hz','PFC4_15Hz','bma22_epm','CaMKII_120_TMT Exposure_5fps'};
list_ID_part = {'_part11', '_part12', '_part21', '_part22'};
% list_title = {'upper left', 'upper right', 'lower left', 'lower right'};
list_avg_radius = [5,6,8,14];
data_ind = 4;
data_name = list_data_names{data_ind};
dir_video = fullfile('../data/data_CNMFE',data_name);
dir_video_raw = dir_video;
dir_video_SNR = fullfile(dir_video,'SUNS_TUnCaT_SF50/network_input');
list_Exp_ID = cellfun(@(x) [data_name,x], list_ID_part,'UniformOutput',false);

num_Exp = length(list_Exp_ID);
list_title = list_Exp_ID;
dir_traces = fullfile(dir_video,'SUNS_TUnCaT_SF50/TUnCaT/alpha= 1.000');
dir_GT_masks = fullfile(dir_video,'GT Masks');
dir_MIN1PIPE = fullfile(dir_video,'min1pipe');
dir_CNMFE = fullfile(dir_video,'CNMFE');
dir_sub_min1pipe = 'cv_save_20221215';
dir_sub_CNMFE = 'cv_save_20221215';
saved_date_MIN1PIPE = '20221215 cv test';
saved_date_CNMFE = '20221215 cv test';
mag=1;
mag_crop=2;
%%
for k=4%:num_Exp_ID
% clear video_SNR video_raw
Exp_ID = list_Exp_ID{k};
load(['CNMFE mat/SNR_max/SNR_max_',Exp_ID,'.mat'],'SNR_max');
% SNR_max=SNR_max';
load(['CNMFE mat/raw_max/raw_max_',Exp_ID,'.mat'],'raw_max');
% raw_max=raw_max';
load(fullfile(dir_traces,[Exp_ID,'.mat']),'traces_nmfdemix'); % raw_traces
unmixed_traces = traces_nmfdemix;
[Lx,Ly] = size(raw_max);
SNR_max = SNR_max(1:Lx,1:Ly);

load(fullfile(dir_GT_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
GT_Masks = logical(FinalMasks);
NGT = size(GT_Masks,3);

% magnify
% mag=4;
mag_kernel = ones(mag,mag,class(SNR_max));
mag_kernel_uint8 = ones(mag_crop,mag_crop,'uint8');
mag_kernel_bool = logical(mag_kernel);
SNR_max_mag = kron(SNR_max,mag_kernel);
[Lxm,Lym] = size(SNR_max_mag);
% GT_Masks_sum_mag=kron(GT_Masks_sum,mag_kernel_bool);
if mag > 1
    GT_Masks_mag=zeros(Lxm,Lym,NGT,'logical');
    for n = 1:NGT
        GT_Masks_mag(:,:,n) = kron(GT_Masks(:,:,n),mag_kernel_bool);
    end
else
    GT_Masks_mag = GT_Masks;
end

xrange=1:Lx; yrange=1:Ly;
Lxc = length(xrange); Lyc = length(yrange); 
xrange_mag=((xrange(1)-1)*mag+1):(xrange(end)*mag); 
yrange_mag=((yrange(1)-1)*mag+1):(yrange(end)*mag);
Lxc_mag = length(xrange_mag); Lyc_mag = length(yrange_mag); 
crop_png=[86,64,Lxc,Lyc];
N_neuron1 = 6; % 378; % 
N_neuron2 = 9; % 
N_neuron3 = 27; % 
rect1=[36,96,80,60];
rect2=[74,36,60,60];
rect3=[134,31,55,90];
rect1_sub=rect1 - [yrange(1)-1,xrange(1)-1,0,0];
rect2_sub=rect2 - [yrange(1)-1,xrange(1)-1,0,0];
rect3_sub=rect3 - [yrange(1)-1,xrange(1)-1,0,0];
SNR_range = [0,8]; % [2,14]; % 

save_folder = sprintf('figures_%d-%d,%d-%d crop',xrange(1),xrange(end),yrange(1),yrange(end));
if ~exist(save_folder,'dir')
    mkdir(save_folder)
end

%% Plot trace of neuron 1
% yrange_zoom = rect1(1):rect1(1)+rect1(3);
% xrange_zoom = rect1(2):rect1(2)+rect1(4);
% COM1 = rect1(1:2) + rect1(3:4)/2;
% COM1 = COM1 + [xrange(1), yrange(1)]-1;
% N_neuron1 = find(FinalMasks(COM1(1),COM1(2),:));
trace_N = unmixed_traces(:,N_neuron1);
[PSNR, loc] = max(trace_N);
high = find(trace_N > min(PSNR*0.8,PSNR-2));
inds = clusterdata(high,0.5);
if max(inds) > 1
    class_highest = inds(high==loc);
    high = high(inds == class_highest);
end
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
SNR_max(start(1):start(1)+count(1)-1,start(2):start(2)+count(2)-1,:) = SNR_high_mean;

figure('Position',[100,650,300,250],'Color','w');
plot(trace_N,'Color',color(3,:),'LineWidth',2);
ylim([-2,16]);
hold on;
plot(high,trace_N(high),'Color',color(7,:),'LineWidth',2);
pos1=1200;
pos2=5;
plot(pos1+[0,250],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+5*[0,1],'k','LineWidth',2);
% text(pos1-20,pos2+0.5,{'SNR','1'},'HorizontalAlignment','right','FontSize',14); % ,'rotation',90
text(pos1-100,pos2+2.5,'SNR=5','HorizontalAlignment','center','rotation',90,'FontSize',14); % 
text(pos1,pos2-1.2,'50 s','FontSize',14); % ,'rotation',0
if save_figures
    saveas(gcf,fullfile(save_folder, [data_name,' ',list_title{k}, ' trace ',num2str(N_neuron1),'.png']));
    saveas(gcf,fullfile(save_folder, [data_name,' ',list_title{k}, ' trace ',num2str(N_neuron1),'.emf']));
end

%% Plot trace of neuron 2
% yrange_zoom = rect2(1):rect2(1)+rect2(3);
% xrange_zoom = rect2(2):rect2(2)+rect2(4);
% COM1 = rect1(1:2) + rect1(3:4)/2;
% COM1 = COM1 + [xrange(1), yrange(1)]-1;
% N_neuron2 = find(FinalMasks(COM1(1),COM1(2),:));
trace_N = unmixed_traces(:,N_neuron2);
[PSNR, loc] = max(trace_N);
high = find(trace_N > min(PSNR*0.8,PSNR-2));
inds = clusterdata(high,0.5);
if max(inds) > 1
    class_highest = inds(high==loc);
    high = high(inds == class_highest);
end
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
SNR_max(start(1):start(1)+count(1)-1,start(2):start(2)+count(2)-1,:) = SNR_high_mean;

figure('Position',[500,650,300,250],'Color','w');
plot(trace_N,'Color',color(3,:),'LineWidth',2);
ylim([-2,16]);
hold on;
plot(high,trace_N(high),'Color',color(7,:),'LineWidth',2);
pos1=1000;
pos2=7;
plot(pos1+[0,250],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+5*[0,1],'k','LineWidth',2);
% text(pos1-20,pos2+0.5,{'SNR','1'},'HorizontalAlignment','right','FontSize',14); % ,'rotation',90
text(pos1-100,pos2+2.5,'SNR=5','HorizontalAlignment','center','rotation',90,'FontSize',14); % 
text(pos1,pos2-1.2,'50 s','FontSize',14); % ,'rotation',0
if save_figures
    saveas(gcf,fullfile(save_folder, [data_name,' ',list_title{k}, ' trace ',num2str(N_neuron2),'.png']));
    saveas(gcf,fullfile(save_folder, [data_name,' ',list_title{k}, ' trace ',num2str(N_neuron2),'.emf']));
end

%% Plot trace of neuron 3
% yrange_zoom = rect2(1):rect2(1)+rect2(3);
% xrange_zoom = rect2(2):rect2(2)+rect2(4);
% COM1 = rect1(1:2) + rect1(3:4)/2;
% COM1 = COM1 + [xrange(1), yrange(1)]-1;
% N_neuron3 = find(FinalMasks(COM1(1),COM1(2),:));
trace_N = unmixed_traces(:,N_neuron3);
[PSNR, loc] = max(trace_N);
high = find(trace_N > min(PSNR*0.8,PSNR-2));
inds = clusterdata(high,0.5);
if max(inds) > 1
    class_highest = inds(high==loc);
    high = high(inds == class_highest);
end
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
SNR_max(start(1):start(1)+count(1)-1,start(2):start(2)+count(2)-1,:) = SNR_high_mean;

figure('Position',[900,650,300,250],'Color','w');
plot(trace_N,'Color',color(3,:),'LineWidth',2);
ylim([-2,16]);
hold on;
plot(high,trace_N(high),'Color',color(7,:),'LineWidth',2);
pos1=1200;
pos2=5;
plot(pos1+[0,250],pos2*[1,1],'k','LineWidth',2);
plot(pos1*[1,1],pos2+5*[0,1],'k','LineWidth',2);
% text(pos1-20,pos2+0.5,{'SNR','1'},'HorizontalAlignment','right','FontSize',14); % ,'rotation',90
text(pos1-100,pos2+2.5,'SNR=5','HorizontalAlignment','center','rotation',90,'FontSize',14); % 
text(pos1,pos2-1.2,'50 s','FontSize',14); % ,'rotation',0
if save_figures
    saveas(gcf,fullfile(save_folder, [data_name,' ',list_title{k}, ' trace ',num2str(N_neuron3),'.png']));
    saveas(gcf,fullfile(save_folder, [data_name,' ',list_title{k}, ' trace ',num2str(N_neuron3),'.emf']));
end

%%
SNR_max_mag = kron(SNR_max,mag_kernel);

%% SUNS + ANE
dir_output_mask_SUNS = fullfile(dir_video,'SUNS_TUnCaT_SF50/4816[1]th4/output_masks');
dir_output_mask = fullfile(dir_output_mask_SUNS,'add_new_blockwise/trained dropout 0.8exp(-8)');

load(fullfile(dir_output_mask_SUNS, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
N_SUNS = size(Masks_SUNS,3);

load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '_added.mat']), 'Masks');
Masks_SUNS_ANE = Masks;
N_SUNS_ANE = size(Masks_SUNS_ANE,3);
N_MF = N_SUNS_ANE - N_SUNS;
Masks_MF = Masks_SUNS_ANE(:,:,N_SUNS+1:N_SUNS_ANE);

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS_ANE,0.5);

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
masks_TP = sum(Masks_SUNS_ANE(:,:,TP_1),3);
masks_FP = sum(Masks_SUNS_ANE(:,:,FP_1),3);
masks_FN = sum(GT_Masks(:,:,FN_1),3);
    
figure('Position',[50,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range,'border','tight'); 
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;

for n = 1:NGT
    temp = GT_Masks_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_SUNS
    temp = Masks_SUNS_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(5,:),'LineWidth',0.5);
    end
end
for n = 1:N_MF
    temp = Masks_MF_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(6,:),'LineWidth',0.5);
    end
end

if ~image_only
    title(sprintf('%s, SUNS + missing finder, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',[215,182,20,6],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks SUNS_ANE ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks SUNS_ANE ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' SUNS noSF h.svg']);
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
    imwrite(permute(cdata,[2,1,3]),fullfile(save_folder,[' Masks SUNS_ANE ',...
        list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom1_mag,[2,1,3]),fullfile(save_folder,[' Masks SUNS_ANE ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom2_mag,[2,1,3]),fullfile(save_folder,[' Masks SUNS_ANE ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom3_mag,[2,1,3]),fullfile(save_folder,[' Masks SUNS_ANE ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
end


%% SUNS TUnCaT
dir_output_mask = fullfile(dir_video,'SUNS_TUnCaT_SF50/4816[1]th4/output_masks');
load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
N_SUNS = size(Masks_SUNS,3);
if mag > 1
    Masks_SUNS_mag=zeros(Lxm,Lym,N_SUNS,'logical');
    for n = 1:N_SUNS
        Masks_SUNS_mag(:,:,n) = kron(Masks_SUNS(:,:,n),mag_kernel_bool);
    end
else
    Masks_SUNS_mag = Masks_SUNS;
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS,0.5);

TP_2 = sum(m,1)>0;
TP_22 = sum(m,2)>0;
FP_2 = sum(m,1)==0;
FN_2 = sum(m,2)==0;
masks_TP = sum(Masks_SUNS(:,:,TP_2),3);
masks_FP = sum(Masks_SUNS(:,:,FP_2),3);
masks_FN = sum(GT_Masks(:,:,FN_2),3);
    
figure('Position',[400,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range,'border','tight'); 
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;
for n = 1:NGT
    temp = GT_Masks_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_SUNS
    temp = Masks_SUNS_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(5,:),'LineWidth',0.5);
    end
end

if ~image_only
    title(sprintf('%s, SUNS TUnCaT, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',2);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks SUNS_TUnCaT ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks SUNS_TUnCaT ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' SUNS noSF h.svg']);
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
zoom1(end-6:end-3,end-22:end-3,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end

zoom2 = cdata(rect2_sub(2)+1:rect2_sub(2)+rect2_sub(4)-1,rect2_sub(1)+1:rect2_sub(1)+rect2_sub(3)-1,:);
zoom2(end-6:end-3,end-22:end-3,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop,size(zoom2,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel_uint8);
end

zoom3 = cdata(rect3_sub(2)+1:rect3_sub(2)+rect3_sub(4)-1,rect3_sub(1)+1:rect3_sub(1)+rect3_sub(3)-1,:);
zoom3(end-6:end-3,end-22:end-3,:) = 255;
zoom3_mag=zeros(size(zoom3,1)*mag_crop,size(zoom3,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom3_mag(:,:,kc)=kron(zoom3(:,:,kc),mag_kernel_uint8);
end

if save_figures
    imwrite(permute(cdata,[2,1,3]),fullfile(save_folder,[' Masks SUNS_TUnCaT ',...
        list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom1_mag,[2,1,3]),fullfile(save_folder,[' Masks SUNS_TUnCaT ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom2_mag,[2,1,3]),fullfile(save_folder,[' Masks SUNS_TUnCaT ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom3_mag,[2,1,3]),fullfile(save_folder,[' Masks SUNS_TUnCaT ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
end


%% SUNS FISSA
dir_output_mask = fullfile(dir_video,'SUNS_FISSA_SF50/4816[1]th2/output_masks');
load(fullfile(dir_output_mask, ['Output_Masks_', Exp_ID, '.mat']), 'Masks');
Masks_SUNS = permute(Masks,[3,2,1]); % FinalMasks; % 
N_SUNS = size(Masks_SUNS,3);
if mag > 1
    Masks_SUNS_mag=zeros(Lxm,Lym,N_SUNS,'logical');
    for n = 1:N_SUNS
        Masks_SUNS_mag(:,:,n) = kron(Masks_SUNS(:,:,n),mag_kernel_bool);
    end
else
    Masks_SUNS_mag = Masks_SUNS;
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_SUNS,0.5);

TP_3 = sum(m,1)>0;
TP_32 = sum(m,2)>0;
FP_3 = sum(m,1)==0;
FN_3 = sum(m,2)==0;
masks_TP = sum(Masks_SUNS(:,:,TP_3),3);
masks_FP = sum(Masks_SUNS(:,:,FP_3),3);
masks_FN = sum(GT_Masks(:,:,FN_3),3);
    
figure('Position',[750,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range,'border','tight'); 
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;
for n = 1:NGT
    temp = GT_Masks_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_SUNS
    temp = Masks_SUNS_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(4,:),'LineWidth',0.5);
    end
end

if ~image_only
    title(sprintf('%s, SUNS FISSA, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',2);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks SUNS_FISSA ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks SUNS_FISSA ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' SUNS noSF h.svg']);
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
    imwrite(permute(cdata,[2,1,3]),fullfile(save_folder,[' Masks SUNS_FISSA ',...
        list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom1_mag,[2,1,3]),fullfile(save_folder,[' Masks SUNS_FISSA ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom2_mag,[2,1,3]),fullfile(save_folder,[' Masks SUNS_FISSA ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom3_mag,[2,1,3]),fullfile(save_folder,[' Masks SUNS_FISSA ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
end


%% MIN1PIPE
dir_output_mask = fullfile(dir_video,'min1pipe',dir_sub_min1pipe);
load(fullfile(dir_output_mask, [Exp_ID, '_Masks.mat']), 'Masks3');
% load(fullfile(dir_output_mask, [Exp_ID, '_Masks_',num2str(thb),'.mat']), 'Masks3');
Masks_min1 = Masks3;
N_min1 = size(Masks_min1,3);

if mag > 1
    Masks_min1_mag=zeros(Lxm,Lym,N_min1,'logical');
    for n = 1:N_min1
        Masks_min1_mag(:,:,n) = kron(Masks_min1(:,:,n),mag_kernel_bool);
    end
else
    Masks_min1_mag = Masks_min1;
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_min1,0.5);

TP_4 = sum(m,1)>0;
TP_42 = sum(m,2)>0;
FP_4 = sum(m,1)==0;
FN_4 = sum(m,2)==0;
masks_TP = sum(Masks_min1(:,:,TP_4),3);
masks_FP = sum(Masks_min1(:,:,FP_4),3);
masks_FN = sum(GT_Masks(:,:,FN_4),3);
    
figure('Position',[1100,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range,'border','tight');
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;

for n = 1:NGT
    temp = GT_Masks_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_min1
    temp = Masks_min1_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(1,:),'LineWidth',0.5);
    end
end

if ~image_only
    title(sprintf('%s, MIN1PIPE, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',2);
% rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks min1pipe ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks min1pipe ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' CaImAn Batch h.svg']);
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
    imwrite(permute(cdata,[2,1,3]),fullfile(save_folder,[' Masks min1pipe ',...
        list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom1_mag,[2,1,3]),fullfile(save_folder,[' Masks min1pipe ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom2_mag,[2,1,3]),fullfile(save_folder,[' Masks min1pipe ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom3_mag,[2,1,3]),fullfile(save_folder,[' Masks min1pipe ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
end


%% CNMF-E
dir_output_mask = fullfile(dir_video,'CNMFE',dir_sub_CNMFE);
load(fullfile(dir_output_mask, [Exp_ID, '_Masks.mat']), 'Masks3');
% load(fullfile(dir_output_mask, [Exp_ID, '_Masks_',num2str(thb),'.mat']), 'Masks3');
Masks_cnmfe = Masks3;
N_cnmfe = size(Masks_cnmfe,3);

if mag > 1
    Masks_cnmfe_mag=zeros(Lxm,Lym,N_cnmfe,'logical');
    for n = 1:N_cnmfe
        Masks_cnmfe_mag(:,:,n) = kron(Masks_cnmfe(:,:,n),mag_kernel_bool);
    end
else
    Masks_cnmfe_mag = Masks_cnmfe;
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_cnmfe,0.5);

TP_5 = sum(m,1)>0;
TP_52 = sum(m,2)>0;
FP_5 = sum(m,1)==0;
FN_5 = sum(m,2)==0;
masks_TP = sum(Masks_cnmfe(:,:,TP_5),3);
masks_FP = sum(Masks_cnmfe(:,:,FP_5),3);
masks_FN = sum(GT_Masks(:,:,FN_5),3);
    
figure('Position',[1450,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range,'border','tight');
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;

for n = 1:NGT
    temp = GT_Masks_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_cnmfe
    temp = Masks_cnmfe_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(2,:),'LineWidth',0.5);
    end
end

if ~image_only
    title(sprintf('%s, CNMF-E, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',2);
% rectangle('Position',[215,182,20,6],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks CNMF-E ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks CNMF-E ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' CaImAn Batch h.svg']);
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
% zoom1(end-6:end-3,end-22:end-3,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end

zoom2 = cdata(rect2_sub(2)+1:rect2_sub(2)+rect2_sub(4)-1,rect2_sub(1)+1:rect2_sub(1)+rect2_sub(3)-1,:);
% zoom2(end-6:end-3,end-22:end-3,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop,size(zoom2,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel_uint8);
end

zoom3 = cdata(rect3_sub(2)+1:rect3_sub(2)+rect3_sub(4)-1,rect3_sub(1)+1:rect3_sub(1)+rect3_sub(3)-1,:);
% zoom3(end-6:end-3,end-22:end-3,:) = 255;
zoom3_mag=zeros(size(zoom3,1)*mag_crop,size(zoom3,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom3_mag(:,:,kc)=kron(zoom3(:,:,kc),mag_kernel_uint8);
end

if save_figures
    imwrite(permute(cdata,[2,1,3]),fullfile(save_folder,[' Masks CNMF-E ',...
        list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom1_mag,[2,1,3]),fullfile(save_folder,[' Masks CNMF-E ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom2_mag,[2,1,3]),fullfile(save_folder,[' Masks CNMF-E ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom3_mag,[2,1,3]),fullfile(save_folder,[' Masks CNMF-E ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
end


%% DeepWonder
% dir_output_mask = fullfile(dir_video,'DeepWonder',Exp_ID,'DeepWonder/mat');
% load(fullfile(dir_output_mask, ['seg_30_',Exp_ID, '_post.mat']), 'network_A_filt');
dir_output_mask = fullfile(dir_video,'DeepWonder');
load(fullfile(dir_output_mask, ['seg_30_',Exp_ID, '_post.mat']), 'network_A_filt');
Masks_DeepWonder = permute(network_A_filt,[2,1,3]); %;%
N_DeepWonder = size(Masks_DeepWonder,3);

if mag > 1
    Masks_DeepWonder_mag=zeros(Lxm,Lym,N_DeepWonder,'logical');
    for n = 1:N_DeepWonder
        Masks_DeepWonder_mag(:,:,n) = kron(Masks_DeepWonder(:,:,n),mag_kernel_bool);
    end
else
    Masks_DeepWonder_mag = Masks_DeepWonder;
end

[Recall, Precision, F1, m] = GetPerformance_Jaccard(dir_GT_masks,Exp_ID,Masks_DeepWonder,0.5);

TP_6 = sum(m,1)>0;
TP_62 = sum(m,2)>0;
FP_6 = sum(m,1)==0;
FN_6 = sum(m,2)==0;
masks_TP = sum(Masks_DeepWonder(:,:,TP_6),3);
masks_FP = sum(Masks_DeepWonder(:,:,FP_6),3);
masks_FN = sum(GT_Masks(:,:,FN_6),3);
    
figure('Position',[1450,50,600,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max_mag(xrange_mag,yrange_mag),SNR_range,'border','tight');
else
    imagesc(SNR_max_mag(xrange_mag,yrange_mag),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;

for n = 1:NGT
    temp = GT_Masks_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
for n = 1:N_DeepWonder
    temp = Masks_DeepWonder_mag(xrange_mag,yrange_mag,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',colors_multi(21,:),'LineWidth',0.5);
    end
end

if ~image_only
    title(sprintf('%s, DeepWonder, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end
rectangle('Position',rect1_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect2_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',rect3_sub,'EdgeColor',color_box,'LineWidth',2);
rectangle('Position',[215,182,20,6],'FaceColor','w','LineStyle','None'); % 20 um scale bar

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks DeepWonder ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name, ' Masks DeepWonder ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' CaImAn Batch h.svg']);
end

%% Save tif images with zoomed regions
% % figure(1)
% img_all=getframe(gcf,crop_png);
img_all=getframe(gcf);
cdata=img_all.cdata;
% cdata=permute(cdata,[2,1,3]);
% % figure; imshow(cdata);
% 
zoom1 = cdata(rect1_sub(2)+1:rect1_sub(2)+rect1_sub(4)-1,rect1_sub(1)+1:rect1_sub(1)+rect1_sub(3)-1,:);
zoom1(end-6:end-3,end-22:end-3,:) = 255;
zoom1_mag=zeros(size(zoom1,1)*mag_crop,size(zoom1,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom1_mag(:,:,kc)=kron(zoom1(:,:,kc),mag_kernel_uint8);
end

zoom2 = cdata(rect2_sub(2)+1:rect2_sub(2)+rect2_sub(4)-1,rect2_sub(1)+1:rect2_sub(1)+rect2_sub(3)-1,:);
zoom2(end-6:end-3,end-22:end-3,:) = 255;
zoom2_mag=zeros(size(zoom2,1)*mag_crop,size(zoom2,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom2_mag(:,:,kc)=kron(zoom2(:,:,kc),mag_kernel_uint8);
end

zoom3 = cdata(rect3_sub(2)+1:rect3_sub(2)+rect3_sub(4)-1,rect3_sub(1)+1:rect3_sub(1)+rect3_sub(3)-1,:);
zoom3(end-6:end-3,end-22:end-3,:) = 255;
zoom3_mag=zeros(size(zoom3,1)*mag_crop,size(zoom3,2)*mag_crop,3,'uint8');
for kc=1:3
    zoom3_mag(:,:,kc)=kron(zoom3(:,:,kc),mag_kernel_uint8);
end

if save_figures
    imwrite(permute(cdata,[2,1,3]),fullfile(save_folder,[' Masks DeepWonder ',...
        list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom1_mag,[2,1,3]),fullfile(save_folder,[' Masks DeepWonder ',...
        num2str(N_neuron1),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom2_mag,[2,1,3]),fullfile(save_folder,[' Masks DeepWonder ',...
        num2str(N_neuron2),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    imwrite(permute(zoom3_mag,[2,1,3]),fullfile(save_folder,[' Masks DeepWonder ',...
        num2str(N_neuron3),' ',list_title{k},' ',mat2str(SNR_range),'.tif']));
end

end