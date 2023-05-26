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
num_Exp = 10;
list_Exp_ID = arrayfun(@(x) ['sim_',num2str(x)],0:(num_Exp-1), 'UniformOutput',false);
list_title = list_Exp_ID;

scale_lowBG = 5e3;
scale_noise = 1;
results_folder = sprintf('lowBG=%.0e,poisson=%g',scale_lowBG,scale_noise);
list_data_names={results_folder};
data_ind = 1;
data_name = list_data_names{data_ind};
dir_video = fullfile('../data/data_simulation',data_name);
dir_video_raw = dir_video;
% varname = '/mov';
% dir_video_SNR = fullfile(dir_video,'SUNS_TUnCaT_SF25/network_input');
% dir_traces = fullfile(dir_video,'SUNS_TUnCaT_SF25/TUnCaT/alpha= 1.000');
dir_GT_masks = fullfile(dir_video,'GT Masks');
dir_MIN1PIPE = fullfile(dir_video,'min1pipe');
dir_CNMFE = fullfile(dir_video,'CNMFE');
dir_sub_min1pipe = 'cv_save_20230218';
dir_sub_CNMFE = 'cv_save_20230217';
saved_date_MIN1PIPE = '20230218 cv test'; % '20220723'; % 
saved_date_CNMFE = '20230217 cv test'; % '20220721'; % '20220809 cv';
mag=1;
mag_crop=4;
%%
for k=5%1:num_Exp
% clear video_SNR video_raw
Exp_ID = list_Exp_ID{k};
load([data_name,' mat/SNR_max/SNR_max_',Exp_ID,'.mat'],'SNR_max');
% SNR_max=SNR_max';
load([data_name,' mat/raw_max/raw_max_',Exp_ID,'.mat'],'raw_max');
% raw_max=raw_max';
% load(fullfile(dir_traces,[Exp_ID,'.mat']),'traces_nmfdemix'); % raw_traces
% unmixed_traces = traces_nmfdemix;
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
% edge_GT_Masks_sum_mag=kron(edge_GT_Masks_sum,mag_kernel_bool);

xrange=1:Lx; yrange=1:Ly;
Lxc = length(xrange); Lyc = length(yrange); 
xrange_mag=((xrange(1)-1)*mag+1):(xrange(end)*mag); 
yrange_mag=((yrange(1)-1)*mag+1):(yrange(end)*mag);
Lxc_mag = length(xrange_mag); Lyc_mag = length(yrange_mag); 
crop_png=[86,64,Lxc,Lyc];
SNR_range = [2,6]; % [0,10]; % [2,14]; % 

save_folder = sprintf('figures_%d-%d,%d-%d crop',xrange(1),xrange(end),yrange(1),yrange(end));
if ~exist(save_folder,'dir')
    mkdir(save_folder)
end

%%
SNR_max_mag = kron(SNR_max,mag_kernel);


%% SUNS TUnCaT
dir_output_mask = fullfile(dir_video,'SUNS_TUnCaT_SF25/4816[1]th4/output_masks');
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
rectangle('Position',[Ly-30,8,20,6],'FaceColor','w','LineStyle','None'); % 20 um scale bar
for n = 1:NGT
    contour(GT_Masks_mag(xrange_mag,yrange_mag,n), 'EdgeColor',color(3,:),'LineWidth',0.5);
end
for n = 1:N_SUNS
    contour(Masks_SUNS_mag(xrange_mag,yrange_mag,n), 'EdgeColor',color(5,:),'LineWidth',0.5);
end

if ~image_only
    title(sprintf('%s, SUNS TUnCaT, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks SUNS_TUnCaT ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks SUNS_TUnCaT ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' SUNS noSF h.svg']);
end


%% SUNS FISSA
dir_output_mask = fullfile(dir_video,'SUNS_FISSA_SF25/4816[1]th2/output_masks');
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
    contour(GT_Masks_mag(xrange_mag,yrange_mag,n), 'EdgeColor',color(3,:),'LineWidth',0.5);
end
for n = 1:N_SUNS
    contour(Masks_SUNS_mag(xrange_mag,yrange_mag,n), 'EdgeColor',color(4,:),'LineWidth',0.5);
end

if ~image_only
    title(sprintf('%s, SUNS FISSA, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks SUNS_FISSA ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks SUNS_FISSA ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' SUNS noSF h.svg']);
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
    contour(GT_Masks_mag(xrange_mag,yrange_mag,n), 'EdgeColor',color(3,:),'LineWidth',0.5);
end
for n = 1:N_min1
    contour(Masks_min1_mag(xrange_mag,yrange_mag,n), 'EdgeColor',color(1,:),'LineWidth',0.5);
end

if ~image_only
    title(sprintf('%s, MIN1PIPE, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks min1pipe ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks min1pipe ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' CaImAn Batch h.svg']);
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
    contour(GT_Masks_mag(xrange_mag,yrange_mag,n), 'EdgeColor',color(3,:),'LineWidth',0.5);
end
for n = 1:N_cnmfe
    contour(Masks_cnmfe_mag(xrange_mag,yrange_mag,n), 'EdgeColor',color(2,:),'LineWidth',0.5);
end

if ~image_only
    title(sprintf('%s, CNMF-E, F1 = %1.2f',Exp_ID,F1),'FontSize',12,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);
end

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks CNMF-E ',list_title{k},' ',mat2str(SNR_range),'.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks CNMF-E ',list_title{k},' ',mat2str(SNR_range),'.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' CaImAn Batch h.svg']);
end

end