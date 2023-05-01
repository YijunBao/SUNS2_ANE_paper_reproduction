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

save_figures = true;
image_only = false;
show_zoom = true;
% mag_kernel = ones(mag,mag,'uint8');
alpha = 0.8;

%% neurons and masks frame
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
list_title = list_Exp_ID;
rate_hz = 20; % frame rate of each video
radius = 9;
data_name = 'TENASPIS';
dir_video = fullfile('../data/data_TENASPIS/added_refined_masks');
dir_video_raw = dir_video;
% varname = '/mov';
dir_GT_masks = fullfile(dir_video,'GT Masks');

%%
for k=1:num_Exp
% clear video_SNR video_raw
Exp_ID = list_Exp_ID{k};
load(['TENASPIS mat/SNR_max/SNR_max_',Exp_ID,'.mat'],'SNR_max');
% SNR_max=SNR_max';
load(['TENASPIS mat/raw_max/raw_max_',Exp_ID,'.mat'],'raw_max');
% raw_max=raw_max';
[Lx,Ly] = size(raw_max);
% SNR_max = SNR_max(1:Lx,1:Ly);

load(fullfile(dir_GT_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
GT_Masks = logical(FinalMasks);
NGT = size(GT_Masks,3);

mag = 1;
xrange=1:Lx; yrange=1:Ly;
Lxc = length(xrange); Lyc = length(yrange); 
xrange_mag=((xrange(1)-1)*mag+1):(xrange(end)*mag); 
yrange_mag=((yrange(1)-1)*mag+1):(yrange(end)*mag);
Lxc_mag = length(xrange_mag); Lyc_mag = length(yrange_mag); 
crop_png=[86,64,Lxc,Lyc];
SNR_range = [2,14]; % [0,5]; % [0,10]; % 
raw_range = [0,4000]; % [2,6]; % [0,10]; % 
fontsize = 24;

save_folder = sprintf('figures_%d-%d,%d-%d raw',xrange(1),xrange(end),yrange(1),yrange(end));
if ~exist(save_folder,'dir')
    mkdir(save_folder)
end
% disp([min(raw_max,[],'all'),max(raw_max,[],'all')])

%% Plot raw_max
figure('Position',[100,50,700,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(raw_max(xrange,yrange),[]); % raw_range
else
    imagesc(raw_max(xrange,yrange)); % raw_range
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;
rectangle('Position',[Ly-40,10,32,10],'FaceColor','w','LineStyle','None'); % 20 um scale bar
for n = 1:NGT
    temp = GT_Masks(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end
%%
if ~image_only
    title(Exp_ID,'FontSize',fontsize,'Interpreter','None');
    set(gca,'clim',[0,min(4000,500*ceil(max(raw_max,[],'all')/500))]);
    h=colorbar('Ticks',0:500:4000);
%     if mod(k,4)
        set(get(h,'Label'),'String','Maximum Intensity');
%     end
    set(h,'FontSize',fontsize);
end
%%
if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks GT raw.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks GT raw.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' SUNS noSF h.svg']);
end

%% Plot SNR_max
figure('Position',[700,50,700,500],'Color','w');
%     imshow(raw_max,[0,1024]);
if image_only
    imshow(SNR_max(xrange,yrange),SNR_range); 
else
    imagesc(SNR_max(xrange,yrange),SNR_range); 
    axis('image'); colormap gray;
end
xticklabels({}); yticklabels({});
hold on;
rectangle('Position',[Ly-40,10,32,10],'FaceColor','w','LineStyle','None'); % 20 um scale bar
for n = 1:NGT
    temp = GT_Masks(xrange,yrange,n);
    if any(temp,'all')
        contour(temp, 'EdgeColor',color(3,:),'LineWidth',0.5);
    end
end

if ~image_only
    title(Exp_ID,'FontSize',fontsize,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',fontsize);
end

if save_figures
    if image_only
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks GT SNR.tif']));
    else
        saveas(gcf,fullfile(save_folder, [data_name,' ',Exp_ID, ' Masks GT SNR.png']));
    end
    % saveas(gcf,['figure 2/',Exp_ID,' SUNS noSF h.svg']);
end

end