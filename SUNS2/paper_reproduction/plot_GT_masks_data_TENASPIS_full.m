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
mag=1;
mag_kernel = ones(mag,mag);
mag_kernel_bool = logical(mag_kernel);
SNR_range = [2,14];
raw_range = [0,500];
addpath(genpath('C:\Matlab Files\STNeuroNet-master\Software'))

%% neurons and masks frame
list_Exp_ID={'Mouse_1K', 'Mouse_2K', 'Mouse_3K', 'Mouse_4K', ...
             'Mouse_1M', 'Mouse_2M', 'Mouse_3M', 'Mouse_4M'};
num_Exp = length(list_Exp_ID);
dir_parent = 'D:\data_TENASPIS\original_masks';
dir_SNR = fullfile(dir_parent,'complete_TUnCaT_SF25\network_input');
dir_GT_masks = fullfile(dir_parent,'GT Masks refined');

for k=1:num_Exp
    % clear video_SNR video_raw
    Exp_ID = list_Exp_ID{k};
    Y = h5read(fullfile(dir_parent,[Exp_ID,'.h5']),'/mov');
    video_raw = Y - min(Y,[],3);
    raw_max = max(video_raw,[],3);
    video_SNR = h5read(fullfile(dir_SNR,[Exp_ID,'.h5']),'/network_input');
    SNR_max = max(video_SNR,[],3);
    [Lx,Ly] = size(raw_max);

    load(fullfile(dir_GT_masks, ['FinalMasks_', Exp_ID, '.mat']), 'FinalMasks');
    GT_Masks = logical(FinalMasks);
%     edge_Masks = 0*FinalMasks;
%     for nn = 1:size(FinalMasks,3)
%         edge_Masks(:,:,nn) = edge(FinalMasks(:,:,nn));
%     end
%     % FinalMasks = permute(FinalMasks,[2,1,3]);
%     GT_Masks_sum = sum(GT_Masks,3);
%     edge_Masks_sum = sum(edge_Masks,3);
% 
%     %% magnify
%     SNR_max_mag = kron(SNR_max,mag_kernel);
%     [Lxm,Lym] = size(SNR_max_mag);
%     GT_Masks_sum_mag=kron(GT_Masks_sum,mag_kernel_bool);
%     edge_Masks_sum_mag=kron(edge_Masks_sum,mag_kernel_bool);
% 
%     xrange=1:Lxm; yrange=1:Lym;
%     % xrange = 334:487; yrange=132:487;
%     % xrange = 1:487; yrange=1:487;
%     crop_png=[86,64,length(yrange),length(xrange)];
% 
%     %% 
%     % Style 2: Three colors
%     % figure(98)
%     % subplot(2,3,1)
%     figure('Position',[50,50,400,300],'Color','w');
%     %     imshow(raw_max,[0,1024]);
%     imshow(SNR_max_mag(xrange,yrange),SNR_range); axis('image'); colormap gray;
%     hold on;
%     alphaImg = ones(Lx*mag,Ly*mag).*reshape(yellow,1,1,3);
%     alpha = 0.8;
%     image(alphaImg,'Alphadata',alpha*(edge_Masks_sum_mag(xrange,yrange)));  
%%
    plot_masks_color(FinalMasks,SNR_max);
    set(gca,'clim',SNR_range);
    xticklabels({}); yticklabels({});

    % title(sprintf('SUNS, F1 = %1.2f',F1),'FontSize',12);
    title(Exp_ID,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Peak SNR');
    set(h,'FontSize',12);

%     rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

    if save_figures
        saveas(gcf,[Exp_ID, ' SNR GT Masks refined.png']);
    end
%%
    plot_masks_color(FinalMasks,raw_max);
    set(gca,'clim',[prctile(raw_max,0,'all'),prctile(raw_max,95,'all')]);
    xticklabels({}); yticklabels({});

    % title(sprintf('SUNS, F1 = %1.2f',F1),'FontSize',12);
    title(Exp_ID,'Interpreter','None');
    h=colorbar;
    set(get(h,'Label'),'String','Max intensity');
    set(h,'FontSize',12);

%     rectangle('Position',[3,65,5,26],'FaceColor','w','LineStyle','None'); % 20 um scale bar

    if save_figures
        saveas(gcf,[Exp_ID, ' raw GT Masks refined.png']);
    end
end